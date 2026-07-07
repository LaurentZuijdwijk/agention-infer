//! Llama-family model implementation.
//!
//! Covers: Llama, Mistral, Qwen2, LFM2 — all dense transformer
//! architectures with the same layer structure:
//!
//!   attn_norm → q/k/v → rope → attention → o_proj → residual
//!   ffn_norm  → gate/up → silu_mul → down → residual

use crate::error::{GgufError, Result};
use crate::model::{KvCache, Model, ModelConfig, WeightMap};
use crate::ops::AnyBackend;
use crate::types::GgufFile;
use rayon::prelude::*;

// ── Ad-hoc CPU phase tracing (GGUF_TRACE_CPU=1) ──────────────────────────
// Prints cumulative per-phase wall time every 10 forward passes. Diagnostic
// only — not on any hot path unless the env var is set.
mod trace_cpu {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub static T_ATTN: AtomicU64 = AtomicU64::new(0);
    pub static T_GATED_ATTN: AtomicU64 = AtomicU64::new(0);
    pub static T_GDN: AtomicU64 = AtomicU64::new(0);
    pub static T_SHORTCONV: AtomicU64 = AtomicU64::new(0);
    pub static T_FFN: AtomicU64 = AtomicU64::new(0);
    pub static T_LMHEAD: AtomicU64 = AtomicU64::new(0);
    pub static T_EMBED: AtomicU64 = AtomicU64::new(0);
    pub static N_CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("GGUF_TRACE_CPU").is_ok())
    }

    pub fn now(trace: bool) -> Option<Instant> {
        trace.then(Instant::now)
    }

    pub fn add(counter: &AtomicU64, t0: Option<Instant>) {
        if let Some(t0) = t0 {
            counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub fn maybe_report() {
        let n = N_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10 != 0 {
            return;
        }
        let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        eprintln!(
            "[GGUF_TRACE_CPU after {n} tokens] embed={:.1}ms attn={:.1}ms gated_attn={:.1}ms gdn={:.1}ms shortconv={:.1}ms ffn={:.1}ms lm_head={:.1}ms",
            ms(&T_EMBED),
            ms(&T_ATTN),
            ms(&T_GATED_ATTN),
            ms(&T_GDN),
            ms(&T_SHORTCONV),
            ms(&T_FFN),
            ms(&T_LMHEAD),
        );
    }
}

/// The token-mixing operator of a layer. Dense Llama/Qwen layers are always
/// `Attention`; LFM2 interleaves `Attention` and `ShortConv` layers; Qwen3.5
/// interleaves `GatedAttention` (full attention, every `full_attention_interval`
/// layers) and `GatedDeltaNet` (linear attention, the rest).
enum Mixer {
    /// Self-attention: Q/K/V/O projections, optional per-head QK-norm (Qwen3),
    /// RoPE, and GQA.
    Attention {
        wq: String,
        wk: String,
        wv: String,
        wo: String,
        /// Per-head RMSNorm on Q before RoPE (Qwen3/LFM2). `None` otherwise.
        q_norm: Option<String>,
        /// Per-head RMSNorm on K before RoPE (Qwen3/LFM2).
        k_norm: Option<String>,
    },
    /// LFM2 short convolution: `in_proj` → gated depthwise causal conv → `out_proj`.
    ShortConv(ShortConv),
    /// Qwen3.5/Qwen3-Next full-attention layer: Q projection is fused with a
    /// sigmoid output gate (`wqg` outputs `[Q(head_dim) | gate(head_dim)]` per
    /// head), RoPE is partial (only the first `n_rot` dims of `head_dim` are
    /// rotated), and the attention output is gated by `sigmoid(gate)` before `wo`.
    GatedAttention {
        wqg: String,
        wk: String,
        wv: String,
        wo: String,
        q_norm: String,
        k_norm: String,
    },
    /// Qwen3.5/Qwen3-Next linear-attention layer ("Gated DeltaNet"): causal
    /// depthwise conv over fused QKV, then a per-head delta-rule recurrence
    /// with a learned scalar decay and write gate, then a gated RMSNorm.
    GatedDeltaNet(GatedDeltaNet),
}

/// Weight handles + pre-dequantized small tensors for a Gated DeltaNet layer.
struct GatedDeltaNet {
    /// `[d_model] → [key_dim*2 + value_dim]` (fused Q/K/V, pre-conv).
    wqkv: String,
    /// `[d_model] → [value_dim]`, the output gate `z` (`attn_gate.weight`).
    wgate: String,
    /// Causal depthwise conv1d kernel, pre-dequantized, laid out
    /// `[conv_dim][d_conv]` (each channel's taps contiguous).
    conv_weight: Vec<f32>,
    /// Per-(value-)head decay multiplier (`ssm_a`, `[n_v_heads]`), typically negative.
    ssm_a: Vec<f32>,
    /// Per-head softplus bias (`ssm_dt.bias`, `[n_v_heads]`).
    ssm_dt_bias: Vec<f32>,
    /// `[d_model] → [n_v_heads]`, write-gate logits (sigmoid'd to get beta).
    ssm_beta: String,
    /// `[d_model] → [n_v_heads]`, decay logits (softplus'd, then scaled by `ssm_a`).
    ssm_alpha: String,
    /// Per-channel gated-RMSNorm weight (`[head_v_dim]`), applied per head.
    ssm_norm: Vec<f32>,
    /// `[value_dim] → [d_model]`.
    ssm_out: String,
}

/// Dimensions shared by every Gated DeltaNet layer in the model (Qwen3.5-style
/// hybrid architectures only have one linear-attention configuration).
#[derive(Clone, Copy)]
struct GdnDims {
    head_k_dim: usize,
    head_v_dim: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    d_conv: usize,
}

/// Weight handles + pre-dequantized conv kernel for an LFM2 short-conv layer.
struct ShortConv {
    /// `[d_model] → [3 * d_model]`, producing the `(B, C, x)` gates.
    in_proj: String,
    /// `[d_model] → [d_model]`.
    out_proj: String,
    /// Depthwise conv kernel, pre-dequantized to f32, laid out `[d_model][l_cache]`
    /// (each channel's `l_cache` taps are contiguous).
    conv_weight: Vec<f32>,
}

/// Weight handles for a single transformer layer.
/// Norm weights are cached as pre-dequantized f32 at model load time.
/// Matmul tensors store names — actual data access goes through WeightMap.
struct LayerWeights {
    attn_norm: Vec<f32>,
    mixer: Mixer,
    ffn_norm: Vec<f32>,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

/// Pre-allocated scratch buffers for one forward pass.
///
/// All activation storage — residual stream, normalized activations, Q/K/V,
/// attention output, FFN intermediates, logits, and the attention-score
/// scratchpad — is allocated once at model load time and reused on every
/// forward pass. No per-token heap allocation happens in the hot path.
pub struct InferenceState {
    /// Residual stream, length `embedding_length`.
    x: Vec<f32>,
    /// Normalized activations (RMSNorm output), length `embedding_length`.
    xn: Vec<f32>,
    /// Query projection, length `head_count * head_dim`.
    q: Vec<f32>,
    /// Key projection, length `head_count_kv * head_dim`.
    k: Vec<f32>,
    /// Value projection, length `head_count_kv * head_dim`.
    v: Vec<f32>,
    /// Attention output, length `head_count * head_dim`.
    attn_out: Vec<f32>,
    /// Projection scratch reused by o_proj and ffn_down, length `embedding_length`.
    proj: Vec<f32>,
    /// FFN gate projection, length `feed_forward_length`.
    gate: Vec<f32>,
    /// FFN up projection, length `feed_forward_length`.
    up: Vec<f32>,
    /// SwiGLU activation `silu(gate) * up`, length `feed_forward_length`.
    ffn_act: Vec<f32>,
    /// Output logits, length `vocab_size`.
    logits: Vec<f32>,
    /// Per-head attention-score scratchpad, length `context_length`.
    scores: Vec<f32>,

    // ── LFM2 short-conv scratch (unused/empty for dense models) ──
    /// `in_proj` output holding the `(B, C, x)` gates, length `3 * embedding_length`.
    bcx: Vec<f32>,
    /// Elementwise gate `B * x`, length `embedding_length`.
    bx: Vec<f32>,
    /// Depthwise conv output, length `embedding_length`.
    conv_out: Vec<f32>,
    /// Persistent recurrent conv state (NOT per-forward scratch): for each layer,
    /// the last `l_cache - 1` `bx` vectors, flat as `[(l_cache-1) * d]`. Empty for
    /// attention layers. Survives across forward passes via the take/restore of
    /// `LlamaModel::state`, giving the causal conv its history.
    conv_state: Vec<Vec<f32>>,

    // ── Qwen3.5 GatedAttention scratch (unused/empty unless present) ──
    /// Raw fused Q+gate projection output, length `2 * head_count * head_dim`.
    ga_qg_raw: Vec<f32>,
    /// Contiguous output gate extracted from `ga_qg_raw`, length `head_count * head_dim`.
    ga_gate: Vec<f32>,

    // ── Qwen3.5 GatedDeltaNet scratch (unused/empty unless present) ──
    /// Fused Q/K/V projection output (pre-conv), length `conv_dim`.
    gdn_qkv: Vec<f32>,
    /// Post-conv, post-SiLU Q/K/V, length `conv_dim`.
    gdn_conv_out: Vec<f32>,
    /// Output gate `z` (`attn_gate` projection), length `value_dim`.
    gdn_z: Vec<f32>,
    /// Raw write-gate logits, length `n_v_heads` (sigmoid'd into beta in place).
    gdn_beta_raw: Vec<f32>,
    /// Raw decay logits, length `n_v_heads`.
    gdn_alpha_raw: Vec<f32>,
    /// Delta-net recurrence output (then gated-RMSNorm output in place), length `value_dim`.
    gdn_out: Vec<f32>,
    /// Per-head predicted-value scratch, length `head_v_dim`.
    gdn_vpred: Vec<f32>,
    /// Per-head delta scratch, length `head_v_dim`.
    gdn_delta: Vec<f32>,
    /// Persistent causal-conv history per layer: `(d_conv-1) * conv_dim` values,
    /// empty for non-GatedDeltaNet layers.
    gdn_conv_state: Vec<Vec<f32>>,
    /// Persistent delta-rule recurrent state per layer: `n_v_heads * head_k_dim *
    /// head_v_dim` values (row-major `[head][a][b]`), empty for other layers.
    gdn_recurrent_state: Vec<Vec<f32>>,
}

impl InferenceState {
    /// Allocate all buffers up front from the model configuration.
    fn new(
        cfg: &ModelConfig,
        l_cache: usize,
        layers: &[LayerWeights],
        gdn_dims: Option<GdnDims>,
    ) -> Self {
        let e = cfg.embedding_length as usize;
        let head_dim = cfg.head_dim as usize;
        let q_dim = cfg.head_count as usize * head_dim;
        let kv_dim = cfg.head_count_kv as usize * head_dim;
        let ff = cfg.feed_forward_length as usize;
        let vocab = cfg.vocab_size as usize;
        let seq = cfg.context_length as usize;

        // Conv scratch only needs to exist if the model has short-conv layers.
        let has_conv = layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::ShortConv(_)));
        let hist = l_cache.saturating_sub(1) * e; // per-layer conv history length
        let conv_state = layers
            .iter()
            .map(|l| {
                if matches!(l.mixer, Mixer::ShortConv(_)) {
                    vec![0.0; hist]
                } else {
                    Vec::new()
                }
            })
            .collect();

        let has_gated_attn = layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::GatedAttention { .. }));

        let gdn_hist = gdn_dims
            .map(|d| d.d_conv.saturating_sub(1) * d.conv_dim)
            .unwrap_or(0);
        let gdn_state_len = gdn_dims
            .map(|d| d.n_v_heads * d.head_k_dim * d.head_v_dim)
            .unwrap_or(0);
        let gdn_conv_state = layers
            .iter()
            .map(|l| {
                if matches!(l.mixer, Mixer::GatedDeltaNet(_)) {
                    vec![0.0; gdn_hist]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let gdn_recurrent_state = layers
            .iter()
            .map(|l| {
                if matches!(l.mixer, Mixer::GatedDeltaNet(_)) {
                    vec![0.0; gdn_state_len]
                } else {
                    Vec::new()
                }
            })
            .collect();

        Self {
            x: vec![0.0; e],
            xn: vec![0.0; e],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; q_dim],
            proj: vec![0.0; e],
            gate: vec![0.0; ff],
            up: vec![0.0; ff],
            ffn_act: vec![0.0; ff],
            logits: vec![0.0; vocab],
            scores: vec![0.0; seq],
            bcx: vec![0.0; if has_conv { 3 * e } else { 0 }],
            bx: vec![0.0; if has_conv { e } else { 0 }],
            conv_out: vec![0.0; if has_conv { e } else { 0 }],
            conv_state,
            ga_qg_raw: vec![0.0; if has_gated_attn { 2 * q_dim } else { 0 }],
            ga_gate: vec![0.0; if has_gated_attn { q_dim } else { 0 }],
            gdn_qkv: vec![0.0; gdn_dims.map(|d| d.conv_dim).unwrap_or(0)],
            gdn_conv_out: vec![0.0; gdn_dims.map(|d| d.conv_dim).unwrap_or(0)],
            gdn_z: vec![0.0; gdn_dims.map(|d| d.value_dim).unwrap_or(0)],
            gdn_beta_raw: vec![0.0; gdn_dims.map(|d| d.n_v_heads).unwrap_or(0)],
            gdn_alpha_raw: vec![0.0; gdn_dims.map(|d| d.n_v_heads).unwrap_or(0)],
            gdn_out: vec![0.0; gdn_dims.map(|d| d.value_dim).unwrap_or(0)],
            gdn_vpred: vec![0.0; gdn_dims.map(|d| d.head_v_dim).unwrap_or(0)],
            gdn_delta: vec![0.0; gdn_dims.map(|d| d.head_v_dim).unwrap_or(0)],
            gdn_conv_state,
            gdn_recurrent_state,
        }
    }
}

/// Llama-family dense transformer model.
pub struct LlamaModel<'a> {
    cfg: ModelConfig,
    weights: WeightMap<'a>,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
    lm_head: String,
    token_embd: String,
    /// Short-conv kernel width (LFM2 `shortconv.l_cache`); 0 for dense models.
    l_cache: usize,
    /// Gated DeltaNet dimensions (Qwen3.5-style hybrid models); `None` for
    /// architectures with no linear-attention layers.
    gdn_dims: Option<GdnDims>,
    /// Partial-RoPE dimension count for `GatedAttention` layers (Qwen3.5); the
    /// first `n_rot` dims of each head are rotated, the rest left untouched.
    /// Equals `cfg.head_dim` (full rotation) for architectures without this split.
    n_rot: usize,
    /// Pre-allocated activation buffers, reused across forward passes.
    state: Option<InferenceState>,
    /// Optional GPU backend for accelerated matmuls.
    backend: Option<AnyBackend>,
    /// Raw tensor data slice.
    raw_data: &'a [u8],
    /// Pre-uploaded GPU tensor cache: tensor name → GpuWeightHandle.
    #[cfg(feature = "wgpu")]
    gpu_tensors: std::collections::HashMap<String, crate::ops::wgpu_backend::GpuWeightHandle>,
    /// Pre-uploaded GPU norm vectors (attn_norm/ffn_norm per layer index, plus
    /// output_norm), for the fully GPU-resident forward path (`run_gpu_resident`).
    #[cfg(feature = "wgpu")]
    gpu_attn_norms: std::collections::HashMap<usize, cubecl::server::Handle>,
    #[cfg(feature = "wgpu")]
    gpu_ffn_norms: std::collections::HashMap<usize, cubecl::server::Handle>,
    #[cfg(feature = "wgpu")]
    gpu_output_norm: Option<cubecl::server::Handle>,
}

impl<'a> LlamaModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, data: &'a [u8]) -> Result<Self> {
        Self::from_gguf_with_backend(gguf, data, None)
    }

    /// Create a model with an optional GPU backend for accelerated matmuls.
    pub fn from_gguf_with_backend(
        gguf: &'a GgufFile,
        data: &'a [u8],
        backend: Option<AnyBackend>,
    ) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(gguf)?;
        let weights = WeightMap::from_gguf(gguf, data);

        // Validate that key tensors exist
        let token_embd = "token_embd.weight".to_string();
        if weights.get(&token_embd).is_none() {
            return Err(GgufError::MissingMetadata(
                "token_embd.weight tensor".into(),
            ));
        }

        // Short-conv kernel width (LFM2). Absent → dense model (no conv layers).
        let l_cache = gguf
            .get_u64(&format!("{}.shortconv.l_cache", cfg.architecture))
            .unwrap_or(0) as usize;

        // Gated DeltaNet dims (Qwen3.5/Qwen3-Next). Absent → no linear-attention layers.
        let arch_key = |field: &str| format!("{}.{field}", cfg.architecture);
        let gdn_dims = {
            let d_state = gguf.get_u64(&arch_key("ssm.state_size"));
            let n_group = gguf.get_u64(&arch_key("ssm.group_count"));
            let d_inner = gguf.get_u64(&arch_key("ssm.inner_size"));
            let dt_rank = gguf.get_u64(&arch_key("ssm.time_step_rank"));
            let d_conv = gguf.get_u64(&arch_key("ssm.conv_kernel"));
            match (d_state, n_group, d_inner, dt_rank, d_conv) {
                (Some(d_state), Some(n_group), Some(d_inner), Some(dt_rank), Some(d_conv)) => {
                    let head_k_dim = d_state as usize;
                    let n_k_heads = n_group as usize;
                    let n_v_heads = dt_rank as usize;
                    let head_v_dim = d_inner as usize / n_v_heads;
                    let key_dim = head_k_dim * n_k_heads;
                    let value_dim = head_v_dim * n_v_heads;
                    Some(GdnDims {
                        head_k_dim,
                        head_v_dim,
                        n_k_heads,
                        n_v_heads,
                        key_dim,
                        value_dim,
                        conv_dim: key_dim * 2 + value_dim,
                        d_conv: d_conv as usize,
                    })
                }
                _ => None,
            }
        };

        // Partial-RoPE dim count (Qwen3.5's `GatedAttention` layers only rotate
        // the first `n_rot` dims of each head). Falls back to full rotation.
        let n_rot = gguf
            .get_u64(&arch_key("rope.dimension_count"))
            .unwrap_or(cfg.head_dim) as usize;

        // Build layer weight handles. A layer is a short-conv layer iff it has a
        // `shortconv.conv` tensor; a Gated DeltaNet layer iff it has an
        // `ssm_alpha` tensor; otherwise it is an attention layer (fused-QG
        // `GatedAttention` iff `attn_q`'s output is `2x` the expected Q width).
        let mut layers = Vec::with_capacity(cfg.block_count as usize);
        for i in 0..cfg.block_count {
            let conv_name = format!("blk.{i}.shortconv.conv.weight");
            let ssm_alpha_name = format!("blk.{i}.ssm_alpha.weight");
            let mixer = if weights.get(&conv_name).is_some() {
                Mixer::ShortConv(ShortConv {
                    in_proj: format!("blk.{i}.shortconv.in_proj.weight"),
                    out_proj: format!("blk.{i}.shortconv.out_proj.weight"),
                    // Conv kernel is tiny and F32 — dequantize once at load.
                    conv_weight: weights.dequant_tensor(&conv_name)?,
                })
            } else if weights.get(&ssm_alpha_name).is_some() {
                let conv_w_name = format!("blk.{i}.ssm_conv1d.weight");
                Mixer::GatedDeltaNet(GatedDeltaNet {
                    wqkv: format!("blk.{i}.attn_qkv.weight"),
                    wgate: format!("blk.{i}.attn_gate.weight"),
                    conv_weight: weights.dequant_tensor(&conv_w_name)?,
                    ssm_a: weights.dequant_1d(&format!("blk.{i}.ssm_a"))?,
                    ssm_dt_bias: weights.dequant_1d(&format!("blk.{i}.ssm_dt.bias"))?,
                    ssm_beta: format!("blk.{i}.ssm_beta.weight"),
                    ssm_alpha: ssm_alpha_name,
                    ssm_norm: weights.dequant_1d(&format!("blk.{i}.ssm_norm.weight"))?,
                    ssm_out: format!("blk.{i}.ssm_out.weight"),
                })
            } else {
                let wq_name = format!("blk.{i}.attn_q.weight");
                let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
                let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
                let wq_tensor = weights
                    .get(&wq_name)
                    .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {wq_name}")))?;
                let expected_q_dim = cfg.head_count as usize * cfg.head_dim as usize;
                let wq_out_dim = wq_tensor.dims[1..].iter().product::<u64>() as usize;
                if wq_out_dim == 2 * expected_q_dim {
                    // Qwen3.5-style fused Q+gate attention layer.
                    Mixer::GatedAttention {
                        wqg: wq_name,
                        wk: format!("blk.{i}.attn_k.weight"),
                        wv: format!("blk.{i}.attn_v.weight"),
                        wo: format!("blk.{i}.attn_output.weight"),
                        q_norm: q_norm_name,
                        k_norm: k_norm_name,
                    }
                } else {
                    Mixer::Attention {
                        wq: wq_name,
                        wk: format!("blk.{i}.attn_k.weight"),
                        wv: format!("blk.{i}.attn_v.weight"),
                        wo: format!("blk.{i}.attn_output.weight"),
                        q_norm: weights.get(&q_norm_name).map(|_| q_norm_name.clone()),
                        k_norm: weights.get(&k_norm_name).map(|_| k_norm_name.clone()),
                    }
                }
            };
            // ffn_norm: Qwen3.5 names this `post_attention_norm` instead.
            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");
            let ffn_norm = if weights.get(&ffn_norm_name).is_some() {
                weights.dequant_1d(&ffn_norm_name)?
            } else {
                weights.dequant_1d(&format!("blk.{i}.post_attention_norm.weight"))?
            };
            layers.push(LayerWeights {
                attn_norm: weights.dequant_1d(&format!("blk.{i}.attn_norm.weight"))?,
                mixer,
                ffn_norm,
                ffn_gate: format!("blk.{i}.ffn_gate.weight"),
                ffn_up: format!("blk.{i}.ffn_up.weight"),
                ffn_down: format!("blk.{i}.ffn_down.weight"),
            });
        }

        // Output norm and lm_head
        let output_norm = if weights.get("output_norm.weight").is_some() {
            weights.dequant_1d("output_norm.weight")?
        } else if weights.get("token_embd_norm.weight").is_some() {
            // LFM2 uses token_embd_norm instead of output_norm
            weights.dequant_1d("token_embd_norm.weight")?
        } else {
            return Err(GgufError::MissingMetadata(
                "output_norm.weight or token_embd_norm.weight".into(),
            ));
        };

        // lm_head: may be tied to token_embd
        let lm_head = if weights.get("output.weight").is_some() {
            "output.weight".to_string()
        } else {
            // Weight tying: reuse embedding table for lm_head
            token_embd.clone()
        };

        let state = Some(InferenceState::new(&cfg, l_cache, &layers, gdn_dims));

        Ok(Self {
            cfg,
            weights,
            layers,
            output_norm,
            lm_head,
            token_embd,
            l_cache,
            gdn_dims,
            n_rot,
            state,
            backend,
            raw_data: data,
            #[cfg(feature = "wgpu")]
            gpu_tensors: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_attn_norms: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_ffn_norms: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_output_norm: None,
        })
    }

    /// Pre-upload all GPU-dequantizable weight tensors to GPU.
    /// Called once after model creation, before first forward pass.
    #[cfg(feature = "wgpu")]
    pub fn pre_upload_gpu(&mut self) {
        use std::time::Instant;

        let wgpu = match &self.backend {
            Some(AnyBackend::Wgpu(b)) => b,
            _ => return,
        };

        // Collect all tensor names on the matmul hot path
        let mut to_upload: Vec<String> = Vec::new();
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Attention { wq, wk, wv, wo, .. } => {
                    to_upload.push(wq.clone());
                    to_upload.push(wk.clone());
                    to_upload.push(wv.clone());
                    to_upload.push(wo.clone());
                }
                Mixer::ShortConv(sc) => {
                    to_upload.push(sc.in_proj.clone());
                    to_upload.push(sc.out_proj.clone());
                }
                Mixer::GatedAttention { wqg, wk, wv, wo, .. } => {
                    to_upload.push(wqg.clone());
                    to_upload.push(wk.clone());
                    to_upload.push(wv.clone());
                    to_upload.push(wo.clone());
                }
                Mixer::GatedDeltaNet(gdn) => {
                    to_upload.push(gdn.wqkv.clone());
                    to_upload.push(gdn.wgate.clone());
                    to_upload.push(gdn.ssm_beta.clone());
                    to_upload.push(gdn.ssm_alpha.clone());
                    to_upload.push(gdn.ssm_out.clone());
                }
            }
            to_upload.push(layer.ffn_gate.clone());
            to_upload.push(layer.ffn_up.clone());
            to_upload.push(layer.ffn_down.clone());
        }
        to_upload.push(self.lm_head.clone());
        to_upload.push(self.token_embd.clone());

        eprintln!("  uploading {} tensors to GPU...", to_upload.len());
        let mut uploaded = 0usize;
        for name in &to_upload {
            if let Some(tensor) = self.weights.get(name) {
                if crate::ops::GPU_DEQUANT_DTYPES.contains(&tensor.ggml_type) {
                    let t0 = Instant::now();
                    let tensor_data = &self.raw_data[tensor.byte_offset as usize
                        ..tensor.byte_offset as usize + tensor.byte_size()];
                    let in_dim = tensor.dims[0] as usize;
                    let handle = wgpu.upload_weight(tensor.ggml_type, tensor_data, in_dim);
                    self.gpu_tensors.insert(name.clone(), handle);
                    uploaded += 1;
                    let mb = tensor.byte_size() as f64 / 1048576.0;
                    eprintln!(
                        "    [{uploaded}] {name} ({mb:.1} MB) in {:.2}s",
                        t0.elapsed().as_secs_f32()
                    );
                }
            }
        }
        eprintln!("  pre-uploaded {uploaded} GPU tensors");

        // Norm vectors are always f32 and already dequantized in memory
        // (see LayerWeights/output_norm) — upload as-is, no packing needed.
        // These feed `run_gpu_resident`'s fully GPU-resident layer chain.
        for (i, layer) in self.layers.iter().enumerate() {
            self.gpu_attn_norms
                .insert(i, wgpu.upload_activation(&layer.attn_norm));
            self.gpu_ffn_norms
                .insert(i, wgpu.upload_activation(&layer.ffn_norm));
        }
        self.gpu_output_norm = Some(wgpu.upload_activation(&self.output_norm));
    }

    /// Non-cfg version for non-wgpu builds.
    #[cfg(not(feature = "wgpu"))]
    pub fn pre_upload_gpu(&mut self) {}

    /// Compile the GPU matmul kernel up front (one dummy launch per dtype
    /// actually present in this model's weights), so the ~7s shader compile
    /// happens with a visible message at load time instead of silently
    /// stalling on the model's first forward pass.
    ///
    /// Must run after `pre_upload_gpu`, which populates `gpu_tensors` with
    /// every GPU-dequantizable weight actually present in this model.
    #[cfg(feature = "wgpu")]
    pub fn warmup_gpu_kernels(&self) {
        let backend = match &self.backend {
            Some(b) => b,
            None => return,
        };

        let mut shapes: Vec<(crate::types::GgmlType, usize)> = Vec::new();
        for handle in self.gpu_tensors.values() {
            let (dtype, in_dim) = handle.shape();
            if !shapes.iter().any(|(dt, _)| *dt == dtype) {
                shapes.push((dtype, in_dim));
            }
        }

        crate::ops::warmup(backend, &shapes);
    }

    /// Non-cfg version for non-wgpu builds.
    #[cfg(not(feature = "wgpu"))]
    pub fn warmup_gpu_kernels(&self) {}

    /// RMS normalization into a caller-provided buffer using a cached weight vector.
    /// `out[i] = (x[i] / rms) * weight[i]`.
    fn rms_norm_into(&self, x: &[f32], weight: &[f32], out: &mut [f32]) {
        let eps = self.cfg.layer_norm_rms_epsilon;
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + eps).sqrt();
        for ((&xi, &wi), o) in x.iter().zip(weight.iter()).zip(out.iter_mut()) {
            *o = wi * xi / rms;
        }
    }

    /// Per-head RMSNorm over `head_dim`, applied in place to each of the
    /// `n_heads` contiguous head slices of `x`. Used for Qwen3 QK-norm.
    fn qk_norm(&self, x: &mut [f32], weight_name: &str, n_heads: usize) -> Result<()> {
        let weight = self.weights.dequant_1d(weight_name)?;
        let head_dim = self.cfg.head_dim as usize;
        let eps = self.cfg.layer_norm_rms_epsilon;
        debug_assert_eq!(weight.len(), head_dim);

        for h in 0..n_heads {
            let head = &mut x[h * head_dim..h * head_dim + head_dim];
            let sum_sq: f32 = head.iter().map(|v| v * v).sum();
            let rms = (sum_sq / head_dim as f32 + eps).sqrt();
            for (d, xi) in head.iter_mut().enumerate() {
                *xi = weight[d] * *xi / rms;
            }
        }
        Ok(())
    }

    /// Rotary position embedding applied per-head.
    fn rope(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;
        let theta = self.cfg.rope_freq_base;

        // Apply RoPE to each Q head
        for h in 0..n_heads {
            let start = h * head_dim;
            let half = head_dim / 2;
            for d in 0..half {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let (sin_val, cos_val) = freq.sin_cos();
                let x0 = q[start + d];
                let x1 = q[start + d + half];
                q[start + d] = x0 * cos_val - x1 * sin_val;
                q[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }

        // Apply RoPE to each K head
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            let half = head_dim / 2;
            for d in 0..half {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let (sin_val, cos_val) = freq.sin_cos();
                let x0 = k[start + d];
                let x1 = k[start + d + half];
                k[start + d] = x0 * cos_val - x1 * sin_val;
                k[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /// Naive attention into a caller-provided buffer: computes full attention
    /// scores for all cached positions and writes the attended output for one
    /// token into `out` (length `n_heads * head_dim`). `scores` is a reused
    /// scratchpad that must hold at least `k_all.len()` values.
    fn attention_into(
        &self,
        q: &[f32],
        k_all: &[Vec<f32>],
        v_all: &[Vec<f32>],
        out: &mut [f32],
        scores: &mut [f32],
    ) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let group_size = self.cfg.gqa_group_size() as usize;
        let seq_len = k_all.len(); // number of cached positions

        // out is reused across tokens, so clear the region we accumulate into.
        for o in out.iter_mut() {
            *o = 0.0;
        }

        for h in 0..n_heads {
            let kv_head = h / group_size; // which KV head does this Q head attend to
            let q_start = h * head_dim;
            let q_head = &q[q_start..q_start + head_dim];

            // Attention scores for this head against all cached positions.
            let scores = &mut scores[..seq_len];
            for t in 0..seq_len {
                let k_start = kv_head * head_dim;
                let k_head = &k_all[t][k_start..k_start + head_dim];
                let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                scores[t] = dot / (head_dim as f32).sqrt();
            }

            // Softmax
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V
            for t in 0..seq_len {
                let v_start = kv_head * head_dim;
                let v_head = &v_all[t][v_start..v_start + head_dim];
                for d in 0..head_dim {
                    out[h * head_dim + d] += scores[t] * v_head[d];
                }
            }
        }
    }

    /// CPU fallback FFN gate+up matmul: computes both `gate = W_gate * x` and
    /// `up = W_up * x` in one pass over the weight data, then applies silu.
    /// Avoids reading `x` twice and spawning two rayon thread pools. Used by
    /// `matmul_ffn_into` when the GPU-chained path isn't available.
    fn ffn_gate_up(
        &self,
        gate_name: &str,
        up_name: &str,
        x: &[f32],
        ffn_act: &mut [f32],
    ) -> Result<()> {
        let gate_tensor = self
            .weights
            .get(gate_name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {gate_name}")))?;
        let up_tensor = self
            .weights
            .get(up_name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {up_name}")))?;

        let out_dim = gate_tensor.dims[1] as usize;
        debug_assert_eq!(ffn_act.len(), out_dim);
        debug_assert_eq!(gate_tensor.ggml_type, up_tensor.ggml_type);

        let data = self.raw_data;

        ffn_act[..out_dim]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row_idx, o)| -> Result<()> {
                let g = crate::quant::dot_row(gate_tensor, data, row_idx, x)?;
                let u = crate::quant::dot_row(up_tensor, data, row_idx, x)?;
                let silu = g / (1.0 + (-g).exp());
                *o = silu * u;
                Ok(())
            })
    }

    /// Matrix-vector product, dispatching to GPU backend when available.
    /// Falls back to CPU (rayon-parallel fused dequant+dot via WeightMap) otherwise.
    fn matmul_into(&self, name: &str, x: &[f32], out: &mut [f32]) -> Result<()> {
        // Try pre-uploaded GPU tensor first (already resident, no re-upload)
        #[cfg(feature = "wgpu")]
        if let Some(ref backend) = self.backend {
            if backend.name() != "cpu" {
                if let Some(handle) = self.gpu_tensors.get(name) {
                    use crate::ops::AnyBackend;
                    match backend {
                        AnyBackend::Wgpu(b) => {
                            b.matmul_dequant_preloaded(handle, x, out)?;
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fallback: ad-hoc GPU dispatch (packs + uploads on every call)
        if let Some(ref backend) = self.backend {
            if backend.name() != "cpu" {
                let tensor = self
                    .weights
                    .get(name)
                    .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;

                let out_dim: usize = if tensor.n_dims as usize == 1 {
                    1
                } else {
                    tensor.dims[1..].iter().product::<u64>() as usize
                };

                let row_data = &self.raw_data
                    [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

                if crate::ops::GPU_DEQUANT_DTYPES.contains(&tensor.ggml_type) {
                    backend.matmul_dequant(tensor.ggml_type, row_data, x, &mut out[..out_dim])?;
                    return Ok(());
                }
            }
        }

        // CPU fallback (rayon-parallel, fast)
        self.weights.matmul_into(name, x, out)
    }

    /// Q/K/V projection, batched into a single GPU round-trip when all three
    /// weights are GPU-resident (they all read the same input `x`). Falls
    /// back to three independent `matmul_into` calls otherwise.
    fn matmul_qkv_into(
        &self,
        wq: &str,
        wk: &str,
        wv: &str,
        x: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "wgpu")]
        if let Some(AnyBackend::Wgpu(b)) = &self.backend {
            if let (Some(hq), Some(hk), Some(hv)) = (
                self.gpu_tensors.get(wq),
                self.gpu_tensors.get(wk),
                self.gpu_tensors.get(wv),
            ) {
                return b.matmul_dequant_qkv(hq, hk, hv, x, q, k, v);
            }
        }

        self.matmul_into(wq, x, q)?;
        self.matmul_into(wk, x, k)?;
        self.matmul_into(wv, x, v)
    }

    /// FFN block (`gate`/`up`/SiLU-combine/`down`), chained entirely on GPU
    /// with a single readback when all three weights are GPU-resident (no
    /// CPU-side dependency sits between them, unlike attention). Falls back
    /// to `ffn_gate_up` + `matmul_into` otherwise. `ffn_act` is scratch space
    /// used only by the CPU fallback path.
    fn matmul_ffn_into(
        &self,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        x: &[f32],
        ffn_act: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(feature = "wgpu")]
        if let Some(AnyBackend::Wgpu(b)) = &self.backend {
            if let (Some(hg), Some(hu), Some(hd)) = (
                self.gpu_tensors.get(gate_name),
                self.gpu_tensors.get(up_name),
                self.gpu_tensors.get(down_name),
            ) {
                return b.matmul_dequant_ffn(hg, hu, hd, x, out);
            }
        }

        self.ffn_gate_up(gate_name, up_name, x, ffn_act)?;
        self.matmul_into(down_name, ffn_act, out)
    }

    /// LFM2 short-convolution mixer for layer `layer_idx`. Reads the operator-
    /// normed activations from `state.xn` and writes the mixer output into
    /// `state.proj` (ready for the residual add). Updates the layer's recurrent
    /// conv state in place.
    ///
    /// Pipeline (matching LiquidAI's `Lfm2ShortConv`):
    ///   `(B, C, x) = split(in_proj(xn))`
    ///   `Bx = B * x`
    ///   `conv_out = depthwise_causal_conv1d(Bx)`   (kernel width `l_cache`)
    ///   `y = C * conv_out`
    ///   `out = out_proj(y)`
    fn short_conv(
        &self,
        layer_idx: usize,
        sc: &ShortConv,
        state: &mut InferenceState,
    ) -> Result<()> {
        // in_proj: xn[d] → bcx[3d], split into B, C, x gates.
        self.matmul_into(&sc.in_proj, &state.xn, &mut state.bcx)?;
        self.short_conv_gate_and_conv(layer_idx, sc, state);
        // out_proj: conv_out[d] → proj[d].
        self.matmul_into(&sc.out_proj, &state.conv_out, &mut state.proj)
    }

    /// The elementwise-gate + depthwise-causal-conv math between `in_proj`
    /// and `out_proj`: reads `state.bcx` (already populated by `in_proj`),
    /// writes `state.conv_out`, and advances the layer's recurrent conv
    /// history. Small (`d` channels, `l_cache` taps) and inherently
    /// sequential (per-channel recurrence), so this stays on CPU — shared by
    /// both `short_conv` (CPU/ad-hoc-GPU matmul dispatch) and
    /// `run_gpu_resident` (GPU-resident matmul dispatch).
    fn short_conv_gate_and_conv(&self, layer_idx: usize, sc: &ShortConv, state: &mut InferenceState) {
        let d = self.cfg.embedding_length as usize;
        let l = self.l_cache;

        let (b, rest) = state.bcx.split_at(d);
        let (c, xg) = rest.split_at(d);

        // Bx = B * x (elementwise gate).
        for j in 0..d {
            state.bx[j] = b[j] * xg[j];
        }

        // Depthwise causal conv1d. For channel `ch`, tap `k` aligns with the
        // input at time `t - (l-1) + k`: taps 0..l-2 come from the recurrent
        // history (oldest first), tap l-1 is the current `bx`.
        let history = &state.conv_state[layer_idx];
        let w = &sc.conv_weight; // [ch][l], contiguous taps per channel
        for ch in 0..d {
            let tap = &w[ch * l..ch * l + l];
            let mut acc = 0.0f32;
            for k in 0..l - 1 {
                acc += tap[k] * history[k * d + ch];
            }
            acc += tap[l - 1] * state.bx[ch];
            // y = C * conv_out, fused into conv_out.
            state.conv_out[ch] = c[ch] * acc;
        }

        // Advance the recurrent state: drop the oldest slot, append current bx.
        let history = &mut state.conv_state[layer_idx];
        for k in 0..l.saturating_sub(2) {
            for ch in 0..d {
                history[k * d + ch] = history[(k + 1) * d + ch];
            }
        }
        if l >= 2 {
            let last = (l - 2) * d;
            history[last..last + d].copy_from_slice(&state.bx);
        }
    }

    /// Partial rotary embedding: like `rope`, but only the first `self.n_rot`
    /// dims of each `head_dim`-sized head are rotated (Qwen3.5's `GatedAttention`
    /// layers leave the remaining "NoPE" tail dims untouched).
    fn rope_partial(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;
        let theta = self.cfg.rope_freq_base;
        let n_rot = self.n_rot;
        let half = n_rot / 2;

        for h in 0..n_heads {
            let start = h * head_dim;
            for d in 0..half {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / n_rot as f32);
                let (sin_val, cos_val) = freq.sin_cos();
                let x0 = q[start + d];
                let x1 = q[start + d + half];
                q[start + d] = x0 * cos_val - x1 * sin_val;
                q[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }

        for h in 0..n_kv_heads {
            let start = h * head_dim;
            for d in 0..half {
                let freq = pos as f32 / theta.powf(2.0 * d as f32 / n_rot as f32);
                let (sin_val, cos_val) = freq.sin_cos();
                let x0 = k[start + d];
                let x1 = k[start + d + half];
                k[start + d] = x0 * cos_val - x1 * sin_val;
                k[start + d + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /// Qwen3.5's fused-QG full-attention layer: `wqg` projects to `[Q(head_dim)
    /// | gate(head_dim)]` per head; the gate is applied (sigmoid) to the
    /// attention output before `wo`. RoPE is partial (see `rope_partial`).
    /// Writes the mixer output into `state.proj`.
    #[allow(clippy::too_many_arguments)]
    fn gated_attention(
        &self,
        layer_idx: usize,
        wqg: &str,
        wk: &str,
        wv: &str,
        wo: &str,
        q_norm: &str,
        k_norm: &str,
        pos: usize,
        kv_cache: &mut KvCache,
        state: &mut InferenceState,
    ) -> Result<()> {
        let head_dim = self.cfg.head_dim as usize;
        let n_heads = self.cfg.head_count as usize;
        let n_kv_heads = self.cfg.head_count_kv as usize;

        // Fused Q+gate projection, then split the interleaved-per-head layout
        // `[Q(head_dim) | gate(head_dim)]` into contiguous Q and gate buffers.
        self.matmul_into(wqg, &state.xn, &mut state.ga_qg_raw)?;
        for h in 0..n_heads {
            let src = h * 2 * head_dim;
            state.q[h * head_dim..h * head_dim + head_dim]
                .copy_from_slice(&state.ga_qg_raw[src..src + head_dim]);
            state.ga_gate[h * head_dim..h * head_dim + head_dim]
                .copy_from_slice(&state.ga_qg_raw[src + head_dim..src + 2 * head_dim]);
        }

        self.matmul_into(wk, &state.xn, &mut state.k)?;
        self.matmul_into(wv, &state.xn, &mut state.v)?;

        self.qk_norm(&mut state.q, q_norm, n_heads)?;
        self.qk_norm(&mut state.k, k_norm, n_kv_heads)?;

        self.rope_partial(&mut state.q, &mut state.k, pos);

        kv_cache.write(layer_idx, pos, &state.k, &state.v);
        let (k_all, v_all) = kv_cache.read_up_to(layer_idx, pos);
        self.attention_into(
            &state.q,
            k_all,
            v_all,
            &mut state.attn_out,
            &mut state.scores,
        );

        // Output gate: attn_out *= sigmoid(gate), applied before wo.
        for (o, &g) in state.attn_out.iter_mut().zip(state.ga_gate.iter()) {
            *o *= 1.0 / (1.0 + (-g).exp());
        }

        self.matmul_into(wo, &state.attn_out, &mut state.proj)
    }

    /// Qwen3.5's linear-attention layer ("Gated DeltaNet"): fused QKV → causal
    /// depthwise conv (SiLU) → per-head L2-norm on Q/K → tile-repeat Q/K from
    /// `n_k_heads` to `n_v_heads` → per-token delta-rule recurrence with a
    /// learned scalar decay and write gate → gated RMSNorm → `ssm_out`.
    /// Writes the mixer output into `state.proj`.
    ///
    /// Per-head recurrence (state `S` is `head_k_dim x head_v_dim`, persistent
    /// across positions, matching llama.cpp's `build_delta_net_autoregressive`):
    ///   `v_pred[b] = sum_a S[a,b] * k[a]`
    ///   `delta[b]  = beta * (v[b] - v_pred[b])`
    ///   `S[a,b]    = decay * S[a,b] + k[a] * delta[b]`
    ///   `out[b]    = sum_a (q[a] * scale) * S[a,b]`,  `scale = 1/sqrt(head_k_dim)`
    fn gated_delta_net(
        &self,
        layer_idx: usize,
        gdn: &GatedDeltaNet,
        dims: &GdnDims,
        state: &mut InferenceState,
    ) -> Result<()> {
        self.matmul_into(&gdn.wqkv, &state.xn, &mut state.gdn_qkv)?;
        self.matmul_into(&gdn.wgate, &state.xn, &mut state.gdn_z)?;
        self.matmul_into(&gdn.ssm_beta, &state.xn, &mut state.gdn_beta_raw)?;
        self.matmul_into(&gdn.ssm_alpha, &state.xn, &mut state.gdn_alpha_raw)?;

        let n_v_heads = dims.n_v_heads;
        let n_k_heads = dims.n_k_heads;
        let head_k_dim = dims.head_k_dim;
        let head_v_dim = dims.head_v_dim;
        let key_dim = dims.key_dim;
        let value_dim = dims.value_dim;
        let eps = self.cfg.layer_norm_rms_epsilon;

        // beta = sigmoid(beta_raw); decay = exp(ssm_a * softplus(alpha_raw + dt_bias)).
        // Overwritten in place — the raw logits aren't needed again this token.
        for h in 0..n_v_heads {
            state.gdn_beta_raw[h] = 1.0 / (1.0 + (-state.gdn_beta_raw[h]).exp());
            let x = state.gdn_alpha_raw[h] + gdn.ssm_dt_bias[h];
            let softplus = if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() };
            state.gdn_alpha_raw[h] = (gdn.ssm_a[h] * softplus).exp();
        }

        causal_depthwise_conv(
            &mut state.gdn_conv_state[layer_idx],
            &gdn.conv_weight,
            dims.conv_dim,
            dims.d_conv,
            &state.gdn_qkv,
            &mut state.gdn_conv_out,
            |acc| acc / (1.0 + (-acc).exp()), // SiLU
        );

        // L2-normalize each Q/K head (dim head_k_dim) in place.
        for h in 0..n_k_heads {
            for base in [h * head_k_dim, key_dim + h * head_k_dim] {
                let seg = &mut state.gdn_conv_out[base..base + head_k_dim];
                let sum_sq: f32 = seg.iter().map(|v| v * v).sum();
                let norm = (sum_sq + eps).sqrt();
                for v in seg.iter_mut() {
                    *v /= norm;
                }
            }
        }

        // Per-(value-)head delta-rule recurrence. Q/K heads are tile-repeated
        // (cyclic) up to the value-head count, matching `ggml_repeat_4d`.
        let scale = 1.0 / (head_k_dim as f32).sqrt();
        for h in 0..n_v_heads {
            let kh = h % n_k_heads;
            let q_h_start = kh * head_k_dim;
            let k_h_start = key_dim + kh * head_k_dim;
            let v_h_start = 2 * key_dim + h * head_v_dim;

            let s = &mut state.gdn_recurrent_state[layer_idx]
                [h * head_k_dim * head_v_dim..(h + 1) * head_k_dim * head_v_dim];

            for b in 0..head_v_dim {
                let mut acc = 0.0f32;
                for a in 0..head_k_dim {
                    acc += s[a * head_v_dim + b] * state.gdn_conv_out[k_h_start + a];
                }
                state.gdn_vpred[b] = acc;
            }
            for b in 0..head_v_dim {
                state.gdn_delta[b] = state.gdn_beta_raw[h]
                    * (state.gdn_conv_out[v_h_start + b] - state.gdn_vpred[b]);
            }
            let decay = state.gdn_alpha_raw[h];
            for a in 0..head_k_dim {
                let k_val = state.gdn_conv_out[k_h_start + a];
                for b in 0..head_v_dim {
                    let idx = a * head_v_dim + b;
                    s[idx] = decay * s[idx] + k_val * state.gdn_delta[b];
                }
            }
            for b in 0..head_v_dim {
                let mut acc = 0.0f32;
                for a in 0..head_k_dim {
                    acc += scale * state.gdn_conv_out[q_h_start + a] * s[a * head_v_dim + b];
                }
                state.gdn_out[h * head_v_dim + b] = acc;
            }
        }

        // Gated RMSNorm per head: norm(out) * silu(z), in place.
        for h in 0..n_v_heads {
            let base = h * head_v_dim;
            let sum_sq: f32 = state.gdn_out[base..base + head_v_dim]
                .iter()
                .map(|v| v * v)
                .sum();
            let rms = (sum_sq / head_v_dim as f32 + eps).sqrt();
            for d in 0..head_v_dim {
                let normed = gdn.ssm_norm[d] * state.gdn_out[base + d] / rms;
                let g = state.gdn_z[base + d];
                let silu = g / (1.0 + (-g).exp());
                state.gdn_out[base + d] = normed * silu;
            }
        }

        self.matmul_into(&gdn.ssm_out, &state.gdn_out[..value_dim], &mut state.proj)
    }
}

/// Generic causal depthwise conv1d over `channels` channels with kernel width
/// `kernel`. `history` holds `(kernel-1) * channels` values (oldest first,
/// layout `[tap][channel]`) and is advanced in place — the recurrent state
/// that gives the conv its history across forward passes. `activation` is
/// applied to each channel's raw conv output before it's written to `output`.
///
/// Shared by Qwen3.5's Gated DeltaNet mixer. (LFM2's `short_conv_gate_and_conv`
/// keeps its own copy since it also fuses in an elementwise `B*x` gate before
/// the conv, which this generic version has no notion of.)
fn causal_depthwise_conv(
    history: &mut [f32],
    conv_weight: &[f32],
    channels: usize,
    kernel: usize,
    input: &[f32],
    output: &mut [f32],
    activation: impl Fn(f32) -> f32,
) {
    for ch in 0..channels {
        let tap = &conv_weight[ch * kernel..ch * kernel + kernel];
        let mut acc = 0.0f32;
        for k in 0..kernel - 1 {
            acc += tap[k] * history[k * channels + ch];
        }
        acc += tap[kernel - 1] * input[ch];
        output[ch] = activation(acc);
    }

    // Advance the recurrent history: drop the oldest slot, append current input.
    for k in 0..kernel.saturating_sub(2) {
        for ch in 0..channels {
            history[k * channels + ch] = history[(k + 1) * channels + ch];
        }
    }
    if kernel >= 2 {
        let last = (kernel - 2) * channels;
        history[last..last + channels].copy_from_slice(&input[..channels]);
    }
}

impl<'a> LlamaModel<'a> {
    /// Forward pass against the pre-allocated `state` buffers.
    ///
    /// Takes `&self` (weights/config, read-only) and `&mut state` (activation
    /// scratch) as disjoint borrows, so every buffer is reused in place with no
    /// per-token heap allocation apart from the returned logits `Vec`.
    fn run(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let trace = trace_cpu::enabled();

        // 1. Embedding lookup → residual stream.
        let t0 = trace_cpu::now(trace);
        let embd = self.weights.dequant_row(&self.token_embd, token as usize)?;
        state.x.copy_from_slice(&embd);
        trace_cpu::add(&trace_cpu::T_EMBED, t0);

        // 2. Transformer layers.
        for (i, layer) in self.layers.iter().enumerate() {
            // --- Token mixer (attention or short-conv) ---
            // Operator prenorm → xn, then the mixer writes its output to proj.
            self.rms_norm_into(&state.x, &layer.attn_norm, &mut state.xn);

            let t0 = trace_cpu::now(trace);
            match &layer.mixer {
                Mixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    self.matmul_qkv_into(wq, wk, wv, &state.xn, &mut state.q, &mut state.k, &mut state.v)?;

                    // Qwen3/LFM2 QK-norm: per-head RMSNorm on Q and K before RoPE.
                    if let Some(q_norm) = q_norm {
                        self.qk_norm(&mut state.q, q_norm, self.cfg.head_count as usize)?;
                    }
                    if let Some(k_norm) = k_norm {
                        self.qk_norm(&mut state.k, k_norm, self.cfg.head_count_kv as usize)?;
                    }

                    self.rope(&mut state.q, &mut state.k, pos);

                    // Write K/V to cache, then read all cached K/V.
                    kv_cache.write(i, pos, &state.k, &state.v);
                    let (k_all, v_all) = kv_cache.read_up_to(i, pos);

                    // Attention into attn_out (scores as reused scratch).
                    self.attention_into(
                        &state.q,
                        k_all,
                        v_all,
                        &mut state.attn_out,
                        &mut state.scores,
                    );

                    // Output projection.
                    self.matmul_into(wo, &state.attn_out, &mut state.proj)?;
                }
                Mixer::ShortConv(sc) => {
                    self.short_conv(i, sc, state)?;
                }
                Mixer::GatedAttention {
                    wqg,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    self.gated_attention(i, wqg, wk, wv, wo, q_norm, k_norm, pos, kv_cache, state)?;
                }
                Mixer::GatedDeltaNet(gdn) => {
                    let dims = self
                        .gdn_dims
                        .expect("gdn_dims must be set when a GatedDeltaNet layer exists");
                    self.gated_delta_net(i, gdn, &dims, state)?;
                }
            }
            trace_cpu::add(
                match &layer.mixer {
                    Mixer::Attention { .. } => &trace_cpu::T_ATTN,
                    Mixer::ShortConv(_) => &trace_cpu::T_SHORTCONV,
                    Mixer::GatedAttention { .. } => &trace_cpu::T_GATED_ATTN,
                    Mixer::GatedDeltaNet(_) => &trace_cpu::T_GDN,
                },
                t0,
            );

            // Mixer residual.
            for j in 0..state.x.len() {
                state.x[j] += state.proj[j];
            }

            // --- FFN (SwiGLU) — fused gate+up+down ---
            self.rms_norm_into(&state.x, &layer.ffn_norm, &mut state.xn);

            let t0 = trace_cpu::now(trace);
            self.matmul_ffn_into(
                &layer.ffn_gate,
                &layer.ffn_up,
                &layer.ffn_down,
                &state.xn,
                &mut state.ffn_act,
                &mut state.proj,
            )?;
            trace_cpu::add(&trace_cpu::T_FFN, t0);

            for j in 0..state.x.len() {
                state.x[j] += state.proj[j];
            }
        }

        // 3. Final norm + project to vocab.
        self.rms_norm_into(&state.x, &self.output_norm, &mut state.xn);
        let t0 = trace_cpu::now(trace);
        self.matmul_into(&self.lm_head, &state.xn, &mut state.logits)?;
        trace_cpu::add(&trace_cpu::T_LMHEAD, t0);
        if trace {
            trace_cpu::maybe_report();
        }

        Ok(state.logits.clone())
    }

    /// Returns the GPU backend if every tensor this model needs (all layer
    /// weights — attention or short-conv — all norm vectors, lm_head) is
    /// GPU-resident — the preconditions for `run_gpu_resident`, which keeps
    /// the residual stream on GPU for the whole forward pass instead of
    /// crossing the CPU/GPU boundary once per matmul (see [[gpu-sync-bottleneck]]
    /// in project memory).
    #[cfg(feature = "wgpu")]
    fn gpu_resident_ready(&self) -> Option<&crate::ops::wgpu_backend::WgpuBackend> {
        let Some(AnyBackend::Wgpu(b)) = &self.backend else {
            return None;
        };
        if !self.gpu_tensors.contains_key(&self.lm_head) || self.gpu_output_norm.is_none() {
            return None;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if !self.gpu_attn_norms.contains_key(&i) || !self.gpu_ffn_norms.contains_key(&i) {
                return None;
            }
            let mixer_ready = match &layer.mixer {
                Mixer::Attention { wq, wk, wv, wo, .. } => [wq, wk, wv, wo]
                    .into_iter()
                    .all(|name| self.gpu_tensors.contains_key(name)),
                Mixer::ShortConv(sc) => [&sc.in_proj, &sc.out_proj]
                    .into_iter()
                    .all(|name| self.gpu_tensors.contains_key(name)),
                // Qwen3.5's GatedAttention/GatedDeltaNet mixers don't have a
                // GPU-resident path yet — always fall back to `run()`.
                Mixer::GatedAttention { .. } | Mixer::GatedDeltaNet(_) => false,
            };
            let ffn_ready = [&layer.ffn_gate, &layer.ffn_up, &layer.ffn_down]
                .into_iter()
                .all(|name| self.gpu_tensors.contains_key(name));
            if !mixer_ready || !ffn_ready {
                return None;
            }
        }
        Some(b)
    }

    /// Fully GPU-resident forward pass: the residual stream (`x`) stays on
    /// GPU as a `Handle` for the entire layer stack. RMSNorm and residual-add
    /// also run on GPU, so a mixer's output feeds straight into the FFN chain
    /// without ever landing on CPU — only the mixer's own CPU-bound step
    /// (attention: q/k/v for RoPE + the CPU-side KV cache; short-conv: the
    /// small sequential gate+conv recurrence) and the final logits cross back
    /// to CPU. That's ~1 sync point per layer instead of ~3-5
    /// (see [[gpu-sync-bottleneck]]).
    #[cfg(feature = "wgpu")]
    fn run_gpu_resident(
        &self,
        b: &crate::ops::wgpu_backend::WgpuBackend,
        token: u32,
        pos: usize,
        kv_cache: &mut KvCache,
        state: &mut InferenceState,
    ) -> Result<Vec<f32>> {
        let eps = self.cfg.layer_norm_rms_epsilon;
        let d_model = self.cfg.embedding_length as usize;

        let embd = self.weights.dequant_row(&self.token_embd, token as usize)?;
        let mut x_handle = b.upload_activation(&embd);

        for (i, layer) in self.layers.iter().enumerate() {
            let attn_norm_h = &self.gpu_attn_norms[&i];
            let xn_handle = b.launch_rms_norm(&x_handle, attn_norm_h, d_model, eps);

            match &layer.mixer {
                Mixer::Attention {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let hq = &self.gpu_tensors[wq];
                    let hk = &self.gpu_tensors[wk];
                    let hv = &self.gpu_tensors[wv];
                    let ho = &self.gpu_tensors[wo];

                    // RoPE + attention (KV cache, causal softmax) stay
                    // CPU-side — this is the one unavoidable sync point per
                    // attention layer.
                    b.matmul_dequant_qkv_from_handle(
                        hq,
                        hk,
                        hv,
                        &xn_handle,
                        &mut state.q,
                        &mut state.k,
                        &mut state.v,
                    )?;

                    if let Some(q_norm) = q_norm {
                        self.qk_norm(&mut state.q, q_norm, self.cfg.head_count as usize)?;
                    }
                    if let Some(k_norm) = k_norm {
                        self.qk_norm(&mut state.k, k_norm, self.cfg.head_count_kv as usize)?;
                    }
                    self.rope(&mut state.q, &mut state.k, pos);

                    kv_cache.write(i, pos, &state.k, &state.v);
                    let (k_all, v_all) = kv_cache.read_up_to(i, pos);
                    self.attention_into(
                        &state.q,
                        k_all,
                        v_all,
                        &mut state.attn_out,
                        &mut state.scores,
                    );

                    // wo → residual-add, both on GPU — no read back to CPU.
                    let attn_out_handle = b.upload_activation(&state.attn_out);
                    let wo_out_handle = b.launch_only(ho, &attn_out_handle);
                    x_handle = b.launch_residual_add(&x_handle, &wo_out_handle, d_model);
                }
                Mixer::ShortConv(sc) => {
                    let hin = &self.gpu_tensors[&sc.in_proj];
                    let hout = &self.gpu_tensors[&sc.out_proj];

                    // in_proj launches straight off xn_handle (GPU), but the
                    // gate+conv math is small and inherently sequential
                    // (per-channel recurrence) — this is the one unavoidable
                    // sync point per short-conv layer.
                    let bcx_handle = b.launch_only(hin, &xn_handle);
                    state.bcx.copy_from_slice(&b.read_handle(bcx_handle, 3 * d_model));
                    self.short_conv_gate_and_conv(i, sc, state);

                    // out_proj → residual-add, both on GPU — no read back.
                    let conv_out_handle = b.upload_activation(&state.conv_out);
                    let out_handle = b.launch_only(hout, &conv_out_handle);
                    x_handle = b.launch_residual_add(&x_handle, &out_handle, d_model);
                }
                Mixer::GatedAttention { .. } | Mixer::GatedDeltaNet(_) => {
                    // `gpu_resident_ready` never returns `Some` when any layer
                    // uses these mixers, so this path is unreachable.
                    unreachable!("GatedAttention/GatedDeltaNet has no GPU-resident path")
                }
            }

            // FFN, fully chained, residual-add on GPU — still no read.
            let ffn_norm_h = &self.gpu_ffn_norms[&i];
            let xn2_handle = b.launch_rms_norm(&x_handle, ffn_norm_h, d_model, eps);
            let hg = &self.gpu_tensors[&layer.ffn_gate];
            let hu = &self.gpu_tensors[&layer.ffn_up];
            let hd = &self.gpu_tensors[&layer.ffn_down];
            let down_handle = b.ffn_chain_from_handle(hg, hu, hd, &xn2_handle);
            x_handle = b.launch_residual_add(&x_handle, &down_handle, d_model);
        }

        // Final norm + lm_head — the one remaining read is the logits.
        let out_norm_h = self.gpu_output_norm.as_ref().unwrap();
        let xn_final = b.launch_rms_norm(&x_handle, out_norm_h, d_model, eps);
        let h_lm = &self.gpu_tensors[&self.lm_head];
        let logits_handle = b.launch_only(h_lm, &xn_final);
        Ok(b.read_handle(logits_handle, h_lm.out_dim()))
    }
}

impl<'a> Model for LlamaModel<'a> {
    fn forward(&mut self, token: u32, pos: usize, kv_cache: &mut KvCache) -> Result<Vec<f32>> {
        // Take the pre-allocated buffers out so `run` can borrow `&self`
        // (weights/config) and `&mut state` disjointly. Restored unconditionally
        // so an error mid-pass doesn't leave the state missing.
        let mut state = self
            .state
            .take()
            .expect("InferenceState missing (forward is not re-entrant)");

        #[cfg(feature = "wgpu")]
        let result = match self.gpu_resident_ready() {
            Some(b) => self.run_gpu_resident(b, token, pos, kv_cache, &mut state),
            None => self.run(token, pos, kv_cache, &mut state),
        };
        #[cfg(not(feature = "wgpu"))]
        let result = self.run(token, pos, kv_cache, &mut state);

        self.state = Some(state);
        result
    }

    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
}
