//! Llama-family model implementation.
//!
//! Covers: Llama, Mistral, Qwen2, LFM2 — all dense transformer
//! architectures with the same layer structure:
//!
//!   attn_norm → q/k/v → rope → attention → o_proj → residual
//!   ffn_norm  → gate/up → silu_mul → down → residual
//!
//! Split across: `types.rs` (mixer/layer data defs), `state.rs`
//! (`InferenceState` + CPU phase tracing), `cpu_path.rs` (the
//! CPU-orchestrated forward pass, one matmul at a time), `gpu_resident.rs`
//! (the fully GPU-resident forward pass for mixers that support it).

mod cpu_path;
mod gpu_resident;
mod state;
mod types;

use crate::error::{GgufError, Result};
use crate::model::{KvCache, Model, ModelConfig, WeightMap};
use crate::ops::AnyBackend;
use crate::types::GgufFile;
pub use state::InferenceState;
use types::{GatedDeltaNet, GdnDims, LayerWeights, Mixer, ShortConv};

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
    /// Pre-uploaded static conv-tap weights per `ShortConv` layer index, for
    /// the GPU-resident `short_conv` kernel.
    #[cfg(feature = "wgpu")]
    gpu_conv_weights: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Persistent GPU-resident conv history per `ShortConv` layer index —
    /// mutated in place by the `short_conv` kernel every forward pass, so
    /// the causal conv never has to round-trip its recurrent state to CPU.
    #[cfg(feature = "wgpu")]
    gpu_conv_history: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Pre-uploaded QK-norm weights (tensor name → handle), shared across
    /// heads within a layer — for the GPU-resident attention path.
    #[cfg(feature = "wgpu")]
    gpu_qk_norm: std::collections::HashMap<String, cubecl::server::Handle>,
    /// Persistent GPU-resident KV cache per `Attention` layer index, lazily
    /// allocated (sized to the caller's `KvCache::max_seq_len()`) on first
    /// use — mutated in place by `kv_cache_write` every forward pass.
    #[cfg(feature = "wgpu")]
    gpu_kv_cache_k: std::collections::HashMap<usize, cubecl::server::Handle>,
    #[cfg(feature = "wgpu")]
    gpu_kv_cache_v: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Reused attention-score scratch buffer, `[head_count, max_seq]`.
    #[cfg(feature = "wgpu")]
    gpu_attn_scores: Option<cubecl::server::Handle>,
    /// Reused softmax-weight scratch buffer, `[head_count, max_seq]` — cached
    /// separately from `gpu_attn_scores` so the attention kernel computes
    /// `exp()` once per cached position instead of once per (position,
    /// output-dim) pair.
    #[cfg(feature = "wgpu")]
    gpu_attn_weights: Option<cubecl::server::Handle>,
    /// `max_seq_len` the GPU KV cache / scores buffer above were sized for.
    #[cfg(feature = "wgpu")]
    gpu_kv_max_seq: usize,
    /// Pre-uploaded conv1d kernel weights per GatedDeltaNet layer index.
    #[cfg(feature = "wgpu")]
    gpu_gdn_conv_weights: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Persistent GPU-resident conv history per GatedDeltaNet layer index —
    /// mutated in place by the `causal_conv1d_silu` kernel every forward pass.
    #[cfg(feature = "wgpu")]
    gpu_gdn_conv_history: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Persistent GPU-resident delta-rule state matrix per GatedDeltaNet layer
    /// index, `[n_v_heads, head_k_dim, head_v_dim]`, mutated in place.
    #[cfg(feature = "wgpu")]
    gpu_gdn_recurrent_state: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Per-channel gated-RMSNorm weight `[head_v_dim]` per GatedDeltaNet layer.
    #[cfg(feature = "wgpu")]
    gpu_gdn_ssm_norm: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Per-head decay multiplier `[n_v_heads]` per GatedDeltaNet layer.
    #[cfg(feature = "wgpu")]
    gpu_gdn_ssm_a: std::collections::HashMap<usize, cubecl::server::Handle>,
    /// Per-head softplus bias `[n_v_heads]` per GatedDeltaNet layer.
    #[cfg(feature = "wgpu")]
    gpu_gdn_ssm_dt_bias: std::collections::HashMap<usize, cubecl::server::Handle>,
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
            #[cfg(feature = "wgpu")]
            gpu_conv_weights: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_conv_history: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_qk_norm: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_kv_cache_k: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_kv_cache_v: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_attn_scores: None,
            #[cfg(feature = "wgpu")]
            gpu_attn_weights: None,
            #[cfg(feature = "wgpu")]
            gpu_kv_max_seq: 0,
            #[cfg(feature = "wgpu")]
            gpu_gdn_conv_weights: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_gdn_conv_history: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_gdn_recurrent_state: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_gdn_ssm_norm: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_gdn_ssm_a: std::collections::HashMap::new(),
            #[cfg(feature = "wgpu")]
            gpu_gdn_ssm_dt_bias: std::collections::HashMap::new(),
        })
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
        let result = if self.gpu_resident_ready_backend().is_some() {
            // `max_seq_len` isn't known until the caller builds its
            // `KvCache`, so the GPU-resident KV cache is allocated lazily
            // here rather than in `pre_upload_gpu`.
            self.ensure_gpu_kv_cache(kv_cache.max_seq_len());
            let b = self
                .gpu_resident_ready_backend()
                .expect("checked Some(_) above; backend/model didn't change in between");
            self.run_gpu_resident(b, token, pos)
        } else {
            self.run(token, pos, kv_cache, &mut state)
        };
        #[cfg(not(feature = "wgpu"))]
        let result = self.run(token, pos, kv_cache, &mut state);

        self.state = Some(state);
        result
    }

    fn forward_batch(
        &mut self,
        tokens: &[u32],
        pos_start: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = self
            .state
            .take()
            .expect("InferenceState missing (forward_batch is not re-entrant)");

        #[cfg(feature = "wgpu")]
        let result = if self.gpu_resident_ready_backend().is_some() {
            self.ensure_gpu_kv_cache(kv_cache.max_seq_len());
            let b = self
                .gpu_resident_ready_backend()
                .expect("checked Some(_) above; backend/model didn't change in between");
            // Stage 3 will batch these dispatches; for now a correct
            // per-position loop keeps the GPU path identical to sequential
            // prefill while the CPU path gets true batching.
            let mut logits = Vec::new();
            let mut outcome = Ok(());
            for (i, &tok) in tokens.iter().enumerate() {
                match self.run_gpu_resident(b, tok, pos_start + i) {
                    Ok(l) => logits = l,
                    Err(e) => {
                        outcome = Err(e);
                        break;
                    }
                }
            }
            outcome.map(|()| logits)
        } else {
            self.run_batch(tokens, pos_start, kv_cache, &mut state)
        };
        #[cfg(not(feature = "wgpu"))]
        let result = self.run_batch(tokens, pos_start, kv_cache, &mut state);

        self.state = Some(state);
        result
    }

    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
}
