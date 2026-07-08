//! Pre-allocated per-forward-pass scratch buffers, plus ad-hoc CPU phase
//! tracing (`GGUF_TRACE_CPU=1`).

use super::types::{GdnDims, LayerWeights, Mixer};
use crate::model::ModelConfig;

// ── Ad-hoc CPU phase tracing (GGUF_TRACE_CPU=1) ──────────────────────────
// Prints cumulative per-phase wall time every 10 forward passes. Diagnostic
// only — not on any hot path unless the env var is set.
pub(crate) mod trace_cpu {
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

/// Pre-allocated scratch buffers for one forward pass.
///
/// All activation storage — residual stream, normalized activations, Q/K/V,
/// attention output, FFN intermediates, logits, and the attention-score
/// scratchpad — is allocated once at model load time and reused on every
/// forward pass. No per-token heap allocation happens in the hot path.
pub struct InferenceState {
    /// Residual stream, length `embedding_length`.
    pub(crate) x: Vec<f32>,
    /// Normalized activations (RMSNorm output), length `embedding_length`.
    pub(crate) xn: Vec<f32>,
    /// Query projection, length `head_count * head_dim`.
    pub(crate) q: Vec<f32>,
    /// Key projection, length `head_count_kv * head_dim`.
    pub(crate) k: Vec<f32>,
    /// Value projection, length `head_count_kv * head_dim`.
    pub(crate) v: Vec<f32>,
    /// Attention output, length `head_count * head_dim`.
    pub(crate) attn_out: Vec<f32>,
    /// Projection scratch reused by o_proj and ffn_down, length `embedding_length`.
    pub(crate) proj: Vec<f32>,
    /// FFN gate projection, length `feed_forward_length`.
    pub(crate) gate: Vec<f32>,
    /// FFN up projection, length `feed_forward_length`.
    pub(crate) up: Vec<f32>,
    /// SwiGLU activation `silu(gate) * up`, length `feed_forward_length`.
    pub(crate) ffn_act: Vec<f32>,
    /// Output logits, length `vocab_size`.
    pub(crate) logits: Vec<f32>,
    /// Per-head attention-score scratchpad, length `context_length`.
    pub(crate) scores: Vec<f32>,

    // ── LFM2 short-conv scratch (unused/empty for dense models) ──
    /// `in_proj` output holding the `(B, C, x)` gates, length `3 * embedding_length`.
    pub(crate) bcx: Vec<f32>,
    /// Elementwise gate `B * x`, length `embedding_length`.
    pub(crate) bx: Vec<f32>,
    /// Depthwise conv output, length `embedding_length`.
    pub(crate) conv_out: Vec<f32>,
    /// Persistent recurrent conv state (NOT per-forward scratch): for each layer,
    /// the last `l_cache - 1` `bx` vectors, flat as `[(l_cache-1) * d]`. Empty for
    /// attention layers. Survives across forward passes via the take/restore of
    /// `LlamaModel::state`, giving the causal conv its history.
    pub(crate) conv_state: Vec<Vec<f32>>,

    // ── Qwen3.5 GatedAttention scratch (unused/empty unless present) ──
    /// Raw fused Q+gate projection output, length `2 * head_count * head_dim`.
    pub(crate) ga_qg_raw: Vec<f32>,
    /// Contiguous output gate extracted from `ga_qg_raw`, length `head_count * head_dim`.
    pub(crate) ga_gate: Vec<f32>,

    // ── Qwen3.5 GatedDeltaNet scratch (unused/empty unless present) ──
    /// Fused Q/K/V projection output (pre-conv), length `conv_dim`.
    pub(crate) gdn_qkv: Vec<f32>,
    /// Post-conv, post-SiLU Q/K/V, length `conv_dim`.
    pub(crate) gdn_conv_out: Vec<f32>,
    /// Output gate `z` (`attn_gate` projection), length `value_dim`.
    pub(crate) gdn_z: Vec<f32>,
    /// Raw write-gate logits, length `n_v_heads` (sigmoid'd into beta in place).
    pub(crate) gdn_beta_raw: Vec<f32>,
    /// Raw decay logits, length `n_v_heads`.
    pub(crate) gdn_alpha_raw: Vec<f32>,
    /// Delta-net recurrence output (then gated-RMSNorm output in place), length `value_dim`.
    pub(crate) gdn_out: Vec<f32>,
    /// Per-head predicted-value scratch, length `head_v_dim`.
    pub(crate) gdn_vpred: Vec<f32>,
    /// Per-head delta scratch, length `head_v_dim`.
    pub(crate) gdn_delta: Vec<f32>,
    /// Persistent causal-conv history per layer: `(d_conv-1) * conv_dim` values,
    /// empty for non-GatedDeltaNet layers.
    pub(crate) gdn_conv_state: Vec<Vec<f32>>,
    /// Persistent delta-rule recurrent state per layer: `n_v_heads * head_k_dim *
    /// head_v_dim` values (row-major `[head][a][b]`), empty for other layers.
    pub(crate) gdn_recurrent_state: Vec<Vec<f32>>,
}

impl InferenceState {
    /// Allocate all buffers up front from the model configuration.
    pub(crate) fn new(
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
