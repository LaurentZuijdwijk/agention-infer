//! Layer/mixer data definitions — pure structs and enums, no compute logic.
//! See `cpu_path.rs` for the CPU implementations and `gpu_resident.rs` for
//! the GPU-resident implementations of each mixer.

/// The token-mixing operator of a layer. Dense Llama/Qwen layers are always
/// `Attention`; LFM2 interleaves `Attention` and `ShortConv` layers; Qwen3.5
/// interleaves `GatedAttention` (full attention, every `full_attention_interval`
/// layers) and `GatedDeltaNet` (linear attention, the rest).
pub(crate) enum Mixer {
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
pub(crate) struct GatedDeltaNet {
    /// `[d_model] → [key_dim*2 + value_dim]` (fused Q/K/V, pre-conv).
    pub(crate) wqkv: String,
    /// `[d_model] → [value_dim]`, the output gate `z` (`attn_gate.weight`).
    pub(crate) wgate: String,
    /// Causal depthwise conv1d kernel, pre-dequantized, laid out
    /// `[conv_dim][d_conv]` (each channel's taps contiguous).
    pub(crate) conv_weight: Vec<f32>,
    /// Per-(value-)head decay multiplier (`ssm_a`, `[n_v_heads]`), typically negative.
    pub(crate) ssm_a: Vec<f32>,
    /// Per-head softplus bias (`ssm_dt.bias`, `[n_v_heads]`).
    pub(crate) ssm_dt_bias: Vec<f32>,
    /// `[d_model] → [n_v_heads]`, write-gate logits (sigmoid'd to get beta).
    pub(crate) ssm_beta: String,
    /// `[d_model] → [n_v_heads]`, decay logits (softplus'd, then scaled by `ssm_a`).
    pub(crate) ssm_alpha: String,
    /// Per-channel gated-RMSNorm weight (`[head_v_dim]`), applied per head.
    pub(crate) ssm_norm: Vec<f32>,
    /// `[value_dim] → [d_model]`.
    pub(crate) ssm_out: String,
}

/// Dimensions shared by every Gated DeltaNet layer in the model (Qwen3.5-style
/// hybrid architectures only have one linear-attention configuration).
#[derive(Clone, Copy)]
pub(crate) struct GdnDims {
    pub(crate) head_k_dim: usize,
    pub(crate) head_v_dim: usize,
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_dim: usize,
    pub(crate) d_conv: usize,
}

/// Weight handles + pre-dequantized conv kernel for an LFM2 short-conv layer.
pub(crate) struct ShortConv {
    /// `[d_model] → [3 * d_model]`, producing the `(B, C, x)` gates.
    pub(crate) in_proj: String,
    /// `[d_model] → [d_model]`.
    pub(crate) out_proj: String,
    /// Depthwise conv kernel, pre-dequantized to f32, laid out `[d_model][l_cache]`
    /// (each channel's `l_cache` taps are contiguous).
    pub(crate) conv_weight: Vec<f32>,
}

/// Weight handles for a single transformer layer.
/// Norm weights are cached as pre-dequantized f32 at model load time.
/// Matmul tensors store names — actual data access goes through WeightMap.
pub(crate) struct LayerWeights {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) mixer: Mixer,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) ffn_gate: String,
    pub(crate) ffn_up: String,
    pub(crate) ffn_down: String,
}
