mod llama;

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::{GgufError, Result};
use crate::types::{GgmlType, GgufFile, TensorInfo};

pub use llama::{InferenceState, LlamaModel};

// ── Inspection types (used by gguf-info CLI) ──────────────────────────

/// Typed view over GGUF metadata for a specific model architecture.
///
/// Fields that are optional in the GGUF spec are `Option<_>` here.
/// Used by the inspection CLI — always shows something useful for any GGUF file.
#[derive(Debug)]
pub struct ModelInfo {
    pub architecture: String,
    pub name: Option<String>,
    pub license: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    /// Defaults to `head_count` when absent (MHA, no GQA).
    pub head_count_kv: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub rope_freq_base: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub layer_norm_rms_epsilon: Option<f32>,
    pub moe: Option<MoeInfo>,
    pub file_type: Option<u32>,
}

/// MoE-specific model parameters.
#[derive(Debug)]
pub struct MoeInfo {
    pub expert_count: u64,
    pub expert_used_count: u64,
    pub expert_feed_forward_length: u64,
    pub expert_shared_count: Option<u64>,
}

impl ModelInfo {
    /// Extract typed model info from a parsed GGUF file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .get_string("general.architecture")
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".into()))?
            .to_string();

        let prefix = &architecture;
        let key = |field: &str| format!("{prefix}.{field}");

        let context_length = gguf.get_u64(&key("context_length"));
        let embedding_length = gguf.get_u64(&key("embedding_length"));
        let block_count = gguf.get_u64(&key("block_count"));
        let head_count = gguf.get_u64(&key("attention.head_count"));
        let head_count_kv = gguf.get_u64(&key("attention.head_count_kv"));

        let feed_forward_length = gguf.get_u64(&key("feed_forward_length"));
        let rope_freq_base = gguf.get_f32(&key("rope.freq_base"));
        let rope_scaling_type = gguf
            .get_string(&key("rope.scaling.type"))
            .map(|s| s.to_string());
        let rope_scaling_factor = gguf.get_f32(&key("rope.scaling.factor"));
        let layer_norm_rms_epsilon = gguf.get_f32(&key("attention.layer_norm_rms_epsilon"));

        // MoE fields
        let moe = if let Some(expert_count) = gguf.get_u64(&key("expert_count")) {
            Some(MoeInfo {
                expert_count,
                expert_used_count: gguf
                    .get_u64(&key("expert_used_count"))
                    .unwrap_or(expert_count),
                expert_feed_forward_length: gguf
                    .get_u64(&key("expert_feed_forward_length"))
                    .unwrap_or(feed_forward_length.unwrap_or(0)),
                expert_shared_count: gguf.get_u64(&key("expert_shared_count")),
            })
        } else {
            None
        };

        let file_type = gguf.get_u64("general.file_type").map(|v| v as u32);

        Ok(Self {
            architecture,
            name: gguf.get_string("general.name").map(|s| s.to_string()),
            license: gguf.get_string("general.license").map(|s| s.to_string()),
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            rope_freq_base,
            rope_scaling_type,
            rope_scaling_factor,
            layer_norm_rms_epsilon,
            moe,
            file_type,
        })
    }

    /// Effective head_count_kv: falls back to head_count when absent (MHA).
    pub fn effective_head_count_kv(&self) -> Option<u64> {
        self.head_count_kv.or(self.head_count)
    }

    /// Head dimension = embedding_length / head_count
    pub fn head_dim(&self) -> Option<u64> {
        self.head_count
            .zip(self.embedding_length)
            .map(|(h, e)| e / h)
    }

    /// Group size for GQA = head_count / head_count_kv
    pub fn gqa_group_size(&self) -> Option<u64> {
        let kv = self.effective_head_count_kv()?;
        let h = self.head_count?;
        if kv == 0 {
            Some(1)
        } else {
            Some(h / kv)
        }
    }

    /// KV cache memory per token per layer in bytes (f16 storage).
    /// See [`Self::kv_bytes_per_token_per_layer_for`] for other KV dtypes.
    pub fn kv_bytes_per_token_per_layer(&self) -> Option<u64> {
        self.kv_bytes_per_token_per_layer_for(KvDtype::F16)
    }

    /// KV cache memory per token per layer in bytes, for a given [`KvDtype`]
    /// — accounts for block/scale overhead (e.g. Q8_0's per-32-element f16
    /// scale), not just a flat bits-per-element estimate.
    pub fn kv_bytes_per_token_per_layer_for(&self, dtype: KvDtype) -> Option<u64> {
        let kv = self.effective_head_count_kv()?;
        let hd = self.head_dim()?;
        let head_dim_kv = (kv * hd) as usize;
        // K + V, each `dtype.bytes_per_token(head_dim_kv)`.
        Some(2 * dtype.bytes_per_token(head_dim_kv) as u64)
    }

    /// KV cache total bytes for a given context length (f16).
    pub fn kv_cache_bytes(&self, context_len: u64) -> Option<u64> {
        self.kv_cache_bytes_for(context_len, KvDtype::F16)
    }

    /// KV cache total bytes for a given context length and [`KvDtype`].
    pub fn kv_cache_bytes_for(&self, context_len: u64, dtype: KvDtype) -> Option<u64> {
        let per_token = self.kv_bytes_per_token_per_layer_for(dtype)?;
        let layers = self.block_count?;
        Some(per_token * layers * context_len)
    }

    /// Maximum context length that fits in `available_bytes` of KV cache
    /// at the given bits-per-element (flat estimate, ignores block overhead
    /// — see [`Self::max_context_at_kv_dtype`] for an exact, dtype-aware version).
    pub fn max_context_at_kv_bits(&self, available_bytes: u64, bits: u32) -> Option<u64> {
        let kv = self.effective_head_count_kv()?;
        let hd = self.head_dim()?;
        let layers = self.block_count?;
        let bytes_per_token_per_layer = 2 * kv * hd * bits as u64 / 8;
        if bytes_per_token_per_layer == 0 {
            return Some(0);
        }
        let total_per_token = bytes_per_token_per_layer * layers;
        Some(available_bytes / total_per_token)
    }

    /// Maximum context length that fits in `available_bytes` of KV cache for
    /// a given [`KvDtype`] — exact, accounts for block/scale overhead.
    pub fn max_context_at_kv_dtype(&self, available_bytes: u64, dtype: KvDtype) -> Option<u64> {
        let per_token = self.kv_bytes_per_token_per_layer_for(dtype)?;
        let layers = self.block_count?;
        let total_per_token = per_token * layers;
        if total_per_token == 0 {
            return Some(0);
        }
        Some(available_bytes / total_per_token)
    }

    /// Model size in bytes (sum of all tensor data).
    pub fn model_bytes(&self, gguf: &GgufFile) -> u64 {
        gguf.total_tensor_bytes()
    }
}

/// Configuration extracted from GGUF metadata. Architecture-agnostic.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub head_dim: u64,
    pub feed_forward_length: u64,
    pub rope_freq_base: f32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub layer_norm_rms_epsilon: f32,
    pub vocab_size: u64,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .get_string("general.architecture")
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".into()))?
            .to_string();

        let key = |field: &str| format!("{architecture}.{field}");

        let head_count = gguf
            .get_u64(&key("attention.head_count"))
            .ok_or_else(|| GgufError::MissingMetadata(key("attention.head_count")))?;
        let embedding_length = gguf
            .get_u64(&key("embedding_length"))
            .ok_or_else(|| GgufError::MissingMetadata(key("embedding_length")))?;
        // Head dimension is explicit in some architectures (Qwen3 decouples it
        // from embedding_length / head_count). Prefer key_length, then fall back
        // to the classic embedding_length / head_count.
        let head_dim = gguf
            .get_u64(&key("attention.key_length"))
            .unwrap_or_else(|| embedding_length / head_count);
        // head_count_kv may be a scalar (dense models) or a per-layer array
        // (LFM2: 0 for short-conv layers, N for attention layers). When it is an
        // array, the KV cache is sized by the largest attention layer's KV heads.
        let head_count_kv = gguf
            .get_u64(&key("attention.head_count_kv"))
            .or_else(|| {
                gguf.get_i64_array(&key("attention.head_count_kv"))
                    .and_then(|arr| arr.into_iter().filter(|&v| v > 0).max())
                    .map(|v| v as u64)
            })
            .unwrap_or(head_count);

        // Infer vocab size from token_embd tensor dims
        let vocab_size = gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| {
                // dims[1] is the vocab dimension (outer dimension in GGUF)
                if t.n_dims >= 2 {
                    t.dims[1]
                } else {
                    t.dims[0]
                }
            })
            .ok_or_else(|| GgufError::MissingMetadata("token_embd.weight tensor".into()))?;

        Ok(Self {
            architecture: architecture.clone(),
            context_length: gguf.get_u64(&key("context_length")).unwrap_or(4096),
            embedding_length,
            block_count: gguf.get_u64(&key("block_count")).unwrap_or(32),
            head_count,
            head_count_kv,
            head_dim,
            feed_forward_length: gguf.get_u64(&key("feed_forward_length")).unwrap_or(0),
            rope_freq_base: gguf.get_f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_scaling_type: gguf
                .get_string(&key("rope.scaling.type"))
                .map(|s| s.to_string()),
            rope_scaling_factor: gguf.get_f32(&key("rope.scaling.factor")),
            layer_norm_rms_epsilon: gguf
                .get_f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            vocab_size,
        })
    }

    pub fn gqa_group_size(&self) -> u64 {
        self.head_count / self.head_count_kv
    }
}

/// Map from tensor name to TensorInfo, with a reference to the data section.
/// This is the primary way the model accesses weights.
pub struct WeightMap<'a> {
    tensors: HashMap<String, &'a TensorInfo>,
    data: &'a [u8],
}

impl<'a> WeightMap<'a> {
    pub fn from_gguf(gguf: &'a GgufFile, data: &'a [u8]) -> Self {
        let tensors: HashMap<String, &TensorInfo> =
            gguf.tensors.iter().map(|t| (t.name.clone(), t)).collect();
        Self { tensors, data }
    }

    /// Get tensor info by name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).copied()
    }

    /// Dequantize an entire tensor into f32 values.
    /// For 2D+ tensors this flattens all rows.
    pub fn dequant_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;
        let n_elements = tensor.n_elements();
        let _data_offset = 0; // data is already the tensor data section
        let row_data = &self.data[tensor.byte_offset as usize..];

        match tensor.ggml_type {
            GgmlType::F32 => {
                let bytes = &row_data[..n_elements * 4];
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            GgmlType::F16 => {
                let bytes = &row_data[..n_elements * 2];
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect())
            }
            GgmlType::Q8_0 => crate::quant::q8_0::dequant_q8_0(row_data, n_elements),
            GgmlType::Q5_0 => crate::quant::q5_0::dequant_q5_0(row_data, n_elements),
            GgmlType::Q4_K => crate::quant::q4_k::dequant_q4_k(row_data, n_elements),
            GgmlType::Q5_K => crate::quant::q5_k::dequant_q5_k(row_data, n_elements),
            GgmlType::Q6_K => crate::quant::q6_k::dequant_q6_k(row_data, n_elements),
            GgmlType::Q2_K => crate::quant::q2_k::dequant_q2_k(row_data, n_elements),
            other => Err(GgufError::BackendError(format!(
                "dequant not implemented for {other}"
            ))),
        }
    }

    /// Dequantize a single row of a tensor.
    pub fn dequant_row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let tensor = self
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;
        crate::quant::dequant_row(tensor, self.data, row)
    }

    /// Dequantize a 1D tensor (norm weights, biases).
    pub fn dequant_1d(&self, name: &str) -> Result<Vec<f32>> {
        self.dequant_row(name, 0)
    }

    /// Fused, parallel matrix-vector product: `out = W * x`.
    ///
    /// `W` is the named weight tensor of shape `[in_dim, out_dim]` (GGUF order),
    /// `x` has length `in_dim`, and `out` receives `out_dim` values. Each output
    /// row is an independent fused dequant+dot (see [`crate::quant::dot_row`]),
    /// so the weight matrix is never materialized as `f32`, and the rows are
    /// computed in parallel across rayon's thread pool.
    pub fn matmul_into(&self, name: &str, x: &[f32], out: &mut [f32]) -> Result<()> {
        let tensor = self
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;

        let n_rows: usize = if tensor.n_dims as usize == 1 {
            1
        } else {
            tensor.dims[1..].iter().product::<u64>() as usize
        };

        debug_assert!(
            out.len() >= n_rows,
            "matmul_into: output buffer len {} < {n_rows} rows",
            out.len()
        );

        let data = self.data;

        // Q5_K has an int8×int8 fused kernel against a Q8_K-quantized `x`
        // (see [`crate::quant::q8_k`]) that's faster than per-row f32
        // dequant+dot once `x` is shared across enough rows to amortize the
        // one-time quantization cost — true for every matmul here.
        if tensor.ggml_type == crate::types::GgmlType::Q5_K {
            let q8k = crate::quant::q8_k::quantize_row_q8_k(x);
            return out[..n_rows]
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row_idx, o)| -> Result<()> {
                    *o = crate::quant::dot_row_q5k_q8k(tensor, data, row_idx, &q8k)?;
                    Ok(())
                });
        }

        out[..n_rows]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(row_idx, o)| -> Result<()> {
                *o = crate::quant::dot_row(tensor, data, row_idx, x)?;
                Ok(())
            })
    }

    /// Batched fused matmul: `out[t] = W * x[t]` for each of `batch` input rows.
    ///
    /// Each weight row is dequantized **once** and dotted against all `batch`
    /// input rows — amortizing the dequant that the per-token [`matmul_into`]
    /// pays `batch` times. This is the batched-prefill win: one weight read per
    /// layer instead of one per prompt token.
    ///
    /// `x` is `[batch * in_dim]` token-major and `out` is `[batch * out_dim]`
    /// token-major, so token `t`'s result is `out[t*out_dim ..][.. out_dim]`.
    pub fn matmul_batch_into(
        &self,
        name: &str,
        x: &[f32],
        out: &mut [f32],
        batch: usize,
    ) -> Result<()> {
        let tensor = self
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor {name}")))?;

        let out_dim: usize = if tensor.n_dims as usize == 1 {
            1
        } else {
            tensor.dims[1..].iter().product::<u64>() as usize
        };

        debug_assert!(batch > 0);
        debug_assert_eq!(x.len() % batch, 0, "matmul_batch_into: x not divisible by batch");
        let in_dim = x.len() / batch;
        debug_assert_eq!(
            out.len(),
            batch * out_dim,
            "matmul_batch_into: output buffer len {} != {batch}*{out_dim}",
            out.len()
        );

        let data = self.data;

        // Compute output-row-major `[out_dim][batch]` in parallel across output
        // rows (dequant shared over the batch), then transpose to token-major.
        let mut rowmajor = vec![0f32; out_dim * batch];
        rowmajor
            .par_chunks_mut(batch)
            .enumerate()
            .try_for_each(|(r, chunk)| -> Result<()> {
                let wr = crate::quant::dequant_row(tensor, data, r)?;
                for (t, o) in chunk.iter_mut().enumerate() {
                    let xt = &x[t * in_dim..t * in_dim + in_dim];
                    *o = wr.iter().zip(xt).map(|(w, xi)| w * xi).sum();
                }
                Ok(())
            })?;

        for r in 0..out_dim {
            for t in 0..batch {
                out[t * out_dim + r] = rowmajor[r * batch + t];
            }
        }
        Ok(())
    }
}

/// The Model trait — the core abstraction.
///
/// Architecture-specific implementations (Llama, LFM2, MoE) all implement this.
/// The caller never needs to know which architecture is running.
pub trait Model: Send + Sync {
    /// Run a forward pass for a single token at the given position.
    ///
    /// `token` is the input token ID.
    /// `pos` is the position in the sequence (0-based).
    /// `kv_cache` is mutable — K/V are written during the forward pass.
    ///
    /// Returns the logits vector `[vocab_size]`.
    fn forward(&mut self, token: u32, pos: usize, kv_cache: &mut KvCache) -> Result<Vec<f32>>;

    /// Run a forward pass for a contiguous batch of tokens (prefill).
    ///
    /// `tokens` are the input token IDs; the first is placed at `pos_start`,
    /// the next at `pos_start + 1`, and so on. K/V are written for **every**
    /// position in the batch, but only the logits for the **last** position are
    /// returned (prefill only needs the last row to seed decoding).
    ///
    /// The default implementation loops [`Model::forward`] one token at a time,
    /// so it is behaviourally identical to sequential prefill. Architectures
    /// that can share a single weight read across the batch override this.
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        pos_start: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            logits = self.forward(tok, pos_start + i, kv_cache)?;
        }
        Ok(logits)
    }

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Get the vocabulary size.
    fn vocab_size(&self) -> u64 {
        self.config().vocab_size
    }

    /// Pre-upload all GPU-dequantizable weight tensors to GPU. Called once
    /// after model creation, before first forward pass. Default is a no-op.
    fn pre_upload_gpu(&mut self) {}

    /// Compile the GPU matmul kernel up front (one dummy launch per dtype
    /// actually present in this model's weights), so the ~7s shader compile
    /// happens with a visible message at load time instead of silently
    /// stalling on the model's first forward pass.
    fn warmup_gpu_kernels(&self) {}

    /// Returns true if the GPU-resident path is ready (all weights uploaded
    /// and the GPU backend is available). Default is false.
    fn gpu_resident_ready(&self) -> bool {
        false
    }
}

/// KV cache storage dtype — llama.cpp `-ctk`/`-ctv` parity. `F16` (default)
/// halves memory/bandwidth vs the original f32 store with no block/scale
/// overhead; `Q8_0` quarters it (GGUF-style block-32, 1 f16 scale + 32 i8 per
/// block) at a small precision cost. Designed so a future TurboQuant variant
/// (Phase 6: 3-bit WHT + Lloyd-Max) is just another enum arm in `pack`/
/// `unpack` below — no restructuring of `KvCache` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F16,
    Q8_0,
}

impl KvDtype {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            _ => None,
        }
    }

    /// Bytes needed to store one token's `head_dim_kv`-length K or V vector.
    pub fn bytes_per_token(&self, head_dim_kv: usize) -> usize {
        match self {
            Self::F16 => head_dim_kv * 2,
            // GGUF Q8_0 block: 2-byte f16 scale + 32 i8 values = 34 bytes/32 elems.
            Self::Q8_0 => head_dim_kv.div_ceil(32) * 34,
        }
    }

    /// Bits per element — used by the memory-budget calculators in [`ModelInfo`].
    pub fn bits_per_element(&self) -> u32 {
        match self {
            Self::F16 => 16,
            Self::Q8_0 => 9, // 34 bytes / 32 elems * 8 bits, rounded
        }
    }

    fn pack(&self, x: &[f32], out: &mut [u8]) {
        match self {
            Self::F16 => {
                for (v, chunk) in x.iter().zip(out.chunks_exact_mut(2)) {
                    chunk.copy_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes());
                }
            }
            Self::Q8_0 => {
                for (block, out_block) in x.chunks(32).zip(out.chunks_mut(34)) {
                    let amax = block.iter().fold(0f32, |a, &v| a.max(v.abs()));
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    out_block[..2].copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                    for (v, o) in block.iter().zip(out_block[2..].iter_mut()) {
                        *o = (*v / scale).round().clamp(-127.0, 127.0) as i8 as u8;
                    }
                }
            }
        }
    }

    fn unpack(&self, bytes: &[u8], out: &mut [f32]) {
        match self {
            Self::F16 => {
                for (chunk, v) in bytes.chunks_exact(2).zip(out.iter_mut()) {
                    *v = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                }
            }
            Self::Q8_0 => {
                for (in_block, out_block) in bytes.chunks(34).zip(out.chunks_mut(32)) {
                    let scale = half::f16::from_bits(u16::from_le_bytes([in_block[0], in_block[1]])).to_f32();
                    for (b, v) in in_block[2..].iter().zip(out_block.iter_mut()) {
                        *v = (*b as i8) as f32 * scale;
                    }
                }
            }
        }
    }
}

/// KV cache. Storage is packed per [`KvDtype`] (`f16` by default); reads
/// dequantize into internal scratch buffers reused across calls (no
/// per-token heap allocation), writes quantize in place.
pub struct KvCache {
    dtype: KvDtype,
    /// Per-layer packed K/V bytes: `[max_seq_len * bytes_per_token]` each.
    k: Vec<Vec<u8>>,
    v: Vec<Vec<u8>>,
    /// Dequantized scratch, reused across `read_up_to` calls: `[max_seq_len *
    /// head_dim_kv]`, position-major (`[pos][head_dim_kv]`).
    k_scratch: Vec<f32>,
    v_scratch: Vec<f32>,
    n_layers: usize,
    head_dim_kv: usize,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self::new_with_dtype(n_layers, n_kv_heads, head_dim, max_seq_len, KvDtype::F16)
    }

    pub fn new_with_dtype(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: KvDtype,
    ) -> Self {
        let head_dim_kv = n_kv_heads * head_dim;
        let bytes_per_token = dtype.bytes_per_token(head_dim_kv);
        let k = vec![vec![0u8; max_seq_len * bytes_per_token]; n_layers];
        let v = vec![vec![0u8; max_seq_len * bytes_per_token]; n_layers];
        Self {
            dtype,
            k,
            v,
            k_scratch: vec![0.0; max_seq_len * head_dim_kv],
            v_scratch: vec![0.0; max_seq_len * head_dim_kv],
            n_layers,
            head_dim_kv,
            max_seq_len,
        }
    }

    /// The KV storage dtype this cache was built with.
    pub fn dtype(&self) -> KvDtype {
        self.dtype
    }

    /// Write K and V for a given layer and position (quantizes in place).
    pub fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert!(layer < self.n_layers);
        debug_assert_eq!(k.len(), self.head_dim_kv);
        debug_assert_eq!(v.len(), self.head_dim_kv);
        let bpt = self.dtype.bytes_per_token(self.head_dim_kv);
        self.dtype.pack(k, &mut self.k[layer][pos * bpt..(pos + 1) * bpt]);
        self.dtype.pack(v, &mut self.v[layer][pos * bpt..(pos + 1) * bpt]);
    }

    /// Write K and V for a contiguous range of positions starting at
    /// `pos_start`. `k` and `v` are laid out `[n_positions * head_dim_kv]`
    /// (position-major), i.e. position `p` occupies `[p*head_dim_kv ..
    /// (p+1)*head_dim_kv]`. Equivalent to calling [`KvCache::write`] once per
    /// position.
    pub fn write_range(&mut self, layer: usize, pos_start: usize, k: &[f32], v: &[f32]) {
        debug_assert!(layer < self.n_layers);
        debug_assert_eq!(k.len(), v.len());
        debug_assert_eq!(k.len() % self.head_dim_kv, 0);
        let n = k.len() / self.head_dim_kv;
        for p in 0..n {
            let src = p * self.head_dim_kv..(p + 1) * self.head_dim_kv;
            self.write(layer, pos_start + p, &k[src.clone()], &v[src]);
        }
    }

    /// Read (dequantizing) all K and V up to (and including) `pos` for a
    /// given layer. Returns flat, position-major slices of shape `[(pos+1) *
    /// head_dim_kv]` — position `t`'s vector is `[t*head_dim_kv ..
    /// (t+1)*head_dim_kv]`. Backed by internal scratch reused across calls.
    pub fn read_up_to(&mut self, layer: usize, pos: usize) -> (&[f32], &[f32]) {
        debug_assert!(layer < self.n_layers);
        let n = pos + 1;
        let bpt = self.dtype.bytes_per_token(self.head_dim_kv);
        let hd = self.head_dim_kv;
        for p in 0..n {
            self.dtype.unpack(
                &self.k[layer][p * bpt..(p + 1) * bpt],
                &mut self.k_scratch[p * hd..(p + 1) * hd],
            );
            self.dtype.unpack(
                &self.v[layer][p * bpt..(p + 1) * bpt],
                &mut self.v_scratch[p * hd..(p + 1) * hd],
            );
        }
        (&self.k_scratch[..n * hd], &self.v_scratch[..n * hd])
    }

    /// Maximum sequence length this cache was allocated for.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Create a model from a GGUF file.
/// Dispatches to the correct architecture implementation.
pub fn create_model<'a>(gguf: &'a GgufFile, data: &'a [u8]) -> Result<Box<dyn Model + 'a>> {
    create_model_with_backend(gguf, data, None)
}

/// Create a model with an optional GPU backend.
pub fn create_model_with_backend<'a>(
    gguf: &'a GgufFile,
    data: &'a [u8],
    backend: Option<crate::ops::AnyBackend>,
) -> Result<Box<dyn Model + 'a>> {
    let arch = gguf
        .get_string("general.architecture")
        .ok_or_else(|| GgufError::MissingMetadata("general.architecture".into()))?;

    match arch {
        "llama" | "qwen2" | "qwen3" | "qwen35" | "lfm2" => {
            let mut model = LlamaModel::from_gguf_with_backend(gguf, data, backend)?;
            model.pre_upload_gpu();
            model.warmup_gpu_kernels();
            Ok(Box::new(model))
        }
        other => Err(GgufError::UnsupportedArchitecture(other.into())),
    }
}
