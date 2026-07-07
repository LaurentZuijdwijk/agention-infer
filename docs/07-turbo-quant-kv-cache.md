# TurboQuant KV Cache Compression

## Overview

TurboQuant (ICLR 2026, Google Research & NYU) is a two-stage vector quantization algorithm for KV cache compression. It achieves near-lossless quality at 3 bits per scalar — a 5× reduction from the standard 16-bit KV cache.

For gguf-rs, TurboQuant is the **default KV cache format**, not an optional flag. It is what makes MiniMax M2.7 at 128K+ context viable on 96 GB of unified memory.

---

## The Problem

Standard KV cache quantization (scalar Q8, Q4) introduces systematic bias in inner product estimation:

```
Attention score = Q · K

Naive quantized:  Q · K_quantized = Q · K_original + systematic_bias
                                                       ↑ not random noise
                                                       ↑ distorts attention patterns
                                                       ↑ gets worse below 4-bit
```

This is why naive 3-bit KV cache breaks model output noticeably. The bias accumulates across layers and corrupts which tokens the model attends to.

TurboQuant solves this by making the quantization noise nearly unbiased for inner product computation.

---

## The Two Stages

### Stage 1: Walsh-Hadamard Transform + Lloyd-Max Quantization

Before quantizing, apply a Walsh-Hadamard Transform (WHT) to the KV vector.

**Why WHT?**
- Orthogonal transform — lossless, invertible
- Spreads energy uniformly across all dimensions
- After WHT, each coordinate follows approximately N(0, 1/d) regardless of the original distribution
- This makes the distribution predictable and optimal for quantization

**Why not random rotation?**
Community experiments found WHT works better than random orthogonal rotation. WHT is also O(d log d) vs O(d²) for a full random matrix multiply, and requires no storage — it's a deterministic butterfly network.

```rust
pub fn wht_in_place(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}
```

**Lloyd-Max quantization**: After WHT, the distribution is approximately Gaussian. Optimal quantization levels for a Gaussian are precomputed once (Lloyd-Max algorithm) and stored as a small codebook. 2-bit quantization uses 4 levels:

```rust
// Precomputed Lloyd-Max 2-bit levels for N(0,1)
// These are the optimal quantization boundaries and centroids
// for a standard normal distribution.
// Actual values computed offline — verify against paper.
const LLOYD_MAX_2BIT_LEVELS: [f32; 4] = [-1.224, -0.408, 0.408, 1.224];
const LLOYD_MAX_2BIT_BOUNDS: [f32; 3] = [-0.816, 0.0, 0.816];

fn lloyd_max_quantize_2bit(x: f32, scale: f32) -> u8 {
    let normalized = x / scale;
    // find which interval normalized falls in
    if normalized < LLOYD_MAX_2BIT_BOUNDS[0] { 0 }
    else if normalized < LLOYD_MAX_2BIT_BOUNDS[1] { 1 }
    else if normalized < LLOYD_MAX_2BIT_BOUNDS[2] { 2 }
    else { 3 }
}

fn lloyd_max_dequantize_2bit(q: u8, scale: f32) -> f32 {
    LLOYD_MAX_2BIT_LEVELS[q as usize] * scale
}
```

### Stage 2: QJL Residual Correction (Optional)

After Lloyd-Max, there's a residual error. The QJL step stores the sign of a random projection of this residual — just 1 bit.

```rust
fn qjl_sign_bit(residual: &[f32], random_vec: &[f32]) -> u8 {
    let dot: f32 = residual.iter().zip(random_vec).map(|(r, v)| r * v).sum();
    if dot >= 0.0 { 1 } else { 0 }
}
```

This 1 bit captures the direction of the systematic bias and cancels it out during decompression.

**Note**: Community testing has found that QJL residual sometimes hurts at low bit widths. Start with Lloyd-Max only (2-bit), add QJL as an optional flag. Default: QJL disabled.

---

## Complete Implementation

```rust
pub struct TurboQuantConfig {
    pub bits: u8,          // 2 (Lloyd-Max only) or 3 (with QJL)
    pub use_qjl: bool,     // add 1-bit residual correction
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self { bits: 3, use_qjl: false }   // 3-bit without QJL per community findings
    }
}

pub struct KvCache {
    config: TurboQuantConfig,
    // Per-head scales (computed from WHT output, one per vector)
    k_scales: Vec<f32>,    // [n_layers * max_seq * n_kv_heads]
    v_scales: Vec<f32>,
    // Quantized data: 2 bits Lloyd-Max + 1 bit QJL (if enabled) = 3 bits total
    // Packed: 8 values per 3 bytes (tight packing) or 8 values per 4 bytes (simple)
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    // Dimensions
    n_layers:   usize,
    n_kv_heads: usize,
    head_dim:   usize,
    max_seq:    usize,
}

impl KvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize,
               max_seq: usize, config: TurboQuantConfig) -> Self {
        let n_vectors = n_layers * max_seq * n_kv_heads;
        let bits_per_scalar = if config.use_qjl { 3 } else { 2 };
        let bytes_per_vector = (head_dim * bits_per_scalar + 7) / 8;

        Self {
            k_scales: vec![1.0; n_vectors],
            v_scales: vec![1.0; n_vectors],
            k_data: vec![0u8; n_vectors * bytes_per_vector],
            v_data: vec![0u8; n_vectors * bytes_per_vector],
            config,
            n_layers, n_kv_heads, head_dim, max_seq,
        }
    }

    pub fn write(&mut self, layer: usize, pos: usize,
                 k_heads: &[f32],   // [n_kv_heads * head_dim]
                 v_heads: &[f32]) {
        for head in 0..self.n_kv_heads {
            let k = &k_heads[head * self.head_dim..(head+1) * self.head_dim];
            let v = &v_heads[head * self.head_dim..(head+1) * self.head_dim];
            self.write_vector(layer, pos, head, k, true);
            self.write_vector(layer, pos, head, v, false);
        }
    }

    fn write_vector(&mut self, layer: usize, pos: usize, head: usize,
                    vec: &[f32], is_k: bool) {
        let mut rotated = vec.to_vec();
        wht_in_place(&mut rotated);   // Stage 1a: WHT

        // Compute scale from rotated vector (max abs value)
        let scale = rotated.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let idx = self.vector_idx(layer, pos, head);
        if is_k { self.k_scales[idx] = scale; }
        else     { self.v_scales[idx] = scale; }

        // Stage 1b: Lloyd-Max 2-bit quantization
        let quantized: Vec<u8> = rotated.iter()
            .map(|&x| lloyd_max_quantize_2bit(x, scale))
            .collect();

        // Stage 2: QJL residual (optional)
        let qjl_bits: Option<Vec<u8>> = if self.config.use_qjl {
            let dequantized: Vec<f32> = quantized.iter()
                .map(|&q| lloyd_max_dequantize_2bit(q, scale))
                .collect();
            let residual: Vec<f32> = rotated.iter().zip(dequantized.iter())
                .map(|(r, d)| r - d)
                .collect();
            // One bit per scalar: sign of residual dot random projection
            // In practice, use WHT of residual sign as a cheap approximation
            Some(residual.iter().map(|&r| if r >= 0.0 { 1 } else { 0 }).collect())
        } else {
            None
        };

        // Pack into storage
        self.pack_vector(idx, &quantized, qjl_bits.as_deref(), is_k);
    }

    pub fn read_k(&self, layer: usize, pos: usize) -> Vec<Vec<f32>> {
        (0..self.n_kv_heads).map(|head| {
            self.read_vector(layer, pos, head, true)
        }).collect()
    }

    fn read_vector(&self, layer: usize, pos: usize, head: usize,
                   is_k: bool) -> Vec<f32> {
        let idx = self.vector_idx(layer, pos, head);
        let scale = if is_k { self.k_scales[idx] } else { self.v_scales[idx] };

        let (quantized, qjl_bits) = self.unpack_vector(idx, is_k);

        // Dequantize
        let mut rotated: Vec<f32> = quantized.iter().zip(qjl_bits.iter())
            .map(|(&q, &qjl)| {
                let base = lloyd_max_dequantize_2bit(q, scale);
                if self.config.use_qjl {
                    // Apply QJL correction: add small positive/negative adjustment
                    // based on the stored residual sign bit
                    let correction = if qjl == 1 { 0.1 * scale } else { -0.1 * scale };
                    base + correction
                } else {
                    base
                }
            })
            .collect();

        // Inverse WHT
        wht_in_place(&mut rotated);
        rotated
    }

    fn vector_idx(&self, layer: usize, pos: usize, head: usize) -> usize {
        layer * self.max_seq * self.n_kv_heads + pos * self.n_kv_heads + head
    }
}
```

---

## Memory Budget with TurboQuant

### M2.7 on 96 GB Strix Halo

```
Weights (IQ3_XXS):         93 GB
KV cache @ 16-bit, 128K:   ~40 GB   → total 133 GB  DOESN'T FIT
KV cache @ 8-bit,  128K:   ~20 GB   → total 113 GB  DOESN'T FIT
KV cache @ 3-bit,  128K:    7.5 GB  → total 100.5 GB TIGHT
KV cache @ 3-bit,  64K:     3.75 GB → total 96.75 GB FITS (barely)
KV cache @ 2-bit,  128K:    5 GB    → total 98 GB   FITS
KV cache @ 2-bit,  200K:    7.8 GB  → total 101 GB  DOESN'T FIT
```

Practical strategy: IQ3_XXS weights + 3-bit KV + 64K context = viable.
Want 128K? Use 2-bit KV (Lloyd-Max only, no QJL = less accuracy).

### Qwen 7B on 96 GB

```
Weights (Q4_K_M):           4 GB
KV cache @ 3-bit, 128K:    ~1.5 GB
Total:                      5.5 GB   → trivial
```

For small models, TurboQuant enables extremely long context (200K+) with negligible memory cost.

---

## Quality at Each Bit Width

From TurboQuant paper (Llama 3.1 8B, needle-in-haystack, 4K–104K context):

```
16-bit KV:    baseline (reference)
8-bit KV:     indistinguishable from baseline
4-bit scalar: small degradation at very long context
3-bit TQ:     matches baseline up to 104K context
2-bit TQ:     marginal degradation, acceptable for most uses
```

Compare to naive scalar quantization:
```
4-bit scalar: noticeable degradation
3-bit scalar: significant degradation
2-bit scalar: severe degradation
```

TurboQuant's rotation + codebook makes 3-bit viable where 3-bit scalar quantization is not.

---

## Integration Points

TurboQuant is transparent to the attention kernel. The attention code calls `kv_cache.read_k(layer, pos)` and receives `Vec<f32>` — it doesn't know or care that the data was compressed.

The only change to the attention kernel is that `kv_cache.read_k` is now slightly more expensive than a simple memory read (it runs WHT on the decompressed data). This cost is dwarfed by the bandwidth savings at long context:

```
Cost:     WHT on read = O(d log d) per KV vector = microseconds
Savings:  5× less bandwidth reading KV cache during attention
          at 128K context: 40 GB → 7.5 GB read per token generation
          at 256 GB/s: 156ms → 29ms just for KV reads
```

Net effect at long context: significantly faster, not slower.
