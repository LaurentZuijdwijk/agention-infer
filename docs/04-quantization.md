# Quantization Formats

## Overview

Quantization reduces weight precision to shrink model size and improve memory bandwidth utilization. There are two distinct families in GGUF, plus emerging novel formats.

```
K-quants    Q2_K through Q8_K      superblock structure, nested scales
I-quants    IQ1_S through IQ4_XS   importance-matrix aware, codebook lookup
Novel       TQ1_0, MXFP4, NF4      specialized formats for specific use cases
```

All dequantization produces `f32` output. The weights stay compressed in memory permanently. Dequantization happens block-by-block at compute time, immediately before the dot product — the f32 is never materialized for the full matrix.

---

## Implementation Priority

```
Phase 1 — required for any model:
  F32         trivial memcpy/cast
  F16         one half::f16::to_f32() per value
  BF16        one bfloat16::to_f32() per value
  Q8_0        simplest quantized format, implement first

Phase 2 — required for common models:
  Q4_K_M      most popular for 7B–13B models
  Q6_K        used in XL/dynamic variants for sensitive layers

Phase 3 — required for M2.7 and IQ models:
  IQ4_XS      M2.7's primary format
  IQ3_XXS     fits M2.7 within 96 GB

Phase 4 — completeness:
  Q4_0        legacy, still common
  Q5_K        between Q4_K and Q6_K
  MXFP4       M2.7 MXFP4 variant, dequant to f32
  TQ1_0       DeepSeek ternary models
  IQ4_NL      ARM-optimized 4-bit
```

---

## K-Quants

### Q8_0

The simplest quantized format. Start here.

```
Block: 32 values
  [delta: f16 (2 bytes)] [qs: i8 × 32 (32 bytes)]
  = 34 bytes per 32 values  (vs 128 bytes at f32)

dequant:
  f[i] = f16_to_f32(delta) * qs[i]
```

```rust
fn dequant_q8_0_block(block: &[u8; 34]) -> [f32; 32] {
    let delta = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let mut out = [0f32; 32];
    for i in 0..32 {
        out[i] = delta * block[2 + i] as i8 as f32;
    }
    out
}
```

### Q4_0 (legacy)

```
Block: 32 values
  [delta: f16 (2 bytes)] [nibbles: u8 × 16 (16 bytes)]
  = 18 bytes per 32 values

dequant:
  nibble_lo = byte & 0xF
  nibble_hi = byte >> 4
  f[2i]   = delta * (nibble_lo as i8 - 8)
  f[2i+1] = delta * (nibble_hi as i8 - 8)
```

### Q4_K

The most important K-quant. Uses superblocks with nested scales.

```
Superblock: 256 values (8 sub-blocks of 32)
  [scales_and_mins: 12 bytes]  packed 6-bit scales and mins
  [nibbles: u8 × 128 (128 bytes)]
  = 140 bytes per 256 values

Scale packing (12 bytes for 8 sub-blocks):
  each sub-block has a 6-bit scale and 6-bit min
  8 × 12 bits = 96 bits = 12 bytes
  packed in a specific interleaved layout (see ggml-quants.c)

dequant for sub-block b, nibble pair i:
  scale = unpack_scale(scales_and_mins, b)
  min   = unpack_min(scales_and_mins, b)
  lo = nibble & 0xF
  hi = nibble >> 4
  f[2i]   = scale * lo + min
  f[2i+1] = scale * hi + min
```

Q4_K_S and Q4_K_M differ in which tensors use Q4_K vs Q6_K for scales. M (medium) keeps more accuracy in the scale factors. In practice Q4_K_M is the default you'll encounter.

### Q6_K

6 bits per value. Reconstructed from two arrays (low 4 bits + high 2 bits).

```
Superblock: 256 values
  [ql: u8 × 128]   lower 4 bits of each value
  [qh: u8 × 64]    upper 2 bits of each value (4 packed per byte)
  [scales: f16 × 16]  one scale per 16-value group
  = 210 bytes per 256 values

dequant for value i:
  lo = ql[i/2] >> ((i%2)*4) & 0xF
  hi = qh[i/4] >> ((i%4)*2) & 0x3
  q  = lo | (hi << 4)          → 6-bit value 0..63
  q_signed = q - 32            → centered -32..31
  f[i] = f16_to_f32(scales[i/16]) * q_signed
```

---

## I-Quants (Importance Matrix)

I-quants use a pre-computed importance matrix to allocate bits non-uniformly. High-importance weights get more bits. The resulting distribution is captured in a learned codebook — quantization becomes a table lookup rather than arithmetic.

### IQ4_XS

Used by MiniMax M2.7 (primary format), Unsloth dynamic quants.

```
Superblock: 256 values (8 sub-blocks of 32)
  [scales: 12 bytes]   same 6-bit packed format as Q4_K
  [qs: u8 × 128]       4-bit indices into codebook (2 per byte)
  = 140 bytes per 256 values

Codebook: 16 entries (hardcoded, not stored in file)
  values placed at statistically optimal positions
  for normally-distributed weights near zero

dequant for sub-block b, byte i:
  scale = unpack_scale(scales, b)
  lo_idx = qs[i] & 0xF
  hi_idx = qs[i] >> 4
  f[2i]   = scale * IQ4_CODEBOOK[lo_idx]
  f[2i+1] = scale * IQ4_CODEBOOK[hi_idx]
```

The actual codebook values must be verified against the llama.cpp source (`ggml-quants.c`). They are not evenly spaced — more values are clustered near zero where weights are dense.

### IQ3_XXS

Fits MiniMax M2.7 within 96 GB. More aggressive than IQ4_XS.

```
~93 GB for M2.7 at IQ3_XXS vs ~108 GB at IQ4_XS
Quality tradeoff: small but measurable
Recommended when memory is the constraint
```

### IQ4_NL

Non-linear 4-bit. Simpler structure than IQ4_XS (single scale per block, no sub-block scales). Preferred for ARM where it enables efficient SIMD repacking. Less relevant for RDNA.

---

## Novel Formats

### TQ1_0 / TQ2_0 (Ternary)

For models trained with ternary weights (BitNet, some DeepSeek variants). Weights are exactly `{-1, 0, +1}`.

```
TQ1_0: packs 5 ternary values per byte (3^5 = 243 < 256) → ~1.6 bits/weight
TQ2_0: 2 bits per value → 2.0625 bits/weight (with overhead)

Inference: matmul becomes additions/subtractions only — no multiplications.
           Fast on hardware with good integer units.

Limitation: only useful for models TRAINED as ternary.
            Post-training ternary quantization of a dense model = severe quality loss.
```

### MXFP4 (Microscaling Float 4-bit)

OCP standard. Used in some MiniMax M2.7 variants.

```
Format: E2M1 — 4-bit float with sign + 2 exponent + 1 mantissa
Block scale: E8M0 — 8-bit exponent only, no mantissa (power of two)
Block size: 32 values

Values representable: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0

dequant:
  scale = 2^(scale_byte - 127)          ← E8M0 block scale
  for each nibble pair:
    f = e2m1_to_f32(nibble) * scale

e2m1_to_f32(nibble: u8) -> f32:
  sign = nibble >> 3
  exp  = (nibble >> 1) & 0x3
  mant = nibble & 0x1
  if exp == 0:
    val = mant * 0.5    ← subnormal
  else:
    val = (1.0 + mant * 0.5) * 2^(exp-1)
  return if sign { -val } else { val }
```

Hardware acceleration for MXFP4 requires dedicated FP4 units (NVIDIA Blackwell, future AMD). On current RDNA 3.5, dequantize to f32 and compute normally — no speed difference vs IQ4.

### NF4 (Normal Float 4-bit)

Primarily encountered in LoRA adapters trained with QLoRA. Not commonly found in plain inference GGUF files.

```
16 quantization levels placed at statistical quantiles of a normal distribution
information-theoretically optimal for normally-distributed weights
stored as a BF16 lookup table of 16 entries
```

---

## Dynamic Quantization (UD- prefix)

Unsloth Dynamic quants apply different quantization types to different tensors based on sensitivity analysis. A single model file will contain a mix of types:

```
Typical UD-IQ4_XS distribution:
  token_embd.weight       Q8_0    ← embedding, high impact, kept accurate
  output.weight           Q8_0    ← LM head, high impact
  *.attn_norm.weight      F32     ← tiny tensors, kept exact
  *.ffn_norm.weight       F32
  *.attn_q.weight         IQ4_XS  ← bulk of compute, compressed
  *.attn_k.weight         IQ4_XS
  *.attn_v.weight         IQ4_XS
  *.ffn_gate_exps.weight  IQ4_XS  ← expert FFNs, bulk of size
  sensitive_layers.*      Q6_K    ← some layers marked high sensitivity
  insensitive_layers.*    IQ3_XXS ← some layers marked low sensitivity
```

The engine handles this for free — each `TensorInfo` carries its own `dtype`. The dequant dispatch is per-tensor already. The `gguf-info` tool surfaces the distribution as a summary.

---

## Memory Bandwidth Math

The theoretical decode speed ceiling:

```
tokens/sec = memory_bandwidth / model_size_bytes

Strix Halo at 256 GB/s:
  Q8_0 7B  (~7 GB):   256/7   = 36 tok/s ceiling
  Q4_K 7B  (~4 GB):   256/4   = 64 tok/s ceiling
  IQ4_XS M2.7 (~108 GB): 256/108 = 2.4 tok/s ceiling (weights only)

With MoE sparsity (only 8/256 experts active):
  effective active weight per token ≈ 108 × (8/256 + attn_fraction)
  ≈ 108 × ~0.08 = ~9 GB effective
  256/9 ≈ 28 tok/s ceiling for M2.7
```

Realistic performance is 60–80% of ceiling. A well-optimized engine hits 70–75%.

---

## Dequantization Architecture

```rust
pub trait Dequant {
    fn dequant_block(&self, block: &[u8], out: &mut [f32]);
    fn block_size(&self) -> usize;     // number of values per block
    fn block_bytes(&self) -> usize;    // bytes per block
}

pub fn dequant_tensor(tensor: &Tensor, out: &mut Vec<f32>) {
    let dq: Box<dyn Dequant> = match tensor.dtype {
        GgmlType::F32   => Box::new(F32Dequant),
        GgmlType::F16   => Box::new(F16Dequant),
        GgmlType::Q8_0  => Box::new(Q8_0Dequant),
        GgmlType::Q4_K  => Box::new(Q4KDequant),
        GgmlType::Q6_K  => Box::new(Q6KDequant),
        GgmlType::IQ4_XS => Box::new(IQ4XSDequant),
        GgmlType::IQ3_XXS => Box::new(IQ3XXSDequant),
        GgmlType::MXFP4 => Box::new(MXFP4Dequant),
        GgmlType::TQ1_0 => Box::new(TQ1_0Dequant),
        other           => panic!("unsupported: {:?}", other),
    };

    out.clear();
    for block in tensor.data.chunks_exact(dq.block_bytes()) {
        let start = out.len();
        out.resize(start + dq.block_size(), 0.0);
        dq.dequant_block(block, &mut out[start..]);
    }
}

// Hot path: dequantize one row directly into a matmul accumulator
// Never materializes the full matrix.
pub fn dequant_row_dot(tensor: &Tensor, row: usize, x: &[f32]) -> f32 {
    let row_bytes = tensor.row_bytes();
    let row_data = &tensor.data[row * row_bytes..(row + 1) * row_bytes];
    // dequant block-by-block, accumulate dot product immediately
    // never allocates
}
```
