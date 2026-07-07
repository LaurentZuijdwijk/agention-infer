//! Q8_K activation quantization.
//!
//! Not a weight format — this is the intermediate representation llama.cpp
//! quantizes the *activation* vector into before dotting it against a
//! K-quantized weight row. Instead of dequantizing the weight to f32 and
//! doing a float multiply-accumulate per element (what [`crate::quant::dot_row`]
//! does for formats without a dedicated int8 kernel), both operands become
//! int8 and the dot product runs as int8×int8→int32 accumulation, with the
//! float scale applied once per 256-element block instead of once per element.
//! Quantizing `x` costs `O(n)` once per matmul call and is shared by every
//! output row, so it's a clear win whenever a weight matrix has many rows
//! (true for every matmul in this engine).
//!
//! Block layout (256 elements per block), matching llama.cpp's `block_q8_K`:
//!
//!   struct block_q8_K {
//!       float   d;         // block scale
//!       int8_t  qs[256];   // quantized values
//!       int16_t bsums[16]; // sum of qs in each contiguous group of 16
//!   };
//!
//! Stored here as parallel arrays (`Q8KRow`) rather than an array of block
//! structs, since it's produced once per matmul and read by every row.

const QK_K: usize = 256;

/// A full activation vector quantized into Q8_K blocks.
pub struct Q8KRow {
    /// Per-block scale, length `ceil(n / 256)`.
    pub d: Vec<f32>,
    /// Quantized values, length `ceil(n / 256) * 256` (zero-padded past `n`).
    pub qs: Vec<i8>,
    /// Per-16-group sums of `qs`, length `ceil(n / 256) * 16`.
    pub bsums: Vec<i16>,
    /// Original (unpadded) vector length.
    pub n: usize,
}

impl Q8KRow {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }
}

/// Quantize an f32 activation vector into Q8_K blocks. Matches llama.cpp's
/// `quantize_row_q8_K_ref`: per-block scale keyed off the largest-magnitude
/// (signed) element, round-to-nearest, clamp to `i8::MAX` (no lower clamp —
/// by construction the most negative value rounds to exactly -127).
pub fn quantize_row_q8_k(x: &[f32]) -> Q8KRow {
    let n = x.len();
    let n_blocks = n.div_ceil(QK_K);

    let mut d = vec![0.0f32; n_blocks];
    let mut qs = vec![0i8; n_blocks * QK_K];
    let mut bsums = vec![0i16; n_blocks * (QK_K / 16)];

    for block in 0..n_blocks {
        let start = block * QK_K;
        let end = (start + QK_K).min(n);
        let block_x = &x[start..end];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block_x {
            let av = v.abs();
            if av > amax {
                amax = av;
                max = v;
            }
        }

        let qs_block = &mut qs[block * QK_K..(block + 1) * QK_K];
        if amax > 0.0 {
            let iscale = -127.0f32 / max;
            for (j, &v) in block_x.iter().enumerate() {
                let rounded = (iscale * v).round() as i32;
                qs_block[j] = rounded.min(127) as i8;
            }
            d[block] = 1.0 / iscale;
        }
        // amax == 0.0: d[block] stays 0.0, qs_block stays zero-initialized.

        let bsums_block = &mut bsums[block * (QK_K / 16)..(block + 1) * (QK_K / 16)];
        for (g, bsum) in bsums_block.iter_mut().enumerate() {
            let seg = &qs_block[g * 16..g * 16 + 16];
            *bsum = seg.iter().map(|&v| v as i16).sum();
        }
    }

    Q8KRow { d, qs, bsums, n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_row_q8_k_roundtrip_error() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let q = quantize_row_q8_k(&x);
        assert_eq!(q.n_blocks(), 1);
        assert_eq!(q.d.len(), 1);
        // Dequantized values should be close to the originals (K-quant-grade
        // 8-bit precision, not exact).
        for (i, &xi) in x.iter().enumerate() {
            let dq = q.d[0] * q.qs[i] as f32;
            assert!((dq - xi).abs() < 0.15, "index {i}: {dq} vs {xi}");
        }
    }

    #[test]
    fn test_quantize_row_q8_k_bsums() {
        let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
        let q = quantize_row_q8_k(&x);
        for g in 0..16 {
            let expected: i16 = q.qs[g * 16..g * 16 + 16].iter().map(|&v| v as i16).sum();
            assert_eq!(q.bsums[g], expected);
        }
    }

    #[test]
    fn test_quantize_row_q8_k_all_zero() {
        let x = vec![0.0f32; 256];
        let q = quantize_row_q8_k(&x);
        assert_eq!(q.d[0], 0.0);
        assert!(q.qs.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_row_q8_k_partial_block() {
        // Non-multiple-of-256 length: padded block should still be safe to read.
        let x: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let q = quantize_row_q8_k(&x);
        assert_eq!(q.n, 100);
        assert_eq!(q.n_blocks(), 1);
        assert_eq!(q.qs.len(), 256);
        assert_eq!(q.bsums.len(), 16);
    }
}
