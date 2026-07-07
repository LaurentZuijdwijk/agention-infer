//! Q5_K dequantization.
//!
//! Block layout (176 bytes per block, 256 values per block), matching
//! llama.cpp `block_q5_K`:
//!
//!   struct block_q5_K {
//!       half d;             // super-block scale (2 bytes)
//!       half dmin;          // super-block minimum (2 bytes)
//!       uint8_t scales[12]; // packed sub-block scales and mins (12 bytes)
//!       uint8_t qh[32];     // high bit of each 5-bit quant (32 bytes)
//!       uint8_t qs[128];    // low 4 bits, packed 2 per byte (128 bytes)
//!   };
//!
//! Total: 2 + 2 + 12 + 32 + 128 = 176 bytes.
//!
//! Dequant formula: value = d * sc * quant - min * m
//! where quant is the 5-bit value (0..31): low nibble from `qs` plus the
//! high bit from `qh` (selected by a rotating 2-bit-shifted mask, one bit
//! per 32-value half-group).

use crate::error::{GgufError, Result};

const Q5_K_BLOCK_SIZE: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;

/// Unpack the scale (`d`) and minimum (`m`) for sub-block index `j` (0..7)
/// from the 12-byte `scales` array. Matches llama.cpp `get_scale_min_k4`
/// (same packing as Q4_K).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

pub fn dequant_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = (n_elements + Q5_K_BLOCK_SIZE - 1) / Q5_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q5_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q5_K dequant: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n_blocks * Q5_K_BLOCK_SIZE];

    for block_idx in 0..n_blocks {
        let base = block_idx * Q5_K_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let qs = &data[base + 48..base + 176];

        let block_out = block_idx * Q5_K_BLOCK_SIZE;

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..Q5_K_BLOCK_SIZE).step_by(64) {
            let qs_offset = j / 2;

            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            let (sc1, m1_s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_s as f32;

            for l in 0..32 {
                let hi_bit = if qh[l] & u1 != 0 { 16 } else { 0 };
                let quant = ((qs[qs_offset + l] & 0x0F) + hi_bit) as f32;
                out[block_out + j + l] = d1 * quant - m1;
            }
            for l in 0..32 {
                let hi_bit = if qh[l] & u2 != 0 { 16 } else { 0 };
                let quant = ((qs[qs_offset + l] >> 4) + hi_bit) as f32;
                out[block_out + j + 32 + l] = d2 * quant - m2;
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    out.truncate(n_elements);
    Ok(out)
}

/// Fused Q5_K × Q8_K dot product: both the weight row and the (pre-quantized)
/// activation are int8, so each element contributes an int8×int8 multiply
/// instead of a float dequant-then-multiply — see [`crate::quant::q8_k`] for
/// why this is faster whenever `x` is reused across many rows (every matmul).
///
/// Mirrors llama.cpp's `ggml_vec_dot_q5_K_q8_K`, restructured around a single
/// loop over the 8 sub-blocks (`is` 0..8) instead of the reference's
/// j-stepped-by-64 loop — algebraically identical (verified: the `qh` bit
/// position and elements' index both work out to depend only on `is`), but
/// easier to follow. Per sub-block `is`:
///   - reads a 32-byte window of `qs` at `qs_offset = 32*(is/2)` (`is` even →
///     low nibble, odd → high nibble — the two nibble-halves of one `qs`
///     byte window are two *different* 32-element output sub-blocks); `qh`
///     is indexed by `l` directly (0..32) — it is *not* windowed per `is`,
///     the same 32 bytes are reread for every sub-block, only the bit
///     position changes
///   - the elements it produces live at `elem_offset = 32*is` in both the
///     weight's dequantized row and the (quantized) activation `x`
///   - the `qh` high bit is at bit position `is` directly (see reference:
///     `u1`/`u2` start at 1/2 and shift left by 2 each outer iteration of 4,
///     covering bit positions 0,2,4,6 and 1,3,5,7 — i.e. exactly `is`).
/// `min_sum` reuses the already-computed `bsums` (sum of 16 consecutive
/// quantized activations) instead of re-summing — sub-block `is` spans
/// `bsums[2*is]` and `bsums[2*is+1]`.
pub fn dot_q5_k_q8k(data: &[u8], q8k: &crate::quant::q8_k::Q8KRow) -> Result<f32> {
    let n = q8k.n;
    let n_blocks = (n + Q5_K_BLOCK_SIZE - 1) / Q5_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q5_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q5_K x Q8_K dot: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }
    debug_assert_eq!(n_blocks, q8k.n_blocks());

    let mut sumf = 0.0f32;

    for block_idx in 0..n_blocks {
        let base = block_idx * Q5_K_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let qs = &data[base + 48..base + 176];

        let x_i8 = &q8k.qs[block_idx * Q5_K_BLOCK_SIZE..(block_idx + 1) * Q5_K_BLOCK_SIZE];
        let bsums = &q8k.bsums[block_idx * 16..(block_idx + 1) * 16];
        let dx = q8k.d[block_idx];

        let mut acc_pos: i32 = 0;
        let mut acc_min: i32 = 0;

        for is in 0..8usize {
            let (sc, m) = get_scale_min_k4(is, scales);
            let qs_offset = 32 * (is / 2);
            let elem_offset = 32 * is;
            let low_nibble = is % 2 == 0;
            let bit = is as u32;

            let mut pos_sum: i32 = 0;
            for l in 0..32 {
                let byte = qs[qs_offset + l];
                let nib = if low_nibble { byte & 0x0F } else { byte >> 4 };
                let hi = ((qh[l] >> bit) & 1) << 4;
                let w = (nib | hi) as i32;
                pos_sum += w * x_i8[elem_offset + l] as i32;
            }
            let min_sum = bsums[2 * is] as i32 + bsums[2 * is + 1] as i32;

            acc_pos += sc as i32 * pos_sum;
            acc_min += m as i32 * min_sum;
        }

        sumf += dx * (d * acc_pos as f32 - dmin * acc_min as f32);
    }

    Ok(sumf)
}

/// Fused Q5_K dequant + dot product: `sum_i dequant(row)[i] * x[i]`.
/// Uses `wide::f32x8` for SIMD vectorization — processes 8 quants at a time,
/// same structure as [`crate::quant::q4_k::dot_q4_k`] plus the high-bit lookup.
pub fn dot_q5_k(data: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    let n_blocks = (n + Q5_K_BLOCK_SIZE - 1) / Q5_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q5_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q5_K dot: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut acc = 0.0f32;

    for block_idx in 0..n_blocks {
        let base = block_idx * Q5_K_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let qs = &data[base + 48..base + 176];
        let block_out = block_idx * Q5_K_BLOCK_SIZE;

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..Q5_K_BLOCK_SIZE).step_by(64) {
            let qs_offset = j / 2;

            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            let (sc1, m1_s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_s as f32;

            acc += dot_q5_k_group(
                d1,
                m1,
                d2,
                m2,
                u1,
                u2,
                &qs[qs_offset..qs_offset + 32],
                qh,
                &x[block_out + j..],
                n.saturating_sub(block_out + j),
            );

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(acc)
}

/// SIMD dot product for one Q5_K group of 64 values (32 bytes of qs, plus the
/// shared 32-byte qh high-bit array). Processes 8 values (8 low nibbles + 8
/// high nibbles) per iteration, same layout as `dot_q4_k_group`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dot_q5_k_group(
    d1: f32,
    m1: f32,
    d2: f32,
    m2: f32,
    u1: u8,
    u2: u8,
    qs: &[u8],
    qh: &[u8],
    x: &[f32],
    n: usize,
) -> f32 {
    use wide::f32x8;

    let d1v = f32x8::splat(d1);
    let m1v = f32x8::splat(m1);
    let d2v = f32x8::splat(d2);
    let m2v = f32x8::splat(m2);
    let mut sum = f32x8::ZERO;

    // `u1`/`u2` are always single-bit masks (bit positions 0/2/4/6 and
    // 1/3/5/7 respectively, one pair per outer j-iteration in the caller).
    // Extracting the bit as `(h >> shift) & 1` instead of branching on
    // `h & mask != 0` keeps the per-lane array construction branch-free so
    // it auto-vectorizes instead of taking 8 scalar conditional jumps.
    let shift1 = u1.trailing_zeros();
    let shift2 = u2.trailing_zeros();

    // 4 iterations of 8 bytes each = 32 bytes of qs → 64 values
    let mut base = 0usize;
    while base < 32 {
        let q: [u8; 8] = qs[base..base + 8].try_into().unwrap();
        let h: [u8; 8] = qh[base..base + 8].try_into().unwrap();

        let lo = f32x8::from([
            ((q[0] & 0x0F) | (((h[0] >> shift1) & 1) << 4)) as f32,
            ((q[1] & 0x0F) | (((h[1] >> shift1) & 1) << 4)) as f32,
            ((q[2] & 0x0F) | (((h[2] >> shift1) & 1) << 4)) as f32,
            ((q[3] & 0x0F) | (((h[3] >> shift1) & 1) << 4)) as f32,
            ((q[4] & 0x0F) | (((h[4] >> shift1) & 1) << 4)) as f32,
            ((q[5] & 0x0F) | (((h[5] >> shift1) & 1) << 4)) as f32,
            ((q[6] & 0x0F) | (((h[6] >> shift1) & 1) << 4)) as f32,
            ((q[7] & 0x0F) | (((h[7] >> shift1) & 1) << 4)) as f32,
        ]);
        let hi = f32x8::from([
            ((q[0] >> 4) | (((h[0] >> shift2) & 1) << 4)) as f32,
            ((q[1] >> 4) | (((h[1] >> shift2) & 1) << 4)) as f32,
            ((q[2] >> 4) | (((h[2] >> shift2) & 1) << 4)) as f32,
            ((q[3] >> 4) | (((h[3] >> shift2) & 1) << 4)) as f32,
            ((q[4] >> 4) | (((h[4] >> shift2) & 1) << 4)) as f32,
            ((q[5] >> 4) | (((h[5] >> shift2) & 1) << 4)) as f32,
            ((q[6] >> 4) | (((h[6] >> shift2) & 1) << 4)) as f32,
            ((q[7] >> 4) | (((h[7] >> shift2) & 1) << 4)) as f32,
        ]);

        let x_lo = load_f32x8_partial(&x[base..], n.saturating_sub(base));
        sum += (d1v * lo - m1v) * x_lo;

        let x_hi = load_f32x8_partial(&x[32 + base..], n.saturating_sub(32 + base));
        sum += (d2v * hi - m2v) * x_hi;

        base += 8;
    }

    let arr: [f32; 8] = sum.to_array();
    arr.iter().sum()
}

/// Load up to 8 f32 values into f32x8, padding with 0.0 for out-of-bounds.
#[inline(always)]
fn load_f32x8_partial(slice: &[f32], available: usize) -> wide::f32x8 {
    let len = available.min(8);
    let mut arr = [0.0f32; 8];
    arr[..len].copy_from_slice(&slice[..len]);
    wide::f32x8::from(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_block(d: f32, dmin: f32, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        b.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        b.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&qs);
        b
    }

    #[test]
    fn test_dequant_q5_k_range() {
        // All quants max (0xFF qs -> nibble 15, qh all bits set -> +16 => 31).
        // Scale packing: sub-blocks 0..3 read `scales[0..4]` directly (low 6
        // bits = 63 needs 0xFF, since 0xFF & 0x3F == 63); sub-blocks 4..7 read
        // `scales[8..12] & 0xF` combined with `scales[0..4] >> 6 << 4` — 0xFF's
        // top 2 bits are `11`, so `3 << 4 == 48`, plus `0x0F & 0xF == 15` gives
        // `48 | 15 == 63` too. dmin=0 makes the min-packing irrelevant.
        let scales: [u8; 12] = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F];
        let qh = [0xFFu8; 32];
        let qs = [0xFFu8; 128];
        let data = build_block(1.0, 0.0, scales, qh, qs);
        let row = dequant_q5_k(&data, Q5_K_BLOCK_SIZE).unwrap();
        for v in row {
            assert!((v - (63.0 * 31.0)).abs() < 1.0);
        }
    }

    #[test]
    fn test_dequant_q5_k_zero_scale() {
        let scales: [u8; 12] = [0; 12];
        let qh = [0u8; 32];
        let qs = [0u8; 128];
        let data = build_block(1.0, 0.0, scales, qh, qs);
        let row = dequant_q5_k(&data, Q5_K_BLOCK_SIZE).unwrap();
        assert!(row.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dot_q5_k_matches_dequant() {
        let scales: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 0x3F, 0x2F, 0x1F, 0x0F];
        let qh: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
        let qs: [u8; 128] = core::array::from_fn(|i| (i as u8).wrapping_mul(13));
        let data = build_block(0.05, -0.01, scales, qh, qs);
        let n = Q5_K_BLOCK_SIZE;

        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 1.0)).collect();
        let row = dequant_q5_k(&data, n).unwrap();
        let expected: f32 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();

        let got = dot_q5_k(&data, &x).unwrap();
        assert!(
            (got - expected).abs() < 1e-3,
            "fused dot {got} != reference {expected}"
        );
    }

    #[test]
    fn test_dot_q5_k_q8k_matches_f32_dot() {
        use crate::quant::q8_k::quantize_row_q8_k;

        let scales: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 0x3F, 0x2F, 0x1F, 0x0F];
        let qh: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
        let qs: [u8; 128] = core::array::from_fn(|i| (i as u8).wrapping_mul(13));
        let data = build_block(0.05, -0.01, scales, qh, qs);
        let n = Q5_K_BLOCK_SIZE;

        // Realistic-ish activations (RMSNorm output magnitude).
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.037).sin()) * 1.3)
            .collect();

        let expected = dot_q5_k(&data, &x).unwrap();

        let q8k = quantize_row_q8_k(&x);
        let got = dot_q5_k_q8k(&data, &q8k).unwrap();

        // int8 activation quantization is lossy (~1/127 relative error per
        // element), so this is a much looser tolerance than the f32-exact
        // fused-vs-dequant tests above — it's checking the int8 pipeline is
        // *correct*, not bit-exact with the f32 path.
        let rel_err = (got - expected).abs() / expected.abs().max(1.0);
        assert!(
            rel_err < 0.05,
            "q8k dot {got} too far from f32 dot {expected} (rel err {rel_err})"
        );
    }
}
