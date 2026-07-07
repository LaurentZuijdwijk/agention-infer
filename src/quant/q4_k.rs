//! Q4_K dequantization.
//!
//! Block layout (144 bytes per block, 256 values per block), matching
//! llama.cpp `block_q4_K`:
//!
//!   struct block_q4_K {
//!       half d;             // super-block scale (2 bytes)
//!       half dmin;          // super-block minimum (2 bytes)
//!       uint8_t scales[12]; // packed sub-block scales and mins (12 bytes)
//!       uint8_t qs[128];    // 4-bit packed quants (256 values / 2 per byte)
//!   };
//!
//! Total: 2 + 2 + 12 + 128 = 144 bytes.
//!
//! Dequant formula: value = d * sc * quant - min * m
//! where quant is the 4-bit nibble (0..15).

use crate::error::{GgufError, Result};

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

/// Unpack the scale (`d`) and minimum (`m`) for sub-block index `j` (0..7)
/// from the 12-byte `scales` array. Matches llama.cpp `get_scale_min_k4`.
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

pub fn dequant_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = (n_elements + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q4_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q4_K dequant: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n_blocks * Q4_K_BLOCK_SIZE];

    for block_idx in 0..n_blocks {
        let base = block_idx * Q4_K_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];

        let block_out = block_idx * Q4_K_BLOCK_SIZE;

        let mut is = 0usize;
        for j in (0..Q4_K_BLOCK_SIZE).step_by(64) {
            let qs_offset = j / 2;

            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            let (sc1, m1_s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_s as f32;

            // First 32 values: low nibbles
            for l in 0..32 {
                let quant = (qs[qs_offset + l] & 0x0F) as f32;
                out[block_out + j + l] = d1 * quant - m1;
            }
            // Next 32 values: high nibbles
            for l in 0..32 {
                let quant = (qs[qs_offset + l] >> 4) as f32;
                out[block_out + j + 32 + l] = d2 * quant - m2;
            }

            is += 2;
        }
    }

    out.truncate(n_elements);
    Ok(out)
}

/// Fused Q4_K dequant + dot product: `sum_i dequant(row)[i] * x[i]`.
/// Uses `wide::f32x8` for SIMD vectorization — processes 8 nibbles at a time.
pub fn dot_q4_k(data: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    let n_blocks = (n + Q4_K_BLOCK_SIZE - 1) / Q4_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q4_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q4_K dot: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut acc = 0.0f32;

    for block_idx in 0..n_blocks {
        let base = block_idx * Q4_K_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];
        let block_out = block_idx * Q4_K_BLOCK_SIZE;

        let mut is = 0usize;
        for j in (0..Q4_K_BLOCK_SIZE).step_by(64) {
            let qs_offset = j / 2;

            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;

            let (sc1, m1_s) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_s as f32;

            acc += dot_q4_k_group(
                d1,
                m1,
                d2,
                m2,
                &qs[qs_offset..qs_offset + 32],
                &x[block_out + j..],
                n.saturating_sub(block_out + j),
            );

            is += 2;
        }
    }

    Ok(acc)
}

/// SIMD dot product for one Q4_K group of 64 values (32 bytes of qs).
/// Processes 8 bytes (16 nibbles) per iteration: 8 low nibbles + 8 high nibbles.
#[inline(always)]
fn dot_q4_k_group(d1: f32, m1: f32, d2: f32, m2: f32, qs: &[u8], x: &[f32], n: usize) -> f32 {
    use wide::f32x8;

    let d1v = f32x8::splat(d1);
    let m1v = f32x8::splat(m1);
    let d2v = f32x8::splat(d2);
    let m2v = f32x8::splat(m2);
    let mut sum = f32x8::ZERO;

    // 4 iterations of 8 bytes each = 32 bytes of qs → 64 values
    let mut base = 0usize;
    while base < 32 {
        // Load 8 qs bytes
        let q: [u8; 8] = qs[base..base + 8].try_into().unwrap();

        // Extract low and high nibbles
        let lo = f32x8::from([
            (q[0] & 0x0F) as f32,
            (q[1] & 0x0F) as f32,
            (q[2] & 0x0F) as f32,
            (q[3] & 0x0F) as f32,
            (q[4] & 0x0F) as f32,
            (q[5] & 0x0F) as f32,
            (q[6] & 0x0F) as f32,
            (q[7] & 0x0F) as f32,
        ]);
        let hi = f32x8::from([
            (q[0] >> 4) as f32,
            (q[1] >> 4) as f32,
            (q[2] >> 4) as f32,
            (q[3] >> 4) as f32,
            (q[4] >> 4) as f32,
            (q[5] >> 4) as f32,
            (q[6] >> 4) as f32,
            (q[7] >> 4) as f32,
        ]);

        // Low nibble dot: (d1*quant - m1) * x[base..base+8]
        let x_lo = load_f32x8_partial(&x[base..], n.saturating_sub(base));
        sum += (d1v * lo - m1v) * x_lo;

        // High nibble dot: (d2*quant - m2) * x[base+32..base+40]
        let x_hi = load_f32x8_partial(&x[32 + base..], n.saturating_sub(32 + base));
        sum += (d2v * hi - m2v) * x_hi;

        base += 8;
    }

    // Horizontal sum
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

    /// Build one synthetic 144-byte Q4_K super-block with the given raw fields.
    fn build_block(d: f32, dmin: f32, scales: [u8; 12], quants: [u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q4_K_BLOCK_BYTES);
        b.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        b.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&quants);
        b
    }

    #[test]
    fn test_get_scale_min_k4() {
        let scales: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 0, 0, 0, 0];
        assert_eq!(get_scale_min_k4(0, &scales), (10, 50));
        assert_eq!(get_scale_min_k4(1, &scales), (20, 60));
        assert_eq!(get_scale_min_k4(2, &scales), (30, 6));
        assert_eq!(get_scale_min_k4(3, &scales), (40, 16));

        let scales: [u8; 12] = [
            0xC0, 0x80, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3F, 0x2F, 0x1F, 0x0F,
        ];
        assert_eq!(get_scale_min_k4(4, &scales), (0x3F, 3));
        assert_eq!(get_scale_min_k4(5, &scales), (0x2F, 2));
    }

    #[test]
    fn test_dot_q4_k_matches_dequant() {
        let scales: [u8; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 0x3F, 0x2F, 0x1F, 0x0F];
        let quants: [u8; 128] = core::array::from_fn(|i| (i as u8).wrapping_mul(13));
        let data = build_block(0.05, -0.01, scales, quants);
        let n = Q4_K_BLOCK_SIZE;

        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 1.0)).collect();
        let row = dequant_q4_k(&data, n).unwrap();
        let expected: f32 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();

        let got = dot_q4_k(&data, &x).unwrap();
        assert!(
            (got - expected).abs() < 1e-3,
            "fused dot {got} != reference {expected}"
        );
    }
}
