//! Q6_K dequantization.
//!
//! Super-block layout (210 bytes, 256 values per block), matching ggml's
//! `block_q6_K`:
//!
//!   uint8_t ql[128];     // quants, lower 4 bits  (QK_K/2)
//!   uint8_t qh[64];      // quants, upper 2 bits  (QK_K/4)
//!   int8_t  scales[16];  // per 16-value sub-block scales (QK_K/16), signed
//!   half    d;           // super-block scale
//!
//! Total: 128 + 64 + 16 + 2 = 210 bytes.
//!
//! Each 6-bit quant is `q = (low4 | (high2 << 4)) - 32` (range -32..31), scaled
//! by `d * scales[sub]`. The value/output-index mapping follows ggml exactly
//! (values within a super-block are written in a scattered order).

use crate::error::{GgufError, Result};

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

/// Decode the four 6-bit quants at position `l` within a 128-value half of a
/// super-block. Returns `(q1, q2, q3, q4)` already offset by -32.
#[inline]
fn quads(ql: &[u8], qh: &[u8], l: usize) -> (i32, i32, i32, i32) {
    let qh_l = qh[l];
    let q1 = ((ql[l] & 0x0F) as i32 | (((qh_l >> 0) & 3) as i32) << 4) - 32;
    let q2 = ((ql[l + 32] & 0x0F) as i32 | (((qh_l >> 2) & 3) as i32) << 4) - 32;
    let q3 = ((ql[l] >> 4) as i32 | (((qh_l >> 4) & 3) as i32) << 4) - 32;
    let q4 = ((ql[l + 32] >> 4) as i32 | (((qh_l >> 6) & 3) as i32) << 4) - 32;
    (q1, q2, q3, q4)
}

pub fn dequant_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = (n_elements + Q6_K_BLOCK_SIZE - 1) / Q6_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q6_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q6_K dequant: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n_blocks * Q6_K_BLOCK_SIZE];

    for block_idx in 0..n_blocks {
        let base = block_idx * Q6_K_BLOCK_BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let scales = &data[base + 192..base + 208]; // int8
        let d = half::f16::from_bits(u16::from_le_bytes([data[base + 208], data[base + 209]]))
            .to_f32();

        let block_out = block_idx * Q6_K_BLOCK_SIZE;

        // Two 128-value halves per super-block.
        for h in 0..2 {
            let ql_h = &ql[h * 64..];
            let qh_h = &qh[h * 32..];
            let sc_h = h * 8;
            let y_h = block_out + h * 128;

            for l in 0..32 {
                let is = l / 16;
                let (q1, q2, q3, q4) = quads(ql_h, qh_h, l);
                out[y_h + l] = d * scales[sc_h + is] as i8 as f32 * q1 as f32;
                out[y_h + l + 32] = d * scales[sc_h + is + 2] as i8 as f32 * q2 as f32;
                out[y_h + l + 64] = d * scales[sc_h + is + 4] as i8 as f32 * q3 as f32;
                out[y_h + l + 96] = d * scales[sc_h + is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }

    out.truncate(n_elements);
    Ok(out)
}

/// Fused Q6_K dequant + dot product: `sum_i dequant(row)[i] * x[i]` without
/// materializing the row. Uses the exact same value/scale arithmetic as
/// [`dequant_q6_k`]. `x.len()` equals the number of weights in the row.
pub fn dot_q6_k(data: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    let n_blocks = (n + Q6_K_BLOCK_SIZE - 1) / Q6_K_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q6_K_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q6_K dot: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut acc = 0.0f32;

    for block_idx in 0..n_blocks {
        let base = block_idx * Q6_K_BLOCK_BYTES;
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let scales = &data[base + 192..base + 208];
        let d = half::f16::from_bits(u16::from_le_bytes([data[base + 208], data[base + 209]]))
            .to_f32();

        let block_out = block_idx * Q6_K_BLOCK_SIZE;

        for h in 0..2 {
            let ql_h = &ql[h * 64..];
            let qh_h = &qh[h * 32..];
            let sc_h = h * 8;
            let y_h = block_out + h * 128;

            for l in 0..32 {
                let is = l / 16;
                let (q1, q2, q3, q4) = quads(ql_h, qh_h, l);
                let contrib = |idx: usize, sc: i32, q: i32| -> f32 {
                    if idx < n {
                        d * sc as f32 * q as f32 * x[idx]
                    } else {
                        0.0
                    }
                };
                acc += contrib(y_h + l, scales[sc_h + is] as i8 as i32, q1);
                acc += contrib(y_h + l + 32, scales[sc_h + is + 2] as i8 as i32, q2);
                acc += contrib(y_h + l + 64, scales[sc_h + is + 4] as i8 as i32, q3);
                acc += contrib(y_h + l + 96, scales[sc_h + is + 6] as i8 as i32, q4);
            }
        }
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one synthetic 210-byte Q6_K super-block.
    fn build_block(ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f32) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q6_K_BLOCK_BYTES);
        b.extend_from_slice(&ql);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&scales.map(|s| s as u8));
        b.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        b
    }

    #[test]
    fn test_dot_q6_k_matches_dequant() {
        let ql: [u8; 128] = core::array::from_fn(|i| (i as u8).wrapping_mul(37));
        let qh: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(53).wrapping_add(1));
        let scales: [i8; 16] = core::array::from_fn(|i| (i as i8) - 6);
        let data = build_block(ql, qh, scales, 0.02);
        let n = Q6_K_BLOCK_SIZE;

        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.005 - 0.6)).collect();
        let row = dequant_q6_k(&data, n).unwrap();
        let expected: f32 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();

        let got = dot_q6_k(&data, &x).unwrap();
        assert!(
            (got - expected).abs() < 1e-2,
            "fused dot {got} != reference {expected}"
        );
    }
}
