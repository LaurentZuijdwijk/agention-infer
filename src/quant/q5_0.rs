//! Q5_0 dequantization.
//!
//! Block layout (22 bytes per block, 32 values per block), matching
//! llama.cpp `block_q5_0`:
//!   bytes [0..2]   : f16 scale (delta `d`)
//!   bytes [2..6]   : `qh` — 32 bits, the 5th (high) bit of each quant
//!   bytes [6..22]  : `qs` — 16 bytes, two 4-bit low nibbles per byte
//!
//! For j in 0..16:
//!   x0 = ((qs[j] & 0x0F) | bit(qh, j)      << 4) - 16   → output[j]
//!   x1 = ((qs[j] >>   4) | bit(qh, j + 16) << 4) - 16   → output[j + 16]
//!   value = x * d

use crate::error::{GgufError, Result};

const Q5_0_BLOCK_SIZE: usize = 32;
const Q5_0_BLOCK_BYTES: usize = 22; // 2 (f16 d) + 4 (qh) + 16 (qs)
const HALF: usize = Q5_0_BLOCK_SIZE / 2; // 16

pub fn dequant_q5_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = (n_elements + Q5_0_BLOCK_SIZE - 1) / Q5_0_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q5_0_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q5_0 dequant: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n_blocks * Q5_0_BLOCK_SIZE];

    for block_idx in 0..n_blocks {
        let base = block_idx * Q5_0_BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        let qs = &data[base + 6..base + 6 + HALF];

        let out_base = block_idx * Q5_0_BLOCK_SIZE;
        for j in 0..HALF {
            let xh_0 = ((qh >> j) & 1) as u8; // high bit for x0
            let xh_1 = ((qh >> (j + HALF)) & 1) as u8; // high bit for x1

            let x0 = (((qs[j] & 0x0F) | (xh_0 << 4)) as i32) - 16;
            let x1 = (((qs[j] >> 4) | (xh_1 << 4)) as i32) - 16;

            out[out_base + j] = x0 as f32 * d;
            out[out_base + j + HALF] = x1 as f32 * d;
        }
    }

    out.truncate(n_elements);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference packer: given 32 signed values in [-16, 15] and a scale,
    /// build a Q5_0 block, so we can check the unpacker inverts it.
    fn build_q5_0_block(scale: f32, values: &[i32; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q5_0_BLOCK_BYTES);
        block.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());

        let mut qh: u32 = 0;
        let mut qs = [0u8; HALF];
        for j in 0..HALF {
            let v0 = (values[j] + 16) as u32; // 0..31
            let v1 = (values[j + HALF] + 16) as u32;
            if (v0 >> 4) & 1 == 1 {
                qh |= 1 << j;
            }
            if (v1 >> 4) & 1 == 1 {
                qh |= 1 << (j + HALF);
            }
            qs[j] = ((v0 & 0x0F) | ((v1 & 0x0F) << 4)) as u8;
        }
        block.extend_from_slice(&qh.to_le_bytes());
        block.extend_from_slice(&qs);
        block
    }

    #[test]
    fn roundtrip_single_block() {
        let scale = 0.5f32;
        let mut values = [0i32; 32];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as i32 % 32) - 16; // spans -16..15
        }
        let data = build_q5_0_block(scale, &values);
        let out = dequant_q5_0(&data, 32).unwrap();
        assert_eq!(out.len(), 32);
        for i in 0..32 {
            let expected = values[i] as f32 * scale;
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "idx {i}: got {}, want {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn high_bit_extremes() {
        // Value 15 needs the 5th bit set; -16 needs all-zero code.
        let scale = 1.0f32;
        let mut values = [0i32; 32];
        values[0] = 15; // code 31 → high bit set
        values[HALF] = -16; // code 0
        values[1] = -1; // code 15 → high bit clear, low nibble 1111
        let data = build_q5_0_block(scale, &values);
        let out = dequant_q5_0(&data, 32).unwrap();
        assert_eq!(out[0], 15.0);
        assert_eq!(out[HALF], -16.0);
        assert_eq!(out[1], -1.0);
    }
}
