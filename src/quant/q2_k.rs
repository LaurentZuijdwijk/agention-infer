//! Q2_K dequantization.
//!
//! Block layout (84 bytes per 256 elements = 2.625 bits/weight):
//!
//! ```c
//! typedef struct {
//!     uint8_t scales[QK_K/16]; // 16 bytes: scales and mins, 4 bits each
//!     uint8_t qs[QK_K/4];     // 64 bytes: 2-bit quantized values
//!     ggml_half d;             // 2 bytes: super-block scale
//!     ggml_half dmin;          // 2 bytes: super-block min scale
//! } block_q2_K;
//! ```
//!
//! Each super-block (256 elements) is split into 2 groups of 128.
//! Each group processes 4 sub-iterations with bit shifts 0,2,4,6.
//! Each sub-iteration reads 2 scales (lo/hi nibble) and produces
//! 2×16 = 32 values from the `qs` array.

use crate::error::{GgufError, Result};

const QK_K: usize = 256;

/// Byte size of one Q2_K super-block.
pub const BLOCK_BYTES: usize = 16 + 64 + 2 + 2; // scales + qs + d + dmin = 84

/// Dequantize a row of Q2_K data.
///
/// `data` starts at the beginning of the row's Q2_K blocks.
/// `row_len` is the number of f32 elements in the row (must be a multiple of 256).
pub fn dequant_q2_k(data: &[u8], row_len: usize) -> Result<Vec<f32>> {
    if row_len % QK_K != 0 {
        return Err(GgufError::BackendError(format!(
            "Q2_K row length {row_len} is not a multiple of {QK_K}"
        )));
    }

    let nb = row_len / QK_K;
    let expected_bytes = nb * BLOCK_BYTES;
    if data.len() < expected_bytes {
        return Err(GgufError::BackendError(format!(
            "Q2_K data too short: need {expected_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(row_len);

    for i in 0..nb {
        let block_start = i * BLOCK_BYTES;
        let block = &data[block_start..block_start + BLOCK_BYTES];

        // Parse block layout: scales[16] | qs[64] | d(f16) | dmin(f16)
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(block[80], block[81]);
        let dmin = f16_to_f32(block[82], block[83]);

        let mut is = 0usize; // scale index
        let mut q_offset = 0usize; // offset into qs

        // Process 2 groups of 128 elements each
        for _group in 0..2 {
            let mut shift: u8 = 0;
            // 4 sub-iterations per group
            for _j in 0..4 {
                // First sub-block: 16 elements from qs[q_offset..q_offset+16]
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let q_val = qs[q_offset + l];
                    let quant = ((q_val >> shift) & 3) as i8;
                    out.push(dl * quant as f32 - ml);
                }

                // Second sub-block: 16 elements from qs[q_offset+16..q_offset+32]
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    let q_val = qs[q_offset + 16 + l];
                    let quant = ((q_val >> shift) & 3) as i8;
                    out.push(dl * quant as f32 - ml);
                }

                shift += 2;
            }
            q_offset += 32; // Each group consumes 32 bytes of qs
        }
    }

    Ok(out)
}

/// Convert a little-endian f16 (IEEE 754 half) to f32.
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q2_k_block_size() {
        assert_eq!(BLOCK_BYTES, 84);
    }

    #[test]
    fn test_q2_k_zero_block() {
        // Create a zero block: all scales=0, qs=0, d=0, dmin=0
        let mut block = vec![0u8; BLOCK_BYTES];
        // Set d = f16(1.0) so we get non-trivial scaling
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[80] = (d_bits & 0xFF) as u8;
        block[81] = (d_bits >> 8) as u8;

        let result = dequant_q2_k(&block, QK_K).unwrap();
        assert_eq!(result.len(), QK_K);
        // With all qs=0 and scales=0, all values should be 0
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q2_k_basic_pattern() {
        // Create a block with known values:
        // Set scale[0] = 0x21 (lo=1, hi=2) so dl = d*1, ml = dmin*2
        // Set first 32 qs bytes to alternating values
        let mut block = vec![0u8; BLOCK_BYTES];

        // d = 2.0
        let d_bits = half::f16::from_f32(2.0).to_bits();
        block[80] = (d_bits & 0xFF) as u8;
        block[81] = (d_bits >> 8) as u8;

        // dmin = 0.5
        let dmin_bits = half::f16::from_f32(0.5).to_bits();
        block[82] = (dmin_bits & 0xFF) as u8;
        block[83] = (dmin_bits >> 8) as u8;

        // scales[0] = 0x21: lo nibble = 1, hi nibble = 2
        block[0] = 0x21;
        // scales[1] = 0x43: lo nibble = 3, hi nibble = 4
        block[1] = 0x43;

        // qs[0] = 0xFF (all 2-bit values = 3 for shift=0)
        for i in 16..48 {
            block[i] = 0xFF;
        }

        let result = dequant_q2_k(&block, QK_K).unwrap();
        assert_eq!(result.len(), QK_K);

        // First 16 values: dl = 2.0 * 1 = 2.0, ml = 0.5 * 2 = 1.0
        // quant = 3, so value = 2.0 * 3 - 1.0 = 5.0
        for i in 0..16 {
            assert!(
                (result[i] - 5.0).abs() < 1e-5,
                "result[{i}] = {}",
                result[i]
            );
        }

        // Next 16 values: dl = 2.0 * 3 = 6.0, ml = 0.5 * 4 = 2.0
        // quant = 3, so value = 6.0 * 3 - 2.0 = 16.0
        for i in 16..32 {
            assert!(
                (result[i] - 16.0).abs() < 1e-5,
                "result[{i}] = {}",
                result[i]
            );
        }
    }
}
