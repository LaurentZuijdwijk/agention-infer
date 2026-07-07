//! Q8_0 dequantization.
//!
//! Block layout (34 bytes per block, 32 values per block):
//!   bytes [0..2]  : f16 scale (delta)
//!   bytes [2..34] : 32 × i8 quantized values
//!
//! Dequant: value[i] = delta * q[i]

use crate::error::{GgufError, Result};

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (i8 values)

pub fn dequant_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let n_blocks = (n_elements + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q8_0_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q8_0 dequant: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let base = block_idx * Q8_0_BLOCK_BYTES;

        // Read f16 scale
        let scale_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // Dequant 32 values
        let block_elements = if block_idx == n_blocks - 1 {
            // Last block may be partial
            n_elements - block_idx * Q8_0_BLOCK_SIZE
        } else {
            Q8_0_BLOCK_SIZE
        };

        for i in 0..block_elements {
            let q_byte = data[base + 2 + i];
            // Sign-extend u8 to i8
            let q_val = q_byte as i8 as f32;
            out.push(scale * q_val);
        }
    }

    Ok(out)
}

/// Fused Q8_0 dequant + dot product: computes `sum_i dequant(row)[i] * x[i]`
/// one block at a time, without ever materializing the row as `Vec<f32>`.
///
/// The scale is factored out of the inner loop — since every value in a block
/// shares the same `delta`, `sum(delta*q[i]*x[i]) == delta * sum(q[i]*x[i])`.
pub fn dot_q8_0(data: &[u8], x: &[f32]) -> Result<f32> {
    let n = x.len();
    let n_blocks = (n + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    let needed_bytes = n_blocks * Q8_0_BLOCK_BYTES;

    if data.len() < needed_bytes {
        return Err(GgufError::BackendError(format!(
            "Q8_0 dot: need {needed_bytes} bytes, have {}",
            data.len()
        )));
    }

    let mut acc = 0.0f32;

    for block_idx in 0..n_blocks {
        let base = block_idx * Q8_0_BLOCK_BYTES;

        let scale_bits = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        let block_start = block_idx * Q8_0_BLOCK_SIZE;
        let block_elements = (n - block_start).min(Q8_0_BLOCK_SIZE);

        let mut block_dot = 0.0f32;
        for i in 0..block_elements {
            let q = data[base + 2 + i] as i8 as f32;
            block_dot += q * x[block_start + i];
        }
        acc += scale * block_dot;
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_q8_0_block(scale_f16: u16, values: &[i8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        block.extend_from_slice(&scale_f16.to_le_bytes());
        for &v in values {
            block.push(v as u8);
        }
        // Pad to 34 bytes
        while block.len() < Q8_0_BLOCK_BYTES {
            block.push(0);
        }
        block
    }

    #[test]
    fn test_dequant_q8_0_single_block() {
        // Scale = 2.0 in f16 = 0x4000
        let scale_bits = half::f16::from_f32(2.0).to_bits();
        let values: Vec<i8> = (0..32).map(|i| (i % 5 - 2) as i8).collect();
        let data = build_q8_0_block(scale_bits, &values);

        let result = dequant_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!(
                (result[i] - 2.0 * values[i] as f32).abs() < 1e-6,
                "mismatch at index {i}: got {}, expected {}",
                result[i],
                2.0 * values[i] as f32
            );
        }
    }

    #[test]
    fn test_dequant_q8_0_partial_block() {
        // 33 elements → 2 blocks, second block has only 1 element
        let scale_bits = half::f16::from_f32(1.0).to_bits();
        let mut data = Vec::new();

        // Block 0: full
        let vals0: Vec<i8> = (0..32).map(|i| (i as i8)).collect();
        data.extend(build_q8_0_block(scale_bits, &vals0));

        // Block 1: partial (1 value = 42)
        let vals1 = [
            42i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
        ];
        data.extend(build_q8_0_block(scale_bits, &vals1));

        let result = dequant_q8_0(&data, 33).unwrap();
        assert_eq!(result.len(), 33);
        assert_eq!(result[32], 42.0);
    }

    #[test]
    fn test_dot_q8_0_matches_dequant() {
        // Two blocks + a partial third; fused dot must equal dequant-then-dot.
        let scale_bits = half::f16::from_f32(0.75).to_bits();
        let mut data = Vec::new();
        for b in 0..3 {
            let vals: Vec<i8> = (0..32).map(|i| ((i + b) % 7 - 3) as i8).collect();
            data.extend(build_q8_0_block(scale_bits, &vals));
        }
        let n = 70; // spans 3 blocks, last is partial (70 - 64 = 6 elements)

        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 - 3.0)).collect();
        let row = dequant_q8_0(&data, n).unwrap();
        let expected: f32 = row.iter().zip(&x).map(|(w, xi)| w * xi).sum();

        let got = dot_q8_0(&data, &x).unwrap();
        assert!(
            (got - expected).abs() < 1e-3,
            "fused dot {got} != reference {expected}"
        );
    }
}
