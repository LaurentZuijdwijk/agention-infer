pub mod q2_k;
pub mod q4_k;
pub mod q5_0;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_k;

// TODO(Phase 2b) — remaining dequant implementations:
//   - Q2_K         (256 bytes → 256 f32 values) — used in LFM2
//   - Q6_K         (210 bytes → 256 f32 values)
//   - IQ4_XS       (136 bytes → 256 f32, codebook lookup vs llama.cpp)
//   - IQ3_XXS      (98 bytes → 256 f32, codebook lookup)
//   - MXFP4        (E2M1 + E8M0 scale)
//   - Q4_K scale packing must be verified against llama.cpp ggml.c
//
// Happy path (Phase 4) only needs the types present in the target model.
// LFM2 uses Q4_K + F32 + Q2_K, so Q2_K is next in line.

use crate::error::{GgufError, Result};
use crate::types::{GgmlType, TensorInfo};

/// Dequantize one row of a tensor into f32 values.
///
/// `data` is the full tensor data section (starting at `GgufFile::data_offset`).
/// `row` is the 0-based row index.
///
/// Returns the dequantized f32 values for that row. For 1D tensors (norm weights),
/// the entire tensor is one "row".
pub fn dequant_row(tensor: &TensorInfo, data: &[u8], row: usize) -> Result<Vec<f32>> {
    let (row_data, row_len) = row_view(tensor, data, row)?;

    match tensor.ggml_type {
        GgmlType::F32 => Ok(dequant_f32(row_data, row_len)),
        GgmlType::F16 => Ok(dequant_f16(row_data, row_len)),
        GgmlType::BF16 => Ok(dequant_bf16(row_data, row_len)),
        GgmlType::Q8_0 => q8_0::dequant_q8_0(row_data, row_len),
        GgmlType::Q5_0 => q5_0::dequant_q5_0(row_data, row_len),
        GgmlType::Q4_K => q4_k::dequant_q4_k(row_data, row_len),
        GgmlType::Q5_K => q5_k::dequant_q5_k(row_data, row_len),
        GgmlType::Q6_K => q6_k::dequant_q6_k(row_data, row_len),
        GgmlType::Q2_K => q2_k::dequant_q2_k(row_data, row_len),
        other => Err(GgufError::BackendError(format!(
            "dequantization not yet implemented for {other}"
        ))),
    }
}

/// Fused Q5_K × Q8_K dot product for one tensor row, against an activation
/// already quantized once (by the caller, shared across every row of the
/// matmul) via [`q8_k::quantize_row_q8_k`]. See [`q5_k::dot_q5_k_q8k`].
pub fn dot_row_q5k_q8k(
    tensor: &TensorInfo,
    data: &[u8],
    row: usize,
    q8k: &q8_k::Q8KRow,
) -> Result<f32> {
    debug_assert_eq!(tensor.ggml_type, GgmlType::Q5_K);
    let (row_data, row_len) = row_view(tensor, data, row)?;
    debug_assert_eq!(row_len, q8k.n);
    q5_k::dot_q5_k_q8k(row_data, q8k)
}

/// Fused dequant + dot product of one tensor row with `x`.
///
/// Computes `sum_i dequant(row)[i] * x[i]` without materializing the row as a
/// `Vec<f32>`. Q8_0 and Q4_K have dedicated fused kernels; other formats fall
/// back to dequant-then-dot (they are not on the hot matmul path). `x.len()`
/// must equal the row length (`tensor.dims[0]`).
pub fn dot_row(tensor: &TensorInfo, data: &[u8], row: usize, x: &[f32]) -> Result<f32> {
    let (row_data, row_len) = row_view(tensor, data, row)?;
    debug_assert_eq!(
        row_len,
        x.len(),
        "dot_row: input length {} != row length {row_len}",
        x.len()
    );

    match tensor.ggml_type {
        GgmlType::Q8_0 => q8_0::dot_q8_0(row_data, x),
        GgmlType::Q4_K => q4_k::dot_q4_k(row_data, x),
        GgmlType::Q5_K => q5_k::dot_q5_k(row_data, x),
        GgmlType::Q6_K => q6_k::dot_q6_k(row_data, x),
        // Cold path: formats without a fused kernel dequant into a temp Vec.
        _ => {
            let row = dequant_row(tensor, data, row)?;
            Ok(row.iter().zip(x).map(|(w, xi)| w * xi).sum())
        }
    }
}

/// Resolve the byte slice and element count for one row of a tensor.
///
/// GGUF dimension order: `dims[0]` is the innermost (contiguous) dimension.
/// For a 2D weight tensor `[in_dim, out_dim]`, each row in memory has `dims[0]`
/// elements and there are `product(dims[1..])` rows. 1D tensors are a single row.
fn row_view<'a>(tensor: &TensorInfo, data: &'a [u8], row: usize) -> Result<(&'a [u8], usize)> {
    let n_dims = tensor.n_dims as usize;
    let ggml_type = tensor.ggml_type;

    let row_len: usize = if n_dims >= 1 {
        tensor.dims[0] as usize
    } else {
        return Err(GgufError::BackendError("tensor has 0 dimensions".into()));
    };

    let n_rows: usize = if n_dims == 1 {
        1
    } else {
        tensor.dims[1..].iter().product::<u64>() as usize
    };

    let row_byte_offset = if n_dims == 1 {
        tensor.byte_offset as usize
    } else {
        if row >= n_rows {
            return Err(GgufError::BackendError(format!(
                "row index {row} out of range (tensor has {n_rows} rows)"
            )));
        }
        let bytes_per_row = ggml_type.type_size(row_len);
        tensor.byte_offset as usize + row * bytes_per_row
    };

    if row_byte_offset + ggml_type.type_size(row_len) > data.len() {
        return Err(GgufError::BackendError(format!(
            "tensor row {row} extends beyond data section (offset {}, need {} bytes, have {})",
            row_byte_offset,
            ggml_type.type_size(row_len),
            data.len() - row_byte_offset
        )));
    }

    Ok((&data[row_byte_offset..], row_len))
}

/// Dequantize a row of F32 values (no-op, just reinterpret).
fn dequant_f32(data: &[u8], n: usize) -> Vec<f32> {
    let bytes = &data[..n * 4];
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Dequantize a row of F16 values using the `half` crate.
fn dequant_f16(data: &[u8], n: usize) -> Vec<f32> {
    let bytes = &data[..n * 2];
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Dequantize a row of BF16 values.
fn dequant_bf16(data: &[u8], n: usize) -> Vec<f32> {
    let bytes = &data[..n * 2];
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // BF16 is the upper 16 bits of an IEEE 754 f32
            let f32_bits = (bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_f32() {
        let data: Vec<u8> = [1.0f32, 2.0, -3.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequant_f32(&data, 3);
        assert_eq!(result, vec![1.0, 2.0, -3.5]);
    }

    #[test]
    fn test_dequant_f16() {
        let data: Vec<u8> = [half::f16::from_f32(1.5), half::f16::from_f32(-2.25)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequant_f16(&data, 2);
        assert!((result[0] - 1.5).abs() < 1e-3);
        assert!((result[1] - (-2.25)).abs() < 1e-3);
    }

    #[test]
    fn test_dequant_bf16() {
        // BF16: 1.0 = 0x3F80
        let data: Vec<u8> = [0x80u8, 0x3F, 0x00, 0xC0] // 1.0 and -2.0
            .to_vec();
        let result = dequant_bf16(&data, 2);
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - (-2.0)).abs() < 1e-2);
    }
}
