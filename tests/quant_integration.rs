use gguf_rs::{load, quant::dequant_row};

/// Dequantize real tensors from the Qwen3-0.6B model (Q8_0 weights) and sanity
/// check that values are finite and non-trivial. This is the model the
/// inference pipeline is exercised against end-to-end.
#[test]
fn test_dequant_real_model() {
    let path = std::path::Path::new("./model/Qwen3-0.6B-Q8_0.gguf");
    if !path.exists() {
        eprintln!("Skipping: model file not found");
        return;
    }

    let (gguf, mmap) = load(path).unwrap();

    // token_embd.weight is Q8_0 in this model.
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    assert_eq!(tensor.ggml_type, gguf_rs::GgmlType::Q8_0);

    let data_offset = gguf.data_offset as usize;
    let data = &mmap[data_offset..];

    // Dequantize row 0 (one token's embedding vector).
    let row = dequant_row(tensor, data, 0).unwrap();
    assert_eq!(row.len(), 1024); // embedding dimension

    let has_nan = row.iter().any(|v| !v.is_finite());
    assert!(!has_nan, "dequant produced NaN/inf values");
    let sum: f32 = row.iter().sum();
    assert!(
        sum.abs() > 0.001,
        "dequant produced near-zero values (sum={sum})"
    );

    // Also test an F32 norm weight.
    let norm = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
        .unwrap();
    assert_eq!(norm.ggml_type, gguf_rs::GgmlType::F32);

    let norm_row = dequant_row(norm, data, 0).unwrap();
    assert_eq!(norm_row.len(), 1024);
    let has_nan = norm_row.iter().any(|v| !v.is_finite());
    assert!(!has_nan, "F32 dequant produced NaN");
}
