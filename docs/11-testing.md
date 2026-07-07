# Testing Strategy

## Philosophy

**llama.cpp is the oracle, not the codebase.**

At temperature=0 (greedy decode), a correct LLM inference engine produces
deterministic, mathematically defined output. llama.cpp's output on the same
model and prompt is the ground truth. Every component in gguf-rs is validated
against either a mathematical reference (numpy/Python) or llama.cpp's output.
We never guess whether output is correct — we verify it.

Three tiers:

```
Unit tests      — no real model files, no GPU, fast, run on every commit
Integration     — require real GGUF files, CPU only, run on merge
Golden tests    — require real GGUF files + llama.cpp, full validation
Performance     — benchmarks, run manually before releases
```

---

## Tier 1: Unit Tests

No model files. No GPU. No network. Run in under 10 seconds. Every component
tested in isolation with hand-crafted inputs.

### Parser Tests

`tests/parser_tests.rs`

Hand-craft minimal valid GGUF byte sequences and verify the parser handles them
correctly, including all error cases.

```rust
fn gguf_header(version: u32, n_tensors: u64, n_kv: u64) -> Vec<u8> {
    let mut b = vec![];
    b.extend_from_slice(b"GGUF");              // magic
    b.extend_from_slice(&version.to_le_bytes());
    b.extend_from_slice(&n_tensors.to_le_bytes());
    b.extend_from_slice(&n_kv.to_le_bytes());
    b
}

fn gguf_string(s: &str) -> Vec<u8> {
    let mut b = vec![];
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
    b
}

fn gguf_kv_u32(key: &str, val: u32) -> Vec<u8> {
    let mut b = gguf_string(key);
    b.extend_from_slice(&4u32.to_le_bytes());  // type = UINT32
    b.extend_from_slice(&val.to_le_bytes());
    b
}

#[test]
fn parses_empty_model() {
    let mut bytes = gguf_header(3, 0, 0);
    // 32-byte alignment padding
    bytes.extend(vec![0u8; 32 - (bytes.len() % 32)]);

    let result = parse(&bytes);
    assert!(result.is_ok());
    let gguf = result.unwrap();
    assert_eq!(gguf.version, 3);
    assert!(gguf.tensors.is_empty());
    assert!(gguf.metadata.is_empty());
}

#[test]
fn parses_string_metadata() {
    let mut bytes = gguf_header(3, 0, 1);
    bytes.extend(gguf_string("general.architecture"));
    bytes.extend_from_slice(&8u32.to_le_bytes());  // type = STRING
    bytes.extend(gguf_string("llama"));
    bytes.extend(vec![0u8; 32]);

    let gguf = parse(&bytes).unwrap();
    assert_eq!(
        gguf.get_string("general.architecture").unwrap(),
        "llama"
    );
}

#[test]
fn parses_array_metadata() {
    let mut bytes = gguf_header(3, 0, 1);
    bytes.extend(gguf_string("tokenizer.ggml.tokens"));
    bytes.extend_from_slice(&9u32.to_le_bytes());   // type = ARRAY
    bytes.extend_from_slice(&8u32.to_le_bytes());   // elem type = STRING
    bytes.extend_from_slice(&3u64.to_le_bytes());   // count = 3
    bytes.extend(gguf_string("<unk>"));
    bytes.extend(gguf_string("<s>"));
    bytes.extend(gguf_string("</s>"));
    bytes.extend(vec![0u8; 32]);

    let gguf = parse(&bytes).unwrap();
    let arr = gguf.get_string_array("tokenizer.ggml.tokens").unwrap();
    assert_eq!(arr, vec!["<unk>", "<s>", "</s>"]);
}

#[test]
fn rejects_bad_magic() {
    let bytes = b"GGML\x03\x00\x00\x00".to_vec();
    assert!(matches!(parse(&bytes), Err(GgufError::InvalidMagic)));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&1u32.to_le_bytes());  // version 1 = unsupported
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert!(matches!(parse(&bytes), Err(GgufError::UnsupportedVersion(1))));
}

#[test]
fn rejects_truncated_file() {
    let bytes = b"GGUF\x03\x00\x00\x00".to_vec();  // truncated after version
    assert!(matches!(parse(&bytes), Err(GgufError::UnexpectedEof(_))));
}

#[test]
fn parses_tensor_info() {
    let mut bytes = gguf_header(3, 1, 0);
    // one tensor: "token_embd.weight", shape [32000, 4096], F32, offset 0
    bytes.extend(gguf_string("token_embd.weight"));
    bytes.extend_from_slice(&2u32.to_le_bytes());      // n_dims = 2
    bytes.extend_from_slice(&32000u64.to_le_bytes());  // dim 0
    bytes.extend_from_slice(&4096u64.to_le_bytes());   // dim 1
    bytes.extend_from_slice(&0u32.to_le_bytes());      // dtype = F32
    bytes.extend_from_slice(&0u64.to_le_bytes());      // offset = 0
    // padding
    let pad = 32 - (bytes.len() % 32);
    bytes.extend(vec![0u8; pad]);
    // tensor data (minimal)
    bytes.extend(vec![0u8; 32000 * 4096 * 4]);

    let gguf = parse(&bytes).unwrap();
    assert_eq!(gguf.tensors.len(), 1);
    assert_eq!(gguf.tensors[0].name, "token_embd.weight");
    assert_eq!(gguf.tensors[0].shape, vec![32000, 4096]);
    assert_eq!(gguf.tensors[0].dtype, GgmlType::F32);
}

#[test]
fn moe_detection_from_metadata() {
    let mut bytes = gguf_header(3, 0, 3);
    bytes.extend(gguf_kv_string("general.architecture", "mixtral"));
    bytes.extend(gguf_kv_u32("mixtral.expert_count", 8));
    bytes.extend(gguf_kv_u32("mixtral.expert_used_count", 2));
    bytes.extend(vec![0u8; 32]);

    let gguf = parse(&bytes).unwrap();
    let info = ModelInfo::from_gguf(&gguf).unwrap();
    assert!(info.is_moe());
    assert_eq!(info.moe.as_ref().unwrap().expert_count, 8);
    assert_eq!(info.moe.as_ref().unwrap().expert_used_count, 2);
}

#[test]
fn dense_model_has_no_moe() {
    let mut bytes = gguf_header(3, 0, 1);
    bytes.extend(gguf_kv_string("general.architecture", "llama"));
    bytes.extend(vec![0u8; 32]);

    let gguf = parse(&bytes).unwrap();
    let info = ModelInfo::from_gguf(&gguf).unwrap();
    assert!(!info.is_moe());
    assert!(info.moe.is_none());
}
```

### Dequantization Tests

`tests/quant_tests.rs`

Round-trip each quantization format: quantize known values, dequantize, verify
within tolerance. Cross-check against precomputed values from Python/numpy.

```rust
// Precomputed with Python:
// import numpy as np
// values = np.array([0.5, -0.3, 1.2, -0.8, 0.0, 0.7, -1.1, 0.4, ...], dtype=np.float32)
// # quantize to Q8_0
// delta = np.max(np.abs(values)) / 127.0
// qs = np.round(values / delta).astype(np.int8)
// # store block: [delta as f16][qs as i8 x 32]

const Q8_0_BLOCK: [u8; 34] = [
    // delta = 0.00944 as f16 LE = [0x0D, 0x24]
    0x0D, 0x24,
    // 32 quantized i8 values
    53, 0xE2u8 as u8, 127, 0xABu8 as u8, 0, 74, 0x94u8 as u8, 42,
    // ... (fill with precomputed values)
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
];

#[test]
fn q8_0_dequant_matches_reference() {
    let out = dequant_q8_0_block(&Q8_0_BLOCK);
    // first value: delta * qs[0] = 0.00944 * 53 = 0.500...
    assert!((out[0] - 0.500).abs() < 1e-3);
    // second value: delta * (-30) = 0.00944 * (-30) = -0.283...
    assert!((out[1] - (-0.283)).abs() < 1e-3);
}

#[test]
fn q8_0_round_trip_preserves_values() {
    let original: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let block = quantize_q8_0(&original);
    let recovered = dequant_q8_0_block(&block);

    for (orig, rec) in original.iter().zip(recovered.iter()) {
        // Q8_0 should recover to within ~1% of original range
        assert!((orig - rec).abs() < 0.02, "orig={orig} rec={rec}");
    }
}

#[test]
fn f16_dequant_exact() {
    // f16 value 1.5 = 0x3E00 in little-endian
    let bytes = [0x00u8, 0x3E];
    let result = dequant_f16_single(&bytes);
    assert!((result - 1.5).abs() < 1e-6);
}

#[test]
fn iq4_xs_uses_codebook() {
    // Verify that dequant produces values from the codebook, not arbitrary values
    let block = [0u8; 140];  // all indices = 0
    let out = dequant_iq4_xs_block(&block, 0);  // sub-block 0, scale=1.0
    // All nibbles are 0 → all values should be IQ4_CODEBOOK[0]
    for &v in &out {
        assert!((v - IQ4_CODEBOOK[0]).abs() < 1e-6);
    }
}

#[test]
fn mxfp4_e2m1_values() {
    // Test specific E2M1 bit patterns
    assert!((e2m1_to_f32(0b0000) - 0.0).abs() < 1e-6);   // +0
    assert!((e2m1_to_f32(0b0001) - 0.5).abs() < 1e-6);   // +0.5
    assert!((e2m1_to_f32(0b0010) - 1.0).abs() < 1e-6);   // +1.0
    assert!((e2m1_to_f32(0b0011) - 1.5).abs() < 1e-6);   // +1.5
    assert!((e2m1_to_f32(0b0100) - 2.0).abs() < 1e-6);   // +2.0
    assert!((e2m1_to_f32(0b1000) - 0.0).abs() < 1e-6);   // -0
    assert!((e2m1_to_f32(0b1001) - (-0.5)).abs() < 1e-6); // -0.5
}
```

### Ops Tests

`tests/ops_tests.rs`

Each operation verified against precomputed numpy values. The Python script
that generates expected values lives in `tests/fixtures/generate.py` and is
committed to the repo.

```python
# tests/fixtures/generate.py
# Run once to regenerate expected values
import numpy as np, json

def rms_norm(x, w, eps=1e-5):
    rms = np.sqrt(np.mean(x**2) + eps)
    return w * (x / rms)

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def silu(x):
    return x / (1 + np.exp(-x))

def rope_one_head(h, pos, theta=10000.0):
    d = len(h) // 2
    result = h.copy()
    for i in range(d):
        freq = pos / (theta ** (2 * i / len(h)))
        result[i]     = h[i] * np.cos(freq) - h[i+d] * np.sin(freq)
        result[i+d]   = h[i] * np.sin(freq) + h[i+d] * np.cos(freq)
    return result

np.random.seed(42)
x     = np.random.randn(16).astype(np.float32)
w     = np.random.randn(16).astype(np.float32)
mat_a = np.random.randn(4, 8).astype(np.float32)
vec_b = np.random.randn(8).astype(np.float32)

fixtures = {
    "rms_norm_input":    x.tolist(),
    "rms_norm_weight":   w.tolist(),
    "rms_norm_expected": rms_norm(x, w).tolist(),
    "softmax_input":     x.tolist(),
    "softmax_expected":  softmax(x).tolist(),
    "silu_input":        x.tolist(),
    "silu_expected":     silu(x).tolist(),
    "matmul_a":          mat_a.tolist(),
    "matmul_b":          vec_b.tolist(),
    "matmul_expected":   (mat_a @ vec_b).tolist(),
    "rope_head":         x.tolist(),
    "rope_pos":          5,
    "rope_expected":     rope_one_head(x, 5).tolist(),
}
print(json.dumps(fixtures, indent=2))
```

```rust
// tests/ops_tests.rs
fn load_fixtures() -> serde_json::Value {
    let s = std::fs::read_to_string("tests/fixtures/ops_expected.json").unwrap();
    serde_json::from_str(&s).unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "{label}[{i}]: got {x}, expected {y}, diff {}",
            (x-y).abs()
        );
    }
}

#[test]
fn rms_norm_matches_numpy() {
    let fix = load_fixtures();
    let x: Vec<f32> = fix["rms_norm_input"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let w: Vec<f32> = fix["rms_norm_weight"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let expected: Vec<f32> = fix["rms_norm_expected"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let result = rms_norm(&x, &w, 1e-5);
    assert_close(&result, &expected, 1e-5, "rms_norm");
}

#[test]
fn softmax_matches_numpy() {
    let fix = load_fixtures();
    let mut x: Vec<f32> = fix["softmax_input"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let expected: Vec<f32> = fix["softmax_expected"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    softmax(&mut x);
    assert_close(&x, &expected, 1e-5, "softmax");
}

#[test]
fn softmax_sums_to_one() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    softmax(&mut x);
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
}

#[test]
fn softmax_stable_with_large_values() {
    // naive softmax overflows on large inputs — stable version must not
    let mut x = vec![1000.0f32, 1001.0, 1002.0];
    softmax(&mut x);
    assert!(x.iter().all(|v| v.is_finite()), "softmax produced inf/nan");
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn matmul_matches_numpy() {
    let fix = load_fixtures();
    let a: Vec<f32> = /* load mat_a flat */ ...;
    let b: Vec<f32> = /* load vec_b */ ...;
    let expected: Vec<f32> = /* load matmul_expected */ ...;

    // build a fake F32 tensor for a
    let tensor = Tensor::from_f32(&a, &[4, 8]);
    let mut out = vec![0f32; 4];
    matmul(&mut out, &b, &tensor);
    assert_close(&out, &expected, 1e-4, "matmul");
}

#[test]
fn rope_matches_numpy() {
    let fix = load_fixtures();
    let head: Vec<f32> = fix["rope_head"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let pos = fix["rope_pos"].as_u64().unwrap() as usize;
    let expected: Vec<f32> = fix["rope_expected"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let mut q = head.clone();
    let mut k = head.clone();
    rope_one_head(&mut q, pos, 10000.0);
    assert_close(&q, &expected, 1e-4, "rope");
}

#[test]
fn no_nan_propagation() {
    // If any intermediate produces NaN, catch it early
    let x = vec![0.0f32; 4096];
    let w = vec![0u8; 4096 / 32 * 34];   // Q8_0 zero block
    let mut out = vec![0f32; 4096];

    let tensor = Tensor { data: &w, dtype: GgmlType::Q8_0, shape: vec![4096, 4096] };
    matmul(&mut out, &x, &tensor);

    assert!(
        out.iter().all(|v| v.is_finite()),
        "matmul produced nan/inf on zero inputs"
    );
}
```

### TurboQuant Tests

`tests/turbo_quant_tests.rs`

```rust
#[test]
fn wht_is_its_own_inverse() {
    let original: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
    let mut x = original.clone();
    wht_in_place(&mut x);
    wht_in_place(&mut x);  // apply twice = identity
    for (orig, result) in original.iter().zip(x.iter()) {
        assert!((orig - result).abs() < 1e-4, "WHT not self-inverse");
    }
}

#[test]
fn wht_preserves_inner_products() {
    // Orthogonal transforms preserve dot products
    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.05).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32 * 0.03).collect();

    let dot_before: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let mut a_rot = a.clone();
    let mut b_rot = b.clone();
    wht_in_place(&mut a_rot);
    wht_in_place(&mut b_rot);

    let dot_after: f32 = a_rot.iter().zip(b_rot.iter()).map(|(x, y)| x * y).sum();

    assert!((dot_before - dot_after).abs() < 1e-3,
        "WHT changed dot product: {} vs {}", dot_before, dot_after);
}

#[test]
fn turbo_quant_round_trip_quality() {
    // At 3-bit, recovered vectors should be close to original
    let head: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();
    let config = TurboQuantConfig { bits: 3, use_qjl: false };
    let mut cache = KvCache::new(1, 1, 128, 1, config);

    cache.write(0, 0, &head, &head);
    let (k_recovered, _) = cache.read_k_head(0, 0, 0);

    // Compute cosine similarity — should be very high
    let dot: f32 = head.iter().zip(k_recovered.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = head.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = k_recovered.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = dot / (norm_a * norm_b);

    assert!(cosine > 0.95, "TurboQuant cosine similarity too low: {cosine}");
}

#[test]
fn turbo_quant_memory_reduction() {
    let n_layers = 32;
    let n_kv_heads = 8;
    let head_dim = 128;
    let max_seq = 1000;

    let cache_16bit_bytes = n_layers * max_seq * n_kv_heads * head_dim * 2 * 2;
    let config = TurboQuantConfig { bits: 3, use_qjl: false };
    let cache = KvCache::new(n_layers, n_kv_heads, head_dim, max_seq, config);
    let cache_3bit_bytes = cache.memory_bytes();

    let ratio = cache_16bit_bytes as f32 / cache_3bit_bytes as f32;
    assert!(ratio > 4.0, "Expected >4× compression, got {ratio:.1}×");
}
```

### Tokenizer Tests

`tests/tokenizer_tests.rs`

```rust
#[test]
fn encode_decode_round_trip() {
    // Build a minimal tokenizer from known vocab
    let vocab = vec!["<unk>", "▁Hello", "▁world", "!"];
    let tokenizer = Tokenizer::from_vocab(&vocab, &[0.0; 4], 0, 1);

    let tokens = tokenizer.encode("Hello world!");
    let text = tokenizer.decode(&tokens);
    assert_eq!(text, "Hello world!");
}

#[test]
fn special_tokens_not_in_output() {
    let tokenizer = /* load from test fixture */;
    let tokens = vec![BOS_TOKEN, 1234, 5678, EOS_TOKEN];
    let text = tokenizer.decode(&tokens);
    assert!(!text.contains("<s>"), "BOS should not appear in decoded text");
    assert!(!text.contains("</s>"), "EOS should not appear in decoded text");
}

#[test]
fn empty_string_encodes_to_bos_only() {
    let tokenizer = /* load from test fixture */;
    let tokens = tokenizer.encode("");
    // Some models: [] others: [BOS]. Either is valid, just be consistent.
    assert!(tokens.len() <= 1);
}

#[test]
fn leading_space_changes_token() {
    // " hello" and "hello" are different tokens in SentencePiece
    let tokenizer = /* load from test fixture */;
    let a = tokenizer.encode("hello");
    let b = tokenizer.encode(" hello");
    assert_ne!(a, b, "leading space should produce different token");
}
```

---

## Tier 2: Integration Tests

Require real GGUF model files. Gated behind a feature flag so CI doesn't
require models.

```toml
# Cargo.toml
[features]
integration = []   # enable with: cargo test --features integration
```

```rust
// Only compiled when --features integration is passed
#[cfg(feature = "integration")]
mod integration {
    const TEST_MODEL: &str = "test-models/llama-3.2-1b-q8_0.gguf";

    fn model_path() -> PathBuf {
        let p = PathBuf::from(TEST_MODEL);
        if !p.exists() {
            panic!(
                "Integration test model not found at {TEST_MODEL}.\n\
                 Download: huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
                 Llama-3.2-1B-Instruct-Q8_0.gguf --local-dir test-models/"
            );
        }
        p
    }
}
```

### Model Loading Tests

```rust
#[test]
#[cfg(feature = "integration")]
fn loads_llama_1b_metadata() {
    let mmap  = load_mmap(&model_path()).unwrap();
    let gguf  = parse(&mmap).unwrap();
    let info  = ModelInfo::from_gguf(&gguf).unwrap();

    assert_eq!(info.architecture, "llama");
    assert_eq!(info.n_layers, 16);
    assert_eq!(info.n_heads, 32);
    assert_eq!(info.n_kv_heads, 8);
    assert_eq!(info.embedding_dim, 2048);
    assert!(!info.is_moe());
}

#[test]
#[cfg(feature = "integration")]
fn tensor_map_complete() {
    let mmap = load_mmap(&model_path()).unwrap();
    let gguf = parse(&mmap).unwrap();

    // Every expected tensor name should be present
    let required = [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];
    let names: std::collections::HashSet<&str> =
        gguf.tensors.iter().map(|t| t.name.as_str()).collect();

    for req in &required {
        assert!(names.contains(req), "Missing tensor: {req}");
    }
}

#[test]
#[cfg(feature = "integration")]
fn embedding_lookup_sane() {
    let mmap  = load_mmap(&model_path()).unwrap();
    let gguf  = parse(&mmap).unwrap();
    let model = LlamaModel::from_gguf(&gguf, &CpuBackend::new()).unwrap();

    // Token 0 and token 1 should produce different vectors
    let emb0 = model.embed(0);
    let emb1 = model.embed(1);

    assert_ne!(emb0, emb1);
    // Vector should be finite and not all zeros
    assert!(emb0.iter().all(|v| v.is_finite()));
    assert!(emb0.iter().any(|v| *v != 0.0));
}
```

### Tokenizer Integration

```rust
#[test]
#[cfg(feature = "integration")]
fn tokenizer_matches_expected() {
    // Known encode outputs verified against transformers / llama.cpp tokenizer
    let mmap = load_mmap(&model_path()).unwrap();
    let gguf = parse(&mmap).unwrap();
    let tok  = Tokenizer::from_gguf(&gguf).unwrap();

    // These are ground truth from running llama.cpp's tokenize tool
    assert_eq!(tok.encode("Hello world"), vec![9906, 1917]);
    assert_eq!(tok.encode(" the"), vec![279]);
    assert_eq!(tok.decode(&[9906, 1917]), "Hello world");
}
```

---

## Tier 3: Golden Tests

The most important tests. Run the full forward pass and compare output
token-for-token against llama.cpp at temperature=0 (greedy, deterministic).
Any divergence = a bug.

```bash
# Generate golden outputs with llama.cpp
./llama-cli \
    -m test-models/llama-3.2-1b-q8_0.gguf \
    -p "The capital of France is" \
    --temp 0 \
    -n 20 \
    --no-display-prompt \
    > tests/fixtures/golden/llama-1b-france.txt

# More diverse golden prompts
./llama-cli -m ... -p "def fibonacci(n):" --temp 0 -n 30 > tests/fixtures/golden/llama-1b-code.txt
./llama-cli -m ... -p "1 + 1 = " --temp 0 -n 5 > tests/fixtures/golden/llama-1b-math.txt
./llama-cli -m ... -p "Once upon a time" --temp 0 -n 25 > tests/fixtures/golden/llama-1b-story.txt
```

```rust
#[test]
#[cfg(feature = "integration")]
fn golden_france_prompt() {
    let expected = std::fs::read_to_string(
        "tests/fixtures/golden/llama-1b-france.txt"
    ).unwrap();

    let mmap   = load_mmap(&model_path()).unwrap();
    let gguf   = parse(&mmap).unwrap();
    let model  = LlamaModel::from_gguf(&gguf, &CpuBackend::new()).unwrap();
    let tok    = Tokenizer::from_gguf(&gguf).unwrap();
    let mut kv = KvCache::new_f16(&model.config);  // f16, no TurboQuant for golden test

    let tokens = tok.encode("The capital of France is");
    let output_tokens = generate_greedy(&model, &tokens, 20, &mut kv);
    let output = tok.decode(&output_tokens);

    assert_eq!(output.trim(), expected.trim(),
        "Golden test failed — output diverges from llama.cpp reference");
}

fn generate_greedy(model: &LlamaModel, tokens: &[u32],
                   n: usize, kv: &mut KvCache) -> Vec<u32> {
    // Prefill
    let mut pos = 0;
    for &t in &tokens[..tokens.len() - 1] {
        model.forward(t, pos, kv);
        pos += 1;
    }
    // Decode
    let mut out = vec![];
    let mut token = *tokens.last().unwrap();
    for _ in 0..n {
        let logits = model.forward(token, pos, kv);
        token = argmax(&logits) as u32;
        if token == model.eos_token { break; }
        out.push(token);
        pos += 1;
    }
    out
}
```

### GPU Golden Tests

Once CubeCL GPU backend is implemented, golden tests run on both backends
and outputs must match:

```rust
#[test]
#[cfg(all(feature = "integration", feature = "rocm"))]
fn gpu_matches_cpu_golden() {
    let mmap = load_mmap(&model_path()).unwrap();
    let gguf = parse(&mmap).unwrap();

    let model_cpu  = LlamaModel::from_gguf(&gguf, &CpuBackend::new()).unwrap();
    let model_gpu  = LlamaModel::from_gguf(&gguf, &RocmBackend::new(0)).unwrap();
    let tok        = Tokenizer::from_gguf(&gguf).unwrap();

    let tokens = tok.encode("The capital of France is");

    let mut kv_cpu = KvCache::new_f16(&model_cpu.config);
    let mut kv_gpu = KvCache::new_f16(&model_gpu.config);

    let cpu_out = generate_greedy(&model_cpu, &tokens, 20, &mut kv_cpu);
    let gpu_out = generate_greedy(&model_gpu, &tokens, 20, &mut kv_gpu);

    assert_eq!(cpu_out, gpu_out,
        "GPU and CPU outputs diverge — kernel bug\n\
         CPU: {:?}\n\
         GPU: {:?}", tok.decode(&cpu_out), tok.decode(&gpu_out));
}
```

### TurboQuant Golden Test

Verify TurboQuant KV cache produces acceptable quality (not token-for-token
identical — it's lossy compression — but semantically close):

```rust
#[test]
#[cfg(feature = "integration")]
fn turbo_quant_quality_acceptable() {
    let mmap  = load_mmap(&model_path()).unwrap();
    let gguf  = parse(&mmap).unwrap();
    let model = LlamaModel::from_gguf(&gguf, &CpuBackend::new()).unwrap();
    let tok   = Tokenizer::from_gguf(&gguf).unwrap();

    let tokens = tok.encode("The capital of France is");

    let mut kv_exact = KvCache::new_f16(&model.config);
    let mut kv_turbo = KvCache::new_turbo(&model.config,
        TurboQuantConfig { bits: 3, use_qjl: false });

    let exact_out = generate_greedy(&model, &tokens, 20, &mut kv_exact);
    let turbo_out = generate_greedy(&model, &tokens, 20, &mut kv_turbo);

    // Not expecting exact match — expecting high overlap
    let matching = exact_out.iter().zip(turbo_out.iter())
        .filter(|(a, b)| a == b).count();
    let overlap = matching as f32 / exact_out.len() as f32;

    assert!(overlap > 0.80,
        "TurboQuant overlap too low: {:.0}% ({}/{} tokens match)",
        overlap * 100.0, matching, exact_out.len());
}
```

---

## Tier 4: Performance Benchmarks

`benches/`

Run manually before releases to catch regressions. Not in CI.

```rust
// benches/matmul.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for (out_dim, in_dim) in [(4096, 4096), (14336, 4096), (128256, 4096)] {
        let x = vec![0.1f32; in_dim];
        let w_q8  = make_q8_0_tensor(out_dim, in_dim);
        let w_q4k = make_q4_k_tensor(out_dim, in_dim);

        group.bench_with_input(
            BenchmarkId::new("q8_0", format!("{out_dim}x{in_dim}")),
            &(&x, &w_q8),
            |b, (x, w)| {
                let mut out = vec![0f32; out_dim];
                b.iter(|| matmul(&mut out, x, w))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("q4_k", format!("{out_dim}x{in_dim}")),
            &(&x, &w_q4k),
            |b, (x, w)| {
                let mut out = vec![0f32; out_dim];
                b.iter(|| matmul(&mut out, x, w))
            }
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
```

```rust
// benches/e2e.rs — tokens per second
fn bench_e2e(c: &mut Criterion) {
    let mmap  = load_mmap("test-models/llama-3.2-1b-q8_0.gguf").unwrap();
    let gguf  = parse(&mmap).unwrap();
    let model = LlamaModel::from_gguf(&gguf, &CpuBackend::new()).unwrap();
    let tok   = Tokenizer::from_gguf(&gguf).unwrap();
    let tokens = tok.encode("The quick brown fox");

    c.bench_function("1b_q8_cpu_decode", |b| {
        b.iter(|| {
            let mut kv = KvCache::new_f16(&model.config);
            generate_greedy(&model, &tokens, 50, &mut kv)
        })
    });
}
```

---

## Test Model Manifest

Canonical set of models used for testing. Download once, reference by hash:

```toml
# tests/models.toml
[[model]]
name     = "llama-3.2-1b-q8_0"
path     = "test-models/Llama-3.2-1B-Instruct-Q8_0.gguf"
sha256   = "..."   # fill after download
hf_repo  = "bartowski/Llama-3.2-1B-Instruct-GGUF"
hf_file  = "Llama-3.2-1B-Instruct-Q8_0.gguf"
use_for  = ["unit_golden", "integration", "benchmark_cpu"]

[[model]]
name     = "qwen2.5-7b-q4km"
path     = "test-models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
sha256   = "..."
hf_repo  = "Qwen/Qwen2.5-7B-Instruct-GGUF"
hf_file  = "qwen2.5-7b-instruct-q4_k_m.gguf"
use_for  = ["integration", "benchmark_gpu", "moe_baseline"]

[[model]]
name     = "mixtral-8x7b-q4km"
path     = "test-models/mixtral-8x7b-q4_k_m.gguf"
sha256   = "..."
hf_repo  = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
hf_file  = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
use_for  = ["moe_integration", "routing_visualization"]
```

Download script:

```bash
#!/usr/bin/env bash
# tests/download_models.sh
set -e
mkdir -p test-models

huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q8_0.gguf \
    --local-dir test-models/

echo "Models downloaded. Run integration tests with:"
echo "  cargo test --features integration"
```

---

## CI Configuration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test                    # unit tests only, no models needed
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --check

  integration:
    runs-on: ubuntu-latest
    # Only on main branch — requires model download
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Download test models
        run: bash tests/download_models.sh
      - name: Generate golden outputs
        run: |
          # Install llama.cpp
          pip install llama-cpp-python
          python tests/fixtures/generate_golden.py
      - run: cargo test --features integration
```

---

## Debug Assertions

Add throughout the codebase behind `#[cfg(debug_assertions)]`.
Compiled out in release builds, zero cost in production:

```rust
fn forward(&self, token: u32, pos: usize, kv: &mut KvCache) -> Vec<f32> {
    debug_assert!(token < self.vocab_size as u32,
        "token {token} out of range (vocab_size={})", self.vocab_size);
    debug_assert!(pos < self.config.context_length,
        "pos {pos} exceeds context_length {}", self.config.context_length);

    let mut x = self.embed(token);

    debug_assert_no_nan(&x, "embedding output");

    for (i, layer) in self.layers.iter().enumerate() {
        x = self.forward_layer(layer, x, pos, kv);
        debug_assert_no_nan(&x, &format!("layer {i} output"));
    }

    let logits = self.lm_head(&x);
    debug_assert_no_nan(&logits, "logits");
    debug_assert_eq!(logits.len(), self.vocab_size,
        "logits length mismatch");

    logits
}

fn debug_assert_no_nan(x: &[f32], label: &str) {
    #[cfg(debug_assertions)]
    if x.iter().any(|v| !v.is_finite()) {
        let nan_count = x.iter().filter(|v| v.is_nan()).count();
        let inf_count = x.iter().filter(|v| v.is_infinite()).count();
        panic!("{label}: {nan_count} NaN, {inf_count} Inf values detected");
    }
}
```

---

## Testing Checklist by Phase

```
Phase 1 (parser):
  ✓ parse valid headers of all versions
  ✓ parse all 13 metadata value types
  ✓ parse tensor info with various shapes
  ✓ reject bad magic bytes
  ✓ reject unsupported versions
  ✓ reject truncated files
  ✓ MoE detection from metadata
  ✓ dense model has no MoE info

Phase 2 (quant + ops):
  ✓ Q8_0 round-trip within tolerance
  ✓ Q4_K round-trip within tolerance
  ✓ IQ4_XS codebook lookup correct
  ✓ MXFP4 E2M1 bit patterns
  ✓ F16 exact conversion
  ✓ matmul vs numpy
  ✓ rms_norm vs numpy
  ✓ softmax vs numpy + sum-to-one + numerical stability
  ✓ rope vs numpy
  ✓ silu_mul vs numpy
  ✓ no NaN propagation from zero inputs

Phase 3 (tokenizer):
  ✓ encode/decode round trip
  ✓ special tokens not in output
  ✓ leading space changes token
  ✓ integration: matches llama.cpp tokenize tool

Phase 4-5 (forward pass + decode):
  ✓ golden: "The capital of France is" → correct continuation
  ✓ golden: code prompt → correct completion
  ✓ integration: tensor map complete
  ✓ integration: embedding lookup sane

Phase 6 (MoE):
  ✓ router scores finite
  ✓ top-k count matches expert_used_count
  ✓ selected weights sum ≈ 1.0
  ✓ golden: Mixtral prompt matches llama.cpp

Phase 7 (TurboQuant):
  ✓ WHT is self-inverse
  ✓ WHT preserves inner products
  ✓ round-trip cosine similarity > 0.95
  ✓ memory reduction > 4×
  ✓ quality test: >80% token overlap with exact KV

Phase 8 (CubeCL GPU):
  ✓ GPU output matches CPU output for each op
  ✓ GPU golden test matches CPU golden test
  ✓ no GPU memory leaks across multiple forward passes
```
