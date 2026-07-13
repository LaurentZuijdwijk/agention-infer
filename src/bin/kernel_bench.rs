//! Per-dispatch kernel micro-benchmark. Tests whether GPU decode is
//! dispatch-count-bound (every kernel ≈ a fixed dispatch floor → fusion is the
//! lever) or compute-bound (kernels differ by their work → optimize the slow
//! ones). See `WgpuBackend::probe_kernel_costs`.
//!
//!   cargo run --release --features wgpu --bin kernel_bench -- models/Qwen3.5-9B-Q4_K_M.gguf

#[cfg(feature = "wgpu")]
fn main() {
    use gguf_rs::loader::load;
    use gguf_rs::model::ModelConfig;
    use gguf_rs::ops::wgpu_backend::WgpuBackend;
    use gguf_rs::ops::GPU_DEQUANT_DTYPES;
    use std::path::Path;
    use std::process::exit;

    let model = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: kernel_bench <model.gguf>");
        exit(2);
    });
    let path = Path::new(&model);
    let (gguf, mmap) = load(path).unwrap_or_else(|e| panic!("load: {e}"));
    let data = &mmap[gguf.data_offset as usize..];
    let cfg = ModelConfig::from_gguf(&gguf).unwrap();
    let d = cfg.embedding_length as usize;
    let ff = cfg.feed_forward_length as usize;

    // A representative attention/ffn projection weight for the matmul row.
    let tensor = gguf
        .tensors
        .iter()
        .filter(|t| t.n_dims >= 2 && GPU_DEQUANT_DTYPES.contains(&t.ggml_type))
        .filter(|t| {
            let n = &t.name;
            !n.contains("token_embd") && !n.contains("output") && !n.contains("embd")
        })
        .max_by_key(|t| t.n_elements()) // a real d×ff/d×d projection (embd/output excluded)
        .expect("no projection weight");
    let in_dim = tensor.dims[0] as usize;
    let bytes = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

    println!(
        "model={model}\n  d={d} ff={ff} | matmul weight {} dtype={:?} in_dim={in_dim} out_dim={}",
        tensor.name,
        tensor.ggml_type,
        tensor.dims[1..].iter().product::<u64>(),
    );

    let n_heads = cfg.head_count as usize;
    let n_kv_heads = cfg.head_count_kv as usize;
    let head_dim = cfg.head_dim as usize;
    let backend = WgpuBackend::new();
    let h = backend.upload_weight(tensor.ggml_type, bytes, in_dim);
    backend.probe_kernel_costs(&h, d, ff, n_heads, head_dim, 500);

    let max_seq = 2048;
    let positions = [0usize, 15, 63, 255, 1023, 2047];
    backend.probe_attention_costs(n_heads, n_kv_heads, head_dim, max_seq, &positions, 200);
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("kernel_bench requires --features wgpu");
}
