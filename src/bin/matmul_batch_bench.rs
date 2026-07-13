//! Settle the load-bearing Stage-3 question: does the batched matmul
//! (`matmul_dequant_wgpu_batch`, one workgroup per row looping N tokens inside)
//! actually read the weight tensor **once** and amortize it across the batch,
//! or does looping N inside the workgroup just cost N× like N separate matmuls?
//!
//! Times `n` separate single-token matmuls vs one batched matmul against a real
//! model weight tensor, for several batch sizes. If batched wins as N grows,
//! the full batched-prefill forward is worth building; if it's flat/worse on
//! this GPU, we've saved ourselves the rewrite.
//!
//!   cargo run --release --features wgpu --bin matmul_batch_bench -- models/Qwen3.5-9B-Q4_K_M.gguf

#[cfg(feature = "wgpu")]
fn main() {
    use gguf_rs::loader::load;
    use gguf_rs::ops::wgpu_backend::WgpuBackend;
    use gguf_rs::ops::GPU_DEQUANT_DTYPES;
    use std::path::Path;
    use std::process::exit;

    let model = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: matmul_batch_bench <model.gguf>");
        exit(2);
    });
    let path = Path::new(&model);
    let (gguf, mmap) = load(path).unwrap_or_else(|e| panic!("load: {e}"));
    let data = &mmap[gguf.data_offset as usize..];

    // Pick a representative mid-layer projection weight — the kind of matmul
    // that dominates prefill weight bandwidth. Constrain to Q8_0 with in_dim
    // (dims[0]) <= 4096, since the cooperative kernel under test is Q8_0-only
    // and holds one dequantized row in LDS. Exclude the giant token_embd /
    // output(lm_head) tensors (vocab-sized; not batched in the inner loop).
    use gguf_rs::types::GgmlType;
    let _ = GPU_DEQUANT_DTYPES;
    let tensor = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.n_dims >= 2
                && matches!(t.ggml_type, GgmlType::Q8_0 | GgmlType::Q4_K)
                && t.dims[0] <= 4096
        })
        .filter(|t| {
            let n = &t.name;
            !n.contains("token_embd") && !n.contains("output") && !n.contains("embd")
        })
        .max_by_key(|t| t.n_elements())
        .expect("no Q8_0 2D projection weight with in_dim<=4096 (coop test needs a Q8_0 model)");

    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1..].iter().product::<u64>() as usize;
    let bytes = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

    println!(
        "tensor {} dtype={:?} in_dim={in_dim} out_dim={out_dim}\n",
        tensor.name, tensor.ggml_type
    );

    let backend = WgpuBackend::new();
    let h = backend.upload_weight(tensor.ggml_type, bytes, in_dim);

    let iters = 10;
    println!(
        "{:>4}  {:>11}  {:>11}  {:>11}  {:>9}  {:>9}  {:>10}",
        "N", "single(ms)", "loopN(ms)", "coop(ms)", "loopN×", "coop×", "coop/tok"
    );
    let mut worst_err = 0f32;
    for &n in &[1usize, 2, 4, 8, 16, 32] {
        let (single, batch, coop, err) = backend.probe_batched_matmul(&h, n, iters);
        worst_err = worst_err.max(err);
        let single_ms = single / iters as f64 * 1e3;
        let batch_ms = batch / iters as f64 * 1e3;
        let coop_ms = coop / iters as f64 * 1e3;
        println!(
            "{n:>4}  {single_ms:>11.3}  {batch_ms:>11.3}  {coop_ms:>11.3}  {:>8.2}x  {:>8.2}x  {:>10.3}",
            single_ms / batch_ms.max(1e-9),
            single_ms / coop_ms.max(1e-9),
            coop_ms / n as f64,
        );
    }
    println!("\ncoop correctness: max_abs_err vs single = {worst_err:.4e}");
    println!(
        "Read: `coop×` is the cooperative kernel's speedup vs N separate matmuls. If it\n\
         grows with N (coop/tok drops), the matmul was dequant-COMPUTE-bound and\n\
         dequant-once-then-N-dots amortizes → batched prefill is worth building. If coop\n\
         stays ~1× like loopN, batching doesn't help GPU prefill on this hardware."
    );
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("matmul_batch_bench requires --features wgpu");
}
