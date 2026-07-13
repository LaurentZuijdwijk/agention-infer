//! GPU micro-benchmarks: per-kernel timing with parameter sweeps.
//!
//! Measures wall time for each kernel type across different parameter ranges.
//! This is Phase 2 of the GPU benchmark plan — Phase 1 was the `GGUF_TRACE_KERNEL`
//! per-kernel timer infrastructure (already in `src/ops/trace.rs`).
//!
//! Usage:
//!   cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul --dims 4096,8192 --batch 1,16,64
//!   cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel all
//!
//! Kernel types:
//!   matmul, rms_norm, rope, attention, qkv, ffn_chain, add_residual,
//!   silu_mul, sigmoid_mul, split_qg, kv_cache_write, short_conv,
//!   gdn_gate_decay, gdn_recurrence, gdn_gated_norm, causal_conv1d,
//!   l2_norm_heads

#[cfg(feature = "wgpu")]
use gguf_rs::loader::load;
#[cfg(feature = "wgpu")]
use gguf_rs::ops::wgpu_backend::WgpuBackend;
#[cfg(feature = "wgpu")]
use gguf_rs::ops::Backend;
#[cfg(feature = "wgpu")]
use gguf_rs::ops::GPU_DEQUANT_DTYPES;
#[cfg(feature = "wgpu")]
use gguf_rs::types::{GgmlType, GgufFile, TensorInfo};
#[cfg(feature = "wgpu")]
use std::path::Path;
#[cfg(feature = "wgpu")]
use std::process::exit;

#[cfg(feature = "wgpu")]
fn main() {

    let model = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: kernels_bench <model.gguf> [--kernel <type>] [--dims <list>] [--batch <list>]");
        exit(2);
    });
    let path = Path::new(&model);
    let (gguf, mmap) = load(path).unwrap_or_else(|e| panic!("load: {e}"));
    let data = &mmap[gguf.data_offset as usize..];

    // Parse CLI args
    let mut args = std::env::args();
    let _ = args.next(); // binary name
    let _ = args.next(); // model path
    let mut kernel_filter: Option<String> = None;
    let mut dims_filter: Option<Vec<usize>> = None;
    let mut batch_filter: Option<Vec<usize>> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--kernel" => {
                kernel_filter = args.next();
            }
            "--dims" => {
                dims_filter = args
                    .next()
                    .map(|s| s.split(',').map(|d| d.parse::<usize>().unwrap()).collect());
            }
            "--batch" => {
                batch_filter = args
                    .next()
                    .map(|s| s.split(',').map(|b| b.parse::<usize>().unwrap()).collect());
            }
            _ => {}
        }
    }

    // If no specific kernel requested, run all
    let kernels: Vec<String> = if let Some(kf) = kernel_filter {
        if kf == "all" {
            vec![
                String::from("matmul"), String::from("rms_norm"), String::from("rope"),
                String::from("attention"), String::from("qkv"), String::from("ffn_chain"),
                String::from("add_residual"), String::from("silu_mul"), String::from("sigmoid_mul"),
                String::from("split_qg"), String::from("kv_cache_write"), String::from("short_conv"),
                String::from("gdn_gate_decay"), String::from("gdn_recurrence"),
                String::from("gdn_gated_norm"), String::from("causal_conv1d"), String::from("l2_norm_heads"),
            ]
        } else {
            vec![kf]
        }
    } else {
        vec![
            String::from("matmul"), String::from("rms_norm"), String::from("rope"),
            String::from("attention"), String::from("qkv"), String::from("ffn_chain"),
            String::from("add_residual"), String::from("silu_mul"), String::from("sigmoid_mul"),
            String::from("split_qg"), String::from("kv_cache_write"), String::from("short_conv"),
            String::from("gdn_gate_decay"), String::from("gdn_recurrence"),
            String::from("gdn_gated_norm"), String::from("causal_conv1d"), String::from("l2_norm_heads"),
        ]
    };

    let backend = WgpuBackend::new();

    println!("GPU micro-benchmarks — {}", model);
    println!("Backend: {}\n", backend.name());
    println!("Kernels: {}\n", kernels.join(", "));

    for kernel in kernels {
        eprintln!("--- {} ---", kernel);
        run_kernel(&backend, &gguf, data, kernel.as_str(), &dims_filter, &batch_filter);
        println!();
    }
}

fn run_kernel(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    kernel: &str,
    dims_filter: &Option<Vec<usize>>,
    batch_filter: &Option<Vec<usize>>,
) {
    match kernel {
        "matmul" => bench_matmul(backend, gguf, data, dims_filter, batch_filter),
        "rms_norm" => bench_rms_norm(backend, gguf, data, dims_filter),
        "rope" => bench_rope(backend, gguf, data, dims_filter),
        "attention" => bench_attention(backend, gguf, data, dims_filter),
        "qkv" => bench_qkv(backend, gguf, data, dims_filter, batch_filter),
        "ffn_chain" => bench_ffn_chain(backend, gguf, data, dims_filter, batch_filter),
        "add_residual" => bench_add_residual(backend, gguf, data, dims_filter),
        "silu_mul" => bench_silu_mul(backend, gguf, data, dims_filter),
        "sigmoid_mul" => bench_sigmoid_mul(backend, gguf, data, dims_filter),
        "split_qg" => bench_split_qg(backend, gguf, data, dims_filter),
        "kv_cache_write" => bench_kv_cache_write(backend, gguf, data, dims_filter),
        "short_conv" => bench_short_conv(backend, gguf, data, dims_filter),
        "gdn_gate_decay" => bench_gdn_gate_decay(backend, gguf, data, dims_filter),
        "gdn_recurrence" => bench_gdn_recurrence(backend, gguf, data, dims_filter),
        "gdn_gated_norm" => bench_gdn_gated_norm(backend, gguf, data, dims_filter),
        "causal_conv1d" => bench_causal_conv1d(backend, gguf, data, dims_filter),
        "l2_norm_heads" => bench_l2_norm_heads(backend, gguf, data, dims_filter),
        _ => eprintln!("unknown kernel: {}", kernel),
    }
}

fn bench_matmul(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
    batch_filter: &Option<Vec<usize>>,
) {
    let _ = GPU_DEQUANT_DTYPES;

    // Filter tensors by dtype and in_dim
    let dtypes = vec![
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ];
    let in_dims = dims_filter
        .clone()
        .unwrap_or_else(|| vec![4096, 8192, 16384, 3072]);
    let batch_sizes = batch_filter
        .clone()
        .unwrap_or_else(|| vec![1, 16, 64]);

    let t0 = std::time::Instant::now();

    for dtype in &dtypes {
        for in_dim in &in_dims {
            let tensors: Vec<_> = gguf
                .tensors
                .iter()
                .filter(|t| {
                    t.n_dims >= 2
                        && t.ggml_type == *dtype
                        && t.dims[0] as usize == *in_dim
                        && !t.name.contains("token_embd")
                        && !t.name.contains("output")
                        && !t.name.contains("embd")
                })
                .collect();

            if tensors.is_empty() {
                continue;
            }

            for tensor in tensors {
                let out_dim = tensor.dims[1..].iter().product::<u64>() as usize;
                let bytes = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

                let h = backend.upload_weight(*dtype, bytes, *in_dim);

                for batch in &batch_sizes {
                    let mut x = vec![0f32; *in_dim * batch];
                    let x_handle = backend.import_f32(x.as_slice());

                    let t0 = std::time::Instant::now();
                    let out_handle = if *batch == 1 {
                        backend.launch_only(&h, &x_handle)
                    } else {
                        backend.launch_only_batch(&h, &x_handle, *batch)
                    };
                    let elapsed = t0.elapsed();

                    eprintln!(
                        "matmul {} in_dim={} out_dim={} batch={} time={}ms",
                        dtype,
                        in_dim,
                        out_dim,
                        batch,
                        elapsed.as_secs_f64() * 1e3
                    );
                }
            }
        }
    }
    eprintln!("matmul done in {:.1}s", t0.elapsed().as_secs_f32());
}

fn bench_rms_norm(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let len_dims = dims_filter
        .clone()
        .unwrap_or_else(|| vec![4096, 8192, 16384, 3072]);

    for len in &len_dims {
        let mut x = vec![0f32; *len];
        let mut weight = vec![1.0f32; *len];
        let x_handle = backend.import_f32(x.as_slice());
        let weight_handle = backend.import_f32(weight.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_rms_norm(&x_handle, &weight_handle, *len, 1e-5);
        let elapsed = t0.elapsed();

        eprintln!("rms_norm len={} time={}ms", len, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_rope(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // Use actual attention head dimensions from the model
    let rope_configs: Vec<(usize, usize, usize)> = vec![
        (8, 64, 64),
        (16, 128, 128),
        (32, 256, 256),
    ];

    for (n_heads, head_dim, n_rot) in &rope_configs {
        let mut q = vec![0f32; n_heads * head_dim];
        let q_handle = backend.import_f32(q.as_slice());

        let t0 = std::time::Instant::now();
        backend.launch_rope(&q_handle, *n_heads, *head_dim, *n_rot, 0, 10000.0);
        let elapsed = t0.elapsed();

        eprintln!(
            "rope n_heads={} head_dim={} n_rot={} time={}ms",
            n_heads, head_dim, n_rot, elapsed.as_secs_f64() * 1e3
        );
    }
}

fn bench_attention(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // Attention configs: (pos, n_heads, head_dim)
    let configs: Vec<(usize, usize, usize)> = vec![
        (16, 8, 64),
        (64, 16, 128),
        (256, 32, 256),
        (1024, 32, 256),
        (4096, 32, 256),
    ];

    for (pos, n_heads, head_dim) in &configs {
        let n_kv_heads = if *n_heads >= 16 { 8 } else { 4 };
        let max_seq = *pos + 64; // some context

        let q = vec![0f32; n_heads * head_dim];
        let k_cache = vec![0f32; max_seq * n_kv_heads * head_dim];
        let v_cache = vec![0f32; max_seq * n_kv_heads * head_dim];
        let scores = vec![0f32; n_heads * max_seq];
        let weights = vec![0f32; n_heads * max_seq];

        let q_handle = backend.import_f32(q.as_slice());
        let k_handle = backend.import_f32(k_cache.as_slice());
        let v_handle = backend.import_f32(v_cache.as_slice());
        let scores_handle = backend.import_f32(scores.as_slice());
        let weights_handle = backend.import_f32(weights.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_attention(
            &q_handle, &k_handle, &v_handle,
            &scores_handle, &weights_handle,
            *pos, *head_dim, *n_heads, n_kv_heads, max_seq,
        );
        let elapsed = t0.elapsed();

        eprintln!(
            "attention pos={} n_heads={} head_dim={} time={}ms",
            pos, n_heads, head_dim, elapsed.as_secs_f64() * 1e3
        );
    }
}

fn bench_qkv(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
    batch_filter: &Option<Vec<usize>>,
) {
    // Use actual attention weight tensors
    let attention_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.name.contains("attn_q") || t.name.contains("attn_k") || t.name.contains("attn_v")
        })
        .collect();

    if attention_tensors.is_empty() {
        eprintln!("no attention tensors found");
        return;
    }

    // Group by input dimension
    let mut groups: std::collections::HashMap<usize, Vec<&TensorInfo>> =
        std::collections::HashMap::new();
    for t in attention_tensors {
        groups.entry(t.dims[0] as usize).or_default().push(t);
    }

    for (in_dim, tensors) in groups {
        let batch_sizes = batch_filter
            .clone()
            .unwrap_or_else(|| vec![1, 16, 64]);

        let mut x = vec![0f32; in_dim];
        let x_handle = backend.import_f32(x.as_slice());

        // Upload the Q, K, V weight tensors
        let bytes_q = &data[tensors[0].byte_offset as usize..tensors[0].byte_offset as usize + tensors[0].byte_size()];
        let bytes_k = &data[tensors[1].byte_offset as usize..tensors[1].byte_offset as usize + tensors[1].byte_size()];
        let bytes_v = &data[tensors[2].byte_offset as usize..tensors[2].byte_offset as usize + tensors[2].byte_size()];

        let h_q = backend.upload_weight(tensors[0].ggml_type, bytes_q, in_dim);
        let h_k = backend.upload_weight(tensors[1].ggml_type, bytes_k, in_dim);
        let h_v = backend.upload_weight(tensors[2].ggml_type, bytes_v, in_dim);

        let t0 = std::time::Instant::now();
        let _result = backend.launch_qkv(&h_q, &h_k, &h_v, &x_handle);
        let elapsed = t0.elapsed();

        eprintln!("qkv in_dim={} time={}ms", in_dim, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_ffn_chain(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
    batch_filter: &Option<Vec<usize>>,
) {
    // Find FFN weight tensors
    let ffn_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| {
            t.name.contains("ffn_gate") || t.name.contains("ffn_up") || t.name.contains("ffn_down")
        })
        .collect();

    if ffn_tensors.len() < 3 {
        eprintln!("not enough FFN tensors");
        return;
    }

    // Group by input dimension
    let mut groups: std::collections::HashMap<usize, Vec<&TensorInfo>> =
        std::collections::HashMap::new();
    for t in ffn_tensors {
        groups.entry(t.dims[0] as usize).or_default().push(t);
    }

    for (in_dim, tensors) in groups {
        let out_dim = tensors[0].dims[1..].iter().product::<u64>() as usize;
        let batch_sizes = batch_filter
            .clone()
            .unwrap_or_else(|| vec![1, 16, 64]);

        let bytes_gate = &data[tensors[0].byte_offset as usize..tensors[0].byte_offset as usize + tensors[0].byte_size()];
        let bytes_up = &data[tensors[1].byte_offset as usize..tensors[1].byte_offset as usize + tensors[1].byte_size()];
        let bytes_down = &data[tensors[2].byte_offset as usize..tensors[2].byte_offset as usize + tensors[2].byte_size()];

        let h_gate = backend.upload_weight(tensors[0].ggml_type, bytes_gate, in_dim);
        let h_up = backend.upload_weight(tensors[1].ggml_type, bytes_up, in_dim);
        let h_down = backend.upload_weight(tensors[2].ggml_type, bytes_down, in_dim);

        let mut x = vec![0f32; in_dim];
        let x_handle = backend.import_f32(x.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.ffn_chain_from_handle(&h_gate, &h_up, &h_down, &x_handle);
        let elapsed = t0.elapsed();

        eprintln!("ffn_chain in_dim={} out_dim={} time={}ms", in_dim, out_dim, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_add_residual(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let len_dims = dims_filter
        .clone()
        .unwrap_or_else(|| vec![4096, 8192, 16384, 3072]);

    for len in &len_dims {
        let mut x = vec![0f32; *len];
        let mut delta = vec![0f32; *len];
        let mut weight = vec![1.0f32; *len];

        let x_handle = backend.import_f32(x.as_slice());
        let delta_handle = backend.import_f32(delta.as_slice());
        let weight_handle = backend.import_f32(weight.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_add_residual_rms_norm(
            &x_handle, &delta_handle, &weight_handle,
            *len, 1e-5,
        );
        let elapsed = t0.elapsed();

        eprintln!("add_residual len={} time={}ms", len, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_silu_mul(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let len_dims = dims_filter
        .clone()
        .unwrap_or_else(|| vec![4096, 8192, 16384, 3072]);

    for len in &len_dims {
        let mut a = vec![0f32; *len];
        let mut b = vec![1.0f32; *len];

        let a_handle = backend.import_f32(a.as_slice());
        let b_handle = backend.import_f32(b.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_silu_mul(&a_handle, &b_handle, *len);
        let elapsed = t0.elapsed();

        eprintln!("silu_mul len={} time={}ms", len, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_sigmoid_mul(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let len_dims = dims_filter
        .clone()
        .unwrap_or_else(|| vec![4096, 8192, 16384, 3072]);

    for len in &len_dims {
        let mut a = vec![0f32; *len];
        let mut b = vec![1.0f32; *len];

        let a_handle = backend.import_f32(a.as_slice());
        let b_handle = backend.import_f32(b.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_sigmoid_mul(&a_handle, &b_handle, *len);
        let elapsed = t0.elapsed();

        eprintln!("sigmoid_mul len={} time={}ms", len, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_split_qg(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // Split Q/G: 2 * head_dim * n_heads
    let configs: Vec<(usize, usize)> = vec![
        (8, 64),
        (16, 128),
        (32, 256),
    ];

    for (n_heads, head_dim) in &configs {
        let len = 2 * n_heads * head_dim;
        let mut qg = vec![0f32; len];
        let qg_handle = backend.import_f32(qg.as_slice());

        let t0 = std::time::Instant::now();
        let _out = backend.launch_split_qg(&qg_handle, *head_dim, *n_heads);
        let elapsed = t0.elapsed();

        eprintln!("split_qg n_heads={} head_dim={} time={}ms", n_heads, head_dim, elapsed.as_secs_f64() * 1e3);
    }
}

fn bench_kv_cache_write(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let configs: Vec<(usize, usize, usize)> = vec![
        (0, 8, 64),
        (1024, 8, 64),
        (4096, 8, 64),
        (0, 16, 128),
        (4096, 16, 128),
    ];

    for (pos, n_heads, head_dim) in &configs {
        let kv_dim = n_heads * head_dim;
        let max_seq = pos + 64;
        let mut k_cache = vec![0f32; max_seq * kv_dim];
        let mut v_cache = vec![0f32; max_seq * kv_dim];
        let mut new_k = vec![0f32; kv_dim];
        let mut new_v = vec![0f32; kv_dim];

        let k_handle = backend.import_f32(k_cache.as_slice());
        let v_handle = backend.import_f32(v_cache.as_slice());
        let new_k_handle = backend.import_f32(new_k.as_slice());
        let new_v_handle = backend.import_f32(new_v.as_slice());

        let t0 = std::time::Instant::now();
        backend.launch_kv_cache_write(
            &k_handle, &v_handle, &new_k_handle, &new_v_handle,
            *pos, kv_dim,
        );
        let elapsed = t0.elapsed();

        eprintln!(
            "kv_cache_write pos={} n_heads={} head_dim={} time={}ms",
            pos, n_heads, head_dim, elapsed.as_secs_f64() * 1e3
        );
    }
}

fn bench_short_conv(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // LFM2 ShortConv tensors
    let short_conv_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.contains("short_conv"))
        .collect();

    if short_conv_tensors.is_empty() {
        eprintln!("no short_conv tensors found");
        return;
    }

    let tensor = short_conv_tensors[0];
    let conv_dim = tensor.dims[0] as usize;
    let l = tensor.dims[1] as usize;
    let bytes = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

    let mut bcx = vec![0f32; 3 * conv_dim];
    let mut history = vec![0f32; conv_dim * l.saturating_sub(1)];

    let bcx_handle = backend.import_f32(bcx.as_slice());
    let weight_handle = backend.upload_weight(tensor.ggml_type, bytes, conv_dim);
    let history_handle = backend.import_f32(history.as_slice());

    let t0 = std::time::Instant::now();
    let _out = backend.launch_short_conv(
        &bcx_handle, &weight_handle.handle_ref(), &history_handle,
        l, conv_dim,
    );
    let elapsed = t0.elapsed();

    eprintln!("short_conv conv_dim={} l={} time={}ms", conv_dim, l, elapsed.as_secs_f64() * 1e3);
}

fn bench_gdn_gate_decay(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // GDN gate/decay tensors
    let gdn_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.contains("gdn") && (t.name.contains("decay") || t.name.contains("a_ssm")))
        .collect();

    if gdn_tensors.len() < 4 {
        eprintln!("not enough GDN tensors (need beta, alpha, ssm_a, dt_bias)");
        return;
    }

    let n_v_heads = gdn_tensors[0].dims[0] as usize;
    let beta_raw = backend.import_f32(vec![0f32; n_v_heads].as_slice());
    let alpha_raw = backend.import_f32(vec![0f32; n_v_heads].as_slice());
    let ssm_a = backend.import_f32(vec![0f32; n_v_heads].as_slice());
    let dt_bias = backend.import_f32(vec![0f32; n_v_heads].as_slice());

    let t0 = std::time::Instant::now();
    backend.launch_gdn_gate_decay(
        &beta_raw, &alpha_raw, &ssm_a, &dt_bias,
        n_v_heads,
    );
    let elapsed = t0.elapsed();

    eprintln!("gdn_gate_decay n_v_heads={} time={}ms", n_v_heads, elapsed.as_secs_f64() * 1e3);
}

fn bench_gdn_recurrence(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // GDN recurrence tensors
    let gdn_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.contains("gdn") && (t.name.contains("recurrence") || t.name.contains("x")))
        .collect();

    if gdn_tensors.is_empty() {
        eprintln!("not enough GDN recurrence tensors");
        return;
    }

    // Estimate parameters from tensor shapes
    let state_size = gdn_tensors[0].dims[0] as usize;
    let conv_dim = gdn_tensors[0].dims[1] as usize;
    let n_v_heads = state_size / (conv_dim / 2);
    let head_v_dim = conv_dim / n_v_heads;

    let state = backend.import_f32(vec![0f32; state_size].as_slice());
    let conv_out = backend.import_f32(vec![0f32; conv_dim].as_slice());
    let beta = backend.import_f32(vec![0f32; n_v_heads].as_slice());
    let decay = backend.import_f32(vec![0f32; n_v_heads].as_slice());

    let t0 = std::time::Instant::now();
    let _out = backend.launch_gdn_recurrence(
        &state, &conv_out, &beta, &decay,
        n_v_heads, n_v_heads / 2, head_v_dim, head_v_dim,
        conv_dim / 2, conv_dim, 1.0,
    );
    let elapsed = t0.elapsed();

    eprintln!("gdn_recurrence n_v_heads={} head_v_dim={} conv_dim={} time={}ms", n_v_heads, head_v_dim, conv_dim, elapsed.as_secs_f64() * 1e3);
}

fn bench_gdn_gated_norm(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // GDN gated norm tensors
    let gdn_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.contains("gdn") && t.name.contains("norm"))
        .collect();

    if gdn_tensors.len() < 2 {
        eprintln!("not enough GDN gated norm tensors");
        return;
    }

    let handle = backend.import_f32(vec![0f32; gdn_tensors[0].dims[0] as usize].as_slice());
    let weight = backend.import_f32(vec![1.0f32; gdn_tensors[1].dims[0] as usize].as_slice());
    let gate = backend.import_f32(vec![0f32; gdn_tensors[0].dims[0] as usize].as_slice());
    let n_heads = gdn_tensors[0].dims[0] as usize;
    let head_dim = gdn_tensors[1].dims[0] as usize;

    let t0 = std::time::Instant::now();
    backend.launch_gdn_gated_norm(
        &handle, &weight, &gate,
        n_heads, head_dim, 1e-5,
    );
    let elapsed = t0.elapsed();

    eprintln!("gdn_gated_norm n_heads={} head_dim={} time={}ms", n_heads, head_dim, elapsed.as_secs_f64() * 1e3);
}

fn bench_causal_conv1d(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    // GDN causal conv1d tensors
    let gdn_tensors: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.contains("gdn") && t.name.contains("conv"))
        .collect();

    if gdn_tensors.is_empty() {
        eprintln!("no GDN causal conv1d tensors");
        return;
    }

    let tensor = gdn_tensors[0];
    let conv_dim = tensor.dims[0] as usize;
    let d_conv = tensor.dims[1] as usize;
    let bytes = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

    let input = backend.import_f32(vec![0f32; conv_dim].as_slice());
    let weight = backend.upload_weight(tensor.ggml_type, bytes, conv_dim * d_conv);
    let history = backend.import_f32(vec![0f32; conv_dim * d_conv.saturating_sub(1)].as_slice());

    let t0 = std::time::Instant::now();
    let _out = backend.launch_causal_conv1d_silu(
        &input, &weight.handle_ref(), &history,
        conv_dim, d_conv,
    );
    let elapsed = t0.elapsed();

    eprintln!("causal_conv1d conv_dim={} d_conv={} time={}ms", conv_dim, d_conv, elapsed.as_secs_f64() * 1e3);
}

fn bench_l2_norm_heads(
    backend: &WgpuBackend,
    gguf: &GgufFile,
    data: &[u8],
    dims_filter: &Option<Vec<usize>>,
) {
    let configs: Vec<(usize, usize)> = vec![
        (8, 64),
        (16, 128),
        (32, 256),
    ];

    for (n_heads, head_dim) in &configs {
        let total_len = 2 * n_heads * head_dim;
        let mut handle = vec![0f32; total_len];
        let handle_handle = backend.import_f32(handle.as_slice());

        let t0 = std::time::Instant::now();
        backend.launch_l2_norm_heads(
            &handle_handle,
            0, *n_heads * *head_dim,
            *n_heads, *head_dim, 1e-5,
            total_len,
        );
        let elapsed = t0.elapsed();

        eprintln!("l2_norm_heads n_heads={} head_dim={} time={}ms", n_heads, head_dim, elapsed.as_secs_f64() * 1e3);
    }
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("kernels_bench requires --features wgpu");
}
