//! Quick GPU matmul correctness check: computes a Q4_K matmul on both CPU and
//! GPU for the first weight tensor of a real model, then compares the outputs.
//! Usage: cargo run --features wgpu --release --bin verify_gpu -- <model.gguf>

use anyhow::Result;
use gguf_rs::{load, GgmlType};
use gguf_rs::ops::{create_backend, BackendPreference, AnyBackend};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: verify_gpu <model.gguf>");
    let (gguf, mmap) = load(std::path::Path::new(&path))?;
    let data = &mmap[gguf.data_offset as usize..];

    // Find a specific tensor or the first Q4_K tensor with a reasonable size.
    let target = std::env::args().nth(2);
    let tensor = if let Some(ref name) = target {
        gguf.tensors.iter().find(|t| t.name == name.as_str()).expect("named tensor not found")
    } else {
        gguf.tensors.iter()
            .find(|t| t.ggml_type == GgmlType::Q4_K && t.n_dims >= 2)
            .expect("no Q4_K tensor found")
    };

    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let dtype = tensor.ggml_type;
    println!("Testing tensor '{}' dtype={} in={} out={}", tensor.name, dtype, in_dim, out_dim);

    let w = &data[tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.byte_size()];

    // Random-ish x input (deterministic).
    let x: Vec<f32> = (0..in_dim).map(|i| ((i * 1234567 + 42) % 1000) as f32 / 500.0 - 1.0).collect();

    // CPU result.
    let cpu_backend = AnyBackend::Cpu(gguf_rs::ops::cpu::CpuBackend::new());
    let mut cpu_out = vec![0f32; out_dim];
    cpu_backend.matmul_dequant(dtype, w, &x, &mut cpu_out)?;
    println!("CPU[0..4] = {:?}", &cpu_out[..4.min(out_dim)]);

    // GPU result.
    #[cfg(feature = "wgpu")]
    {
        let gpu_backend = create_backend(BackendPreference::Wgpu);
        let mut gpu_out = vec![0f32; out_dim];
        gpu_backend.matmul_dequant(dtype, w, &x, &mut gpu_out)?;
        println!("GPU[0..4] = {:?}", &gpu_out[..4.min(out_dim)]);

        // Compare.
        let max_err = cpu_out.iter().zip(&gpu_out).map(|(c, g)| (c - g).abs()).fold(0f32, f32::max);
        let rel_max = cpu_out.iter().zip(&gpu_out).map(|(c, g)| {
            if c.abs() > 1e-6 { (c - g).abs() / c.abs() } else { (c - g).abs() }
        }).fold(0f32, f32::max);
        println!("max_abs_err={:.6e}  max_rel_err={:.4}%  -> {}", max_err, rel_max * 100.0,
            if max_err < 1e-3 { "PASS" } else { "FAIL" });
    }

    Ok(())
}
