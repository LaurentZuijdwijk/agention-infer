//! Compare CPU vs GPU logits for the first token of a model.
//! Usage: cargo run --features wgpu --release --bin compare_backends -- <model.gguf>

use anyhow::Result;
use gguf_rs::load;
use gguf_rs::model::{create_model_with_backend, KvCache};
use gguf_rs::ops::{create_backend, BackendPreference};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: compare_backends <model.gguf>");
    let (gguf, mmap) = load(std::path::Path::new(&path))?;
    let data = &mmap[gguf.data_offset as usize..];

    let token: u32 = 100; // arbitrary test token

    // CPU run
    eprintln!("=== CPU run ===");
    let cpu_backend = create_backend(BackendPreference::Cpu);
    let mut cpu_model = create_model_with_backend(&gguf, data, Some(cpu_backend))?;
    let cfg = cpu_model.config().clone();
    let mut cpu_kv = KvCache::new(
        cfg.block_count as usize,
        cfg.head_count_kv as usize,
        cfg.head_dim as usize,
        16,
    );
    let cpu_logits = cpu_model.forward(token, 0, &mut cpu_kv)?;
    let (cpu_top_idx, _) = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let cpu_sum: f32 = cpu_logits.iter().map(|x| x.abs()).sum();
    eprintln!("CPU: top={} logit[0]={:.6} sum_abs={:.3}", cpu_top_idx, cpu_logits[0], cpu_sum);
    eprintln!("CPU top-5: {:?}", top5(&cpu_logits));

    // GPU run
    eprintln!("\n=== GPU run ===");
    #[cfg(feature = "wgpu")]
    {
        let gpu_backend = create_backend(BackendPreference::Wgpu);
        let mut gpu_model = create_model_with_backend(&gguf, data, Some(gpu_backend))?;
        let mut gpu_kv = KvCache::new(
            cfg.block_count as usize,
            cfg.head_count_kv as usize,
            cfg.head_dim as usize,
            16,
        );
        let gpu_logits = gpu_model.forward(token, 0, &mut gpu_kv)?;
        let (gpu_top_idx, _) = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let gpu_sum: f32 = gpu_logits.iter().map(|x| x.abs()).sum();
        eprintln!("GPU: top={} logit[0]={:.6} sum_abs={:.3}", gpu_top_idx, gpu_logits[0], gpu_sum);
        eprintln!("GPU top-5: {:?}", top5(&gpu_logits));

        // Compare
        let max_err = cpu_logits.iter().zip(&gpu_logits)
            .map(|(c, g)| (c - g).abs())
            .fold(0f32, f32::max);
        let match_top = cpu_top_idx == gpu_top_idx;
        eprintln!("\nmax_abs_err={:.6e}  top_match={}", max_err, match_top);
    }

    Ok(())
}

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    indexed.truncate(5);
    indexed
}
