use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use gguf_rs::{GgufFile, ModelInfo};

/// Inspect GGUF model files.
#[derive(Parser, Debug)]
#[command(name = "gguf-info", version, about)]
struct Args {
    /// Path to the GGUF file
    model: PathBuf,

    /// Show tensor details
    #[arg(long, short = 't')]
    tensors: bool,

    /// Show all metadata keys and values
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Show memory budget for different KV cache bit depths
    #[arg(long, short = 'm')]
    memory: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (gguf, _mmap) = gguf_rs::load(&args.model)
        .with_context(|| format!("failed to load GGUF file: {}", args.model.display()))?;

    print_summary(&gguf)?;

    if args.verbose {
        print_metadata(&gguf)?;
    }

    if args.memory {
        print_memory_budget(&gguf)?;
    }

    if args.tensors {
        print_tensors(&gguf)?;
    }

    Ok(())
}

fn fmt_opt<T: std::fmt::Display>(opt: Option<T>) -> String {
    match opt {
        Some(v) => v.to_string(),
        None => "—".to_string(),
    }
}

fn print_summary(gguf: &GgufFile) -> Result<()> {
    let info = ModelInfo::from_gguf(gguf)?;

    println!("═══ GGUF File Summary ═══");
    println!();

    if let Some(name) = &info.name {
        println!("  Name:           {name}");
    }
    println!("  Architecture:   {}", info.architecture);
    println!("  GGUF Version:   {}", gguf.version);

    if let Some(v) = info.block_count {
        println!("  Layers:         {v}");
    }
    if let Some(v) = info.context_length {
        println!("  Context Length: {v}");
    }
    if let Some(v) = info.embedding_length {
        println!("  Embedding Dim:  {v}");
    }
    if let Some(v) = info.head_count {
        println!("  Heads (Q):      {v}");
    }
    if let Some(kv) = info.head_count_kv {
        println!("  Heads (KV):     {kv}");
    } else if info.head_count.is_some() {
        // No explicit KV heads → MHA (same as Q heads)
        println!("  Heads (KV):     {} (MHA)", fmt_opt(info.head_count));
    }
    if let Some(v) = info.head_dim() {
        println!("  Head Dim:       {v}");
    }
    if let Some(v) = info.gqa_group_size() {
        println!("  GQA Groups:     {v}");
    }

    if let Some(v) = info.feed_forward_length {
        if v > 0 {
            println!("  FFN Dim:        {v}");
        }
    }

    if let Some(v) = info.rope_freq_base {
        println!("  RoPE Freq Base: {v}");
    }
    if let Some(ref scaling_type) = info.rope_scaling_type {
        println!("  RoPE Scaling:   {scaling_type}");
        if let Some(factor) = info.rope_scaling_factor {
            println!("  RoPE Factor:    {factor}");
        }
    }
    if let Some(v) = info.layer_norm_rms_epsilon {
        println!("  RMSNorm Eps:    {v:.1e}");
    }

    if let Some(ref moe) = info.moe {
        println!();
        println!("  ─── MoE ───");
        println!("  Experts:        {}", moe.expert_count);
        println!("  Active/Token:   {}", moe.expert_used_count);
        println!("  Expert FFN Dim: {}", moe.expert_feed_forward_length);
        if let Some(shared) = moe.expert_shared_count {
            println!("  Shared Experts: {shared}");
        }
    }

    // Param count and size
    let param_count: u64 = gguf.tensors.iter().map(|t| t.n_elements() as u64).sum();
    let model_size = gguf.total_tensor_bytes();
    println!();
    println!("  Parameters:     {}", format_count(param_count));
    println!("  Model Size:     {}", format_bytes(model_size));
    println!("  Tensor Count:   {}", gguf.tensors.len());
    println!("  Metadata Keys:  {}", gguf.metadata.len());

    if let Some(license) = &info.license {
        println!("  License:        {license}");
    }

    if let Some(ft) = info.file_type {
        println!("  File Type:      {ft}");
    }

    // Quantization distribution
    let dist = quant_distribution(gguf);
    if dist.len() > 1 {
        println!();
        println!("  ─── Quantization Mix ───");
        for (qtype, count) in &dist {
            println!("  {qtype:>12}: {count} tensor(s)");
        }
    } else if let Some((qtype, _)) = dist.first() {
        println!("  Quantization:   {qtype}");
    }

    println!();
    Ok(())
}

fn print_metadata(gguf: &GgufFile) -> Result<()> {
    println!("═══ Metadata ═══");
    println!();

    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();

    for key in keys {
        let value = gguf.metadata.get(key).unwrap();
        let display = value.display_value();
        // Truncate very long values
        let truncated = if display.len() > 120 {
            &display[..117]
        } else {
            &display
        };
        println!("  {key} = {truncated}");
    }

    println!();
    Ok(())
}

fn detect_system_ram() -> (u64, u64) {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    // available_memory() is preferred but can be unreliable on some platforms;
    // fall back to free_memory() if it reports 0
    let available = sys.available_memory();
    let free = sys.free_memory();
    (total, available.max(free))
}

fn print_memory_budget(gguf: &GgufFile) -> Result<()> {
    let info = ModelInfo::from_gguf(gguf)?;
    let model_size = info.model_bytes(gguf);

    // Check if we have enough info for memory budget calculation
    if info.effective_head_count_kv().is_none()
        || info.head_dim().is_none()
        || info.block_count.is_none()
    {
        println!("═══ Memory Budget ═══");
        println!();
        println!("  Insufficient metadata to calculate KV cache budget.");
        println!("  (need head_count, embedding_length, and block_count)");
        println!();
        return Ok(());
    }

    let (total_ram, available_ram) = detect_system_ram();

    println!("═══ Memory Budget ═══");
    println!();

    if total_ram > 0 {
        println!("  System RAM:     {}", format_bytes(total_ram));
        println!("  Available RAM:  {}", format_bytes(available_ram));
    }
    println!("  Model size:     {}", format_bytes(model_size));

    if total_ram > 0 {
        let after_load = total_ram.saturating_sub(model_size);
        println!("  After loading:  {}", format_bytes(after_load));
    }
    println!();

    println!("  ┌──────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("  │ System RAM    │ KV @ 16-bit     │ KV @ 8-bit      │ KV @ 3-bit      │");
    println!("  ├──────────────┼─────────────────┼─────────────────┼─────────────────┤");

    let max_ctx = info.context_length.unwrap_or(u64::MAX);

    let base_tiers: Vec<f64> = vec![8.0, 16.0, 32.0, 64.0, 96.0, 128.0, 192.0, 256.0];
    let system_gb = total_ram as f64 / 1_073_741_824.0;

    // Insert the actual system RAM tier if it doesn't match any standard tier
    let mut ram_tiers = base_tiers;
    if total_ram > 0 {
        let matches_existing = ram_tiers.iter().any(|&t| (t - system_gb).abs() / t < 0.05);
        if !matches_existing {
            // Round to nearest GB for display
            let rounded = system_gb.round();
            // Insert in sorted position
            let pos = ram_tiers
                .iter()
                .position(|&t| t > rounded)
                .unwrap_or(ram_tiers.len());
            ram_tiers.insert(pos, rounded);
        }
    }

    for &ram_gb in &ram_tiers {
        let is_current = total_ram > 0 && (ram_gb - system_gb).abs() / ram_gb.max(1.0) < 0.05;
        let ram_bytes = (ram_gb * 1_073_741_824.0) as u64;
        let available = ram_bytes.saturating_sub(model_size);

        let cap_ctx = |ctx: u64| -> (u64, bool) {
            if ctx > max_ctx {
                (max_ctx, true)
            } else {
                (ctx, false)
            }
        };

        // Exact, block-overhead-aware figures for the two `--kv-type` values
        // the engine actually supports (f16, q8_0) — not just a flat
        // bits-per-element estimate.
        let (ctx_16, hit_cap_16) = if available > 0 {
            cap_ctx(
                info.max_context_at_kv_dtype(available, gguf_rs::model::KvDtype::F16)
                    .unwrap_or(0),
            )
        } else {
            (0, false)
        };
        let (ctx_8, hit_cap_8) = if available > 0 {
            cap_ctx(
                info.max_context_at_kv_dtype(available, gguf_rs::model::KvDtype::Q8_0)
                    .unwrap_or(0),
            )
        } else {
            (0, false)
        };
        let (ctx_3, hit_cap_3) = if available > 0 {
            cap_ctx(info.max_context_at_kv_bits(available, 3).unwrap_or(0))
        } else {
            (0, false)
        };

        let marker = if is_current { " ◀" } else { "  " };
        let fmt_ctx = |ctx: u64, hit: bool| {
            if hit {
                format!("{ctx}*")
            } else {
                format!("{ctx}")
            }
        };
        println!(
            "  │ {ram_gb:>7.0} GB{marker}│ {:>13}   │ {:>13}   │ {:>13}   │",
            fmt_ctx(ctx_16, hit_cap_16),
            fmt_ctx(ctx_8, hit_cap_8),
            fmt_ctx(ctx_3, hit_cap_3)
        );
    }

    println!("  └──────────────┴─────────────────┴─────────────────┴─────────────────┘");
    if max_ctx < u64::MAX {
        println!("  * capped at model max context ({max_ctx})");
    }
    println!("  (Context lengths in tokens)");
    println!();
    Ok(())
}

fn print_tensors(gguf: &GgufFile) -> Result<()> {
    println!("═══ Tensors ═══");
    println!();

    // Header
    println!(
        "  {:<45} {:>14} {:>10} {:>10}",
        "Name", "Shape", "Type", "Size"
    );
    println!("  {}", "─".repeat(82));

    let mut total_bytes: u64 = 0;
    for tensor in &gguf.tensors {
        let size = tensor.byte_size();
        total_bytes += size as u64;
        println!(
            "  {:<45} {:>14} {:>10} {:>10}",
            truncate_name(&tensor.name, 44),
            tensor.shape_str(),
            tensor.ggml_type.name(),
            format_bytes(size as u64),
        );
    }

    println!("  {}", "─".repeat(82));
    println!(
        "  {:<45} {:>14} {:>10} {:>10}",
        "",
        "",
        "Total:",
        format_bytes(total_bytes)
    );
    println!();
    Ok(())
}

fn quant_distribution(gguf: &GgufFile) -> Vec<(String, usize)> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for tensor in &gguf.tensors {
        *counts
            .entry(tensor.ggml_type.name().to_string())
            .or_insert(0) += 1;
    }
    let mut result: Vec<_> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        let start = name.len() - max_len + 3;
        format!("...{}", &name[start..])
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn format_count(n: u64) -> String {
    const BILLION: u64 = 1_000_000_000;
    const MILLION: u64 = 1_000_000;
    const THOUSAND: u64 = 1_000;

    if n >= BILLION {
        format!("{:.1}B", n as f64 / BILLION as f64)
    } else if n >= MILLION {
        format!("{:.1}M", n as f64 / MILLION as f64)
    } else if n >= THOUSAND {
        format!("{:.1}K", n as f64 / THOUSAND as f64)
    } else {
        n.to_string()
    }
}
