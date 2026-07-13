//! `generate_trace` — end-to-end text generation with per-kernel timing breakdown.
//!
//! Runs the full forward pass (prefill + decode) and collects per-kernel timings
//! via `GGUF_TRACE_KERNEL=json`. Outputs structured JSON with:
//!   - Summary: kernel count, total time, total bandwidth
//!   - Kernels: list of per-kernel timings with effective bandwidth
//!   - Forward pass: prefill time, decode time, tok/s
//!
//! Usage:
//!   GGUF_TRACE_KERNEL=json cargo run --release --features wgpu --bin generate_trace -- \
//!     models/Qwen3.5-9B-Q4_K_M.gguf --backend wgpu --prompt "Hello world" --max-tokens 32

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use gguf_rs::load;
use gguf_rs::model::{create_model_with_backend, KvCache};
use gguf_rs::ops::{create_backend, BackendPreference};
use gguf_rs::sampler::{Sampler, SamplingConfig};
use gguf_rs::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Generate text with per-kernel timing breakdown")]
struct Args {
    /// Path to the GGUF file.
    model: PathBuf,

    /// Prompt text.
    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(short = 'n', long, default_value_t = 32)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy / deterministic).
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-k filtering (0 disables).
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-p (nucleus) filtering.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// RNG seed.
    #[arg(long, default_value_t = 0xD1CE_5EED)]
    seed: u64,

    /// Greedy decoding (overrides temperature/top-k/top-p).
    #[arg(long)]
    greedy: bool,

    /// Backend to use (cpu, metal, wgpu). Default: cpu.
    #[arg(long, default_value = "cpu")]
    backend: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Load ────────────────────────────────────────────────────────────
    let load_start = Instant::now();
    let (gguf, mmap) =
        load(&args.model).with_context(|| format!("loading {}", args.model.display()))?;
    let data = &mmap[gguf.data_offset as usize..];

    let tokenizer = Tokenizer::from_gguf(&gguf).context("building tokenizer")?;

    // Select backend
    let backend_pref = match args.backend.as_str() {
        #[cfg(feature = "wgpu")]
        "metal" => BackendPreference::Metal,
        #[cfg(feature = "wgpu")]
        "wgpu" => BackendPreference::Wgpu,
        _ => BackendPreference::Cpu,
    };
    let backend = create_backend(backend_pref);
    let backend_name = backend.name().to_string();

    let t0 = Instant::now();
    let mut model =
        create_model_with_backend(&gguf, data, Some(backend)).context("creating model")?;
    let cfg = model.config().clone();

    // ── Tokenize ────────────────────────────────────────────────────────
    let prompt_ids = tokenizer.encode(args.prompt.as_str());
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");

    let max_seq = (prompt_ids.len() + args.max_tokens).min(cfg.context_length as usize);
    let mut kv = KvCache::new(
        cfg.block_count as usize,
        cfg.head_count_kv as usize,
        cfg.head_dim as usize,
        max_seq,
    );

    let sampling = if args.greedy {
        SamplingConfig::greedy()
    } else {
        SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
        }
    };
    let mut sampler = Sampler::new(sampling);

    // ── GPU-resident initialization (if using GPU) ──────────────────────
    #[cfg(feature = "wgpu")]
    if args.backend != "cpu" {
        // Upload all weights to GPU and warm up shader compilation.
        // This is required for the GPU-resident path to be activated.
        eprintln!("\n  Pre-uploading weights to GPU...");
        model.pre_upload_gpu();
        eprintln!("\n  Warmup GPU kernels (shader compilation)...");
        model.warmup_gpu_kernels();

        // Check if resident path will be used
        eprintln!("\n  GPU-resident path ready: {}", model.gpu_resident_ready());
        if !model.gpu_resident_ready() {
            eprintln!("  WARNING: resident path not ready, falling back to CPU-orchestrated path");
        }
    }

    // ── Prefill ─────────────────────────────────────────────────────────
    let prefill_start = Instant::now();
    let mut logits = model.forward_batch(&prompt_ids, 0, &mut kv)?;
    let mut pos = prompt_ids.len();
    let prefill_time = prefill_start.elapsed();

    // ── Decode loop ─────────────────────────────────────────────────────
    let decode_start = Instant::now();
    let mut generated = 0usize;

    while generated < args.max_tokens && pos < max_seq {
        let next = sampler.sample(&logits);

        if Some(next) == tokenizer.eos_token_id {
            break;
        }

        logits = model.forward(next, pos, &mut kv)?;
        pos += 1;
        generated += 1;
    }

    let decode_time = decode_start.elapsed();

    // ── Output ──────────────────────────────────────────────────────────
    // First, dump the kernel trace (if any)
    gguf_rs::ops::trace::dump_trace();

    // Then, output the summary
    let total_time = prefill_time + decode_time;
    let total_tokens = prompt_ids.len() + generated;
    let prefill_tps = prompt_ids.len() as f64 / prefill_time.as_secs_f64().max(1e-9);
    let decode_tps = generated as f64 / decode_time.as_secs_f64().max(1e-9);
    let overall_tps = total_tokens as f64 / total_time.as_secs_f64().max(1e-9);

    if gguf_rs::ops::trace::format() == "json" {
        // JSON output
        eprintln!(
            "{}",
            serde_json::json!({
                "model": args.model.display().to_string(),
                "backend": backend_name,
                "prompt_length": prompt_ids.len(),
                "generated": generated,
                "total_tokens": total_tokens,
                "prefill_time_ms": prefill_time.as_secs_f64() * 1e3,
                "decode_time_ms": decode_time.as_secs_f64() * 1e3,
                "total_time_ms": total_time.as_secs_f64() * 1e3,
                "prefill_tok_s": (prefill_tps * 100.0).round() / 100.0,
                "decode_tok_s": (decode_tps * 100.0).round() / 100.0,
                "overall_tok_s": (overall_tps * 100.0).round() / 100.0,
            })
        );
    } else {
        // Text output
        eprintln!(
            "\n=== Forward Pass Timing ==="
        );
        eprintln!(
            "  prefill: {} tok in {:.2}s ({:.1} tok/s)",
            prompt_ids.len(),
            prefill_time.as_secs_f64(),
            prefill_tps
        );
        eprintln!(
            "  decode: {} tok in {:.2}s ({:.1} tok/s)",
            generated,
            decode_time.as_secs_f64(),
            decode_tps
        );
        eprintln!(
            "  total: {} tok in {:.2}s ({:.1} tok/s)",
            total_tokens,
            total_time.as_secs_f64(),
            overall_tps
        );
    }

    Ok(())
}
