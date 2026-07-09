//! Throughput benchmark: prefill tok/s, decode tok/s, and % of the memory
//! bandwidth ceiling for a given model + backend. Prints one machine-parseable
//! line so later phases can diff against a recorded baseline.
//!
//!   cargo run --release --features wgpu --bin bench -- models/<m>.gguf --backend wgpu

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use gguf_rs::ops::BackendPreference;
use gguf_rs::runner::greedy_run;

#[derive(Parser)]
#[command(about = "Benchmark prefill/decode throughput vs the bandwidth ceiling")]
struct Args {
    /// Path to the GGUF model.
    model: PathBuf,
    /// Backend: cpu | wgpu | metal.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Prompt used for prefill timing.
    #[arg(long, default_value = "Once upon a time, in a land far away,")]
    prompt: String,
    /// Number of tokens to decode (EOS ignored so timing is over a fixed count).
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    /// Memory bandwidth of the target box, in GB/s (Strix Halo ≈ 260).
    #[arg(long, default_value_t = 260.0)]
    bandwidth_gbs: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let pref = match args.backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => BackendPreference::Wgpu,
        #[cfg(feature = "wgpu")]
        "metal" => BackendPreference::Metal,
        "cpu" => BackendPreference::Cpu,
        other => anyhow::bail!("unknown backend {other:?} (try cpu | wgpu | metal)"),
    };

    let file_bytes = std::fs::metadata(&args.model)?.len();
    let r = greedy_run(&args.model, pref, &args.prompt, args.max_tokens, false)?;

    // Bandwidth-bound decode ceiling: one full sweep of the weights per token.
    // For dense models the on-disk size is a good proxy for active bytes/token.
    let ceiling = args.bandwidth_gbs * 1e9 / file_bytes as f64;
    let decode_tps = r.decode_tok_s();
    let pct = decode_tps / ceiling * 100.0;

    let name = args
        .model
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    println!(
        "model={name} backend={} size_gib={:.2} prompt_tok={} gen_tok={} \
         prefill_tok_s={:.1} decode_tok_s={:.2} ceiling_tok_s={:.1} pct_ceiling={:.1}",
        r.backend,
        file_bytes as f64 / (1u64 << 30) as f64,
        r.prompt_ids.len(),
        r.token_ids.len(),
        r.prefill_tok_s(),
        decode_tps,
        ceiling,
        pct,
    );

    Ok(())
}
