//! `generate` — end-to-end text generation from a GGUF model.
//!
//! Loads a model, tokenizes a prompt, runs the prefill + decode loop with a
//! KV cache, samples tokens, and streams decoded text to stdout.

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
#[command(about = "Generate text from a GGUF language model")]
struct Args {
    /// Path to the GGUF file.
    model: PathBuf,

    /// Prompt text.
    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
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

    /// Wrap the prompt in the chat template (<|im_start|> … assistant).
    #[arg(long)]
    chat: bool,

    /// Print the tokenized prompt and timing diagnostics.
    #[arg(long)]
    verbose: bool,

    /// Backend to use (cpu, metal, wgpu). Default: cpu.
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// KV cache storage dtype: f16 (default) or q8_0. llama.cpp `-ctk`/`-ctv`
    /// parity — q8_0 quarters KV memory/bandwidth vs the original f32 store
    /// at a small precision cost.
    #[arg(long, default_value = "f16")]
    kv_type: String,
}

/// eprintln that flushes immediately — makes sure tracing shows up in real
/// time even if stderr is piped rather than a TTY.
macro_rules! trace {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
        std::io::stderr().flush().ok();
    }};
}

fn main() -> Result<()> {
    let args = Args::parse();
    trace!("generate: starting, model={}", args.model.display());

    // ── Load ────────────────────────────────────────────────────────────
    let load_start = Instant::now();
    let (gguf, mmap) =
        load(&args.model).with_context(|| format!("loading {}", args.model.display()))?;
    let data = &mmap[gguf.data_offset as usize..];
    trace!("  gguf parsed + mmapped ({:.2}s)", load_start.elapsed().as_secs_f32());

    let tokenizer = Tokenizer::from_gguf(&gguf).context("building tokenizer")?;
    trace!("  tokenizer built ({:.2}s)", load_start.elapsed().as_secs_f32());

    // Select backend
    trace!("Selecting backend {}...", args.backend);
    let t0 = Instant::now();
    let backend_pref = match args.backend.as_str() {
        #[cfg(feature = "wgpu")]
        "metal" => BackendPreference::Metal,
        #[cfg(feature = "wgpu")]
        "wgpu" => BackendPreference::Wgpu,
        _ => BackendPreference::Cpu,
    };
    let backend = create_backend(backend_pref);
    let backend_name = backend.name().to_string();
    trace!(
        "  backend created: {backend_name} ({:.2}s)",
        t0.elapsed().as_secs_f32()
    );
    let t0 = Instant::now();
    trace!("  creating model (dequant + GPU upload + kernel warmup happen here)...");
    let mut model =
        create_model_with_backend(&gguf, data, Some(backend)).context("creating model")?;
    trace!("  model created ({:.2}s)", t0.elapsed().as_secs_f32());
    let cfg = model.config().clone();

    if args.verbose {
        eprintln!(
            "loaded {} ({} layers, d={}, heads {}/{}, head_dim {}) on {backend_name} in {:.2}s",
            cfg.architecture,
            cfg.block_count,
            cfg.embedding_length,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.head_dim,
            load_start.elapsed().as_secs_f32(),
        );
    }

    // ── Tokenize ────────────────────────────────────────────────────────
    let prompt = if args.chat {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            args.prompt
        )
    } else {
        args.prompt.clone()
    };
    let prompt_ids = tokenizer.encode(&prompt);
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt encoded to zero tokens");

    if args.verbose {
        eprintln!("prompt: {} tokens {:?}", prompt_ids.len(), prompt_ids);
    }

    // ── Set up cache + sampler ──────────────────────────────────────────
    let kv_dtype = gguf_rs::model::KvDtype::parse(&args.kv_type)
        .with_context(|| format!("invalid --kv-type {:?} (expected f16 or q8_0)", args.kv_type))?;
    let max_seq = (prompt_ids.len() + args.max_tokens).min(cfg.context_length as usize);
    let mut kv = KvCache::new_with_dtype(
        cfg.block_count as usize,
        cfg.head_count_kv as usize,
        cfg.head_dim as usize,
        max_seq,
        kv_dtype,
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

    // Echo the prompt itself.
    print!("{prompt}");
    std::io::stdout().flush().ok();

    // ── Prefill ─────────────────────────────────────────────────────────
    trace!("prefill: {} tokens...", prompt_ids.len());
    let prefill_start = Instant::now();
    // One batched pass over the whole prompt (one weight read per layer),
    // rather than a `forward` per token.
    let mut logits = model.forward_batch(&prompt_ids, 0, &mut kv)?;
    let mut pos = prompt_ids.len();
    let prefill_time = prefill_start.elapsed();
    trace!("prefill done ({:.2}s)", prefill_time.as_secs_f32());

    // ── Decode loop ─────────────────────────────────────────────────────
    let decode_start = Instant::now();
    let mut pending = Vec::<u8>::new(); // buffer for partial UTF-8
    let mut generated = 0usize;

    while generated < args.max_tokens && pos < max_seq {
        let next = sampler.sample(&logits);

        if Some(next) == tokenizer.eos_token_id {
            break;
        }

        // Stream: accumulate bytes, flush the valid UTF-8 prefix.
        pending.extend_from_slice(&tokenizer.token_bytes(next));
        flush_utf8(&mut pending);

        logits = model.forward(next, pos, &mut kv)?;
        pos += 1;
        generated += 1;
    }

    // Flush any remainder (lossily if it ends mid-char).
    if !pending.is_empty() {
        print!("{}", String::from_utf8_lossy(&pending));
    }
    println!();
    std::io::stdout().flush().ok();

    if args.verbose {
        let dt = decode_start.elapsed().as_secs_f32();
        let tps = generated as f32 / dt.max(1e-6);
        eprintln!(
            "\nprefill: {} tok in {:.2}s ({:.1} tok/s) · decode: {} tok in {:.2}s ({:.1} tok/s)",
            prompt_ids.len(),
            prefill_time.as_secs_f32(),
            prompt_ids.len() as f32 / prefill_time.as_secs_f32().max(1e-6),
            generated,
            dt,
            tps,
        );
    }

    Ok(())
}

/// Print and drain the longest valid UTF-8 prefix of `buf`, keeping any
/// trailing incomplete multi-byte sequence for the next token.
fn flush_utf8(buf: &mut Vec<u8>) {
    let valid_up_to = match std::str::from_utf8(buf) {
        Ok(s) => s.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid_up_to > 0 {
        // Safe: we only take the validated prefix.
        let s = std::str::from_utf8(&buf[..valid_up_to]).unwrap();
        print!("{s}");
        std::io::stdout().flush().ok();
        buf.drain(..valid_up_to);
    }
}
