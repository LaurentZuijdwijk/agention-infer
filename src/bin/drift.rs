//! Long-generation drift check: CPU (f32) vs GPU (f16 activations) greedy
//! decode over many tokens, asserting token-for-token agreement.
//!
//! The golden fixtures are short (~24 tokens), which won't surface precision
//! drift that only compounds over a long generation — in particular the Gated
//! DeltaNet recurrence accumulates persistent state across every token. This
//! binary drives both backends through the same prompt for 200+ tokens and
//! reports the first divergence (if any), so f16 activations can be trusted for
//! real generations, not just the golden window.
//!
//!   cargo run --release --features wgpu --bin drift -- models/Qwen3.5-9B-Q4_K_M.gguf [n_tokens] [prompt]
//!
//! Exit status is non-zero if the two backends diverge.

#[cfg(feature = "wgpu")]
fn main() {
    use gguf_rs::ops::BackendPreference;
    use gguf_rs::runner::greedy_run;
    use std::path::Path;
    use std::process::exit;

    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| {
        eprintln!("usage: drift <model.gguf> [n_tokens] [prompt]");
        exit(2);
    });
    let n_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let prompt = args
        .next()
        .unwrap_or_else(|| "Count slowly and explain each step as you go: one, two,".to_string());

    let path = Path::new(&model);
    if !path.exists() {
        eprintln!("model not found: {model}");
        exit(2);
    }

    let act = if cfg!(feature = "f32-activations") {
        "f32"
    } else {
        "f16"
    };
    println!("drift: model={model}\n  n_tokens={n_tokens} gpu_activations={act}\n  prompt={prompt:?}\n");

    // Fixed token count (no EOS stop) so both backends generate the same length.
    let cpu = greedy_run(path, BackendPreference::Cpu, &prompt, n_tokens, false)
        .unwrap_or_else(|e| panic!("cpu run: {e}"));
    let gpu = greedy_run(path, BackendPreference::Wgpu, &prompt, n_tokens, false)
        .unwrap_or_else(|e| panic!("gpu run: {e}"));

    println!(
        "  cpu backend={} tokens={} | gpu backend={} tokens={}",
        cpu.backend,
        cpu.token_ids.len(),
        gpu.backend,
        gpu.token_ids.len(),
    );

    let n = cpu.token_ids.len().min(gpu.token_ids.len());
    let mut first_div = None;
    for i in 0..n {
        if cpu.token_ids[i] != gpu.token_ids[i] {
            first_div = Some(i);
            break;
        }
    }

    match first_div {
        None if cpu.token_ids.len() == gpu.token_ids.len() => {
            println!("\n✅ CLEAN — {n} tokens identical (CPU f32 == GPU {act})");
        }
        None => {
            // Prefixes agree but lengths differ (only possible if one hit a
            // guard); still a divergence worth surfacing.
            println!("\n❌ length mismatch: cpu={} gpu={}", cpu.token_ids.len(), gpu.token_ids.len());
            exit(1);
        }
        Some(i) => {
            let lo = i.saturating_sub(3);
            println!("\n❌ DIVERGES at token {i}");
            println!("  cpu[{lo}..{}] = {:?}", i + 1, &cpu.token_ids[lo..=i]);
            println!("  gpu[{lo}..{}] = {:?}", i + 1, &gpu.token_ids[lo..=i]);
            println!("\n  cpu text: {:?}", cpu.text);
            println!("  gpu text: {:?}", gpu.text);
            exit(1);
        }
    }
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("drift requires --features wgpu");
    std::process::exit(2);
}
