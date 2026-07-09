//! Golden regression gate: replay llama.cpp reference completions through our
//! engine's greedy decode and assert token-for-token text equality.
//!
//! Fixtures live in `tests/fixtures/golden/*.txt` and are produced by
//! `scripts/gen_golden.sh` from llama.cpp's `llama-simple` (raw greedy, no chat
//! template). See `docs/roadmap/phase-0-correctness-harness.md`.
//!
//! Model-dependent: each fixture is **skipped** (not failed) when its model
//! file is absent, so `cargo test` stays green on machines without models /
//! in CI. Point at models with `GGUF_MODELS_DIR` (defaults to `./models`).

use std::path::{Path, PathBuf};

use gguf_rs::ops::BackendPreference;
use gguf_rs::runner::greedy_run;

struct Fixture {
    model_file: String,
    prompt: String,
    n_predict: usize,
    completion: String,
    name: String,
}

fn parse_fixture(path: &Path) -> Fixture {
    let raw = std::fs::read_to_string(path).expect("read fixture");
    // Header lines (`# key: value`) then a blank line then the completion.
    let (header, completion) = raw
        .split_once("\n\n")
        .expect("fixture must have a blank line separating header and completion");
    let mut model_file = String::new();
    let mut prompt = String::new();
    let mut n_predict = 0usize;
    for line in header.lines() {
        if let Some(v) = line.strip_prefix("# model:") {
            model_file = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("# prompt:") {
            // Keep leading space after the colon intact except the single
            // separator space we add in the generator.
            prompt = v.strip_prefix(' ').unwrap_or(v).to_string();
        } else if let Some(v) = line.strip_prefix("# n_predict:") {
            n_predict = v.trim().parse().unwrap_or(0);
        }
    }
    Fixture {
        model_file,
        prompt,
        n_predict,
        completion: completion.to_string(),
        name: path.file_stem().unwrap().to_string_lossy().into_owned(),
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden")
}

fn models_dir() -> PathBuf {
    std::env::var("GGUF_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("models"))
}

fn first_divergence(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Run every fixture whose model is present against `pref`. Returns the number
/// actually checked (0 => everything was skipped).
fn run_all(pref: BackendPreference) -> usize {
    let dir = fixtures_dir();
    let mut checked = 0usize;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    entries.sort();

    for path in entries {
        let fx = parse_fixture(&path);
        let model_path = models_dir().join(&fx.model_file);
        if !model_path.exists() {
            eprintln!("SKIP {} (missing {})", fx.name, model_path.display());
            continue;
        }

        let result = greedy_run(&model_path, pref, &fx.prompt, fx.n_predict, false)
            .unwrap_or_else(|e| panic!("greedy_run {}: {e}", fx.name));

        // GOLDEN_REPORT=1 surveys all fixtures without failing (diagnostics).
        if std::env::var("GOLDEN_REPORT").is_ok() {
            if result.text == fx.completion {
                eprintln!("MATCH {} backend={}", fx.name, result.backend);
            } else {
                let d = first_divergence(&result.text, &fx.completion);
                eprintln!(
                    "DIFF  {} backend={} first_diverge_char={d}\n    exp: {:?}\n    got: {:?}",
                    fx.name, result.backend, fx.completion, result.text
                );
            }
            checked += 1;
            continue;
        }

        if result.text != fx.completion {
            let d = first_divergence(&result.text, &fx.completion);
            panic!(
                "GOLDEN MISMATCH [{}] backend={}\n  prompt:   {:?}\n  diverges at char {d}\n  expected: {:?}\n  got:      {:?}",
                fx.name, result.backend, fx.prompt, fx.completion, result.text,
            );
        }
        eprintln!("OK   {} backend={}", fx.name, result.backend);
        checked += 1;
    }
    checked
}

#[test]
fn golden_cpu() {
    let checked = run_all(BackendPreference::Cpu);
    if checked == 0 {
        eprintln!("golden_cpu: no models present, all fixtures skipped");
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn golden_wgpu() {
    let checked = run_all(BackendPreference::Wgpu);
    if checked == 0 {
        eprintln!("golden_wgpu: no models present, all fixtures skipped");
    }
}
