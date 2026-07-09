# Phase 0 — Correctness harness + benchmark + YaRN fix

**Status:** committed (build first). See [00-parity-roadmap.md](00-parity-roadmap.md) for context.

## Goal

Make every future change *safe and measurable*: a committed golden regression gate against
llama.cpp, a repeatable performance benchmark, and a fix for the one known correctness bug (RoPE
scaling / YaRN is parsed but never applied). After this phase, no later change can silently break
correctness or speed without a test going red.

## Why

Phases 1–6 touch numerics (f16, new kernels, MoE, a possible new backend). Without a token-for-token
oracle and a tok/s baseline, regressions are invisible until a human notices bad output. llama.cpp
is the oracle; we operationalize that rule here. The YaRN fix is folded in because it's a latent
correctness bug that the golden harness must be able to catch (and long-context models need it).

## Prerequisites

- A working local `llama-cli` (llama.cpp) to generate reference outputs. Document the exact
  commit/version used in the fixtures.
- The 4 checked-in models in `models/`: `Qwen3.5-9B-Q4_K_M`, `Qwen3-1.7B-Q8_0`,
  `LFM2.5-1.2B-Instruct-Q4_K_M`, `LFM2.5-1.2B-Instruct-Q6_K`.
- Existing `src/bin/compare_backends.rs` (first-token CPU↔GPU logit parity) and `examples/prof.rs`
  (ad-hoc timing) to build on.

## Tasks

### 0.1 Golden test harness (vs llama.cpp, temp=0)
- [ ] Add a fixture generator that shells to local `llama-cli` and captures, per (model, prompt):
      the greedy token sequence (N tokens) and optionally top-k logits at position 0. Store as JSON
      under `tests/fixtures/golden/<model>__<prompt-id>.json`. Record the llama.cpp version in each file.
- [ ] `tests/golden.rs`: load each fixture, run our engine greedy (`SamplingConfig::greedy()`),
      assert token-for-token equality. Run on **both** CPU and (when `--features wgpu`) GPU backends.
- [ ] Gate the model-dependent tests behind a cargo feature (e.g. `golden`) or a `GGUF_MODELS_DIR`
      env check so `cargo test` stays fast/green on machines without the models.
- [ ] Keep at least one tiny prompt per model so the suite runs in seconds.

### 0.2 Promote CPU↔GPU cross-check into the test tier
- [ ] Generalize `compare_backends` beyond the first token: compare top-1 and top-5 over a short
      greedy run, not just position 0. Reuse its `top5` helper.
- [ ] Expose it as a library helper (e.g. `tests/support/`) so both `golden.rs` and the binary use
      one implementation. There is already a `GGUF_CROSSCHECK_GPU` path in the model — reuse it.

### 0.3 Benchmark harness
- [ ] `src/bin/bench.rs` (or `benches/`): for a given model+backend, measure **prefill tok/s**,
      **decode tok/s**, and **% of bandwidth ceiling** (`ceiling = 260e9 / active_bytes_per_token`;
      for dense models active_bytes ≈ model size on disk). Print a one-line, machine-parseable result.
- [ ] Record the current baseline (~9.5 tok/s decode on Qwen3.5-9B Q4_K_M) as a checked-in
      `docs/roadmap/baselines.md` row so later phases can show deltas.
- [ ] Reuse `examples/prof.rs` timing patterns; keep warmup (`warmup_gpu_kernels`) out of the timed region.

### 0.4 Fix RoPE scaling / YaRN (correctness bug)
- [ ] `ModelConfig` already reads `rope_scaling_type` / `rope_scaling_factor`
      (`src/model/mod.rs:242`). Implement the actual scaling in the rope math:
      `cpu_path.rs::rope` (line ~50), `cpu_path.rs::rope_partial` (line ~407), and the GPU
      `launch_rope` / `qk_norm_rope` kernels in `ops/kernels_wgpu.rs`.
- [ ] Support at minimum **linear** scaling and **YaRN** (NTK-by-parts). Match llama.cpp's formula
      (`rope_yarn` in `ggml`), including the attention-factor / mscale term.
- [ ] Add a golden test on a YaRN/long-context model (pick one with `rope.scaling.type = yarn` in
      metadata) that only passes once scaling is applied.

### 0.5 CI
- [ ] GitHub Actions: `cargo build`, `cargo clippy -D warnings`, `cargo test` (fast tier, no models).
      Golden/model tests run only where models are present.

## Design notes & gotchas

- **Keep the harness dtype-agnostic.** Phase 1 switches activations to f16; the golden comparison
  must tolerate a small numeric epsilon on *logits* but still require exact *token* match at temp=0.
  Design the assertion around token equality, with logit-closeness as a secondary diagnostic.
- **llama.cpp determinism:** run llama.cpp greedy with a fixed context and no repeat penalty so the
  reference is reproducible. Pin and record the version — kernels change output slightly across releases.
- **YaRN reference:** the trig table hoist already in `rope()` (per-`d`, out of the head loop) is the
  right place to fold the scaling factor in — do it once per `d`, not per head.
- Don't regress the existing per-`d` trig-table optimization (see `PERFORMANCE_RECOMMENDATIONS.md`
  Priority 4) when adding scaling.

## Verification

```
cargo test                                  # fast tier green
cargo clippy --all-targets -- -D warnings
GGUF_MODELS_DIR=models cargo test --features golden,wgpu   # golden match on all 4 models, CPU+GPU
cargo run --release --features wgpu --bin bench -- models/Qwen3.5-9B-Q4_K_M.gguf   # records baseline
```

## Done-criteria

- Golden suite matches llama.cpp token-for-token on all 4 models on CPU **and** wgpu.
- At least one YaRN model passes only because scaling is now applied.
- `bench` prints prefill+decode tok/s and %-ceiling; baseline row committed to `baselines.md`.
- CI green on push (fast tier).
