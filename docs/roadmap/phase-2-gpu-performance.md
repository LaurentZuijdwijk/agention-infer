# Phase 2 — GPU performance to competitive

**Status:** committed. Depends on **Phase 1** (batched forward + f16 must exist).
See [00-parity-roadmap.md](00-parity-roadmap.md). **This is the gate**: Phases 3–6 start only if this
phase lands competitive numbers and we decide to continue.

## Goal

Turn the Phase 1 refactor into **llama.cpp-competitive tokens/sec** on the target box, and settle the
primary GPU backend. Target: **≥70% of the bandwidth ceiling** on 7–9B Q4_K_M decode, prefill within
~1.5× of llama.cpp.

## Why

Baseline decode is ~9.5 tok/s; the ceiling on 7–9B Q4 at 260 GB/s is ~50–60 tok/s. Being
"competitive on a select set of big models" (the project thesis) means closing most of that gap.
This phase is the price of admission for everything after it.

## Prerequisites

- Phase 0 bench (to measure %-ceiling and compare to llama.cpp) and golden gate (parity must not regress).
- Phase 1 batched forward (flash attention and prefill GEMM need the batched shape).
- Read `docs/PERFORMANCE_RECOMMENDATIONS.md` in full — especially "What we learned about this GPU's
  dispatch cost." Several plausible optimizations already **regressed** and must not be re-attempted.

## Tasks

### 2.1 Backend A/B — decide the primary GPU path (Fork 1)
- [ ] Build a **matmul-only** benchmark comparing `cubecl-wgpu` (Vulkan) against a minimal
      `cubecl-hip` client on gfx1151, at the real decode GEMV and prefill GEMM shapes.
- [ ] Wire the existing native-u8 kernels (`src/ops/kernels_u8.rs`, written for the HIP/CUDA/CPU path)
      to a real `cubecl-hip` client to make the comparison. Set `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- [ ] Investigate Vulkan **cooperative-matrix / subgroup** support on gfx1151 via cubecl — llama.cpp's
      fast Strix Halo path is Vulkan+coopmat.
- [ ] **Decision:** keep Vulkan primary unless HIP shows a clear, reproducible matmul win. Un-stub
      `src/ops/backend_select.rs:73` for whichever wins (today Hip/Cuda fall through to CPU).

### 2.2 Matmul parity (decode GEMV + prefill GEMM)
- [ ] Bring `mul_mat_vec` (decode, batch=1) up to llama.cpp-class on the winning backend. Pursue only
      the **safe levers** (do not reduce dispatch parallelism):
  - [ ] Vectorized byte reads in dequant kernels — `PERFORMANCE_RECOMMENDATIONS.md` Priority 14
        (read whole `u32` words, unpack 4 bytes, instead of per-byte `read_byte_u32`). Q4_K/Q5_K first
        (4-aligned blocks); Q6_K needs unaligned handling.
  - [ ] f16 accumulate where precision allows (already partly enabled by Phase 1).
- [ ] Add a batched `mul_mat` (prefill GEMM) using subgroup/coopmat matmul if available; this is where
      prefill speed comes from.
- [ ] **Do not** reintroduce multi-row/fewer-workgroup matmul designs — measured regressions
      (9.5→6.0 and 9.5→8.1 tok/s). Keep the many-small-workgroups shape.

### 2.3 Flash attention
- [ ] Replace the 3-kernel score/softmax/output split (`ops/kernels_wgpu.rs`,
      `attention_scores`/`attention_softmax`/`attention_output`) with a tiled online-softmax flash
      kernel, now that Phase 1 gives a batched query shape. Preserve the parallelism-preserving design
      (partial max/sum per block, combined) that the perf doc calls out as the non-regressing approach
      (Priority 11 analysis).
- [ ] Keep the naive path available behind a flag as a correctness reference.

### 2.4 Autotune + wave32
- [ ] Confirm cubecl autotune runs on gfx1151 and caches the chosen config to disk at startup
      (`~/.cache/gguf-rs/autotune-<device>.json`). `docs/06-rocm-hardware.md` argues wave32 discovery
      is the biggest llama.cpp gap on this GPU — verify empirically, don't assume.

## Design notes & gotchas

- **This GPU rewards many small independent workgroups.** The scheduler hides memory latency across
  ~40 CUs when there are thousands of small workgroups; consolidating into fewer/fatter ones loses
  occupancy and regresses. Every kernel change must be benched, not reasoned about
  (`PERFORMANCE_RECOMMENDATIONS.md`, closing section).
- **Switching APIs won't fix per-dispatch cost.** A raw-`ash` Vulkan probe (`src/bin/vulkan_poc.rs`)
  showed per-dispatch cost no better than wgpu's. So the A/B in 2.1 is about *matmul kernel quality*
  (coopmat/subgroup, wave32), not about escaping wgpu overhead.
- **Prefill GEMM vs decode GEMV are different regimes.** Prefill (batched) is compute-heavier and
  benefits from coopmat; decode (batch=1) is bandwidth-bound GEMV. Optimize them separately.
- Re-run golden after **every** kernel change — dequant bit-tricks are a real correctness risk (the
  perf doc documents multiple Q4_K/Q5_K debugging sessions).

## Verification

```
cargo test --features golden,wgpu           # parity unchanged
cargo run --release --features wgpu --bin bench -- models/Qwen3.5-9B-Q4_K_M.gguf   # ≥70% ceiling decode
# compare against a local llama.cpp run on the same model/prompt for prefill+decode tok/s
```

## Done-criteria

- Decode ≥70% of bandwidth ceiling on 7–9B Q4_K_M; prefill within ~1.5× of llama.cpp.
- Primary GPU backend decided and un-stubbed; A/B numbers recorded in `docs/roadmap/baselines.md`.
- Flash attention landed with parity preserved; naive path retained as reference.
- **Gate review:** with these numbers in hand, decide whether to proceed to Phases 3–6.
