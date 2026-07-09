# Parity Roadmap — agention-infer on Strix Halo

> **This is the current, authoritative plan.** It supersedes `docs/10-build-roadmap.md`
> (the old 11-phase vision). Where the older `docs/0x` design docs disagree with what is
> actually in the tree, this roadmap and the per-phase docs below are correct.

## The thesis (why this project exists)

llama.cpp already runs on Strix Halo. We are **not** trying to reimplement llama.cpp, and we are
**not** chasing its 70+ architectures. The bet is narrower and sharper:

1. **Be competitive** (matching tokens/sec, correct output) on a **select set of big models** that
   actually matter on a 128 GB unified-memory APU — dense workhorses plus the large MoE models the
   RAM exists to run.
2. **Then differentiate** on capabilities llama.cpp is slow to mainline — TurboQuant-style low-bit
   KV cache, speculative decoding tuned for this box, expert-locality scheduling for huge MoE.

Speed and correctness are the price of admission (Phases 0–2). The differentiators (Phases 3–6)
are the reason to keep going — and we only start them once the box is genuinely fast.

### Target hardware

Ryzen AI Max+ 395 / Strix Halo — Radeon 8060S (RDNA 3.5, gfx1151), **128 GB** unified LPDDR5X,
**~260 GB/s**. Bandwidth-bound, batch-1 decode is the dominant workload. The bandwidth ceiling
(`tok/s ≈ bandwidth / active-bytes-per-token`) is the number every optimization is measured against.

### Non-goals (for now)

Full architecture breadth, training, distributed/multi-node, multimodal, NPU/XDNA offload, Windows.
CUDA. These are explicitly out of scope until the core thesis is delivered.

## Current state (verified against the tree, not the old docs)

Crate `gguf-rs`, ~9.5k LOC. What actually works today:

- **Backends:** CPU (rayon + `wide` SIMD) and a working **wgpu/Vulkan** GPU-resident path via
  CubeCL. `Hip`/`Cuda`/`Metal` are declared but **HIP/CUDA silently fall back to CPU**
  (`src/ops/backend_select.rs`).
- **Architectures:** one `LlamaModel` with a `Mixer` enum (Attention/GQA, GatedAttention,
  GatedDeltaNet SSM, ShortConv) covering `llama/qwen2/qwen3/qwen35/lfm2`. **No MoE forward pass.**
- **Decode only:** one `forward(token, pos)` per token; prefill is a sequential loop (no batching).
- **f32 everywhere**, including a naive `Vec<Vec<Vec<f32>>>` KV cache. The `feat/f16-activations`
  branch has no commits yet — that work is Phase 1.
- **Quant:** dequant for F32/F16/BF16/Q8_0/Q5_0/Q2_K/Q4_K/Q5_K/Q6_K; GPU-fused for
  Q8_0/Q4_K/Q5_K/Q6_K. No I-quants, MXFP4, Q3_K, or legacy Q4_0/Q4_1/Q5_1.
- **Tokenizer:** GPT-2 byte-level BPE only (hard-rejects SPM/Unigram). Chat template is a hardcoded
  ChatML string; GGUF `chat_template` is ignored.
- **Sampling:** temperature / top-k / top-p / greedy. No min-p, penalties, mirostat, grammar.
- **RoPE scaling (YaRN):** parsed into config but **not applied** — a latent correctness bug for
  long-context models.
- **Serving:** none (CLI bins: `generate`, `gguf-info`, `compare_backends`, `verify_gpu`).
- **Tests:** quant round-trips + unit tests. No golden-vs-llama.cpp harness, no bench harness, no CI.
- **Baseline perf:** ~9.5 tok/s decode on Qwen3.5-9B Q4_K_M (see `docs/PERFORMANCE_RECOMMENDATIONS.md`).

Assets worth building on: solid GGUF parser + multi-file mmap; a genuinely good GPU-resident path
with fused kernels; the honest, benchmark-backed `PERFORMANCE_RECOMMENDATIONS.md`; the `Mixer`
abstraction (ahead of baseline llama.cpp for hybrid/SSM models); 4 real models in `models/`.

## Phases

| Phase | Title | Status | Doc |
|---|---|---|---|
| 0 | Correctness harness + bench + YaRN fix | **committed** | [phase-0](phase-0-correctness-harness.md) |
| 1 | Batched prefill + f16 activations + quantized KV | **committed** | [phase-1](phase-1-batched-f16-kv.md) |
| 2 | GPU performance to competitive | **committed** | [phase-2](phase-2-gpu-performance.md) |
| 3 | Model + quant + tokenizer breadth (MoE) | gated on Phase 2 | [phase-3](phase-3-breadth.md) |
| 4 | Sampling + constrained decoding | gated | [phase-4](phase-4-sampling-grammar.md) |
| 5 | OpenAI-compatible server | gated | [phase-5](phase-5-server.md) |
| 6 | Differentiators (spec-decode, TurboQuant KV) | gated | [phase-6](phase-6-differentiators.md) |

**Committed** = build now, in order. **Gated** = we build these only after Phase 2 lands and we
decide the performance is good enough to justify continuing. The gated docs are intentionally
stubs; flesh them out when their gate opens.

### Sequencing rationale

Ordered by **leverage × unblocking**:

- **Phase 0** makes everything after it *safe and measurable*. Without a golden regression gate and
  a bench, every later change can silently break correctness or speed. It also fixes the YaRN bug.
- **Phase 1** is the core engine refactor (batched forward + f16 + quantized KV). It's the single
  biggest bandwidth/latency win *and* it unblocks flash attention, continuous batching, and
  speculative decoding. Nothing in Phases 2–6 is fully realizable without it.
- **Phase 2** turns the refactor into competitive tok/s and picks the primary GPU backend.
- **Phases 3–6** are the payoff: the models and the differentiators — but only if the box is fast.

## The correctness rule (applies to every phase)

At temperature 0 (greedy), output must match llama.cpp **token-for-token** on the same model,
prompt, and context length — on **both** CPU and GPU backends. Any divergence is a bug to fix
before moving on. This is what the Phase 0 golden harness enforces.

## How to use these docs

Each phase doc is written to be handed to an executing agent standalone. It follows one template:

**Goal · Why · Prerequisites · Tasks (checklist with file paths) · Design notes & gotchas ·
Verification · Done-criteria.**

Read this master doc first for context, then the specific phase doc. When in doubt about current
behavior, trust the code and `docs/PERFORMANCE_RECOMMENDATIONS.md` over the older `docs/0x` files.
