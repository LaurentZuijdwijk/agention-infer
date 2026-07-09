# Phase 6 — Differentiators (STUB)

**Status:** gated on Phase 2. This is the *payoff* phase — the features llama.cpp is slow to
mainline, which are the reason to build this engine at all. Stub — expand when the gate opens.

## Goal

Beat llama.cpp on *this box* on capability, not just match it: low-bit KV for long context on huge
MoE, and higher effective decode throughput.

## Rough task list

- **Speculative decoding:** `DraftSource` trait (`docs/09-speedup-multimodel.md`) — small-model draft,
  self-speculative (layer-skip), MTP heads where present. Parallel verify via the Phase 1 batched
  forward. New `src/speculative.rs`. Target 2–3× effective decode.
- **TurboQuant 3-bit KV** (`docs/07-turbo-quant-kv-cache.md`): WHT + Lloyd-Max codebook, slotted into
  the Phase 1 quantized-KV substrate as another KV dtype → 128K+ context on big MoE within 128 GB.
  Needle-in-haystack quality gate vs f16 KV.
- **Prefix / prompt caching:** reuse KV across shared prefixes (system prompts, few-shot).
- **MoE expert locality + prefetch** tuning for the 90–120 GB models.

## Done-criteria (draft)

Speculative acceptance-rate + effective tok/s bench; TurboQuant needle test ≥ f16-KV quality at
3-bit; a 128 GB-class MoE runs at long context within budget.
