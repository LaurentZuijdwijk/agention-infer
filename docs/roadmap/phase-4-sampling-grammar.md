# Phase 4 — Sampling + constrained decoding (STUB)

**Status:** gated on Phase 2. Stub — expand when the gate opens.

## Goal

Bring sampling and structured output up to the level users expect from llama.cpp.

## Rough task list

- **Samplers:** min-p, repetition / frequency / presence penalties, mirostat v1/v2, tail-free,
  typical-p, DRY, XTC, logit bias. Refactor `SamplingConfig` / `Sampler` (`src/sampler.rs`) into a
  composable sampler chain (llama.cpp `llama_sampler` shape) rather than a fixed temp→topk→topp path.
- **Grammar / structured output:** GBNF grammar + JSON-schema-constrained decoding (mask logits to
  grammar-allowed tokens). New `src/grammar.rs`.

## Done-criteria (draft)

Unit tests per sampler; a JSON-schema-constrained prompt always emits schema-valid JSON.
