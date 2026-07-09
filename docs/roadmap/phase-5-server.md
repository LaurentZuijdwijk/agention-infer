# Phase 5 — OpenAI-compatible server (STUB)

**Status:** gated on Phase 2 (and benefits from Phase 1 batching, Phase 4 sampling). Stub — expand
when the gate opens.

## Goal

Usability parity with `llama-server`: an OpenAI-compatible HTTP API so the engine is a daily driver.

## Rough task list

- **HTTP server** (`axum` + `tokio`, add to `Cargo.toml`): `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models`; **SSE streaming**; server-side chat-template rendering (Phase 3);
  sampling + grammar params (Phase 4) wired through. New `src/bin/server.rs`.
- **Continuous batching / slots:** concurrent requests sharing the Phase 1 batched forward with
  paged/slotted KV — where batching finally pays off for throughput.
- **Multi-model:** load several models into the unified pool, route by request `model` field; optional
  cascading (try small first, escalate). New `src/multi_model.rs`. See `docs/09-speedup-multimodel.md`.

## Done-criteria (draft)

`curl` streaming + non-streaming returns correct output; concurrent-request load test passes; an
unmodified OpenAI SDK client works.
