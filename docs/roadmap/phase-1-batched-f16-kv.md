# Phase 1 — Batched prefill + f16 activations + quantized KV

**Status:** committed. Depends on **Phase 0** (golden + bench must exist first).
See [00-parity-roadmap.md](00-parity-roadmap.md).

## Goal

The core engine refactor. Three coupled changes:

1. **Batched forward** — process N tokens in one pass instead of one `forward()` per token.
2. **f16 activations** — halve activation bandwidth (f32 accumulation preserved inside kernels).
3. **Quantized KV cache** — f16 / q8_0 / q4 KV storage (llama.cpp `-ctk/-ctv` parity), replacing the
   naive `Vec<Vec<Vec<f32>>>`.

This is the `feat/f16-activations` branch's actual intent, plus the batching and KV work that share
the same buffers.

## Why

On a bandwidth-bound APU this is the single biggest win *and* the biggest unblocker:

- Prefill today re-reads **every weight tensor once per prompt token** (`bin/generate.rs:171`;
  `PERFORMANCE_RECOMMENDATIONS.md` Priority 13). Batching collapses that to one weight read per layer.
- f16 activations + quantized KV cut per-token bytes moved — directly the bound resource.
- Batched forward is the prerequisite for **flash attention** (Phase 2), **continuous batching**
  (Phase 5), and **speculative verification** (Phase 6). Without it, those are not realizable.

## Prerequisites

- Phase 0 golden harness (to prove the refactor doesn't change temp=0 tokens) and bench (to show the win).
- Familiarity with the two forward paths: `src/model/llama/cpu_path.rs` (`run`, one op at a time) and
  `src/model/llama/gpu_resident.rs` (`run_gpu_resident`, residual stream stays on-GPU).

## Tasks

### 1.1 Batched forward
- [ ] Extend the `Model` trait (`src/model/mod.rs:382`): add
      `forward_batch(&mut self, tokens: &[u32], pos_start: usize, kv: &mut KvCache) -> Result<Vec<f32>>`
      returning logits for the **last** position (prefill only needs the last), while writing K/V for
      all positions. Keep single-token `forward` as `forward_batch(&[t], pos, kv)`.
- [ ] `KvCache::write` → support writing a range of positions (`write_range(layer, pos_start, k, v)`);
      current call sites are `cpu_path.rs:491`, `cpu_path.rs:684`, and the GPU
      `launch_kv_cache_write` at `gpu_resident.rs:383,447`.
- [ ] CPU path: batch the QKV / gate-up / down / o-proj matmuls to `[N, d] × [d, ·]` (matrix-matrix,
      not N matrix-vector calls). `WeightMap::matmul_into` (`model/mod.rs:334`) currently does
      one x-vector; add a batched variant that shares the dequantized/packed weight across the N rows.
- [ ] GPU-resident path: make `x_handle` hold `[N, d]`; batch the projection dispatches; write N KV
      positions in one dispatch. Attention over the batch is the naive path for now (flash comes in Phase 2).
- [ ] `bin/generate.rs`: replace the per-token prefill loop (line ~171) with a single
      `forward_batch(&prompt_ids, 0, &mut kv)`; keep the decode loop as single-token `forward`.

### 1.2 f16 activations
- [ ] Introduce an activation dtype (`half::f16`) for `InferenceState` buffers (`state.rs:64` — all
      the `Vec<f32>` fields: `x`, `xn`, `q`, `k`, `v`, `attn_out`, `proj`, `gate`, `up`, `ffn_act`,
      `scores`, the GDN/ShortConv scratch, etc.) and for the GPU `Handle`s.
- [ ] Generalize the WGSL kernels in `ops/kernels_wgpu.rs` from `Array<f32>` to the activation dtype
      where they carry activations, **keeping f32 accumulators** inside dot products / reductions.
      The matmul weight-scale reads (`read_f16`) already handle f16.
- [ ] Keep logits and the LM head output in f32 (sampling precision).
- [ ] Guard the GDN recurrent state (`gpu_gdn_recurrent_state`) precision — it accumulates across
      tokens; consider keeping *that* state f32 even if other activations are f16 (see gotchas).

### 1.3 Quantized KV cache
- [ ] Replace `KvCache` (`model/mod.rs:401-445`, `Vec<Vec<Vec<f32>>>`) with a packed store
      parameterized by KV dtype: **f16** (default), **q8_0**, **q4**. Write quantizes, read
      dequantizes. Mirror the GPU-resident persistent KV handles.
- [ ] Add a `--kv-type` CLI flag to `generate` (and later the server) — llama.cpp `-ctk/-ctv` parity.
- [ ] Update the memory-budget calculators (`model/mod.rs:133-158`) to report max context per KV dtype.
- [ ] Design the KV store so Phase 6 **TurboQuant** (3-bit WHT + Lloyd-Max) slots in as another
      KV dtype without another rewrite — i.e. a `KvQuant` trait or enum with pack/unpack.

## Design notes & gotchas

- **Prove tokens are unchanged first.** Land batched-forward as a pure refactor (still f32) and get
  golden green before flipping activations to f16. Then flip f16 and re-run golden — token match must
  hold; only logits move within epsilon.
- **f16 drift is the real risk.** The GDN delta-rule recurrence accumulates state across *many*
  tokens, so a single-token parity check won't catch drift. Add a **200+ token CPU-f32 vs GPU-f16
  drift test** (`PERFORMANCE_RECOMMENDATIONS.md` Priority 15 caveat). If drift shows, keep GDN state
  (and possibly RMSNorm accumulators) in f32.
- **Don't collapse parallelism.** When batching matmuls, keep the many-small-workgroups shape that
  won on this GPU — fewer/bigger workgroups regressed 9.5→6.0 tok/s
  (`PERFORMANCE_RECOMMENDATIONS.md` "What we learned about this GPU's dispatch cost"). Batched prefill
  adds *more* total work per dispatch, which is fine; just don't reduce workgroup count for decode.
- **Attention during batched prefill** stays the existing naive kernel here; the flash-attention
  rewrite is Phase 2 and depends on this batched shape existing.

## Verification

```
cargo test --features golden,wgpu           # token-for-token match preserved after refactor + f16
cargo run --release --features wgpu --bin bench -- models/Qwen3.5-9B-Q4_K_M.gguf   # prefill tok/s ↑, decode mem ↓
cargo run --release --features wgpu --bin <drift-test> -- models/Qwen3.5-9B-Q4_K_M.gguf   # 200-token CPU/GPU drift clean
cargo run --release --features wgpu --bin compare_backends -- models/<m>.gguf       # CPU↔GPU parity
```

## Done-criteria

- Golden tokens unchanged on all 4 models (CPU + GPU) with f16 activations on.
- Prefill throughput improved multiple× vs the Phase 0 baseline (single biggest expected metric).
- KV cache supports f16/q8/q4 with a `--kv-type` flag; budget report updated.
- Long-generation drift test clean; KV store designed to accept TurboQuant later.
