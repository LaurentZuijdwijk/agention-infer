# Performance baselines

Recorded by `src/bin/bench.rs` so later phases can show deltas. Reproduce with:

```
cargo run --release --features wgpu --bin bench -- models/<model>.gguf --backend <cpu|wgpu>
```

`ceiling_tok_s = 260 GB/s / on-disk-size` (dense proxy; for hybrid/MoE models active bytes/token is
lower than the file, so % ceiling is a rough lower bound). Decode = 64 greedy tokens, EOS ignored.

## Phase 0 baseline — 2026-07-09

Box: Ryzen AI Max+ 395 / Strix Halo, Radeon 8060S (RADV, Vulkan), 128 GB, ~260 GB/s.
Commit: start of Phase 0 (pre-refactor). Build: `--release`.

| Model | Quant | Backend | Size (GiB) | Prefill tok/s | Decode tok/s | % ceiling |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | Q4_K_M | wgpu | 5.29 | 9.5 | **9.60** | 21.0 |
| Qwen3-1.7B | Q8_0 | wgpu | 1.71 | 15.4 | 15.35 | 10.8 |
| LFM2.5-1.2B | Q4_K_M | wgpu | 0.68 | 49.0 | 49.12 | 13.8 |
| LFM2.5-1.2B | Q6_K | wgpu | 0.90 | 51.2 | 51.04 | 18.9 |
| LFM2.5-1.2B | Q4_K_M | cpu | 0.68 | 21.3 | 22.36 | 6.3 |

### Observations (targets for later phases)

- **Prefill ≈ decode tok/s** on every model — prefill runs one `forward()` per prompt token with no
  batching (Phase 1 fixes this; expect prefill to jump multiple×).
- **9.6 tok/s on Qwen3.5-9B** matches `docs/PERFORMANCE_RECOMMENDATIONS.md`. 21% of the (proxy)
  ceiling — Phase 2 targets ≥70%.
- Qwen3.5-9B is a GatedDeltaNet hybrid, so its active bytes/token < file size; its true ceiling is
  higher than the dense proxy shown, i.e. real headroom is larger than 21%.
