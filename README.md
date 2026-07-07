# gguf-rs Documentation

An inference engine for GGUF language models, built for AMD unified memory APUs.

## Documents

| # | File | Contents |
|---|---|---|
| 01 | [overview.md](01-overview.md) | Project goals, non-goals, target hardware, target models |
| 02 | [architecture.md](02-architecture.md) | Crate layout, module diagram, Backend trait, data flow |
| 03 | [gguf-format.md](03-gguf-format.md) | Binary layout, metadata types, tensor naming, multi-file |
| 04 | [quantization.md](04-quantization.md) | K-quants, I-quants, novel formats, dequant implementation |
| 05 | [inference-pipeline.md](05-inference-pipeline.md) | 6 core ops, forward pass, attention, MoE, decode loop, sampler |
| 06 | [rocm-hardware.md](06-rocm-hardware.md) | Strix Halo specs, ROCm setup, HIP kernels, APU optimization |
| 07 | [turbo-quant-kv-cache.md](07-turbo-quant-kv-cache.md) | TurboQuant, WHT, Lloyd-Max, memory budget math |
| 08 | [moe-routing.md](08-moe-routing.md) | MoE architectures, tracing, visualization, expert analysis |
| 09 | [speedup-multimodel.md](09-speedup-multimodel.md) | Speculative decoding, prefix caching, routing, Self-MoA |
| 10 | [build-roadmap.md](10-build-roadmap.md) | Phase-by-phase checklist, milestones, test strategy |
| 11 | [testing.md](11-testing.md) | Test tiers, unit/integration/golden tests, CI, debug assertions |

## Quick Reference

### Bandwidth Ceiling (Strix Halo, 256 GB/s)

```
Model                  Quant      Size     Ceiling
Llama 3.2 1B           Q8_0       1.3 GB   197 tok/s
Qwen2.5 7B             Q4_K_M     4.1 GB   62 tok/s
MiniMax M2.7           IQ3_XXS    93 GB    ~28 tok/s*
```
*MoE: only ~10B params active per token

### Memory Budget (96 GB)

```
M2.7 IQ3_XXS (93 GB) + 3-bit KV @ 64K context (3.8 GB) = 96.8 GB  ← fits
M2.7 IQ3_XXS (93 GB) + 3-bit KV @ 32K context (1.9 GB) = 94.9 GB  ← fits comfortably
7B Q4_K_M   (4.1 GB) + 3-bit KV @ 200K context (2.5 GB) = 6.6 GB  ← trivial
```

### Build Phase Summary

```
Phase 1:  gguf-info CLI                    → inspect any GGUF file
Phase 2:  math primitives (CPU)            → correct dequant + ops
Phase 3:  tokenizer                        → encode/decode text
Phase 4:  single forward pass              → first token generation
Phase 5:  KV cache + decode loop           → full response generation
Phase 6:  MoE support                      → Mixtral, M2.7
Phase 7:  TurboQuant KV cache             → long context on M2.7
Phase 8:  ROCm backend                    → GPU acceleration
Phase 9:  speculative decoding            → 2-4× throughput
Phase 10: MoE routing visualization       → expert analysis tool
Phase 11: multi-model server              → OpenAI-compatible API
```

### Key Design Decisions

- **Parsing is pure**: `parse(&[u8])` takes bytes, returns data. No I/O inside.
- **Zero copy**: tensors are pointers into mmap, never copied.
- **Backend trait**: model code never knows if it's on CPU or GPU.
- **TurboQuant default**: 3-bit KV cache is the default, not a flag.
- **Typed errors in lib, anyhow in bins**: library errors are matchable.
- **CPU backend always present**: correctness reference, never removed.
