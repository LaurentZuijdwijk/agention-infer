# gguf-rs — Project Overview

## What This Is

gguf-rs is a Rust inference engine for GGUF-format large language models, purpose-built for unified memory APU hardware — specifically AMD Strix Halo and architecturally similar chips (Apple Silicon, future AMD APUs).

It is not a general-purpose inference engine. It is a focused, high-quality implementation that treats unified memory as a first-class architectural primitive rather than an afterthought, and targets advanced features — TurboQuant KV cache compression, MoE routing visualization, multi-model serving — that existing engines handle poorly or not at all.

---

## The Problem

Existing inference engines were designed around discrete GPU hardware:

- Small VRAM (8–24 GB) + large system RAM
- Expensive PCIe transfer between CPU and GPU memory
- Optimization strategy: minimize transfers, maximize VRAM utilization

This mental model is wrong for unified memory APUs. On Strix Halo:

- Single 96 GB memory pool, accessed by both CPU and GPU at 256 GB/s
- No transfer penalty — GPU reads weights directly from the same memory CPU uses
- Optimization strategy: maximize bandwidth utilization, not transfer avoidance

llama.cpp handles Strix Halo adequately via its ROCm backend. Nobody handles it *well*. The APU-specific optimizations — unified memory allocation, wave32 kernel tuning, expert locality scheduling, TurboQuant KV cache — do not exist in any current engine.

---

## Primary Goals

### 1. Correct

Before fast. The engine produces token-for-token identical output to llama.cpp on the same model, same prompt, temperature=0. Every component is unit tested against a reference implementation.

### 2. APU-Native

Unified memory is not an option or a fallback. It is the design target. Weights are mmap'd directly into unified memory. The GPU reads from the same physical addresses as the CPU. No staging buffers. No explicit transfers.

### 3. Long Context via TurboQuant

TurboQuant (ICLR 2026) compresses the KV cache to 3 bits with near-zero quality loss. This is the difference between M2.7 at 8K context and M2.7 at 128K context on 96 GB of unified memory. It is implemented as the default KV cache format, not an optional flag.

### 4. MoE First-Class

MoE models (Mixtral, MiniMax M2.7, DeepSeek V3, Qwen MoE) are primary targets, not afterthoughts. The engine exposes routing internals — which experts fired, with what weights, across which layers — as structured data that tools can consume.

### 5. Multi-Model Serving

Multiple models loaded simultaneously into the unified memory pool. A small router dispatches queries to the right model. Zero swap cost between models. Cascading (try small first, escalate) and Self-MoA (sample one model multiple times, synthesize) built in.

---

## Non-Goals (v1)

- CUDA support — ROCm and CPU only
- Windows — Linux only initially
- Training — inference only
- Cloud/distributed inference — single machine
- Real-time audio/video — text only
- General hardware portability — that is llama.cpp's job

---

## Target Hardware

**Primary:** AMD Strix Halo (Radeon 890M, RDNA 3.5)
- 96 GB LPDDR5X unified memory
- 256 GB/s bandwidth
- ROCm gfx1151

**Secondary:** AMD Strix Point, future AMD APUs with large unified memory pools

**Tertiary:** CPU-only (any x86-64 with AVX2, any ARM64 with NEON) — always supported as correctness reference and for machines without ROCm

---

## Target Models

| Model | Params | Active | Quant | Size | Context |
|---|---|---|---|---|---|
| Llama 3.2 1B | 1B | 1B | Q8_0 | 1.3 GB | 128K |
| Qwen2.5 7B | 7B | 7B | Q4_K_M | 4.1 GB | 128K |
| Llama 3.1 8B | 8B | 8B | Q4_K_M | 4.7 GB | 128K |
| Qwen2.5-Coder 7B | 7B | 7B | Q4_K_M | 4.1 GB | 128K |
| Qwen2.5 32B | 32B | 32B | Q4_K_M | 19 GB | 128K |
| MiniMax M2.7 | 230B | 10B | IQ4_XS | 108 GB | 204K |

MiniMax M2.7 at IQ3_XXS (~93 GB) + TurboQuant KV cache is the flagship target — a 230B parameter model at full 128K+ context on consumer APU hardware.

---

## Why Rust

- Ownership model prevents accidental copies of giant weight tensors
- Zero-cost abstractions — the `Backend` trait dispatches with no overhead
- `unsafe` is contained and explicit — mmap, SIMD, HIP FFI all isolated
- Best-in-class SIMD via `std::simd` or `wide`
- `rayon` for parallelism across CPU cores
- Strong ecosystem: `memmap2`, `half`, `bytemuck`, `clap`, `axum`
- No GC pauses during token generation

---

## Status

Pre-alpha. Actively designing. No released version.

First milestone: `gguf-info` CLI tool that correctly parses and displays metadata for any GGUF file including MoE models.
