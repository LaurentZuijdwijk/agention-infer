# GPU Micro-Benchmark Quick Start

## TL;DR

We need per-kernel micro-benchmarks to find bottlenecks in the GPU-resident forward pass. The existing `bench` binary only gives end-to-end tok/s.

## Phase 1: Per-Kernel Timing — DONE ✅

### 1. Per-Kernel Timing (Fastest to implement)

`GGUF_TRACE_KERNEL=1` is now available to instrument each `launch_*` call in `resident.rs`. Run a small model and see how many milliseconds each kernel takes.

```bash
# CPU resident path (no GPU)
cargo run --release --features wgpu --bin generate -- models/Qwen3-1.7B-Q8_0.gguf --backend cpu --prompt "Hello world"

# GPU resident path with per-kernel timing
GGUF_TRACE_KERNEL=1 cargo run --release --features wgpu --bin generate -- models/Qwen3-1.7B-Q8_0.gguf --backend wgpu --prompt "Hello world"
```

### 2. Dedicated Benchmark Binary (Phase 2 — DONE ✅)

The `kernels_bench` binary provides parameter sweeps for individual kernels:

```bash
# Run all kernels
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel all

# Run only matmul kernels
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul

# Run with custom dimensions
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul --dims 4096,8192

# Run rms_norm
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel rms_norm
```

## Phase 3: Full Forward Pass with Breakdown — TODO

## Phase 4: Memory Bandwidth Benchmarks — TODO

## Phase 5: Dispatch Overhead Benchmarks — TODO

## Full Plan

See [benchmark-plan.md](benchmark-plan.md) for the complete 5-tier plan with all parameters and expected outcomes.

## Progress

See [benchmark-progress.md](benchmark-progress.md) for the current state of each phase.

## Key Questions This Will Answer

1. Is the resident path faster than CPU-orchestrated (after dispatch overhead)?
2. Are small ops (RMSNorm, RoPE) dominated by dispatch overhead?
3. Is the batched matmul for prefill actually amortizing?
4. Are the attention 3-kernel split optimal?
5. How much does f16 activations save on activation bandwidth?
