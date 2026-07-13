# Benchmark Progress

## Phase 1: Per-Kernel Timer Infrastructure — DONE ✅

### Created files
- **`src/ops/trace.rs`** — Timer module with `GGUF_TRACE_KERNEL` env var support
  - `enabled()` — checks if tracing is enabled
  - `Timer::new(name)` — creates a timer that prints elapsed time on drop
  - Only prints if tracing is enabled (cached once at startup)

### Instrumented functions (all in `src/ops/wgpu_backend/resident.rs`)
Every `launch_*` method in the resident path now has a `let _timer = Timer::new("...");` at the start.

**Dequant matmul:**
- `launch_only` — single-token dequant matmul
- `launch_only_f32out` — f32-output variant (LM head)
- `launch_only_batch` — batched dequant matmul (prefill)
- `launch_coop_q8_0_batch` — cooperative batched Q8_0 (prefill, in_dim <= 4096)

**FFN chain:**
- `ffn_chain_from_handle` — fused gate→up→silu_mul→down

**Elementwise ops:**
- `launch_silu_mul` — gate * silu(up)
- `launch_sigmoid_mul` — a * sigmoid(b)

**Norm:**
- `launch_rms_norm` — per-token RMSNorm
- `launch_add_residual_rms_norm` — fused residual+RMSNorm

**Position embedding:**
- `launch_rope` — RoPE rotation
- `launch_qk_norm_rope` — fused QK-norm+RoPE

**Attention:**
- `launch_attention` — full 3-kernel attention (scores→softmax→output)
- `launch_kv_cache_write` — append K/V to cache

**QKV projection:**
- `launch_qkv` — 3-way batched QKV

**Deinterleave:**
- `launch_split_qg` — Q/gate deinterleave (Qwen3.5)

**LFM2 mixer:**
- `launch_short_conv` — depthwise conv1d + gate

**GDN (Qwen3.5):**
- `launch_gdn_gate_decay` — beta/decay gate
- `launch_causal_conv1d_silu` — causal conv1d + SiLU
- `launch_l2_norm_heads` — L2 normalize Q/K heads
- `launch_gdn_recurrence` — delta-rule recurrence
- `launch_gdn_gated_norm` — output gated-RMSNorm

### Visibility changes (needed for Phase 2)
- `WgpuBackend::client` — made `pub`
- `WgpuBackend::resident` — made `pub mod`
- All `launch_*` methods in `resident.rs` — made `pub`
- `GpuWeightHandle::handle_ref()` — new public method to access underlying GPU handle
- `ffn_chain_from_handle` — made `pub`
- `WgpuBackend::import_f32()` — new public method to create GPU handles from f32 slices
- CPU `matmul_dequant` — also instrumented with timer

### Usage
```bash
# Enable per-kernel timing
GGUF_TRACE_KERNEL=1 cargo run --release --features wgpu --bin generate -- models/Qwen3.5-9B-Q4_K_M.gguf --backend wgpu --prompt "Hello world"

# Without timing
cargo run --release --features wgpu --bin generate -- models/Qwen3.5-9B-Q4_K_M.gguf --backend wgpu --prompt "Hello world"
```

## Phase 2: Dedicated Benchmark Binary — DONE ✅

### Created file
- **`src/bin/kernels_bench.rs`** — Parameter-sweep benchmark binary

### Usage
```bash
# Run all kernels
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel all

# Run only matmul kernels
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul

# Run with custom dimensions
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul --dims 4096,8192

# Run with custom batch sizes
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel matmul --batch 1,16,64

# Run rms_norm
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel rms_norm

# Run attention
cargo run --release --features wgpu --bin kernels_bench -- models/Qwen3.5-9B-Q4_K_M.gguf --kernel attention
```

### Supported kernels
matmul, rms_norm, rope, attention, qkv, ffn_chain, add_residual, silu_mul, sigmoid_mul, split_qg, kv_cache_write, short_conv, gdn_gate_decay, gdn_recurrence, gdn_gated_norm, causal_conv1d, l2_norm_heads

## Phase 3: Full Forward Pass with Breakdown — TODO

## Phase 4: Memory Bandwidth Benchmarks — TODO

## Phase 5: Dispatch Overhead Benchmarks — TODO
