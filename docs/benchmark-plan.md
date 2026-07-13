# GPU Micro-Benchmark Plan

## Context

The library (`gguf-rs`) is a GGUF inference engine for AMD unified-memory APUs (Strix Halo), using CubeCL/WGPU for GPU compute. The existing `bench` binary only measures end-to-end throughput (tok/s) for prefill and decode. We need **per-kernel** micro-benchmarks to find bottlenecks and measure optimization impact.

## Existing Benchmarks (What's Already Covered)

| Binary | What it measures | Gap |
|--------|-----------------|-----|
| `bench` | End-to-end tok/s, % bandwidth ceiling | No per-kernel timing |
| `matmul_batch_bench` | Batched vs single-token matmul for one tensor | Only one tensor, no other kernels |
| `compare_backends` | CPU vs GPU correctness (first token) | No timing |

## Benchmark Plan

### Tier 1: Kernel-Level Timings (Immediate)

Each kernel gets a dedicated micro-benchmark that measures **isolated** wall time for the GPU resident path (no CPU round-trips between kernels).

#### 1.1 Dequant+Matmul (per quantization format)

- **What**: Time the `matmul_dequant_wgpu` kernel for each dtype (Q8_0, Q4_K, Q5_K, Q6_K)
- **Parameters to sweep**:
  - `in_dim`: 4096, 8192, 16384, 3072, 4096 (covering common projection sizes)
  - `out_dim`: 4096, 8192, 16384, 3072, 4096
  - Batch size: 1 (decode), 16, 32, 64, 128 (prefill)
- **Output**: ms per matmul, GB/s effective (bytes read + bytes written / time)
- **Why**: This is the dominant op (80%+ of compute). We need to know if the workgroup reduction design is efficient across all dtypes.

#### 1.2 RMSNorm

- **What**: Time `launch_rms_norm` for various lengths
- **Parameters**: `len` = 4096, 8192, 16384, 3072, embedding_length of the model
- **Output**: ms per call
- **Why**: Currently single-threaded on purpose (per-kernel launch cost dominates). Verify this assumption.

#### 1.3 RoPE

- **What**: Time `launch_rope` for various head counts and dimensions
- **Parameters**: `n_heads` = 8, 16, 32; `head_dim` = 64, 128, 256; `n_rot` = 64, 128, 256
- **Output**: ms per call
- **Why**: Small kernel but called twice per layer per token. Verify workgroup sizing is appropriate.

#### 1.4 Attention (3-kernel split)

- **What**: Time `attention_scores`, `attention_softmax`, `attention_output` separately
- **Parameters**:
  - `pos` = 16, 64, 256, 1024, 4096 (sequence length)
  - `n_heads` = 8, 16, 32
  - `head_dim` = 64, 128
- **Output**: ms per kernel, total attention time
- **Why**: The split design is critical — we need to know if the 3-launch approach is better than a single kernel, and if the workgroup sizing is optimal for different sequence lengths.

#### 1.5 QKV Projection (3-way batched)

- **What**: Time `matmul_dequant_qkv` — 3 matmuls with one GPU round-trip
- **Parameters**: `in_dim` = 4096, 8192; `out_dim` = 4096, 8192
- **Output**: ms per 3-way batched call, vs 3× separate calls
- **Why**: The batching saves CPU round-trips. Verify the savings are real.

#### 1.6 FFN Chain (4-way chained)

- **What**: Time `ffn_chain_from_handle` — gate+up+silu_mul+down as one chain
- **Parameters**: `ffn_dim` = 8192, 16384, 28672 (covering 4K→12K, 4K→32K, 4K→4096)
- **Output**: ms per chain, vs 4 separate calls
- **Why**: The fused chain avoids CPU round-trips. Measure the savings.

#### 1.7 Add+Residual+RMSNorm (fused)

- **What**: Time `launch_add_residual_rms_norm`
- **Parameters**: `len` = embedding_length (4096, 8192, 3072)
- **Output**: ms per call
- **Why**: Fused version vs separate add + rms_norm.

#### 1.8 Gated DeltaNet (Qwen3.5)

- **What**: Time the individual GDN kernels: `gdn_gate_decay`, `causal_conv1d_silu`, `l2_norm_heads`, `gdn_recurrence`, `gdn_gated_rms_norm`
- **Parameters**: `conv_dim`, `n_k_heads`, `head_k_dim`, `head_v_dim`, `kernel` (from model config)
- **Output**: ms per kernel
- **Why**: These are novel ops. We need to know if the GPU implementation is efficient.

### Tier 2: Dispatch Overhead (Critical for Small Ops)

#### 2.1 Per-Dipatch Overhead

- **What**: Time a dummy launch (0 elements) to measure the per-dispatch overhead
- **Parameters**: 1, 2, 4, 8, 16, 32, 64 launches in sequence
- **Output**: ms per dispatch, amortized dispatch cost
- **Why**: The resident path chains many small ops. If dispatch overhead is 0.5ms, then 10 ops = 5ms wasted.

#### 2.2 GPU Round-Trip Cost

- **What**: Time `client.read` (blocking readback) vs `read_one_unchecked`
- **Parameters**: read sizes = 1KB, 16KB, 64KB, 256KB, 1MB
- **Output**: ms per read
- **Why**: Every CPU round-trip kills residency. Measure if it's worth keeping ops chained even when the output is small.

### Tier 3: Full Forward Pass Decomposition

#### 3.1 Per-Layer Breakdown

- **What**: Time each layer's complete forward pass (all kernels, fully chained)
- **Parameters**: Each layer in the model (different layers may have different mixers)
- **Output**: ms per layer, breakdown by mixer type
- **Why**: Identify which layers are slowest and why.

#### 3.2 Full Forward Pass (GPU Resident vs CPU)

- **What**: End-to-end timing with per-kernel breakdown
- **Parameters**: Small prompt (8 tokens), medium prompt (64 tokens), long prompt (256 tokens)
- **Output**: ms per kernel, total time, % in each kernel category
- **Why**: See the full picture — which kernels are the bottlenecks in practice.

### Tier 4: Memory Bandwidth

#### 4.1 Weight Read Bandwidth

- **What**: Measure effective bandwidth when reading weight tensors of various sizes
- **Parameters**: tensor sizes = 1MB, 8MB, 64MB, 256MB (covering small to large projections)
- **Output**: GB/s, % of memory bandwidth ceiling
- **Why**: Quantize the memory-bound claim. Are we actually bandwidth-limited?

#### 4.2 Activation Bandwidth

- **What**: Measure bandwidth for activation buffers (f32 vs f16)
- **Parameters**: buffer sizes = 16KB, 64KB, 256KB, 1MB
- **Output**: GB/s, f32 vs f16 comparison
- **Why**: The `Act = f16` design halves activation bandwidth. Verify the gain.

### Tier 5: Pre-Upload vs Ad-Hoc

#### 5.1 Upload Cost

- **What**: Time `upload_weight` for various tensor sizes
- **Parameters**: 1MB, 8MB, 64MB, 256MB
- **Output**: GB/s upload speed, total upload time
- **Why**: Pre-upload is one-time; ad-hoc is per-token. Quantify the difference.

#### 5.2 Warmup vs No Warmup

- **What**: Time the first dispatch with and without kernel warmup
- **Output**: First-call penalty
- **Why**: The ~7s shader compile is visible to the user. Quantify it.

## Implementation Plan

### Phase 1: Per-Kernel Timers (1-2 days)

Add `GGUF_TRACE_KERNEL=1` environment variable that instruments each `launch_*` call in `resident.rs` with start/end timestamps. This is the minimum viable change to get per-kernel data from the resident path.

### Phase 2: Dedicated Benchmark Binaries (2-3 days)

Create `src/bin/kernels_bench.rs` that directly calls each kernel with controlled parameters and reports results. This is the most flexible approach — we can sweep parameters easily.

### Phase 3: Full Forward Pass with Breakdown (1-2 days)

Add per-kernel timing to the `run_gpu_resident` path, outputting a structured report (JSON or CSV) for later analysis.

### Phase 4: Memory Bandwidth Benchmarks (1-2 days)

Create `src/bin/bandwidth_bench.rs` that uploads/reads buffers of various sizes and measures effective bandwidth.

### Phase 5: Dispatch Overhead Benchmarks (1 day)

Create `src/bin/dispatch_bench.rs` that measures per-dispatch and per-readback overhead.

## Key Metrics to Track

1. **Kernel wall time** (ms) — the primary metric
2. **Effective bandwidth** (GB/s) — for memory-bound ops
3. **Dispatch overhead** (ms) — for small ops
4. **CPU round-trips** (count) — for residency analysis
5. **% of ceiling** — for bandwidth-bound ops
6. **f32 vs f16 activation** — for the `Act` type design
7. **Batched vs sequential** — for prefill vs decode
8. **Chained vs separate** — for fused vs unfused paths

## Expected Outcomes

After running these benchmarks, we should be able to answer:

1. Which kernels are the bottlenecks?
2. Is the resident path actually faster than the CPU-orchestrated path (after accounting for dispatch overhead)?
3. Is the batched matmul for prefill actually amortizing weight reads?
4. Are small ops (RMSNorm, RoPE) dominated by dispatch overhead?
5. Is the attention 3-kernel split optimal?
6. How much does f16 activations save on activation bandwidth?
7. What is the per-dispatch overhead on this hardware?

## Risk Assessment

- **CubeCL kernel warmup**: The first dispatch of each kernel type is slow (~7s shader compile). Must warm up before measuring.
- **GPU clock scaling**: On an APU, GPU clocks vary with load. Need many iterations to get stable numbers.
- **PCIe vs unified memory**: On Strix Halo, GPU and CPU share memory — upload is PCIe-bound but readback is also shared. Need to measure both paths.
- **AMD specific**: Vulkan/WGPU on AMD may have different performance characteristics than NVIDIA. Benchmarks are hardware-specific.
