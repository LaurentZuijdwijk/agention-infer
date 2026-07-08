# Performance Optimization Recommendations

This document outlines concrete opportunities to improve inference speed in agention-infer.

---

## Priority 1: GatedDeltaNet GPU-Resident Path ✅ IMPLEMENTED

**Status:** ✅ Implemented and correctness-verified (2026-07-08) — Qwen3.5 hybrid
models now run entirely on GPU. Note: the initial implementation had two bugs
that made it silently produce degenerate repeated-token output (not caught
by "it compiles and runs") — the gated-RMSNorm's result was discarded instead
of applied, and the Q/K L2-norm was ordered before the causal conv instead of
after. Both are fixed; `cargo run --release --features wgpu --bin
compare_backends -- <model.gguf>` and `--bin debug_gdn` now confirm CPU/GPU
logit parity. Run those two after touching this path again.

**Changes made:**
1. Added GPU state buffers to `LlamaModel`:
   - `gpu_gdn_conv_history` - persistent conv history per layer
   - `gpu_gdn_recurrent_state` - persistent state matrix per layer
2. Added GPU weight handles:
   - `gpu_gdn_conv_weights` - conv1d kernel per layer
   - `gpu_gdn_ssm_norm` - per-channel RMSNorm weight
   - `gpu_gdn_ssm_a` - per-head decay multiplier
   - `gpu_gdn_ssm_dt_bias` - per-head softplus bias
3. Updated `pre_upload_gpu()` to upload GatedDeltaNet weights/state
4. Updated `gpu_resident_ready()` to recognize GatedDeltaNet layers
5. Implemented full GatedDeltaNet GPU chain in `run_gpu_resident()`:
   - QKV + gate + beta + alpha projections (batched, one GPU round-trip)
   - L2-norm on Q/K segments in-place
   - Beta/decay gate computation in-place
   - Causal conv1d + SiLU (mutates persistent history)
   - Delta-rule recurrence (mutates persistent state)
   - Gated RMSNorm + ssm_out projection

**Effort:** Medium | **Impact:** High ✅ **COMPLETED**

---

## Priority 2: Shared Memory Optimization for Matmul Kernel ❌ NOT VIABLE AS DESCRIBED

**Status:** ❌ Investigated 2026-07-08, not implemented — the premise doesn't
hold for this kernel.

`matmul_dequant_wgpu`'s `partial_*` functions stride each thread's block
index by 64 (`let mut b = lane; while b < n_blocks { ...; b += 64 }`), so
within one workgroup (one output row) the 64 lanes read **disjoint**
elements of `x` — every element is read exactly once across the workgroup,
not once per thread. There's no redundant per-lane read to eliminate here.

The actual redundancy, if any, is that `x` is re-fetched from global memory
once per **output row** (the matmul dispatches one workgroup per row, and
each independently re-reads all of `x`) — fixing that would mean caching `x`
across the whole grid, which WGSL workgroups can't share (no cross-workgroup
shared memory), so it'd require a separate pass or relying on the GPU's L2
cache, not the `SharedMemory` scratchpad this proposal used.

The proposed code also has a bug independent of the above: `if lane < in_dim
% 64 { x_smem[lane] = x[lane] }` only loads up to 63 elements — `in_dim % 64`
is 0 for any `in_dim` that's a multiple of 64 (the common case), which would
leave `x_smem` entirely uninitialized.

---

## Priority 3: Flash Attention-Style Tiling (CPU Path) ❌ ARCHITECTURE MISMATCH

**Status:** ❌ Investigated 2026-07-08, not implemented — doesn't fit how this
codebase runs attention.

`generate.rs` calls `model.forward()` once per token, including during
prefill (there is no batched multi-query prefill pass — every prompt token
gets its own forward call). So a single `attention_into` call is already
O(n_heads × pos × head_dim), linear in `pos`, not quadratic — the O(pos²)
term the original writeup below cites only shows up cumulatively across all
generated tokens, which is inherent to causal attention and isn't something
K/V tiling changes (tiling saves memory bandwidth from materializing an N×N
score matrix for *batched* queries; there's no such matrix here since it's
one query at a time).

If this codebase later adds batched prefill (processing all prompt tokens in
one multi-query attention pass instead of one `forward()` call per token),
tiling would become relevant there and should be revisited — that's also
likely a bigger win on its own than the tiling itself, since it would cut
prefill from N sequential single-token forward passes to O(1) attention
calls with a batched matmul.

**Original issue description (see architecture note above):**

**Current code** (`cpu_path.rs`):
```rust
fn attention_into(&self, q, k_all, v_all, out, scores) {
    for h in 0..n_heads {
        for t in 0..seq_len {  // For each position
            let dot: f32 = q_head.iter().zip(k_head).map(|(a,b)| a*b).sum(); // O(head_dim)
        }
        for t in 0..seq_len {  // Again over all positions
            for d in 0..head_dim {
                out[h*head_dim + d] += scores[t] * v_head[d];
            }
        }
    }
}
```

**Proposed improvement:** Tile-based attention that processes K/V in blocks:

```rust
fn attention_into_tiled(&self, q, k_all, v_all, out, scores) {
    const TILE_SIZE: usize = 64;  // Cache-friendly tile size
    
    for h in 0..n_heads {
        // Process K/V in tiles
        let mut acc = vec![0.0f32; head_dim];
        
        for tile_start in (0..seq_len).step_by(TILE_SIZE) {
            let tile_end = (tile_start + TILE_SIZE).min(seq_len);
            
            // Load tile of K/V into cache
            let k_tile = load_tile(k_all, tile_start, tile_end);
            let v_tile = load_tile(v_all, tile_start, tile_end);
            
            // Compute partial scores for this tile
            let mut tile_scores = vec![0.0f32; tile_end - tile_start];
            for t in 0..tile_end - tile_start {
                tile_scores[t] = dot_product(&q_head, &k_tile[t]);
            }
            
            // Softmax partial (will need online normalization)
            softmax_partial(&tile_scores);
            
            // Weighted sum for this tile
            for d in 0..head_dim {
                for t in 0..tile_end - tile_start {
                    acc[d] += tile_scores[t] * v_tile[t * head_dim + d];
                }
            }
        }
        
        out[h*head_dim..(h+1)*head_dim].copy_from_slice(&acc);
    }
}
```

**Effort:** High | **Impact:** High | **Estimated speedup:** 2-4x on long contexts

---

## Priority 4: RoPE Frequency Precomputation ✅ IMPLEMENTED

**Status:** ✅ Implemented 2026-07-08, but differently than originally proposed
below. `freq`/`sin`/`cos` depend only on `(pos, d)`, not the head — the real
redundancy was recomputing `theta.powf(...)` and `sin_cos()` once per
`(head, d)` pair when it only needs computing once per `d` and reused across
all `n_heads`/`n_kv_heads` heads. `rope()` and `rope_partial()` in
`cpu_path.rs` now hoist a small `Vec<(f32, f32)>` table of size `head_dim/2`
(or `n_rot/2`) out of the head loop and index into it per head, instead of
recomputing per head.

This gets the same "stop recomputing trig redundantly" win as the original
proposal below without its downsides: no `max_seq_len`-sized table (the
original's `rope_cos`/`rope_sin` fields are `O(max_seq_len × head_dim)`, ~2MB
per the estimate below, for tables where each position is actually visited
exactly once during generation — precomputing them ahead of time doesn't
save anything past the first access), no extra `LlamaModel` fields, and no
seq-len cap to maintain.

**Original issue description:** Trigonometric computation repeated every token for every head dimension.

**Current code** (`cpu_path.rs`):
```rust
fn rope(&self, q, k, pos) {
    let theta = self.cfg.rope_freq_base;
    for h in 0..n_heads {
        for d in 0..half {
            let freq = pos as f32 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let (sin_val, cos_val) = freq.sin_cos();
            // ...
        }
    }
}
```

**Proposed improvement:** Precompute `inv_freq` at model load:

```rust
// In model initialization
pub struct LlamaModel<'a> {
    // ...
    rope_inv_freq: Vec<f32>,  // Precomputed: 1.0 / theta^(2d/dim)
    rope_cos: Vec<Vec<f32>>,  // Position-indexed cos values
    rope_sin: Vec<Vec<f32>>,  // Position-indexed sin values
}

impl LlamaModel {
    fn init_rope_cache(&mut self, max_seq_len: usize) {
        let theta = self.cfg.rope_freq_base;
        let head_dim = self.cfg.head_dim as usize;
        
        // Compute inv_freq once
        self.rope_inv_freq = (0..head_dim/2)
            .map(|d| 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32))
            .collect();
        
        // Optionally precompute sin/cos for common positions
        for pos in 0..max_seq_len {
            self.rope_cos.push(
                (0..head_dim/2).map(|d| {
                    (pos as f32 * self.rope_inv_freq[d]).cos()
                }).collect()
            );
            self.rope_sin.push(
                (0..head_dim/2).map(|d| {
                    (pos as f32 * self.rope_inv_freq[d]).sin()
                }).collect()
            );
        }
    }
}

fn rope_fast(&self, q, k, pos) {
    let half = self.cfg.head_dim as usize / 2;
    for h in 0..self.cfg.head_count as usize {
        let start = h * self.cfg.head_dim as usize;
        for d in 0..half {
            let cos = self.rope_cos[pos][d];
            let sin = self.rope_sin[pos][d];
            let x0 = q[start + d];
            let x1 = q[start + d + half];
            q[start + d] = x0 * cos - x1 * sin;
            q[start + d + half] = x0 * sin + x1 * cos;
        }
    }
    // Same for k...
}
```

**Memory cost:** `max_seq_len * head_dim/2 * 2 * 4 bytes` = ~2MB for seq=4096, head_dim=128

**Effort:** Low | **Impact:** Medium | **Estimated speedup:** 5-10% on CPU path

---

## Priority 5: Q5_K High-Bit Extraction Optimization ✅ IMPLEMENTED

**Status:** ✅ Implemented 2026-07-08. `partial_q5_k` in `kernels_wgpu.rs` now
reads `qs`/`qh` in 4-byte-aligned `u32` word chunks (`qs_byte_off`/
`qh_byte_off` are always word-aligned, being built from multiples of 4) and
extracts all 4 sub-bytes per word via shift+mask, instead of calling
`read_byte_u32` (itself a `w[byte_offset/4]` read + shift) once per byte.
Note the original "not memory-coalesced" framing overstated the actual
issue — `read_byte_u32` was already indexing the same packed `u32` array, so
this is really removing redundant repeated array-index/shift work across 4
adjacent bytes, not fixing an uncoalesced memory access pattern. The
Qwen3.5-9B test model does include Q5_K-quantized rows (`in_dim=4096`
dtype list at kernel-compile time confirms it), so `compare_backends`'
CPU/GPU logit-parity check after this change exercises the new code path
directly.

**Original issue description:** Per-element byte read from `qh` array is not memory-coalesced.

**Current code** (`kernels_wgpu.rs`):
```rust
fn partial_q5_k(...) {
    let qh_byte_off = byte_off + 16;
    let mut l = 0usize;
    while l < 32 {
        let hb = read_byte_u32(w, qh_byte_off + l);  // 32 separate reads
        let hi_bit = (hb >> is) & 1u32;
        // ...
        l += 1;
    }
}
```

**Proposed improvement:** Read in 4-byte chunks and extract all relevant bits:

```rust
fn partial_q5_k_optimized(...) {
    let qh_word_off = byte_off + 16;  // Aligned u32 offset
    let mut l = 0usize;
    while l < 32 {
        let word = w[qh_word_off + l/4];  // 8 reads instead of 32
        for sub_idx in 0..4 {
            let byte = (word >> (8 * sub_idx)) & 0xFF;
            let hi_bit = (byte >> is) & 1u32;
            // process element l + sub_idx
        }
        l += 4;
    }
}
```

**Effort:** Low | **Impact:** Low | **Estimated speedup:** 5% on Q5_K models

---

## Priority 6: QK-Norm → RoPE Kernel Fusion ✅ IMPLEMENTED

**Status:** ✅ Implemented 2026-07-08, with a correctness fix to the proposed
kernel below. As originally written, the proposed kernel computed a single
`freq`/`cos_val`/`sin_val` (from `UNIT_POS`) *outside* the `for d in
0..half` loop and reused that one rotation angle for every dimension pair —
real RoPE needs a distinct frequency per `d` (`theta^(-2d/head_dim)`), so
that snippet would have rotated every pair in a head by the same angle,
which is wrong. The shipped `qk_norm_rope` kernel (`kernels_wgpu.rs`)
recomputes `freq` per `d` inside the rotation loop, matching `launch_rope`'s
existing (correct) per-`d` math, and also takes `n_rot` as a parameter since
Qwen3.5's `GatedAttention` layers only rotate the first `n_rot` dims (the
proposal below only handled full-head rotation). `gpu_resident.rs`'s two
call sites (`Attention` and `GatedAttention` mixers) now call
`launch_qk_norm_rope` instead of separate `launch_qk_norm` +
`launch_rope` calls; the old standalone `launch_qk_norm`/`qk_norm` kernel
was removed as dead code once nothing called it anymore. Verified via
`compare_backends` (CPU/GPU logit parity unchanged) and end-to-end
`generate --backend wgpu`.

**Original issue description:** Two separate kernel launches for QK-norm and RoPE.

**Current code** (`gpu_resident.rs`):
```rust
// Two separate kernel launches
b.launch_qk_norm(&q_h, &self.gpu_qk_norm[qn], n_heads, head_dim, eps);
b.launch_rope(&q_h, n_heads, head_dim, head_dim, pos, self.cfg.rope_freq_base);
```

**Proposed improvement:** Single fused kernel:

```rust
#[cube(launch)]
pub fn qk_norm_rope(
    x: &mut Array<f32>,
    qk_weight: &Array<f32>,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    pos: usize,
    theta: f32,
) {
    let h = CUBE_POS_X as usize;
    if h >= n_heads { terminate!(); }
    
    let start = h * head_dim;
    
    // QK-norm in place (first half of head)
    let mut sum_sq = 0.0f32;
    let mut d = 0usize;
    while d < head_dim {
        let v = x[start + d];
        sum_sq += v * v;
        d += 1;
    }
    let rms = f32::sqrt(sum_sq / (head_dim as f32) + eps);
    d = 0usize;
    while d < head_dim {
        x[start + d] = qk_weight[d] * x[start + d] / rms;
        d += 1;
    }
    
    // RoPE on first half
    let half = head_dim / 2;
    let exp_base = 2.0f32 / (head_dim as f32);
    let exponent = exp_base * (UNIT_POS as f32);
    let freq = (pos as f32) * f32::powf(theta, -exponent);  // pos / theta^exp
    let cos_val = freq.cos();
    let sin_val = freq.sin();
    
    for d in 0..half {
        let x0 = x[start + d];
        let x1 = x[start + d + half];
        x[start + d] = x0 * cos_val - x1 * sin_val;
        x[start + d + half] = x0 * sin_val + x1 * cos_val;
    }
}
```

**Effort:** Medium | **Impact:** Low | **Estimated speedup:** 2-5%

---

## Priority 7: CPU SIMD Vectorization ⏳ NOT ATTEMPTED

**Status:** ⏳ Skipped 2026-07-08 — `std::simd` (`portable_simd`) is still
nightly-only on the toolchain this project builds with (stable 1.96.1, no
`rust-toolchain.toml` pinning nightly), and the rest of the crate is written
against stable. Worth revisiting if the project moves to nightly, or with a
stable alternative (e.g. `std::arch` intrinsics behind a `target_feature`
check, or a crate like `wide`), but that's a bigger toolchain/dependency
decision than the other items on this list.

**Current issue:** Scalar loops in `attention_into`, `gated_delta_net`, and normalization functions.

**Proposed improvement:** Use `std::simd` for dot products:

```rust
use std::simd::f32x8;
use std::simd::SimdFloat;

fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    
    let chunks = a.chunks(8);
    for (av, bv) in chunks.zip(b.chunks(8)) {
        let a_vec = f32x8::from_slice(av);
        let b_vec = f32x8::from_slice(bv);
        sum += a_vec * b_vec;
    }
    
    sum.reduce_sum()
}
```

**Effort:** High | **Impact:** Medium | **Estimated speedup:** 2-3x on CPU path

---

---

## Priority 8: GPU Path — Embedding Dequant + Upload Per Token (decode hot path) ❌ MEASURED, NOT VIABLE

**Status:** ❌ Measured 2026-07-08, not implemented — the premise was right
(this *is* per-token waste) but the actual cost is three orders of magnitude
smaller than the fix's price tag. Built a standalone timing probe
(`weights.dequant_row("token_embd.weight", ...)` in a loop, 1000 calls) against
the real Qwen3.5-9B-Q4_K_M model: **3.08µs per call**. Against a measured
~105ms/token decode budget, that's ~0.003% of total time — not measurable
against run-to-run noise, let alone worth engineering around.

The reason it's this cheap: `dequant_row` only decompresses *one row*
(`d_model=4096` elements, 16 Q4_K blocks), not the whole table — the doc's
original framing ("128k-vocab embedding... per-token CPU dequant") conflated
the table's total size with the cost of a single-row lookup.

The proposed fix (cache the fully-dequantized `[vocab, d]` table) also turns
out to have a real cost the original writeup didn't account for: this model's
vocab is unusually large (**248,320**, not ~128k), so a full f32 cache would
be `248320 × 4096 × 4 bytes ≈ 3.79 GiB` of additional memory — for a fix that
saves ~3µs/token. Not a good trade on this model. (It could look different on
a small-vocab model where the row dequant is proportionally more expensive
relative to a smaller total forward-pass time — hasn't been measured there.)

**Effort:** Low | **Impact:** Negligible (measured, not estimated) — skip

---

## Priority 9: GPU Path — Per-Token `client.empty` Allocation Storm ❌ TESTED, NO EFFECT

**Status:** ❌ Tested 2026-07-08 with a direct experiment, not implemented —
the premise doesn't hold. Temporarily patched `launch_only` (the
highest-frequency allocator, ~15-20 calls/token) to hand out clones of one
pre-allocated 16MB scratch buffer instead of calling `self.client.empty(...)`
fresh each time, then re-ran the real decode benchmark: **no speedup** (8.0-8.6
tok/s vs. the clean 9.3-9.5 tok/s baseline — if anything marginally worse,
within noise). Reverted immediately (it also breaks correctness, since every
"new" buffer aliased the same memory — fine for a timing-only probe, not for
production).

This means `client.empty()`'s cost — whatever it is on this cubecl/wgpu stack —
isn't the per-dispatch bottleneck. A real handle-pool implementation (as
originally proposed here) would cost real engineering time to re-confirm the
same null result. Combined with the `tasks_max` finding in the session notes
below, this rules out both "batch submissions" and "stop allocating" as fixes
for the per-dispatch cost; see the new **"What we learned about this GPU's
dispatch cost"** section for what the cost actually turned out to be.

**Effort:** Medium | **Impact:** None (measured) — do not re-attempt without
new evidence

---

## Priority 10: GPU Path — `std::env::var` Trace Check Per Matmul ✅ IMPLEMENTED

**Status:** ✅ Implemented 2026-07-08 exactly as proposed. Added
`trace_matmul_enabled()` in `wgpu_backend/mod.rs` — a `static OnceLock<bool>`
wrapping `std::env::var("GGUF_TRACE_MATMUL").is_ok()`, same pattern as
`trace_cpu::enabled()` — and replaced all four call sites
(`matmul_dequant_preloaded`, `matmul_dequant_qkv_from_handle`,
`matmul_dequant_multi_from_handle`, `matmul_dequant_ffn`). Zero behavior
change, verified via the full test suite + `compare_backends` +
byte-identical-generation-output checks. As predicted, no measurable effect on
Qwen3.5 decode speed (the resident path never touches this code), but it's
free and correct, so it stays.

**Effort:** Low | **Impact:** Low (only the non-resident fallback), but trivial

---

## Priority 11: GPU Path — Attention Softmax/Score Is Two Extra Launches Per Token ⚠️ ANALYZED, HIGH REGRESSION RISK — NOT ATTEMPTED

**Status:** ⚠️ Analyzed 2026-07-08, deliberately not implemented. The premise
(three dispatches, three sync points, every attention layer, every token) is
correct, but a naive fusion has a parallelism problem that two *other* changes
this session already demonstrated regresses on this GPU (see the new section
below) — so this was reasoned through and rejected rather than tried and
reverted a third time.

`attention_scores` runs one thread per `(head, kv-position)` pair — for this
model/benchmark, `16 heads × 61 positions ≈ 976 threads` across 16
workgroups, each doing a cheap `O(head_dim)=256` dot product.
`attention_softmax` runs at a *different, much coarser* granularity: one
thread per head (16 threads, 1 workgroup), each sequentially scanning all
`pos+1` positions twice (max pass, then exp/sum pass). A straightforward
"fuse them" implementation has to pick one granularity, and picking the
coarser one (16 threads doing everything) turns `attention_scores`' cheap
`O(head_dim)` per-thread work into `O(seq_len × head_dim)` sequential work
per thread — the same "consolidate many small parallel units into fewer,
busier ones" pattern that cost 9.5→6.0 tok/s and, even in a more careful
form, 9.5→8.1 tok/s in the matmul multi-row experiments this session (see
below). It also gets *worse* as `pos` grows over generation, unlike the
matmul case which was a fixed per-call cost.

Separately: `attention_softmax`'s dispatch is already tiny (1 workgroup, 16
threads) — smaller than the 64-workgroup case in the raw-Vulkan probe that
measured near-zero cost at that scale. So the realistic upside of removing it
is small, and the realistic downside (a third parallelism-reduction
regression, this time scaling unfavorably with context length) is real.

A version that avoids the regression — computing partial max/sum per
position-block in parallel and combining them (true flash-attention tiling,
not a naive granularity collapse) — is real engineering, not a quick kernel
merge, and wasn't attempted given the size of the lift for an already-cheap
kernel.

**Effort:** Medium (naive, likely regresses) / High (parallelism-preserving,
untested) | **Impact:** Small even if it works — not recommended without new
evidence changing the analysis above

---

## Priority 12: GPU Path — Ad-Hoc `matmul_dequant` Re-Packs + Re-Uploads Every Call ❌ NOT DONE

**Status:** ❌ Open. `matmul_dequant` (`wgpu_backend/mod.rs:332-358`) is the
fallback used when a tensor isn't in `gpu_tensors` (e.g. a dtype not in
`GPU_DEQUANT_DTYPES`, or a norm vector that should have been pre-uploaded). It
calls `pack_bytes_to_u32` **and** `create_from_slice` on the **entire weight
matrix on every call** — so a single missed pre-upload means re-packing and
re-uploading that weight every token.

`pre_upload_gpu` only uploads weights whose `ggml_type` is in
`GPU_DEQUANT_DTYPES`. Any hot-path tensor that falls outside that set (and isn't
caught by the per-mixer readiness checks in `gpu_resident_ready`) silently
routes through this path. The `gpu_resident_ready` check mostly guards against
this for the resident path, but the CPU-orchestrated path (`cpu_path.rs`) can
still hit `matmul_dequant` via the `AnyBackend` dispatch.

**Proposed improvement:** (a) Audit which dtypes actually appear on the hot path
and ensure all are in `GPU_DEQUANT_DTYPES` or pre-uploaded as f32; (b) make the
ad-hoc path cache its packed handle by `(name, byte_offset)` so even fallback
calls pack+upload once, not every token. (b) is the robust fix and also helps
the `matmul_dequant_preloaded` caller that reuses a handle — i.e. memoize packed
weights in a `HashMap<String, GpuWeightHandle>` inside `WgpuBackend`.

**Effort:** Low-Medium | **Impact:** High *if* any hot tensor currently misses
pre-upload (silent per-token repack); at minimum defensive.

---

## Priority 13: GPU Path — No Prefill Batching (GPU-resident) ❌ NOT DONE

**Status:** ❌ Open (same root cause as Priority 3 / item A, called out here
because it's the dominant GPU-prefill cost). `generate.rs` runs one full
`forward()` per prompt token; on the GPU-resident path each of those is a full
layer stack of dispatches that **re-reads every weight tensor from GPU memory
once per prompt token**. For a prompt of length N, the entire `[vocab×d]` +
`[d×d]` weight set is streamed from VRAM N times during prefill.

**Proposed improvement:** Add a batched prefill that processes all N prompt
tokens in a single forward-shaped pass (one weight read, one KV-cache write per
layer across all positions, one batched QKV matmul of shape `[N, d]×[d, d_ff]`).
On a bandwidth-bound APU this is the single largest prefill-speedup available
and also makes the Priority 3 tiling win real. The GPU-resident machinery
(`launch_only`, persistent KV cache, `launch_attention` over `pos+1` positions)
is already position-parameterized and would extend to batched `x` with modest
changes.

**Effort:** High | **Impact:** High (prefill latency, all prompt lengths)

---

## What We Learned About This GPU's Dispatch Cost (2026-07-08)

A cluster of experiments this session converged on a specific, load-bearing
finding that should inform any future work here — several plausible-looking
optimizations (Priorities 9 and 11 above, plus two matmul kernel rewrites)
turned out to make things *worse*, all for the same underlying reason.

**The setup:** Qwen3.5-9B on an AMD Ryzen AI Max+ 395 (Radeon 8060S / RDNA3.5,
~40 compute units, unified memory). Baseline before this investigation: ~9.5
tok/s decode (up from 7.7 tok/s at the start of the broader session, via the
GPU-residency and kernel-fusion work in Priorities 1/4/5/6 above).

**Experiment 1 — is it `cubecl-wgpu`'s submission batching?** `cubecl-wgpu`
batches multiple dispatches into one command-buffer submit, controlled by
`CUBECL_WGPU_MAX_TASKS` (default 32). Raising it to 100,000 (fewer, bigger
submits) changed decode speed by nothing measurable (9.0-9.5 tok/s across the
whole range). There's also a separate, hardcoded `MAX_TOTAL_TASKS=512` in
`cubecl-wgpu` that forces a `device.poll(Wait)` every 512 cumulative
dispatched tasks regardless of the batching setting — our per-token dispatch
count (~450-500) is already near that ceiling either way, so this wasn't
controllable via config in either direction.

**Experiment 2 — is it wgpu's abstraction overhead vs. raw Vulkan?** Built a
standalone `ash`-based Vulkan probe (`src/bin/vulkan_poc.rs`, feature
`vulkan-poc`) with the same matmul shape as our real kernels (workgroup-per-
row, 64-thread tree reduction). Result: even in Vulkan's *best case* —
2000 dispatches recorded into one command buffer, one submit, one fence-wait
— per-dispatch cost was **289µs** at `out_dim=4096` (4096 workgroups), which
is *not better* than wgpu's own ~220-289µs on the same shape. Naive
"sync-every-dispatch" Vulkan was worse (500µs). **Switching APIs doesn't
help** — this rules out a from-scratch Vulkan backend as a fix for the
dominant cost.

**Experiment 3 — what does the cost actually scale with?** Same Vulkan probe,
varying `out_dim` (and therefore *both* workgroup count and total work
together, which is a confound worth flagging for future readers): 4096
workgroups → 289µs, 64 workgroups → 0.4µs. Read at face value this looks like
"cost scales with workgroup count" — but reducing `out_dim` also reduces the
total FLOPs by the same factor, so this measurement alone doesn't distinguish
"fewer workgroups is faster" from "less total work is faster." That
ambiguity directly caused the next two failed attempts.

**Experiment 4 — acting on the (wrong) inference from Experiment 3.**
Redesigned `matmul_dequant_wgpu` so one workgroup computes multiple output
rows (mirroring llama.cpp's Vulkan `mul_mat_vec` shaders' `NUM_ROWS`
specialization constant), for the *same total work*, on the theory that fewer
workgroups would help per Experiment 3. Two implementations, both reverted:
- **Naive** (independent compute-then-reduce per row, looped `rows_per_wg`
  times, so `sync_cube()` overhead scales with rows/workgroup): **9.5 → 6.0
  tok/s**.
- **Batched reduction** (each row's partial sum still computed completely
  independently — zero changes to `partial_q4_k`/`q5_k`/`q6_k`/`q8_0` — but
  all rows share one barrier-synchronized reduction pass, so `sync_cube()`
  count is fixed regardless of rows/workgroup): **9.5 → 8.1 tok/s**. Better
  than the naive version (confirming barrier count is *part* of the story),
  but still strictly worse than not doing it.

**The actual conclusion:** on this GPU, for this workload, the many-small-
workgroups design we already had is *better* than fewer/bigger workgroups for
the same total work — even when the reduction overhead is minimized. The
likely mechanism: with thousands of small, independent workgroups, the
scheduler has plenty of units to interleave across ~40 CUs, hiding memory
latency well. Consolidating into fewer, "fatter" workgroups (more sequential
per-thread work each) gives the scheduler less to work with, and that
occupancy loss outweighs the reduced dispatch/barrier count. **This is why
Priority 11 (attention softmax fusion) was analyzed and rejected without
being attempted** — it has the identical shape (collapsing ~976 parallel
threads down to 16 sequential-per-thread ones), and would be expected to
regress for the same reason, worse as context grows.

**Practical implication for future work here:** any optimization that
*reduces per-dispatch parallelism* (fewer/bigger workgroups, fewer/busier
threads) to *reduce dispatch or barrier count* should be treated as
high-risk on this hardware and verified empirically before trusting it, even
when the reasoning behind it seems sound. Optimizations that reduce
*dispatch count without touching parallelism* (Priorities 4, 6, and the
residual+RMSNorm fusion earlier in the broader session) are the ones that
have actually worked so far.

---

## Priority 14: GPU Path — Vectorized Byte Reads in Quant Dequant Kernels ⏳ NOT ATTEMPTED (proposed, untested)

**Status:** ⏳ Proposed 2026-07-08, not attempted this session — flagged as
the most promising *remaining* lever precisely because it doesn't touch
thread/workgroup parallelism at all (avoids the failure mode documented
above).

`read_byte_u32` (`kernels_wgpu.rs`) reads one `u32` word from the packed
weight array and extracts a single byte via shift+mask:
```rust
fn read_byte_u32(w: &Array<u32>, byte_offset: usize) -> u32 {
    (w[byte_offset / 4] >> (8u32 * (byte_offset % 4) as u32)) & 0xFFu32
}
```
`partial_q4_k`/`partial_q5_k`'s inner loops call this **once per byte** even
when reading 4 (or more) consecutive bytes from what is often the same
4-byte-aligned `u32` word — e.g. `partial_q4_k`'s `while l < 32 { qb =
read_byte_u32(w, qs_byte_off + l); ... }` re-fetches and re-shifts the same
backing word up to 4 times per word instead of reading it once and unpacking
all 4 bytes. llama.cpp's Vulkan `mul_mat_vec_q4_k.comp` (see
`~/Projects/llama-cpp-turboquant/ggml/src/ggml-vulkan/vulkan-shaders/`)
does exactly this kind of unpacking (`unpack8` on a `u32`, `vec4` loads) —
worth pulling the actual bit-tricks from there rather than re-deriving them,
per the earlier decision to reuse their kernels if we ever invest here.

**Caveat:** block sizes aren't uniformly 4-aligned — Q4_K (144 bytes/block)
and Q5_K (176 bytes/block) are divisible by 4, but Q6_K (210 bytes/block) is
not, so Q6_K's per-block byte offsets won't always land on a word boundary
and would need unaligned-read handling (combining parts of two words) if
vectorized. Q4_K and Q5_K are the safer first targets — and are also the two
dtypes this model uses most (132 and 48 tensors respectively, vs. 22 Q6_K
tensors and 48 Q8_0).

**Effort:** Medium (careful bit-manipulation, real correctness risk requiring
the same rigor as the earlier Q4_K/Q5_K debugging this session) | **Impact:**
Unknown, plausibly Low-Medium — reduces redundant shared/global memory
traffic per thread without changing occupancy, so it's in the "safe shape"
category, but hasn't been measured.

---

## Priority 15: GPU Path — f16 Intermediate Activations ⏳ NOT ATTEMPTED (proposed, untested)

**Status:** ⏳ Proposed 2026-07-08, not attempted — a different lever than
everything above: reducing *data volume* rather than *dispatch count* or
*parallelism shape*.

Every intermediate activation buffer in the resident path (`x_handle`,
`xn_handle`, `q_h`/`k_h`/`v_h`, FFN's `gate_handle`/`up_handle`, the KV cache
itself, etc.) is `f32`. This is a single-token (batch=1) decode workload on a
unified-memory APU, which tends to be memory-bandwidth-bound rather than
compute-bound — halving the byte-width of every intermediate read/write
(switching to `f16`) directly halves bandwidth pressure for all of them,
independent of the dispatch-parallelism findings above. llama.cpp's own
shader generator supports this as a first-class option (`f16acc`
specializations throughout `vulkan-shaders-gen.cpp`), which suggests it's a
real, load-bearing lever on this class of hardware, not a marginal one.

**Risks:** (1) precision — K-quant dequant already loses precision, and f16
accumulation could compound it further, especially in the GDN recurrence's
persistent state (`gpu_gdn_recurrent_state`), which accumulates *across many
tokens* — errors there could compound over a long generation in a way a
single-token accuracy check wouldn't catch. Would need a long-generation
drift check (compare CPU vs. GPU output over hundreds of tokens, not just
one), not just the single-token `compare_backends` check used elsewhere in
this doc. (2) scope — touches nearly every kernel and buffer in the resident
path, the widest-reaching change proposed here.

**Effort:** High (touches every kernel; needs new verification methodology
for cross-token precision drift) | **Impact:** Potentially High if this
workload is genuinely bandwidth-bound (plausible for batch=1 decode on
unified memory, unconfirmed) — the highest-uncertainty, highest-ceiling item
in this document.

---

## Non-Engineering Lever: Model-Level Quantization Choice

Worth stating explicitly since it's easy to overlook while deep in kernel
work: for a memory-bandwidth-bound, batch=1 decode workload, the single
biggest lever available *without touching this codebase at all* is using a
more aggressively quantized model (e.g. Q3_K/Q2_K instead of Q4_K_M) — fewer
bytes per weight directly means fewer bytes streamed from memory per token,
which is exactly the resource this workload is bound by. This trades output
quality for speed and is a user/deployment choice, not an engine
optimization, but it's the lowest-effort "performance improvement" available
if quality loss is acceptable for a given use case.

---

## Summary Table

| # | Optimization | Effort | Impact | Status |
|---|-------------|--------|--------|--------|
| 1 | GatedDeltaNet GPU path | Medium | High | ✅ Done (2 correctness bugs fixed 2026-07-08) |
| 2 | Shared memory x vector | Low | Medium | ❌ Not viable — premise didn't hold |
| 3 | Flash Attention tiling | High | High | ❌ Not applicable — no batched prefill to tile |
| 4 | RoPE precomputation | Low | Medium | ✅ Done (per-`d` table hoisted out of head loop) |
| 5 | Q5_K high-bit coalescing | Low | Low | ✅ Done |
| 6 | QK-norm → RoPE fusion | Medium | Low | ✅ Done (fixed a math bug in the original proposal) |
| 7 | CPU SIMD vectorization | High | Medium | ⏳ Pending — needs nightly `std::simd` or a stable alternative |
| 8 | GPU: embed dequant+upload per token | Low | Medium | ❌ Measured, not viable (3µs/token; 3.8GiB cache cost) |
| 9 | GPU: per-token buffer alloc storm | Medium | Medium-High | ❌ Tested, no effect (revert) |
| 10 | GPU: `env::var` per matmul (trace) | Low | Low | ✅ Done (OnceLock) |
| 11 | GPU: fuse attention softmax into scores | Medium | Medium | ⚠️ Analyzed, high regression risk — not attempted |
| 12 | GPU: ad-hoc matmul re-packs+re-uploads | Low-Medium | High (if hit) | ❌ Open |
| 13 | GPU: no prefill batching | High | High | ❌ Open |
| 14 | GPU: vectorized byte reads in dequant kernels | Medium | Unknown (Low-Med) | ⏳ Proposed, untested — safe parallelism shape |
| 15 | GPU: f16 intermediate activations | High | Potentially High | ⏳ Proposed, untested — highest ceiling, highest risk |

---

## Verification

Priorities 1, 4, 5, and 6 touch GPU-resident numerics directly. After
changing any of them, run:

```
cargo run --release --features wgpu --bin debug_gdn
cargo run --release --features wgpu --bin compare_backends -- <model.gguf>
```

`debug_gdn` checks the low-level GDN kernels against a hand-computed CPU
reference; `compare_backends` runs a full forward pass on both CPU and GPU
backends and compares top-token/top-5 logits. Both were clean (green
top-token match) after the changes above.