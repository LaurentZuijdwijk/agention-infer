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

## Priority 11: GPU Path — Attention Softmax Parallelism ✅ IMPLEMENTED (correcting the 2026-07-08 analysis below)

**Status:** ✅ Done 2026-07-13. The *fusing scores+softmax into one dispatch*
idea analyzed below was correctly identified as high-risk and was **not**
what got built — instead, `attention_softmax` was independently reprofiled
with the same forced-sync micro-benchmark methodology used for Priority 16/17
(`bin/kernel_bench`, `WgpuBackend::probe_attention_costs`), and its own claim
("dispatch is already tiny... realistic upside of removing it is small") was
**checked directly and found wrong at long context**:

| pos | softmax before | after |
|---|---|---|
| 15 | 0.051ms | 0.005ms |
| 1023 | 0.175ms | 0.0095ms |
| 2047 | **0.340ms** | **0.0146ms** (23×, no longer scales with `pos`) |

The root cause was exactly the anti-pattern this doc's Priority 16/17 fixes
describe: one thread per head (16-32 threads) serially scanning `0..=pos`
*twice* (max pass, then exp/sum pass) — the analysis below assumed this was
"already tiny" based on short-context reasoning, but it scales directly with
context length and was comparable to a full matmul at `pos≈2000`.

**Fix:** one workgroup per head, 64 cooperating lanes, two-phase
shared-memory tree reduction (max, then exp+sum) — the exact same pattern as
Priority 16, applied to a different kernel. Hit one new cubecl codegen quirk:
a self-referential compound assignment on the running max panicked
("mutable operation on a const variable"); resolved with `RuntimeCell` for
the per-lane max scan, mirroring the ReLU-style `+=` trick the original
single-threaded kernel already used for ordinary reassignment restrictions.

End-to-end decode at *short* context (the golden/bench prompts, pos<100) is
neutral (~20.9 tok/s, matches pre-fix) — expected, since the old serial
softmax was already cheap there. The win is specifically for long-context
decode, where naive per-head softmax would otherwise become a real cost.

**Original 2026-07-08 analysis (kept for context — the fusion idea it
evaluated is still unattempted and still looks high-risk):**

`attention_scores` runs one thread per `(head, kv-position)` pair — for this
model/benchmark, `16 heads × 61 positions ≈ 976 threads` across 16
workgroups, each doing a cheap `O(head_dim)=256` dot product.
`attention_softmax` ran at a *different, much coarser* granularity: one
thread per head (16 threads, 1 workgroup), each sequentially scanning all
`pos+1` positions twice. A straightforward "fuse them" implementation has to
pick one granularity, and picking the coarser one (16 threads doing
everything) turns `attention_scores`' cheap `O(head_dim)` per-thread work
into `O(seq_len × head_dim)` sequential work per thread — the same
"consolidate many small parallel units into fewer, busier ones" pattern that
cost 9.5→6.0 tok/s in the matmul multi-row experiments. **This fusion idea
remains unattempted and the risk analysis still stands** — what changed is
only that softmax's *own* (unfused) parallelism was fixed separately, which
was lower-risk and didn't require picking a shared granularity with scores.

**Effort:** Medium (naive fusion, likely regresses) / High
(parallelism-preserving fusion, untested) | **Impact:** Small even if fusion
works, now that softmax's own cost is fixed — not recommended without new
evidence.

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

## Priority 14: GPU Path — Vectorized Byte Reads in Quant Dequant Kernels ✅ Q4_K DONE, Q5_K ALREADY DONE (predates this doc), Q6_K NOT THE SAME BUG

**Status:** ✅ Done for Q4_K 2026-07-13 — this was the single biggest win of
that session, exactly as the 2026-07-08 proposal below predicted ("safe
shape," no parallelism change).

`read_byte_u32` (`kernels_wgpu.rs`) read one `u32` word from the packed
weight array and extracted a single byte via shift+mask, called **once per
byte** even when reading 4 consecutive bytes from the same 4-byte-aligned
word — `partial_q4_k`'s inner loop re-fetched and re-shifted the same
backing word up to 4 times per word. Fixed: read one `u32` per 4 bytes
(`qs_byte_off` is always 4-aligned — all its terms are multiples of 4),
unpack all 4 nibbles from one fetch.

| | before | after |
|---|---|---|
| matmul (4096×12288, Q4_K) | 0.338ms | **0.178ms** (1.9×) |
| decode (Qwen3.5-9B) | 16.55 tok/s | **20.92 tok/s** (+26%) |
| % bandwidth ceiling | 36% | **46%** |

Golden token-parity green on all 4 models, CPU + GPU (this is a
bit-manipulation change to quant unpacking — correctness-critical to verify,
and was).

**Correction (2026-07-13, later the same day): both "still open" claims below
were wrong on inspection — checked before doing any more work, not after.**

- **Q5_K was already vectorized** — `partial_q5_k` already reads one `u32`
  word per 4 bytes (`w[(qs_byte_off + l) / 4]`, unpacking 4 shifted extracts)
  for both the quant bytes and the high-bit `qh` array. This predates the
  original 2026-07-08 proposal (commit `343ac0b`, same calendar day, timestamp
  earlier) — the proposal's claim that Q5_K had the same bug was simply wrong
  even when written. Nothing to do here.
- **Q6_K does *not* have the same bug**, on closer reading of its actual
  thread-assignment pattern. `partial_q4_k`/`partial_q5_k` assign *multiple*
  sub-units to each thread via an internal loop (`while l < 32 { ... }`) — that
  loop is what re-read the same word repeatedly. `partial_q6_k` assigns
  exactly **one** lane-specific offset to each thread with no such loop (only
  loops over `block`, a different axis) — `ql0`/`ql1`/`qh_byte` are each a
  single `read_byte_u32` call per thread, and adjacent lanes reading adjacent
  bytes is the normal GPU-coalesced access pattern, not a redundant re-read.
  There *is* a small, genuine instance of the pattern in the **scale** reads
  (`sc0/sc2/sc4/sc6`, four `read_i8_i32` calls 2 bytes apart, pairing into 2
  shared words) — but that's 4 scale bytes vs. Q4_K's 32 quant bytes, and
  `output.weight` (Q6_K's only user, the LM head) is called once per token,
  not ~30× like the FFN/attention matmuls. Bounded to a fraction of one
  dispatch/token — not worth the bit-manipulation correctness risk right now.

**Original 2026-07-08 proposal (superseded by the Q4_K result and the
correction above):** llama.cpp's Vulkan `mul_mat_vec_q4_k.comp` (see
`~/Projects/llama-cpp-turboquant/ggml/src/ggml-vulkan/vulkan-shaders/`) does
the same kind of unpacking — this remains a good reference if the small Q6_K
scale-read optimization is ever revisited.

**Effort:** Medium (careful bit-manipulation, real correctness risk) |
**Impact:** Confirmed High for Q4_K (1.9× the dominant per-token matmul cost);
Q5_K already captured; Q6_K's remaining opportunity is small and bounded.

---

## Priority 15: GPU Path — f16 Intermediate Activations ✅ IMPLEMENTED (smaller win than expected — see Priorities 16/17/14 for why)

**Status:** ✅ Done (Phase 1 Stage 4, before the kernel-parallelism work
below). Default build now stores GPU-resident activations as `f16` (feature
`f32-activations` opts back to `f32` for backends without shader-f16, e.g.
wgpu→Metal). f32 accumulators kept inside every kernel throughout — only
storage narrows. Native f16 compute confirmed working on this Vulkan device
via a standalone probe (`bin/f16_probe.rs`) before committing to the full
kernel rewrite. GDN recurrent state, attention score/softmax scratch, and
logits deliberately kept `f32` per the risk noted below.

**Measured result — smaller than the "potentially High" estimate below
predicted:** decode 9.49 (f32) → 9.72 (f16) tok/s, **~2-3%**, not the
bandwidth-halving win the `f16acc` comparison to llama.cpp implied. 200-token
CPU-f32-vs-GPU-f16 drift test (`bin/drift.rs`) clean on the GDN 9B — the
precision risk called out below didn't materialize.

**Why the win was small, with hindsight from the rest of this document:**
this workload was **not actually bandwidth-bound** at the time f16 was
implemented (21% of the 260GB/s ceiling) — it was bottlenecked by the
serial-kernel and dequant issues Priorities 16/17/14 later found and fixed.
Halving activation bytes doesn't help much when the dominant cost is a
handful of GPU lanes doing wasted serial work, or a matmul re-reading the
same quant byte 4× — neither of which f16 activations touch. The KV cache is
now f16 too, so there's a *latent* benefit that should show up more at long
context (larger KV, more of it moved per token) — not yet re-measured after
Priorities 16/17/14 changed the bandwidth/compute balance.

**Original 2026-07-08 risk analysis (for the record — didn't materialize):**
K-quant dequant already loses precision, and f16 accumulation could compound
it further, especially in the GDN recurrence's persistent state, which
accumulates *across many tokens*. Verified clean via the drift test built for
this purpose.

**Effort:** High (touched every kernel; new drift-test methodology) |
**Impact:** Confirmed Low-Medium in isolation (~2-3%), given this workload
wasn't bandwidth-bound when it landed — revisit once Priorities 16/17/14 push
utilization high enough that bandwidth becomes the binding constraint again.

---

## Priority 16: GPU Path — RMSNorm Ran Single-Threaded ✅ IMPLEMENTED (biggest single win this session)

**Status:** ✅ Done 2026-07-13. Found by building a proper kernel
micro-benchmark (`bin/kernel_bench`, `WgpuBackend::probe_kernel_costs` —
amortized per-dispatch timing via K back-to-back launches + one final sync,
matching how the real forward pass submits) after the user asked to find real
GPU speedups rather than more bandwidth-reduction, since decode was only
~9.72 tok/s at ~21% of the 260GB/s ceiling despite Priorities 1-15 above.

`rms_norm` and `add_residual_rms_norm` ran on a **single GPU thread**
(`CubeDim::new_1d(1)`), serially reducing `embedding_length` (4096) elements
per call. Measured cost: **0.49-0.65ms per call** — more than a full
4096×12288 matmul (0.34ms at the time) — and at ~2 calls/layer × 32-48 layers
this was roughly **half the entire per-token budget**. This directly
contradicts the "sync cost dominates at these vector sizes" rationale the
kernel's original comment gave for staying single-threaded — that assumption
was never measured and was wrong by 1-2 orders of magnitude.

**Fix:** 256-thread shared-memory tree reduction (the same pattern the
matmul kernel already used) — each lane reduces a strided slice, then a
standard `stride/=2` tree combines partials.

| | before | after |
|---|---|---|
| rms_norm(d=4096) | 0.49ms | 0.013ms (38×) |
| add_residual_rms_norm | 0.64ms | 0.014ms (45×) |
| decode (Qwen3.5-9B) | 9.72 tok/s | **16.03 tok/s (+65%)** |
| % bandwidth ceiling | 21% | 35% |

Golden token-parity green on all 4 models, CPU + GPU.

**The broader lesson, restated for future readers:** this project was
**never actually bandwidth-bound** despite Priority 15's framing — it had
serial-kernel bottlenecks nobody had measured. "This kernel is too small to
matter" is an assumption, not a fact, until you force-sync and time it.

**Effort:** Low (same tree-reduction pattern already used elsewhere) |
**Impact:** Confirmed Very High — the single biggest win in this document.

---

## Priority 17: GPU Path — Per-Head Norm Kernels Ran One Thread Per Head ✅ IMPLEMENTED

**Status:** ✅ Done 2026-07-13, same session and methodology as Priority 16,
smaller scale. `qk_norm_rope`, `l2_norm_heads`, and `gdn_gated_rms_norm` each
ran **one thread per head** (`~16-32` heads → `~16-32` total threads),
serially reducing `head_dim` (and, for `qk_norm_rope`, doing per-dim
`powf`/`sin`/`cos` in that same one thread) — the identical anti-pattern to
Priority 16, at smaller absolute cost. `qk_norm_rope` measured 0.0925ms/call
(13× the ~0.007ms dispatch floor).

**Fix:** one workgroup per head (or per segment, for `l2_norm_heads`'s
Q-range/K-range split), 64 cooperating lanes, shared-memory tree reduction —
`qk_norm_rope` 0.0925ms → 0.0044ms (21×). Combined with Priority 16:
16.03 → 16.55 tok/s. Golden green on all 4 models.

**Effort:** Low | **Impact:** Medium (smaller than Priority 16 since these
run fewer times per token, but same fix, same confidence).

---

## Priority 18: GPU Path — Batched-Prefill Matmul ❌ TESTED, DTYPE-DEPENDENT, NOT ADOPTED

**Status:** ❌ Built and measured 2026-07-13 (was Priority 13, "no prefill
batching"). Two designs tested with a purpose-built micro-bench
(`bin/matmul_batch_bench`, `WgpuBackend::probe_batched_matmul`/
`probe_gdn_chain_cost`) rather than committed on theory alone:

1. **Loop-N-inside-one-workgroup** (`matmul_dequant_wgpu_batch`): each
   output-row workgroup loops over all N batch tokens, re-reading the same
   weight row N times, hoping L1 would amortize it. It didn't —
   `batch/tok` was flat regardless of N on Q8_0 (~0.36ms/tok), and actively
   *regressed* on Q4_K as N grew (0.68× at N=32) — the same "fewer,
   busier workgroups lose occupancy" failure mode as the earlier multi-row
   matmul experiments below.
2. **Cooperative dequant-once** (`matmul_q8_0_coop_batch`/
   `matmul_q4_k_coop_batch`): dequantize the weight row into shared memory
   once per workgroup, then dot it against all N tokens from LDS — this
   *did* amortize (proving the matmul is dequant-compute-bound, not
   VRAM-bandwidth-bound): **Q8_0 wins, 1.53× at N=32** (coop/tok
   0.538→0.215ms). But **Q4_K regresses, 0.36-0.48×** (~2× slower/tok) — the
   heavier nibble-unpack-into-LDS cost (2.8× overhead already at N=1) swamps
   the amortization. Both variants exactly correct (`max_abs_err=0` vs the
   single-token reference).

**Since the flagship model (Qwen3.5-9B) is Q4_K, cooperative batched prefill
would slow it down, not speed it up.** Winning on Q4_K would need real
kernel tuning (LDS bank conflicts, footprint, occupancy) starting from a 2-3×
deficit — high-risk, speculative. GPU prefill stays the per-position loop.
Kept the probe kernels as reproducible evidence (à la the raw-Vulkan probe
below).

**Effort:** Medium (built) | **Impact:** Positive for Q8_0-only deployments,
negative for the Q4_K models this project actually targets — not adopted.

---

## Priority 19: GPU Path — GDN Recurrence Kernel ✅ PROFILED, NO FIX NEEDED

**Status:** ✅ Profiled 2026-07-13 (`WgpuBackend::probe_gdn_recurrence_cost`,
`probe_gdn_chain_cost`) on the hypothesis that it might share Priority 16/17's
under-parallelization bug, given Qwen3.5-9B is GDN-heavy (24 of 32 layers).
It doesn't: `CubeDim = head_v_dim` (128), `n_v_heads` (32) workgroups, each
thread owns one state-matrix column with zero cross-thread sync, doing
genuine `O(head_k_dim)` sequential work — not a wasted serial reduction.
Measured 0.059ms/call, ≈1.4ms of the ~48ms/token budget (~3%) across 24 GDN
layers — present, not a bottleneck.

**Side finding:** the full GDN mixer chain (10 dispatches: 4 projections +
gate_decay + conv1d_silu + l2_norm + recurrence + gated_norm + ssm_out) costs
**more chained with no intermediate sync (0.495ms) than the sum of each
step measured in isolation (0.395ms)** — a genuine ~20% pipeline-switching
tax from alternating between 10 different kernel pipelines back-to-back, that
doesn't show up when the same kernel is repeated in a loop. This is real
headroom for **kernel fusion** (not per-kernel speedup) — the 4 same-input
projections (`wqkv`/`wgate`/`ssm_beta`/`ssm_alpha`) are the safest fusion
candidate (independent linear ops, no state), estimated at ~1-3% of total
decode time if built — not pursued this session given the modest ROI vs. the
engineering/correctness-risk cost of a new multi-weight matmul kernel.

**Effort:** N/A (no fix applied) | **Impact:** N/A — documented negative
result plus a scoped, unbuilt fusion candidate for later.

---

## Priority 20: CubeCL Math Micro-Optimizations (`FastMath`, `FastDivmod`) ❌ TESTED, NO EFFECT

**Status:** ❌ Tested 2026-07-13, reverted (zero diff vs. committed state).
CubeCL 0.10 exposes `#[cube(fast_math = FastMath::all())]` (relaxed
IEEE-754 semantics — `NotNaN`/`NotInf`/`ReducedPrecision`/etc., backend-
dependent) and `FastDivmod<T>` (Barrett-reduction-based fast integer
division/modulo for a runtime, non-constant divisor).

Applied `FastMath::all()` to all 9 exp/sqrt-heavy kernels (both RMSNorms,
softmax, both per-head norms, silu/sigmoid, conv-silu, gate-decay) — no
measurable change, confirmed not noise by checking an *untouched* kernel
(`wqkv matmul`) drifted by the same ~10-15% between runs as the touched ones
(ambient/thermal variance, not a real regression or improvement either way).

Applied `FastDivmod<usize>` to `attention_scores`/`attention_output`'s
`idx / seq_len` / `idx % head_dim` index math (the clearest "uniform,
non-constant divisor" case in this codebase) — same null result, if anything
marginally worse (extra multiply-high + shift vs. one division).

**Why these gave nothing, with the rest of this document as context:** by
this point Priorities 16/17/14 had already removed the actual bottlenecks
(serial threads, redundant byte reads). The remaining hot kernels are
dispatch-floor-bound (near their per-launch overhead) or dominated by
100+-op dot-product loops where 1 division or a handful of `exp`/`sqrt`
calls is noise. **This class of instruction-level optimization only pays off
once the real structural bottleneck (parallelism, redundant memory traffic)
is already fixed** — applying it earlier in the session (e.g. to the
still-single-threaded RMSNorm) would likely have shown the same null result,
since the bottleneck there was thread *count*, not per-thread instruction
cost.

**Effort:** Low (both are simple attribute/type additions) | **Impact:**
Confirmed zero on this codebase's current kernels — no reason to revisit
unless a new kernel is written that's genuinely ALU-bound on division or
transcendentals specifically.

---

## Priority 21: CMMA / Cooperative-Matrix (Tensor-Core-Style) Hardware ✅ CAPABILITY CONFIRMED, NOT ADOPTED

**Status:** ✅ Confirmed working 2026-07-13 via an isolated spike
(`/home/laurent/Projects/agention/cmma_spike`, not part of this repo) — not
yet adopted in the engine. See the new "CubeCL Upgrade & Vulkan Library
Investigation" section below for the full narrative (version research,
`vulkan_poc` prior art, capability probing methodology).

**The confirmed fact:** this exact GPU (`Radeon 8060S Graphics (RADV
STRIX_HALO)`), via the RADV Mesa Vulkan driver, advertises
`VK_KHR_cooperative_matrix` with `FLOAT16×FLOAT16→FLOAT32` at 16×16×16,
subgroup scope — real tensor-core-style matrix-multiply-accumulate hardware,
usable through CubeCL's `main` branch (`0.11.0-pre.1`, unreleased) via its
`#[cube]`-level `cmma` API on the `wgpu` runtime's SPIR-V/Vulkan compiler
(`wgpu<spirv>`, **not** the default WGSL compiler, which per CubeCL's own
0.10 README doesn't support tensor cores at all). Verified with an exact
numeric match against CubeCL's own reference test values, not just "it
compiled."

**Why this isn't being adopted now:** two structural mismatches with this
project's actual workload. (1) CMMA operates on **dense** f16 tiles; our
weights are quantized (Q4_K/Q5_K/Q6_K/Q8_0) and would still need dequanting
first — CMMA doesn't remove that cost, it just changes what consumes the
dequantized values. (2) CMMA wants `M≥16` to fill a tile; batch=1 **decode**
(this project's current bottleneck) has `M=1`, wasting 15/16 of the tile —
this would only pay off for **batched prefill** with pre-dequantized f16
weights, and Priority 18 above already shows the naive batched-prefill
approach regresses on Q4_K. Real adoption means hand-writing a Vulkan
compute shader outside CubeCL's existing kernel set, pre-dequantizing
weights to dense f16 (a memory-footprint tradeoff), and restructuring around
batched shapes — by far the largest lift considered this session, and
squarely **Phase 2 (GPU performance / flash-attention) territory**, not a
Phase 1 fit.

**Effort:** Unknown, likely Very High (from-scratch Vulkan compute shader,
new weight-preprocessing pipeline) | **Impact:** Unknown/unmeasured for our
actual quantized-weight, batch=1-decode workload — confirmed hardware
capability, unconfirmed real-world speedup. Revisit for Phase 2 batched
prefill / flash attention.

---

## Priority 22: GPU Path — `short_conv` (LFM2 ShortConv Mixer) ✅ PROFILED, NO FIX NEEDED

**Status:** ✅ Profiled 2026-07-13 (`WgpuBackend::probe_short_conv_cost`) —
this session's Priorities 16/17/19/20 all profiled the Qwen3.5-9B (attention +
GatedDeltaNet) resident path; `short_conv` is LFM2's mixer and hadn't been
checked with this methodology. It's already fine: one thread per channel
(`d=2048` for LFM2.5-1.2B → 32 workgroups, 2048 total threads), no internal
loop over a value axis, no cross-thread sync — the same "already parallel"
shape as `attention_output`, not the "one thread per head" shape that needed
fixing elsewhere. Measured **0.0077ms/dispatch**, at the dispatch floor.
LFM2.5-1.2B decode: 116.5 tok/s, 32.8% of its (much higher, smaller-model)
bandwidth ceiling — already benefits from Priorities 16/17/20's shared
kernels (RMSNorm, softmax, etc. are used by every architecture).

**Effort:** N/A (no fix applied) | **Impact:** N/A — confirms the
"already-parallel" kernel shapes generalize correctly to the other
architecture in this codebase, no hidden bug found.

---

## Priority 23: Loading — mmap Page Cache Accumulates Toward Full Model Size During GPU Upload ✅ FIXED

**Status:** ✅ Fixed 2026-07-14, prompted by the user's own observation while
running the model that CPU memory filled before GPU memory. Not a memory-pool
*misplacement* bug — measured directly (`/sys/class/drm/card1/device/mem_info_
{vram,gtt}_used` alongside process RSS during a real load) and confirmed GPU
buffers correctly land in VRAM (device-local), not the slower GTT
(system-RAM-backed) pool. The real issue was a genuine *transient
double-buffer*: `pre_upload_gpu`'s per-tensor mmap → CPU pack (`Vec<u32>`) →
GPU upload pipeline let the mmap'd page cache accumulate toward the *entire*
file's size by the end of loading (each tensor's temporary packing buffer was
already correctly bounded and dropped per-tensor — no leak there), so the CPU
briefly held close to a full copy of the model concurrently with the growing
VRAM copy.

This matters because this project's own thesis (`00-parity-roadmap.md`)
targets 90–120GB MoE models on a 128GB box, and this test box specifically
splits its 128GB as 30GB system RAM + a 96GB VRAM carve-out — a model file
approaching or exceeding system RAM would risk OOM purely from page-cache
growth during loading, independent of any Rust-level allocation bug.

**Fix:** process tensors in ascending file-offset order (not layer-
declaration order), and after each upload, `madvise(MADV_DONTNEED)` the
newly-consumed, page-aligned prefix of the mmap (`release_mmap_prefix` in
`gpu_resident.rs`). Safe for a read-only file-backed mapping: any later read
(e.g. the CPU-orchestrated fallback path, if GPU-resident isn't fully ready)
transparently re-faults from disk — never wrong, just potentially slower in
that fallback case. `raw_data`'s own start address isn't page-aligned (it's a
sub-slice of the mmap starting at the GGUF data offset), so the release-range
math rounds the start up and the end down to page boundaries on *every* call
— always safe, self-correcting regardless of prior alignment.

| | before | after |
|---|---|---|
| peak CPU RSS during load (Qwen3.5-9B) | ~7.5GB | **~3.0GB**, and *decreasing* as loading progresses (2.9→2.1→0.9→0.3GB) |
| VRAM usage | ~8GB (unaffected) | ~8GB (unaffected) |

Golden green on all 4 models, CPU + GPU; 200-token drift clean; manual LFM2 +
Qwen3.5-9B generate runs produce identical output to before.

**Effort:** Low-Medium (careful page-alignment math, but self-contained —
no API changes outside `gpu_resident.rs`) | **Impact:** High for large-model
loading specifically — directly addresses an OOM risk on the MoE models this
project's roadmap targets, not yet reproducible on today's small test models
but real for tomorrow's big ones.

---

## CubeCL Upgrade & Vulkan Library Investigation (2026-07-13)

Prompted by "should we look at a different GPU library" after Priority 20's
null result. Documenting the full chain of reasoning since each step
corrected the previous one — useful if this question comes up again.

**Is there a newer CubeCL release?** No. `cargo info cubecl` and the crates.io
sparse index both show `0.10.0` as the latest published version — no `0.11`
has been released. `main` on GitHub is `0.11.0-pre.1`, pre-release-branch-cut
(confirmed via `git ls-remote --heads`: `release/0.10` exists, no
`release/0.11`), and CubeCL's own README calls itself "**alpha**... a lot of
rough edges." "Upgrading" would mean depending on an unreleased, unpinned git
commit for a project whose bar is exact token-parity — a materially
different risk profile than the stable 0.10.0 release everything else in
this doc was measured against. **Recommendation stands: don't float on `main`
for the production engine; pin to a specific commit only if/when a specific
capability is confirmed worth the risk** (see below).

**Is a different Vulkan *library* (bypassing CubeCL) worth trying?** Already
answered, and already in this repo: `bin/vulkan_poc.rs` (feature
`vulkan-poc`) is a hand-written raw-`ash` compute-shader dispatch-overhead
probe. Documented result (see "What We Learned About This GPU's Dispatch
Cost" above): raw Vulkan's per-dispatch cost (~289µs) was **not** better than
CubeCL/wgpu's own (~220-289µs) on the same shape — "switching APIs doesn't
help" for *dispatch overhead*. Re-litigating that specific question with a
different Vulkan wrapper crate (`vulkano`, etc.) would just reconfirm it —
the overhead is inherent to this hardware/driver, not a CubeCL tax.

**Is there a *different* capability worth the CubeCL-main risk?**
Yes — cooperative-matrix (CMMA). The 0.10 README states tensor-core
acceleration "isn't supported on WebGPU yet" (CubeCL's wgpu backend,
CUDA/NVIDIA-first). Reading CubeCL's `main` source directly (the user cloned
it to `/home/laurent/Projects/agention/cubecl`) found a full, real CMMA
implementation added since 0.10: an IR instruction (`cubecl-ir/src/cmma.rs`),
SPIR-V codegen (`cubecl-spirv/src/cmma.rs`), a `#[cube]`-level frontend API
(`cubecl-core/src/frontend/cmma.rs`), and CubeCL's own runtime tests
exercising it. This is Priority 21 above — confirmed working on this exact
GPU via an isolated spike crate, not adopted into the engine (structural
mismatch with quantized weights + batch=1 decode; Phase 2 candidate for
batched prefill instead).

**Methodology note, since it mattered twice:** every empirical measurement in
this investigation (and this document generally) needs the GPU to actually be
idle and cool. This session hit two false alarms — a ~2× "regression" in an
untouched kernel traced to a concurrent `llama-server` process sharing the
GPU, and later to residual heat (78°C vs. the ~47-50°C clean baseline) from
back-to-back benchmark runs. **Before trusting any GPU perf number in this
codebase: `ps aux | grep llama-server` and check
`/sys/class/hwmon/hwmon*/temp1_input` (>65°C is a throttle-risk signal).**
Re-measure after confirming both are clear before drawing conclusions.

**How the CMMA capability was confirmed** (reusable methodology): rather than
touch this engine's dependencies at all, a throwaway crate
(`/home/laurent/Projects/agention/cmma_spike`, `cubecl = { path =
"...cubecl/crates/cubecl", features = ["vulkan"] }`) queried
`client.features().matmul.cmma` and ran CubeCL's own reference CMMA kernel,
checking output against CubeCL's own expected values. Two gotchas: (1) the
default `wgpu` feature selects the **WGSL** compiler (`wgpu<wgsl>`, no tensor
cores, matching the README); the **`vulkan`** feature (=`wgpu` +
`cubecl-wgpu/spirv`) is needed to get `wgpu<spirv>`, which does. (2) AMD's
wgpu plane/wavefront size is a runtime-queried range (`plane_size_min..max`,
32-64 here), not a fixed constant like NVIDIA/HIP — must be read from
`client.properties().hardware`, not assumed.

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
| 11 | GPU: attention softmax parallelism | Low | High (at long context) | ✅ Done — 23× at pos=2047; fusion-into-scores idea still not attempted |
| 12 | GPU: ad-hoc matmul re-packs+re-uploads | Low-Medium | High (if hit) | ❌ Open |
| 13 | GPU: no prefill batching | High | High | ❌ Superseded by #18 — tested, dtype-dependent, not adopted |
| 14 | GPU: vectorized byte reads in dequant kernels (Q4_K) | Medium | High | ✅ Q4_K done (matmul 1.9×, decode +26%); Q5_K already done pre-session; Q6_K not the same bug, skip |
| 15 | GPU: f16 intermediate activations | High | Low-Medium (confirmed) | ✅ Done — only ~2-3%, workload wasn't bandwidth-bound yet |
| 16 | GPU: RMSNorm ran single-threaded | Low | Very High | ✅ Done — 38-45× kernel, decode +65% (biggest win this doc) |
| 17 | GPU: per-head norms (qk_norm/l2_norm/gdn_norm) 1 thread/head | Low | Medium | ✅ Done — 21× on qk_norm_rope |
| 18 | GPU: batched-prefill matmul (loop-N / cooperative dequant) | Medium | Dtype-dependent | ❌ Tested — wins Q8_0 (1.5×), regresses Q4_K (0.4×) — not adopted |
| 19 | GPU: GDN recurrence kernel | N/A | N/A | ✅ Profiled, already well-parallelized — no fix; found ~20% chain pipeline-switch tax instead |
| 20 | CubeCL `FastMath`/`FastDivmod` micro-opts | Low | None | ❌ Tested, zero measurable effect — reverted |
| 21 | CMMA / cooperative-matrix hardware | Very High (unbuilt) | Unknown | ✅ Capability confirmed on this GPU (CubeCL `main`); not adopted — Phase 2 candidate |
| 22 | GPU: `short_conv` (LFM2 mixer) | N/A | N/A | ✅ Profiled, already well-parallelized — no fix needed |
| 23 | Loading: mmap page cache accumulates during GPU upload | Low-Medium | High (large models) | ✅ Fixed — incremental madvise(DONTNEED), peak RSS 7.5GB→3.0GB |

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

**Priorities 16-20 (2026-07-13 kernel-parallelism work) additionally require:**

```
GGUF_MODELS_DIR=./models cargo test --release --features wgpu --test golden   # token-parity, all 4 models, CPU+GPU
cargo run --release --features wgpu --bin drift -- <model.gguf> 256           # 200+ token CPU-f32 vs GPU-f16 drift
cargo run --release --features wgpu --bin kernel_bench -- <model.gguf>        # per-kernel forced-sync micro-bench
cargo run --release --features wgpu --bin matmul_batch_bench -- <model.gguf>  # batched/coop matmul probe (Priority 18)
cargo run --release --features vulkan-poc --bin coop_matrix_probe            # CMMA hardware capability check (Priority 21)
```

`kernel_bench` is the tool that found Priorities 16/17/19 — always run it
with the GPU idle and cool (`ps aux | grep llama-server`; check
`/sys/class/hwmon/hwmon*/temp1_input` < ~55°C) or the numbers are unreliable,
per the methodology note under "CubeCL Upgrade & Vulkan Library
Investigation" above.