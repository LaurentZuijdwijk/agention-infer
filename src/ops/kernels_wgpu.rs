//! WGPU-compatible kernels using u32-packed byte arrays.
//!
//! WGSL doesn't support `u8`, so ALL operations use `u32` internally.
//! Bytes are packed 4-per-u32, unpacked inline with bit operations.
//!
//! Same numerical results as the u8 kernels.

use cubecl::prelude::*;

// ── SwiGLU combine: out[i] = silu(gate[i]) * up[i] ────────────────────────
//
// Elementwise — no quantization involved, so this reads `gate`/`up` straight
// from GPU handles (e.g. the outputs of two matmul_dequant_wgpu launches)
// with no CPU round-trip in between.

#[cube(launch)]
pub fn silu_mul(gate: &Array<f32>, up: &Array<f32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= out.len() {
        terminate!();
    }
    let g = gate[i];
    let silu = g / (1.0f32 + (-g).exp());
    out[i] = silu * up[i];
}

// ── Sigmoid gate: out[i] = a[i] * sigmoid(b[i]) ───────────────────────────
//
// Used by Qwen3.5's `GatedAttention` mixer: `attn_out *= sigmoid(gate)`
// before `wo`. Not the same as `silu_mul` (`silu(gate)*up == gate *
// sigmoid(gate) * up`) — this has no extra `gate` factor.

#[cube(launch)]
pub fn sigmoid_mul(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= out.len() {
        terminate!();
    }
    out[i] = a[i] / (1.0f32 + (-b[i]).exp());
}

// ── Split fused Q+gate projection ──────────────────────────────────────────
//
// Qwen3.5's `GatedAttention` fuses the Q projection with a per-head sigmoid
// output gate: `wqg`'s output is `n_heads` blocks of `[Q(head_dim) |
// gate(head_dim)]`. One thread per (head, dim) deinterleaves that into
// contiguous `q`/`gate` buffers.

#[cube(launch)]
pub fn split_qg(qg_raw: &Array<f32>, q: &mut Array<f32>, gate: &mut Array<f32>, head_dim: usize) {
    let idx = ABSOLUTE_POS;
    let n_heads = q.len() / head_dim;
    if idx >= n_heads * head_dim {
        terminate!();
    }
    let h = idx / head_dim;
    let d = idx % head_dim;
    let src = h * 2 * head_dim;
    q[idx] = qg_raw[src + d];
    gate[idx] = qg_raw[src + head_dim + d];
}

// ── RMSNorm: out[i] = (x[i] / rms) * weight[i] ────────────────────────────
//
// Single-threaded on purpose: at these vector sizes (embedding_length, a few
// thousand elements) actual GPU compute time is negligible compared to the
// fixed per-launch sync cost (see gpu-sync-bottleneck memory) — a parallel
// reduction would add complexity without a measurable win.

#[cube(launch)]
pub fn rms_norm(x: &Array<f32>, weight: &Array<f32>, out: &mut Array<f32>, eps: f32) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0usize;
    while i < n {
        let xi = x[i];
        sum_sq += xi * xi;
        i += 1;
    }
    let rms = f32::sqrt(sum_sq / (n as f32) + eps);
    let mut j = 0usize;
    while j < n {
        out[j] = (x[j] / rms) * weight[j];
        j += 1;
    }
}

// ── Fused residual-add + RMSNorm: new_x = x + delta; normed = norm(new_x) ─
//
// Every `residual_add` in the resident forward pass is immediately followed
// by an `rms_norm` on its result (the next mixer's or FFN's prenorm, or the
// final output norm) — folding them into one launch halves the dispatch
// count for this pair. Single-threaded, same rationale as `rms_norm` above.

#[cube(launch)]
pub fn add_residual_rms_norm(
    x: &Array<f32>,
    delta: &Array<f32>,
    weight: &Array<f32>,
    new_x: &mut Array<f32>,
    normed: &mut Array<f32>,
    eps: f32,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }
    let n = x.len();
    let mut sum_sq = 0.0f32;
    let mut i = 0usize;
    while i < n {
        let v = x[i] + delta[i];
        new_x[i] = v;
        sum_sq += v * v;
        i += 1;
    }
    let rms = f32::sqrt(sum_sq / (n as f32) + eps);
    let mut j = 0usize;
    while j < n {
        normed[j] = (new_x[j] / rms) * weight[j];
        j += 1;
    }
}

// ── Short-conv gate + depthwise causal conv1d (LFM2 `ShortConv`) ─────────
//
// Fully data-parallel across the `d` channels: each thread owns one channel
// and only that channel's `l-1` history slots (tap weights are laid out
// `[channel][tap]`, contiguous per channel) — there is no cross-channel
// dependency, unlike the CPU version's per-channel-recurrence framing might
// suggest. `history` is mutated in place: a thread reads its own slots for
// the dot product, then shifts and appends `bx` into those same slots —
// safe because no other thread ever touches this channel's slice.

#[cube(launch)]
pub fn short_conv(
    bcx: &Array<f32>,
    weight: &Array<f32>,
    history: &mut Array<f32>,
    conv_out: &mut Array<f32>,
    l: usize,
) {
    let ch = ABSOLUTE_POS;
    let d = conv_out.len();
    if ch >= d {
        terminate!();
    }

    // bcx = concat(B, C, x_gate), each length d.
    let b = bcx[ch];
    let c = bcx[d + ch];
    let xg = bcx[2 * d + ch];
    let bx = b * xg;

    let tap_base = ch * l;
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k + 1 < l {
        acc += weight[tap_base + k] * history[k * d + ch];
        k += 1;
    }
    acc += weight[tap_base + l - 1] * bx;
    conv_out[ch] = c * acc;

    // Shift history left by one slot (drop oldest), append `bx` as newest.
    if l >= 2 {
        let mut k = 0usize;
        while k + 2 < l {
            history[k * d + ch] = history[(k + 1) * d + ch];
            k += 1;
        }
        history[(l - 2) * d + ch] = bx;
    }
}

// ── Rotary position embedding ─────────────────────────────────────────────
//
// One thread per (head, freq-index) pair, covering both Q (`n_heads`) and K
// (`n_kv_heads`) with the same kernel — caller launches it once per buffer.
// Mutates `x` in place. `n_rot` is the number of dims rotated per head (may
// be less than `head_dim` — Qwen3.5's `GatedAttention` layers only rotate
// the first `n_rot` dims and leave the rest untouched; pass `n_rot ==
// head_dim` for full rotation).

#[cube(launch)]
pub fn rope(x: &mut Array<f32>, n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, theta: f32) {
    let half = n_rot / 2;
    let idx = ABSOLUTE_POS;
    if idx >= n_heads * half {
        terminate!();
    }
    let h = idx / half;
    let d = idx % half;
    let start = h * head_dim;

    let exponent = 2.0f32 * (d as f32) / (n_rot as f32);
    let freq = (pos as f32) / f32::powf(theta, exponent);
    let cos_val = freq.cos();
    let sin_val = freq.sin();

    let x0 = x[start + d];
    let x1 = x[start + d + half];
    x[start + d] = x0 * cos_val - x1 * sin_val;
    x[start + d + half] = x0 * sin_val + x1 * cos_val;
}

// ── Fused QK-norm + RoPE: one thread per head does the RMSNorm reduction ──
// then the rotation, back to back — one kernel launch (and one `sync`-cost
// round trip) instead of two. `n_rot` may be less than `head_dim` (Qwen3.5's
// `GatedAttention` layers only rotate the first `n_rot` dims).

#[cube(launch)]
pub fn qk_norm_rope(
    x: &mut Array<f32>,
    weight: &Array<f32>,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    n_rot: usize,
    pos: usize,
    theta: f32,
) {
    let h = ABSOLUTE_POS;
    if h >= n_heads {
        terminate!();
    }
    let start = h * head_dim;

    let mut sum_sq = 0.0f32;
    let mut d = 0usize;
    while d < head_dim {
        let v = x[start + d];
        sum_sq += v * v;
        d += 1;
    }
    let rms = f32::sqrt(sum_sq / (head_dim as f32) + eps);

    let mut d2 = 0usize;
    while d2 < head_dim {
        x[start + d2] = weight[d2] * x[start + d2] / rms;
        d2 += 1;
    }

    let half = n_rot / 2;
    let mut d3 = 0usize;
    while d3 < half {
        let exponent = 2.0f32 * (d3 as f32) / (n_rot as f32);
        let freq = (pos as f32) / f32::powf(theta, exponent);
        let cos_val = freq.cos();
        let sin_val = freq.sin();
        let x0 = x[start + d3];
        let x1 = x[start + d3 + half];
        x[start + d3] = x0 * cos_val - x1 * sin_val;
        x[start + d3 + half] = x0 * sin_val + x1 * cos_val;
        d3 += 1;
    }
}

// ── L2-normalize `n_heads` segments of `head_dim`, at two base offsets ────
//
// Qwen3.5's Gated DeltaNet normalizes Q and K (no learned weight, divides by
// `sqrt(sum_sq+eps)` — not a mean-based RMS like `qk_norm`). One thread per
// head, `2*n_heads` threads total — the first half handle the Q range
// (`base_offset`), the second half the K range (`base_offset2`), fusing what
// would otherwise be two separate dispatches into one.

#[cube(launch)]
pub fn l2_norm_heads(
    x: &mut Array<f32>,
    base_offset: usize,
    base_offset2: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= 2 * n_heads {
        terminate!();
    }
    let base = if idx < n_heads { base_offset } else { base_offset2 };
    let h = idx % n_heads;
    let start = base + h * head_dim;

    let mut sum_sq = 0.0f32;
    let mut d = 0usize;
    while d < head_dim {
        let v = x[start + d];
        sum_sq += v * v;
        d += 1;
    }
    let norm = f32::sqrt(sum_sq + eps);

    let mut d2 = 0usize;
    while d2 < head_dim {
        x[start + d2] = x[start + d2] / norm;
        d2 += 1;
    }
}

// ── Gated DeltaNet output gated-RMSNorm, per head, in place: ──────────────
// x[h] = weight * (x[h] / rms(x[h])) * silu(gate[h]), matching the CPU
// path's per-head loop (see `cpu_path.rs`'s gated-RMSNorm comment). `weight`
// is `[head_dim]` and reused across all `n_heads` segments.

#[cube(launch)]
pub fn gdn_gated_rms_norm(
    x: &mut Array<f32>,
    weight: &Array<f32>,
    gate: &Array<f32>,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let h = ABSOLUTE_POS;
    if h >= n_heads {
        terminate!();
    }
    let start = h * head_dim;

    let mut sum_sq = 0.0f32;
    let mut d = 0usize;
    while d < head_dim {
        let v = x[start + d];
        sum_sq += v * v;
        d += 1;
    }
    let rms = f32::sqrt(sum_sq / (head_dim as f32) + eps);

    let mut d2 = 0usize;
    while d2 < head_dim {
        let normed = weight[d2] * x[start + d2] / rms;
        let g = gate[start + d2];
        let silu = g / (1.0f32 + (-g).exp());
        x[start + d2] = normed * silu;
        d2 += 1;
    }
}

// ── Gated DeltaNet gate/decay: beta = sigmoid(beta_raw), ──────────────────
// decay = exp(ssm_a * softplus(alpha_raw + dt_bias)), both in place.

#[cube(launch)]
pub fn gdn_gate_decay(
    beta_raw: &mut Array<f32>,
    alpha_raw: &mut Array<f32>,
    ssm_a: &Array<f32>,
    dt_bias: &Array<f32>,
) {
    let h = ABSOLUTE_POS;
    if h >= beta_raw.len() {
        terminate!();
    }
    let braw = beta_raw[h];
    beta_raw[h] = 1.0f32 / (1.0f32 + (-braw).exp());

    let x = alpha_raw[h] + dt_bias[h];
    let softplus = if x > 20.0f32 { x } else { (1.0f32 + x.exp()).ln() };
    alpha_raw[h] = (ssm_a[h] * softplus).exp();
}

// ── Gated DeltaNet causal depthwise conv1d + SiLU ─────────────────────────
//
// Same recurrent-history scheme as `short_conv`, but without the `B*x`/`C`
// gating — runs directly on the raw `wqkv` projection output. One thread
// per channel; mutates the persistent `history` buffer in place.

#[cube(launch)]
pub fn causal_conv1d_silu(
    input: &Array<f32>,
    weight: &Array<f32>,
    history: &mut Array<f32>,
    output: &mut Array<f32>,
    kernel: usize,
) {
    let ch = ABSOLUTE_POS;
    let d = output.len();
    if ch >= d {
        terminate!();
    }

    let tap_base = ch * kernel;
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k + 1 < kernel {
        acc += weight[tap_base + k] * history[k * d + ch];
        k += 1;
    }
    acc += weight[tap_base + kernel - 1] * input[ch];
    output[ch] = acc / (1.0f32 + (-acc).exp());

    // Advance the recurrent history: drop the oldest slot, append current input.
    if kernel >= 2 {
        let mut k2 = 0usize;
        while k2 + 2 < kernel {
            history[k2 * d + ch] = history[(k2 + 1) * d + ch];
            k2 += 1;
        }
        history[(kernel - 2) * d + ch] = input[ch];
    }
}

// ── Gated DeltaNet per-head delta-rule recurrence ─────────────────────────
//
// One workgroup per v-head, one thread per output column `b`. Each thread
// exclusively owns column `b` of that head's `head_k_dim x head_v_dim`
// persistent state matrix, so there's no cross-thread synchronization at
// all: vpred (reduction over `a`) → delta → state update (this thread's
// column only) → output (reduction over `a`, using the just-updated state).
// Q/K heads are tile-repeated (cyclic) up to the value-head count, matching
// `ggml_repeat_4d` (see the CPU reference in `gated_delta_net`).

#[cube(launch)]
pub fn gdn_recurrence(
    state: &mut Array<f32>,
    conv_out: &Array<f32>,
    beta: &Array<f32>,
    decay: &Array<f32>,
    out: &mut Array<f32>,
    n_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    scale: f32,
) {
    let h = CUBE_POS_X as usize;
    let b = UNIT_POS as usize;
    if b >= head_v_dim {
        terminate!();
    }

    let kh = h % n_k_heads;
    let q_h_start = kh * head_k_dim;
    let k_h_start = key_dim + kh * head_k_dim;
    let v_h_start = 2 * key_dim + h * head_v_dim;
    let state_base = h * head_k_dim * head_v_dim;

    let beta_h = beta[h];
    let decay_h = decay[h];

    let mut vpred = 0.0f32;
    let mut a = 0usize;
    while a < head_k_dim {
        vpred += state[state_base + a * head_v_dim + b] * conv_out[k_h_start + a];
        a += 1;
    }
    let delta_b = beta_h * (conv_out[v_h_start + b] - vpred);

    let mut a2 = 0usize;
    while a2 < head_k_dim {
        let idx = state_base + a2 * head_v_dim + b;
        let k_val = conv_out[k_h_start + a2];
        state[idx] = decay_h * state[idx] + k_val * delta_b;
        a2 += 1;
    }

    let mut acc = 0.0f32;
    let mut a3 = 0usize;
    while a3 < head_k_dim {
        acc += scale * conv_out[q_h_start + a3] * state[state_base + a3 * head_v_dim + b];
        a3 += 1;
    }
    out[h * head_v_dim + b] = acc;
}

// ── KV cache write ──────────────────────────────────────────────────────
//
// Appends the current token's K/V (already RoPE'd) into the persistent,
// GPU-resident per-layer cache at slot `pos`. One thread per channel.

#[cube(launch)]
pub fn kv_cache_write(
    k_cache: &mut Array<f32>,
    v_cache: &mut Array<f32>,
    new_k: &Array<f32>,
    new_v: &Array<f32>,
    pos: usize,
) {
    let i = ABSOLUTE_POS;
    let kv_dim = new_k.len();
    if i >= kv_dim {
        terminate!();
    }
    k_cache[pos * kv_dim + i] = new_k[i];
    v_cache[pos * kv_dim + i] = new_v[i];
}

// ── Causal GQA attention ──────────────────────────────────────────────────
//
// Split into three kernels, each parallel over a much wider index space than
// a single "one thread per head" kernel would allow — with only 8-32 heads
// typical, that single-kernel design left almost the whole GPU idle (RDNA
// wavefronts are 32-wide; 8-16 heads doesn't even fill one), and every
// dot-product / weighted-sum loop over `head_dim * pos` ran serially inside
// that one thread. Splitting the head-independent inner dimensions (KV
// position, output dim) out into their own thread axis instead gives
// `n_heads * pos` and `n_heads * head_dim` threads respectively.
//
// 1. `attention_scores`  — one thread per (head, kv-position): scaled dot
//    product, written once into `scores[head, max_seq]`.
// 2. `attention_softmax` — one thread per head: numerically-stable softmax
//    reduction over `scores[head, ..=pos]`, writing unnormalized weights to
//    `weights[head, max_seq]` and the normalizer to `sums[head]`. Still
//    O(pos) per thread (no `head_dim` factor), so cheap even serial.
// 3. `attention_output`  — one thread per (head, output-dim): weighted sum
//    of V using the cached weights, normalized by `sums[head]` on write.
//
// Running max uses a ReLU-style `+=` update (`max_score += max(delta, 0)`)
// rather than a plain reassignment: this cubecl version's `#[cube]` macro
// only handles `var = ...` correctly when the RHS is a method call — a bare
// reassignment (even pure arithmetic, even a free-function call like
// `max(a, b)`) fails to typecheck post-expansion. Compound assignment
// (`+=`) doesn't have this restriction, so every mutable scalar here is
// updated that way (or via `RuntimeCell`, which the macro's error message
// itself points at).

#[cube(launch)]
pub fn attention_scores(
    q: &Array<f32>,
    kv_cache_k: &Array<f32>,
    scores: &mut Array<f32>,
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq: usize,
) {
    let idx = ABSOLUTE_POS;
    let seq_len = pos + 1;
    if idx >= n_heads * seq_len {
        terminate!();
    }
    let h = idx / seq_len;
    let t = idx % seq_len;

    let group = n_heads / n_kv_heads;
    let kv_head = h / group;
    let q_base = h * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let kv_base = kv_head * head_dim;
    let scale = 1.0f32 / f32::sqrt(head_dim as f32);

    let k_off = t * kv_dim + kv_base;
    let mut dot = 0.0f32;
    let mut d = 0usize;
    while d < head_dim {
        dot += q[q_base + d] * kv_cache_k[k_off + d];
        d += 1;
    }
    scores[h * max_seq + t] = dot * scale;
}

#[cube(launch)]
pub fn attention_softmax(
    scores: &Array<f32>,
    weights: &mut Array<f32>,
    sums: &mut Array<f32>,
    pos: usize,
    n_heads: usize,
    max_seq: usize,
) {
    let h = ABSOLUTE_POS;
    if h >= n_heads {
        terminate!();
    }

    let max_cell = RuntimeCell::<f32>::new(-1.0e30f32);
    let mut t = 0usize;
    while t <= pos {
        max_cell.store(max(max_cell.read(), scores[h * max_seq + t]));
        t += 1;
    }
    let max_score = max_cell.read();

    let mut sum_exp = 0.0f32;
    let mut t2 = 0usize;
    while t2 <= pos {
        let w = (scores[h * max_seq + t2] - max_score).exp();
        weights[h * max_seq + t2] = w;
        sum_exp += w;
        t2 += 1;
    }
    sums[h] = sum_exp;
}

#[cube(launch)]
pub fn attention_output(
    kv_cache_v: &Array<f32>,
    weights: &Array<f32>,
    sums: &Array<f32>,
    out: &mut Array<f32>,
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n_heads * head_dim {
        terminate!();
    }
    let h = idx / head_dim;
    let d = idx % head_dim;

    let group = n_heads / n_kv_heads;
    let kv_head = h / group;
    let kv_dim = n_kv_heads * head_dim;
    let kv_base = kv_head * head_dim;

    let mut acc = 0.0f32;
    let mut t = 0usize;
    while t <= pos {
        let v_off = t * kv_dim + kv_base;
        acc += weights[h * max_seq + t] * kv_cache_v[v_off + d];
        t += 1;
    }
    out[h * head_dim + d] = acc / sums[h];
}

// ── Helpers (no u8 anywhere — WGSL compatible) ────────────────────────

/// Read a byte from a u32-packed array. Returns u32 in range 0..255.
#[cube]
fn read_byte_u32(w: &Array<u32>, byte_offset: usize) -> u32 {
    (w[byte_offset / 4] >> (8u32 * (byte_offset % 4) as u32)) & 0xFFu32
}

/// Read f16 from two consecutive bytes in a u32-packed array → f32.
#[cube]
fn read_f16(w: &Array<u32>, byte_offset: usize) -> f32 {
    let lo = read_byte_u32(w, byte_offset);
    let hi = read_byte_u32(w, byte_offset + 1);
    f16_u32_to_f32(lo, hi)
}

/// f16 → f32: takes two u32 values (each 0..255) representing the lo/hi bytes.
#[cube]
fn f16_u32_to_f32(lo: u32, hi: u32) -> f32 {
    let bits: u32 = (hi << 8u32) | lo;
    let sign = (bits >> 15u32) & 1u32;
    let exp = (bits >> 10u32) & 0x1Fu32;
    let mant = bits & 0x3FFu32;

    if exp == 0u32 {
        let sign_factor: f32 = 1.0f32 - 2.0f32 * (sign as f32);
        sign_factor * ((mant as f32) / 1024.0f32) * f32::powf(2.0f32, -14.0f32)
    } else {
        let sign_factor: f32 = 1.0f32 - 2.0f32 * (sign as f32);
        sign_factor
            * (1.0f32 + (mant as f32) / 1024.0f32)
            * f32::powf(2.0f32, (exp as f32) - 15.0f32)
    }
}

/// Read i8 (sign-extended) from a u32-packed array. Returns i32.
#[cube]
fn read_i8_i32(w: &Array<u32>, byte_offset: usize) -> i32 {
    let b = read_byte_u32(w, byte_offset);
    // Sign-extend: if bit 7 is set, fill upper bits with 1s
    if (b & 0x80u32) != 0u32 {
        (b | 0xFFFFFF00u32) as i32
    } else {
        b as i32
    }
}

/// Q4_K scale/min unpack. All u32.
#[cube]
fn get_scale_min_k4_u32(j: u32, s: &Array<u32>, scale_byte_off: usize) -> (u32, u32) {
    if j < 4u32 {
        let sc = read_byte_u32(s, scale_byte_off + j as usize) & 0x3Fu32;
        let m = read_byte_u32(s, scale_byte_off + 4 + j as usize) & 0x3Fu32;
        (sc, m)
    } else {
        let s_lo = read_byte_u32(s, scale_byte_off + 4 + j as usize);
        let s_hi = read_byte_u32(s, scale_byte_off + (j - 4u32) as usize);
        let m_hi = read_byte_u32(s, scale_byte_off + j as usize);
        let sc = (s_lo & 0xFu32) | ((s_hi >> 6u32) & 0x3u32) << 4u32;
        let m = (s_lo >> 4u32) | ((m_hi >> 6u32) & 0x3u32) << 4u32;
        (sc, m)
    }
}

// ── Dequant type ids (match GgmlType discriminants in src/types.rs) ──────

pub const DEQUANT_Q8_0: u32 = 8;
pub const DEQUANT_Q4_K: u32 = 12;
pub const DEQUANT_Q5_K: u32 = 13;
pub const DEQUANT_Q6_K: u32 = 14;

// ── Consolidated matmul with workgroup reduction ──────────────────────────
//
// Launch: one cube (workgroup) per output row, 64 threads per cube.
// Each thread computes a partial dot product (striding through weight blocks
// by 64), then 64 partials are reduced to one scalar via a shared-memory
// tree reduction. Thread 0 writes the final value.
//
// Parallelism vs the old 1-thread-per-row design:
//   Q8_0 at d=2048  → 64 blocks/row, 1 block/thread  — fully occupied
//   Q4_K at d=2048  → 64 sub-units/row (8 per block × 8 blocks), 1/thread
//   Q6_K at d=2048  → 64 l-slots/row  (32 l-values × 2 halves × 8 blocks
//                       would be 512; each thread handles all blocks for its
//                       fixed (h,l) — see partial_q6_k below)

#[cube(launch)]
pub fn matmul_dequant_wgpu(
    w: &Array<u32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    dtype: u32,
    in_dim: usize,
    row_u32s: usize,
    // grid_x is the number of workgroups in the X dimension of the 2-D dispatch.
    // row = CUBE_POS_Y * grid_x + CUBE_POS_X so we support out_dim > 65535.
    grid_x: u32,
) {
    let row = (CUBE_POS_Y * grid_x + CUBE_POS_X) as usize;
    let lane = UNIT_POS as usize;

    if row >= out.len() {
        terminate!();
    }

    let mut partial = 0.0f32;
    if dtype == DEQUANT_Q8_0 {
        partial = partial_q8_0(w, x, row, lane, in_dim, row_u32s);
    } else if dtype == DEQUANT_Q4_K {
        partial = partial_q4_k(w, x, row, lane, in_dim, row_u32s);
    } else if dtype == DEQUANT_Q5_K {
        partial = partial_q5_k(w, x, row, lane, in_dim, row_u32s);
    } else if dtype == DEQUANT_Q6_K {
        partial = partial_q6_k(w, x, row, lane, in_dim, row_u32s);
    }

    // Tree reduction: 64 → 32 → 16 → 8 → 4 → 2 → 1
    let mut smem = SharedMemory::<f32>::new(64usize);
    smem[lane] = partial;
    sync_cube();

    if lane < 32 {
        smem[lane] += smem[lane + 32];
    }
    sync_cube();
    if lane < 16 {
        smem[lane] += smem[lane + 16];
    }
    sync_cube();
    if lane < 8 {
        smem[lane] += smem[lane + 8];
    }
    sync_cube();
    if lane < 4 {
        smem[lane] += smem[lane + 4];
    }
    sync_cube();
    if lane < 2 {
        smem[lane] += smem[lane + 2];
    }
    sync_cube();

    if lane == 0 {
        out[row] = smem[0] + smem[1];
    }
}

// ── Q8_0 partial ─────────────────────────────────────────────────────────
//
// Thread `lane` owns blocks lane, lane+64, lane+128, … (stride 64).
// Each block = 34 bytes: 2B f16 scale + 32B i8 values.

#[cube]
fn partial_q8_0(
    w: &Array<u32>,
    x: &Array<f32>,
    row: usize,
    lane: usize,
    in_dim: usize,
    row_u32s: usize,
) -> f32 {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = (in_dim + 31) / 32;
    let mut sum = 0.0f32;
    let mut b = lane;
    while b < n_blocks {
        let byte_off = w_byte_base + b * 34;
        let scale = read_f16(w, byte_off);
        let block_start = b * 32;
        let mut dot = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let idx = block_start + i;
            if idx < in_dim {
                let q = read_i8_i32(w, byte_off + 2 + i) as f32;
                dot += q * x[idx];
            }
            i += 1;
        }
        sum += scale * dot;
        b += 64;
    }
    sum
}

// ── Q4_K partial ─────────────────────────────────────────────────────────
//
// Each 256-element super-block has 8 sub-units of 32 elements each
// (4 groups × 2 nibble-halves). Thread `lane` owns sub-units
// lane, lane+64, lane+128, … across all super-blocks.
// At d=2048 (8 blocks × 8 sub-units = 64 total) every thread owns exactly 1.

#[cube]
fn partial_q4_k(
    w: &Array<u32>,
    x: &Array<f32>,
    row: usize,
    lane: usize,
    in_dim: usize,
    row_u32s: usize,
) -> f32 {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = ((in_dim as u32) + 255u32) / 256u32;
    let n_units = (n_blocks * 8u32) as usize;
    let mut sum = 0.0f32;
    let mut u = lane;
    while u < n_units {
        let block = (u / 8) as u32;
        let is = (u % 8) as u32;
        let group = is / 2u32;

        let byte_off = w_byte_base + (block * 144u32) as usize;
        let d = read_f16(w, byte_off);
        let dmin = read_f16(w, byte_off + 2);

        let (sc, m) = get_scale_min_k4_u32(is, w, byte_off + 4);
        let d_sc = d * (sc as f32);
        let dmin_m = dmin * (m as f32);

        let qs_byte_off = byte_off + 16 + (group * 32u32) as usize;
        let group_val_base = block * 256u32 + group * 64u32;

        let low = is % 2u32 == 0u32;

        // `if low { A } else { B }` must select a complete final value here
        // (idx, nibble), not an intermediate scalar (e.g. a 0/4 shift amount)
        // combined with further arithmetic afterward — cubecl's wgpu codegen
        // has been observed to silently always take the `else` branch when a
        // runtime if/else's result feeds into more arithmetic before use.
        // See git history for the synthetic single-block repro that caught this.
        let mut l = 0usize;
        while l < 32 {
            let idx = if low {
                group_val_base as usize + l
            } else {
                group_val_base as usize + 32 + l
            };
            if idx < in_dim {
                let qb = read_byte_u32(w, qs_byte_off + l);
                let nibble = if low { qb & 0xFu32 } else { (qb >> 4u32) & 0xFu32 };
                sum += (d_sc * (nibble as f32) - dmin_m) * x[idx];
            }
            l += 1;
        }
        u += 64;
    }
    sum
}

// ── Q5_K partial ─────────────────────────────────────────────────────────
//
// Block layout (176 bytes, 256 values): d(2) dmin(2) scales(12) qh(32) qs(128).
// Same 8-sub-units-per-block structure as Q4_K (thread `lane` owns sub-units
// lane, lane+64, …), but each 32-value sub-unit `is` also needs one high bit
// per element, taken from bit position `is` of the shared 32-byte `qh` array
// (re-read, unshifted by sub-unit — only the bit index changes).

#[cube]
fn partial_q5_k(
    w: &Array<u32>,
    x: &Array<f32>,
    row: usize,
    lane: usize,
    in_dim: usize,
    row_u32s: usize,
) -> f32 {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = ((in_dim as u32) + 255u32) / 256u32;
    let n_units = (n_blocks * 8u32) as usize;
    let mut sum = 0.0f32;
    let mut u = lane;
    while u < n_units {
        let block = (u / 8) as u32;
        let is = (u % 8) as u32;
        let low_nibble = is % 2u32 == 0u32;

        let byte_off = w_byte_base + (block * 176u32) as usize;
        let d = read_f16(w, byte_off);
        let dmin = read_f16(w, byte_off + 2);

        let (sc, m) = get_scale_min_k4_u32(is, w, byte_off + 4);
        let d_sc = d * (sc as f32);
        let dmin_m = dmin * (m as f32);

        let qs_byte_off = byte_off + 48 + ((is / 2u32) * 32u32) as usize;
        let qh_byte_off = byte_off + 16;
        let val_base = (block * 256u32 + is * 32u32) as usize;

        // `qs_byte_off`/`qh_byte_off` are always u32-aligned (built from
        // multiples of 4), so each word covers 4 consecutive elements —
        // one array read + 4 shifted extracts instead of 4 separate reads.
        let mut l = 0usize;
        while l < 32 {
            let qs_word = w[(qs_byte_off + l) / 4];
            let qh_word = w[(qh_byte_off + l) / 4];
            let mut sub = 0usize;
            while sub < 4 {
                let idx = val_base + l + sub;
                if idx < in_dim {
                    let shift = 8u32 * (sub as u32);
                    let qb = (qs_word >> shift) & 0xFFu32;
                    let nibble = if low_nibble { qb & 0xFu32 } else { (qb >> 4u32) & 0xFu32 };
                    let hb = (qh_word >> shift) & 0xFFu32;
                    let hi_bit = (hb >> is) & 1u32;
                    let quant = nibble | (hi_bit << 4u32);
                    sum += (d_sc * (quant as f32) - dmin_m) * x[idx];
                }
                sub += 1;
            }
            l += 4;
        }
        u += 64;
    }
    sum
}

// ── Q6_K partial ─────────────────────────────────────────────────────────
//
// Thread `lane` owns a fixed (h, l) slot — h = lane/32, l = lane%32 —
// and iterates over all 256-element blocks. Each (h, l) pair processes
// 4 elements per block (at offsets l, l+32, l+64, l+96 within the half).
// All 64 threads are active for any valid in_dim.

#[cube]
fn partial_q6_k(
    w: &Array<u32>,
    x: &Array<f32>,
    row: usize,
    lane: usize,
    in_dim: usize,
    row_u32s: usize,
) -> f32 {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = (in_dim + 255) / 256;
    let h = lane / 32;
    let l = lane % 32;
    let is = l / 16;

    let mut sum = 0.0f32;
    let mut block = 0usize;
    while block < n_blocks {
        let byte_off = w_byte_base + block * 210;
        let d = read_f16(w, byte_off + 208);

        let ql_off = byte_off + h * 64;
        let qh_off = byte_off + 128 + h * 32;
        let sc_off = byte_off + 192 + h * 8;
        let y_off = block * 256 + h * 128;

        let sc0 = read_i8_i32(w, sc_off + is) as f32;
        let sc2 = read_i8_i32(w, sc_off + is + 2) as f32;
        let sc4 = read_i8_i32(w, sc_off + is + 4) as f32;
        let sc6 = read_i8_i32(w, sc_off + is + 6) as f32;

        let ql0 = read_byte_u32(w, ql_off + l);
        let ql1 = read_byte_u32(w, ql_off + 32 + l);
        let qh_byte = read_byte_u32(w, qh_off + l);

        let q1 = ((ql0 & 0xFu32) as i32 | ((qh_byte & 3u32) as i32) << 4) - 32;
        let q2 = ((ql1 & 0xFu32) as i32 | (((qh_byte >> 2u32) & 3u32) as i32) << 4) - 32;
        let q3 = (((ql0 >> 4u32) & 0xFu32) as i32 | (((qh_byte >> 4u32) & 3u32) as i32) << 4) - 32;
        let q4 = (((ql1 >> 4u32) & 0xFu32) as i32 | (((qh_byte >> 6u32) & 3u32) as i32) << 4) - 32;

        let idx0 = y_off + l;
        if idx0 < in_dim {
            sum += d * sc0 * (q1 as f32) * x[idx0];
        }
        let idx1 = idx0 + 32;
        if idx1 < in_dim {
            sum += d * sc2 * (q2 as f32) * x[idx1];
        }
        let idx2 = idx0 + 64;
        if idx2 < in_dim {
            sum += d * sc4 * (q3 as f32) * x[idx2];
        }
        let idx3 = idx0 + 96;
        if idx3 < in_dim {
            sum += d * sc6 * (q4 as f32) * x[idx3];
        }

        block += 1;
    }
    sum
}
