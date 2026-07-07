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

// ── Residual add: out[i] = a[i] + b[i] ────────────────────────────────────

#[cube(launch)]
pub fn residual_add(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= out.len() {
        terminate!();
    }
    out[i] = a[i] + b[i];
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
pub const DEQUANT_Q6_K: u32 = 14;

// ── Consolidated matmul: one kernel, runtime dtype branch ────────────────
//
// `in_dim` and `row_u32s` are runtime (dimensional) values, not comptime —
// every distinct tensor shape used to force a brand new kernel compilation
// (hundreds of them across a real model). `dtype` is also runtime: the
// branch is uniform across all threads in a given launch (same weight
// tensor => same dtype for every row), so it costs nothing once compiled.

#[cube(launch)]
pub fn matmul_dequant_wgpu(
    w: &Array<u32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    dtype: u32,
    in_dim: usize,
    row_u32s: usize,
) {
    let row = ABSOLUTE_POS;
    if row >= out.len() {
        terminate!();
    }

    if dtype == DEQUANT_Q8_0 {
        matmul_row_q8_0_wgpu(w, x, out, row, in_dim, row_u32s);
    } else if dtype == DEQUANT_Q4_K {
        matmul_row_q4_k_wgpu(w, x, out, row, in_dim, row_u32s);
    } else if dtype == DEQUANT_Q6_K {
        matmul_row_q6_k_wgpu(w, x, out, row, in_dim, row_u32s);
    }
}

#[cube]
fn matmul_row_q8_0_wgpu(
    w: &Array<u32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_u32s: usize,
) {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = (in_dim + 31) / 32;
    let mut sum = 0.0f32;

    let mut b = 0u32;
    while b < (n_blocks as u32) {
        let byte_off = w_byte_base + (b * 34u32) as usize;
        let scale = read_f16(w, byte_off);
        let block_start = (b * 32u32) as usize;
        let mut block_dot = 0.0f32;
        let mut i = block_start;
        let limit = block_start + 32;
        while i < limit {
            if i < in_dim {
                let q = read_i8_i32(w, byte_off + 2 + (i - block_start)) as f32;
                block_dot += q * x[i];
            }
            i += 1;
        }
        sum += scale * block_dot;
        b += 1u32;
    }
    out[row] = sum;
}

// ── Q4_K ─────────────────────────────────────────────────────────────────

#[cube]
fn matmul_row_q4_k_wgpu(
    w: &Array<u32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_u32s: usize,
) {
    let w_byte_base = row * row_u32s * 4;
    let mut sum = 0.0f32;

    let mut block = 0u32;
    while block < ((in_dim as u32) + 255u32) / 256u32 {
        let byte_off = w_byte_base + (block * 144u32) as usize;

        let d = read_f16(w, byte_off);
        let dmin = read_f16(w, byte_off + 2);

        let mut is: u32 = 0u32;
        let mut group: u32 = 0u32;
        while group < 4u32 {
            let qs_byte_off = byte_off + 16 + (group * 32u32) as usize;

            let (sc0, m0) = get_scale_min_k4_u32(is, w, byte_off + 4);
            let d1 = d * (sc0 as f32);
            let m1 = dmin * (m0 as f32);
            let (sc1, m1_s) = get_scale_min_k4_u32(is + 1u32, w, byte_off + 4);
            let d2 = d * (sc1 as f32);
            let m2 = dmin * (m1_s as f32);

            let val_base: usize = (block * 256u32 + group * 64u32) as usize;

            let mut l: u32 = 0u32;
            while l < 32u32 {
                let idx: usize = val_base + (l as usize);
                if idx < in_dim {
                    let qb = read_byte_u32(w, qs_byte_off + (l as usize));
                    sum += (d1 * (qb & 0xFu32) as f32 - m1) * x[idx];
                }
                l += 1u32;
            }
            let mut l: u32 = 0u32;
            while l < 32u32 {
                let idx: usize = val_base + 32 + (l as usize);
                if idx < in_dim {
                    let qb = read_byte_u32(w, qs_byte_off + (l as usize));
                    sum += (d2 * ((qb >> 4u32) & 0xFu32) as f32 - m2) * x[idx];
                }
                l += 1u32;
            }
            is += 2u32;
            group += 1u32;
        }
        block += 1u32;
    }
    out[row] = sum;
}

// ── Q6_K ─────────────────────────────────────────────────────────────────

#[cube]
fn matmul_row_q6_k_wgpu(
    w: &Array<u32>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_u32s: usize,
) {
    let w_byte_base = row * row_u32s * 4;
    let n_blocks = (in_dim + 255) / 256;
    let mut sum = 0.0f32;

    let mut block = 0u32;
    while block < (n_blocks as u32) {
        let byte_off = w_byte_base + (block * 210u32) as usize;
        let d = read_f16(w, byte_off + 208);

        let mut h: u32 = 0u32;
        while h < 2u32 {
            let ql_off = byte_off + (h * 64u32) as usize;
            let qh_off = byte_off + 128 + (h * 32u32) as usize;
            let sc_off = byte_off + 192 + (h * 8u32) as usize;
            let y_off = (block * 256u32 + h * 128u32) as usize;

            let mut l: u32 = 0u32;
            while l < 32u32 {
                let is = (l / 16u32) as usize;
                let sc0 = read_i8_i32(w, sc_off + is) as f32;
                let sc2 = read_i8_i32(w, sc_off + is + 2) as f32;
                let sc4 = read_i8_i32(w, sc_off + is + 4) as f32;
                let sc6 = read_i8_i32(w, sc_off + is + 6) as f32;

                let ql0 = read_byte_u32(w, ql_off + (l as usize));
                let ql1 = read_byte_u32(w, ql_off + 32 + (l as usize));
                let qh_byte = read_byte_u32(w, qh_off + (l as usize));

                let q1 = ((ql0 & 0xFu32) as i32 | ((qh_byte & 3u32) as i32) << 4) - 32;
                let q2 = ((ql1 & 0xFu32) as i32 | (((qh_byte >> 2u32) & 3u32) as i32) << 4) - 32;
                let q3 = (((ql0 >> 4u32) & 0xFu32) as i32
                    | (((qh_byte >> 4u32) & 3u32) as i32) << 4)
                    - 32;
                let q4 = (((ql1 >> 4u32) & 0xFu32) as i32
                    | (((qh_byte >> 6u32) & 3u32) as i32) << 4)
                    - 32;

                let idx0 = y_off + (l as usize);
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

                l += 1u32;
            }
            h += 1u32;
        }
        block += 1u32;
    }
    out[row] = sum;
}
