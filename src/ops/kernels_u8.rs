//! u8-native kernels for HIP, CUDA, and CPU backends.
//!
//! These use `Array<u8>` directly — no packing overhead.
//! Same numerical results as the wgpu kernels.

use cubecl::prelude::*;

// ── SwiGLU combine: out[i] = silu(gate[i]) * up[i] ────────────────────────
//
// Elementwise — no quantization involved, identical across the u8-native and
// u32-packed kernel sets (see kernels_wgpu.rs).

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
// Single-threaded on purpose — see kernels_wgpu.rs.

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

#[cube]
pub(crate) fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = ((hi as u32) << 8u32) | (lo as u32);
    let sign = ((bits >> 15u32) & 1u32) as f32;
    let exp = ((bits >> 10u32) & 0x1Fu32) as f32;
    let mant = (bits & 0x3FFu32) as f32;
    if exp == 0.0f32 {
        let sign_factor = 1.0f32 - 2.0f32 * sign;
        sign_factor * (mant / 1024.0f32) * f32::powf(2.0f32, -14.0f32)
    } else {
        let sign_factor = 1.0f32 - 2.0f32 * sign;
        sign_factor * (1.0f32 + mant / 1024.0f32) * f32::powf(2.0f32, exp - 15.0f32)
    }
}

#[cube]
pub(crate) fn get_scale_min_k4(
    j: u32,
    s0: u8,
    s1: u8,
    s2: u8,
    s3: u8,
    s4: u8,
    s5: u8,
    s6: u8,
    s7: u8,
    s8: u8,
    s9: u8,
    s10: u8,
    s11: u8,
) -> (u32, u32) {
    if j < 4u32 {
        if j == 0u32 {
            ((s0 & 0x3Fu8) as u32, (s4 & 0x3Fu8) as u32)
        } else if j == 1u32 {
            ((s1 & 0x3Fu8) as u32, (s5 & 0x3Fu8) as u32)
        } else if j == 2u32 {
            ((s2 & 0x3Fu8) as u32, (s6 & 0x3Fu8) as u32)
        } else {
            ((s3 & 0x3Fu8) as u32, (s7 & 0x3Fu8) as u32)
        }
    } else {
        if j == 4u32 {
            let sc = ((s8 & 0xFu8) as u32) | (((s0 >> 6u32) & 0x3u8) as u32) << 4u32;
            let m = ((s8 >> 4u32) as u32) | (((s4 >> 6u32) & 0x3u8) as u32) << 4u32;
            (sc, m)
        } else if j == 5u32 {
            let sc = ((s9 & 0xFu8) as u32) | (((s1 >> 6u32) & 0x3u8) as u32) << 4u32;
            let m = ((s9 >> 4u32) as u32) | (((s5 >> 6u32) & 0x3u8) as u32) << 4u32;
            (sc, m)
        } else if j == 6u32 {
            let sc = ((s10 & 0xFu8) as u32) | (((s2 >> 6u32) & 0x3u8) as u32) << 4u32;
            let m = ((s10 >> 4u32) as u32) | (((s6 >> 6u32) & 0x3u8) as u32) << 4u32;
            (sc, m)
        } else {
            let sc = ((s11 & 0xFu8) as u32) | (((s3 >> 6u32) & 0x3u8) as u32) << 4u32;
            let m = ((s11 >> 4u32) as u32) | (((s7 >> 6u32) & 0x3u8) as u32) << 4u32;
            (sc, m)
        }
    }
}

// ── Dequant type ids (match GgmlType discriminants in src/types.rs) ──────

pub const DEQUANT_Q8_0: u32 = 8;
pub const DEQUANT_Q4_K: u32 = 12;
pub const DEQUANT_Q6_K: u32 = 14;

// ── Consolidated matmul: one kernel, runtime dtype branch ────────────────
//
// `in_dim` and `row_bytes` are runtime (dimensional) values, not comptime —
// every distinct tensor shape used to force a brand new kernel compilation
// (hundreds of them across a real model). `dtype` is also runtime: the
// branch is uniform across all threads in a given launch (same weight
// tensor => same dtype for every row), so it costs nothing once compiled.

#[cube(launch)]
pub fn matmul_dequant(
    w: &Array<u8>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    dtype: u32,
    in_dim: usize,
    row_bytes: usize,
) {
    let row = ABSOLUTE_POS;
    if row >= out.len() {
        terminate!();
    }

    if dtype == DEQUANT_Q8_0 {
        matmul_row_q8_0(w, x, out, row, in_dim, row_bytes);
    } else if dtype == DEQUANT_Q4_K {
        matmul_row_q4_k(w, x, out, row, in_dim, row_bytes);
    } else if dtype == DEQUANT_Q6_K {
        matmul_row_q6_k(w, x, out, row, in_dim, row_bytes);
    }
}

#[cube]
fn matmul_row_q8_0(
    w: &Array<u8>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_bytes: usize,
) {
    let w_base = row * row_bytes;
    let n_blocks = (in_dim + 31) / 32;
    let mut sum = 0.0f32;

    let mut b = 0u32;
    while b < (n_blocks as u32) {
        let base = w_base + (b * 34u32) as usize;
        let scale = f16_to_f32(w[base], w[base + 1]);
        let block_start = (b * 32u32) as usize;
        let mut block_dot = 0.0f32;
        let mut i = block_start;
        let limit = block_start + 32;
        while i < limit {
            if i < in_dim {
                let q = w[base + 2 + (i - block_start)] as i8 as f32;
                block_dot += q * x[i];
            }
            i += 1;
        }
        sum += scale * block_dot;
        b += 1u32;
    }
    out[row] = sum;
}

#[cube]
fn matmul_row_q4_k(
    w: &Array<u8>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_bytes: usize,
) {
    let w_base = row * row_bytes;
    let mut sum = 0.0f32;

    let mut block = 0u32;
    while block < ((in_dim as u32) + 255u32) / 256u32 {
        let block_base = w_base + (block * 144u32) as usize;
        let d = f16_to_f32(w[block_base], w[block_base + 1]);
        let dmin = f16_to_f32(w[block_base + 2], w[block_base + 3]);
        let s0 = w[block_base + 4];
        let s1 = w[block_base + 5];
        let s2 = w[block_base + 6];
        let s3 = w[block_base + 7];
        let s4 = w[block_base + 8];
        let s5 = w[block_base + 9];
        let s6 = w[block_base + 10];
        let s7 = w[block_base + 11];
        let s8 = w[block_base + 12];
        let s9 = w[block_base + 13];
        let s10 = w[block_base + 14];
        let s11 = w[block_base + 15];
        let qs_base = block_base + 16;

        let mut is: u32 = 0u32;
        let mut group: u32 = 0u32;
        while group < 4u32 {
            let qs_offset: usize = (group * 32u32) as usize;
            let (sc0, m0) = get_scale_min_k4(is, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11);
            let d1 = d * (sc0 as f32);
            let m1 = dmin * (m0 as f32);
            let (sc1, m1_s) =
                get_scale_min_k4(is + 1u32, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11);
            let d2 = d * (sc1 as f32);
            let m2 = dmin * (m1_s as f32);
            let val_base: usize = (block * 256u32 + group * 64u32) as usize;

            let mut l: u32 = 0u32;
            while l < 32u32 {
                let idx: usize = val_base + (l as usize);
                if idx < in_dim {
                    sum +=
                        (d1 * (w[qs_base + qs_offset + (l as usize)] & 0xFu8) as f32 - m1) * x[idx];
                }
                l += 1u32;
            }
            let mut l: u32 = 0u32;
            while l < 32u32 {
                let idx: usize = val_base + 32 + (l as usize);
                if idx < in_dim {
                    sum +=
                        (d2 * (w[qs_base + qs_offset + (l as usize)] >> 4u32) as f32 - m2) * x[idx];
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
fn matmul_row_q6_k(
    w: &Array<u8>,
    x: &Array<f32>,
    out: &mut Array<f32>,
    row: usize,
    in_dim: usize,
    row_bytes: usize,
) {
    let w_base = row * row_bytes;
    let n_blocks = (in_dim + 255) / 256;
    let mut sum = 0.0f32;

    let mut block = 0u32;
    while block < (n_blocks as u32) {
        let base = w_base + (block * 210u32) as usize;
        let d = f16_to_f32(w[base + 208], w[base + 209]);
        let mut h: u32 = 0u32;
        while h < 2u32 {
            let ql_off = base + (h * 64u32) as usize;
            let qh_off = base + 128 + (h * 32u32) as usize;
            let sc_off = base + 192 + (h * 8u32) as usize;
            let y_off = (block * 256u32 + h * 128u32) as usize;

            let mut l: u32 = 0u32;
            while l < 32u32 {
                let is = (l / 16u32) as usize;
                let sc0 = w[sc_off + is] as i8 as f32;
                let sc2 = w[sc_off + is + 2] as i8 as f32;
                let sc4 = w[sc_off + is + 4] as i8 as f32;
                let sc6 = w[sc_off + is + 6] as i8 as f32;
                let ql0 = w[ql_off + (l as usize)];
                let ql1 = w[ql_off + 32 + (l as usize)];
                let qh_byte = w[qh_off + (l as usize)];
                let q1 = ((ql0 & 0xFu8) as i32 | (((qh_byte >> 0u32) & 3u8) as i32) << 4) - 32;
                let q2 = ((ql1 & 0xFu8) as i32 | (((qh_byte >> 2u32) & 3u8) as i32) << 4) - 32;
                let q3 = ((ql0 >> 4u32) as i32 | (((qh_byte >> 4u32) & 3u8) as i32) << 4) - 32;
                let q4 = ((ql1 >> 4u32) as i32 | (((qh_byte >> 6u32) & 3u8) as i32) << 4) - 32;
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
