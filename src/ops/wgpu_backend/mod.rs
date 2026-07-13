//! WGPU backend (Metal / Vulkan): the core fused dequant+matmul dispatch —
//! ad-hoc (`matmul_dequant`), pre-uploaded (`matmul_dequant_preloaded`), and
//! batched same-input variants (`matmul_dequant_qkv`, `matmul_dequant_multi`,
//! `matmul_dequant_ffn`). See `resident.rs` for the GPU-resident `launch_*`
//! helpers used by the fully-resident forward pass.

pub mod resident;

use super::Backend;
use crate::error::Result;
use crate::types::GgmlType;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;

/// `GGUF_TRACE_MATMUL=1` enables per-matmul timing eprintln's on the
/// ad-hoc/non-resident dispatch paths below. Checked via `std::env::var` on
/// every call otherwise (a full process-environment walk per matmul) —
/// cached once, same pattern as `trace_cpu::enabled()`.
fn trace_matmul_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("GGUF_TRACE_MATMUL").is_ok())
}

pub struct WgpuBackend {
    pub client: ComputeClient<WgpuRuntime>,
}

impl WgpuBackend {
    pub fn new() -> Self {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        // Default build stores activations as f16; require the device to
        // actually support shader-f16 so a mis-built binary fails loudly here
        // rather than silently corrupting activations (e.g. wgpu→Metal, which
        // should be built with `--features f32-activations`).
        #[cfg(not(feature = "f32-activations"))]
        {
            let f16_ok = client
                .properties()
                .features
                .supports_type(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16));
            assert!(
                f16_ok,
                "f16 activations are the default but this wgpu device does not advertise \
                 shader-f16. Rebuild with `--features f32-activations` for this backend."
            );
        }
        Self { client }
    }

    pub fn new_metal() -> Self {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        Self { client }
    }

    /// Upload quantized weight data to GPU and return a handle.
    /// The handle is reused across forward passes — no re-upload.
    pub fn upload_weight(&self, dtype: GgmlType, w: &[u8], in_dim: usize) -> GpuWeightHandle {
        use cubecl::prelude::*;
        use std::time::Instant;
        let row_bytes = dtype.type_size(in_dim);
        let out_dim = w.len() / row_bytes;
        let row_u32s = (row_bytes + 3) / 4;

        let t0 = Instant::now();
        let packed = pack_bytes_to_u32(w, row_bytes, out_dim, row_u32s);
        let pack_time = t0.elapsed();

        let t0 = Instant::now();
        let handle = self.client.create_from_slice(u32::as_bytes(&packed));
        let upload_time = t0.elapsed();

        if std::env::var("GGUF_TRACE_UPLOAD").is_ok() {
            eprintln!(
                "      pack={:.3}s upload={:.3}s ({} bytes)",
                pack_time.as_secs_f32(),
                upload_time.as_secs_f32(),
                w.len(),
            );
        }

        GpuWeightHandle {
            handle,
            dtype,
            in_dim,
            out_dim,
            row_u32s,
        }
    }

    /// Micro-benchmark: does the batched matmul actually read the weight tensor
    /// once (L1-amortized) instead of `n` times? Times `iters` rounds of (a) `n`
    /// separate single-token `launch_only` calls vs (b) one `launch_only_batch`,
    /// for the same `[n, in_dim]` input against weight `h`. Returns
    /// `(single_secs, batch_secs)` totals. Each round is forced to completion
    /// with a readback so the timing is real. Used by `bin/matmul_batch_bench`.
    /// Returns `(single_secs, loopN_secs, coop_secs, coop_max_abs_err)`. The
    /// coop path is Q8_0-only with `in_dim <= 4096`; pass such a tensor. The
    /// err compares coop's token-0 output against the single-token matmul.
    pub fn probe_batched_matmul(
        &self,
        h: &GpuWeightHandle,
        n: usize,
        iters: usize,
    ) -> (f64, f64, f64, f32) {
        use cubecl::prelude::*;
        use std::time::Instant;

        let x = vec![1.0f32; n * h.in_dim];
        let x_batch = self.upload_act(&x);
        // A single-token x handle (first token's slice) reused for the n separate
        // launches — same weight-read pattern as n independent matmuls.
        let x_one = self.upload_act(&x[..h.in_dim]);

        // Warm up all paths (compile kernels, prime caches).
        for _ in 0..2 {
            let o = self.launch_only(h, &x_one);
            let _ = self.client.read_one_unchecked(o);
            let o = self.launch_only_batch(h, &x_batch, n);
            let _ = self.client.read_one_unchecked(o);
            let o = self.launch_coop_batch(h, &x_batch, n);
            let _ = self.client.read_one_unchecked(o);
        }

        // Correctness: coop token-0 vs single (both dot dequant(W) · ones).
        let ref_bytes = {
            let o = self.launch_only(h, &x_one);
            self.client.read_one_unchecked(o)
        };
        let coop_bytes = {
            let o = self.launch_coop_batch(h, &x_batch, n);
            self.client.read_one_unchecked(o)
        };
        let refv = crate::ops::act_decode(&ref_bytes, h.out_dim);
        let coopv = crate::ops::act_decode(&coop_bytes, h.out_dim);
        let max_abs_err = refv
            .iter()
            .zip(&coopv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        let t0 = Instant::now();
        for _ in 0..iters {
            let mut last = None;
            for _ in 0..n {
                last = Some(self.launch_only(h, &x_one));
            }
            let _ = self.client.read_one_unchecked(last.unwrap());
        }
        let single_secs = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for _ in 0..iters {
            let o = self.launch_only_batch(h, &x_batch, n);
            let _ = self.client.read_one_unchecked(o);
        }
        let batch_secs = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        for _ in 0..iters {
            let o = self.launch_coop_batch(h, &x_batch, n);
            let _ = self.client.read_one_unchecked(o);
        }
        let coop_secs = t0.elapsed().as_secs_f64();

        (single_secs, batch_secs, coop_secs, max_abs_err)
    }

    /// Micro-benchmark the amortized per-dispatch cost of the resident-path
    /// kernels, to test whether decode is dispatch-count-bound (in which case
    /// every kernel costs ~the same dispatch floor regardless of work, and
    /// fusion is the lever) vs compute-bound (kernels differ by their work).
    ///
    /// Each kernel is launched `k` times back-to-back with a single sync at the
    /// end (mirroring the real forward's batched submission), so the reported
    /// per-launch time is the *amortized, pipelined* cost — not the isolated
    /// sync-inflated cost. Prints a table. `h` is a real weight tensor (matmul);
    /// `d` = embedding_length, `ff` = feed_forward_length.
    pub fn probe_kernel_costs(
        &self,
        h: &GpuWeightHandle,
        d: usize,
        ff: usize,
        n_heads: usize,
        head_dim: usize,
        k: usize,
    ) {
        use std::time::Instant;

        // Inputs.
        let x_in = self.upload_act(&vec![1.0f32; h.in_dim]);
        let x_d = self.upload_act(&vec![1.0f32; d]);
        let w_d = self.upload_activation(&vec![1.0f32; d]);
        let g_ff = self.upload_act(&vec![1.0f32; ff]);
        let u_ff = self.upload_act(&vec![1.0f32; ff]);
        let tiny = self.upload_act(&vec![1.0f32; 64]);
        let tiny2 = self.upload_act(&vec![1.0f32; 64]);
        let q_heads = self.upload_act(&vec![1.0f32; n_heads * head_dim]);
        let qk_w = self.upload_activation(&vec![1.0f32; head_dim]);

        // (label, closure returning a handle to sync on).
        let mut rows: Vec<(&str, f64)> = Vec::new();
        macro_rules! bench {
            ($label:expr, $body:expr) => {{
                for _ in 0..3 {
                    let o = $body;
                    let _ = self.client.read_one_unchecked(o);
                }
                let t0 = Instant::now();
                let mut last = None;
                for _ in 0..k {
                    last = Some($body);
                }
                let _ = self.client.read_one_unchecked(last.unwrap());
                rows.push(($label, t0.elapsed().as_secs_f64() / k as f64 * 1e3));
            }};
        }

        bench!("dispatch_floor(silu len64)", self.launch_silu_mul(&tiny, &tiny2, 64));
        bench!("matmul (in_dim x out_dim)", self.launch_only(h, &x_in));
        bench!("rms_norm(d) [1-thread]", self.launch_rms_norm(&x_d, &w_d, d, 1e-5));
        bench!(
            "add_resid_rms_norm(d) [1-thread]",
            { let (_n, r) = self.launch_add_residual_rms_norm(&x_d, &x_d, &w_d, d, 1e-5); r }
        );
        bench!("silu_mul(ff)", self.launch_silu_mul(&g_ff, &u_ff, ff));

        // qk_norm_rope is in-place (one thread per head, serial over head_dim):
        // launch then sync on the mutated buffer.
        {
            for _ in 0..3 {
                self.launch_qk_norm_rope(&q_heads, &qk_w, n_heads, head_dim, 1e-5, head_dim, 1, 1e4);
                let _ = self.client.read_one_unchecked(q_heads.clone());
            }
            let t0 = Instant::now();
            for _ in 0..k {
                self.launch_qk_norm_rope(&q_heads, &qk_w, n_heads, head_dim, 1e-5, head_dim, 1, 1e4);
            }
            let _ = self.client.read_one_unchecked(q_heads.clone());
            rows.push(("qk_norm_rope [1-thread/head]", t0.elapsed().as_secs_f64() / k as f64 * 1e3));
        }

        println!("\n{:>36}  {:>12}", "kernel", "ms/dispatch");
        for (label, ms) in &rows {
            println!("{label:>36}  {ms:>12.4}");
        }
        let floor = rows[0].1;
        println!(
            "\nfloor = {floor:.4} ms/dispatch. If every kernel ≈ floor, decode is\n\
             dispatch-count-bound → fusion (fewer dispatches) is the lever, not faster kernels."
        );
    }

    /// Launch the consolidated dequant matmul using a pre-uploaded weight tensor.
    pub fn matmul_dequant_preloaded(
        &self,
        h: &GpuWeightHandle,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        use cubecl::prelude::*;
        use std::time::Instant;
        let trace = trace_matmul_enabled();

        let t0 = Instant::now();
        // CPU-orchestrated (non-resident) path: `x` is uploaded as the `Act`
        // activation type the matmul reads, and the `f32out` kernel keeps the
        // CPU-side result f32.
        let x_handle = self.client.create_from_slice(&crate::ops::act_encode(x));
        let out_handle = self.client.empty(h.out_dim * core::mem::size_of::<f32>());
        let alloc_time = t0.elapsed();

        let t0 = Instant::now();
        let grid_x = (h.out_dim as u32).min(65535);
        let grid_y = ((h.out_dim as u32) + grid_x - 1) / grid_x;
        unsafe {
            crate::ops::kernels::wgpu::matmul_dequant_wgpu_f32out::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(grid_x, grid_y, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(h.handle.clone(), h.out_dim * h.row_u32s),
                ArrayArg::from_raw_parts(x_handle.clone(), h.in_dim),
                ArrayArg::from_raw_parts(out_handle.clone(), h.out_dim),
                h.dtype as u32,
                h.in_dim,
                h.row_u32s,
                grid_x,
            );
        }
        let launch_time = t0.elapsed();

        let t0 = Instant::now();
        let actual_bytes = self.client.read_one_unchecked(out_handle);
        let read_time = t0.elapsed();
        let actual = f32::from_bytes(&actual_bytes);
        out.copy_from_slice(&actual[..h.out_dim]);

        if trace {
            eprintln!(
                "        matmul in_dim={} out_dim={}: alloc={:.4}s launch={:.4}s read={:.4}s",
                h.in_dim,
                h.out_dim,
                alloc_time.as_secs_f32(),
                launch_time.as_secs_f32(),
                read_time.as_secs_f32(),
            );
        }
        Ok(())
    }

    /// Fused Q/K/V projection: `wq`, `wk`, `wv` all read the same input `x`,
    /// so this uploads `x` once, launches all three kernels back-to-back,
    /// then reads all three results with a single `client.read` — one
    /// blocking GPU round-trip instead of three.
    pub fn matmul_dequant_qkv(
        &self,
        hq: &GpuWeightHandle,
        hk: &GpuWeightHandle,
        hv: &GpuWeightHandle,
        x: &[f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()> {
        let x_handle = self.upload_act(x);
        self.matmul_dequant_qkv_from_handle(hq, hk, hv, &x_handle, q, k, v)
    }

    /// Same as `matmul_dequant_qkv`, but `x` is already a GPU handle (e.g.
    /// the output of `launch_rms_norm`) — no upload needed. Used by the
    /// fully GPU-resident forward pass.
    pub(crate) fn matmul_dequant_qkv_from_handle(
        &self,
        hq: &GpuWeightHandle,
        hk: &GpuWeightHandle,
        hv: &GpuWeightHandle,
        x_handle: &cubecl::server::Handle,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) -> Result<()> {
        use cubecl::prelude::*;
        use std::time::Instant;
        let trace = trace_matmul_enabled();

        let t0 = Instant::now();
        let out_q = self.launch_only_f32out(hq, x_handle);
        let out_k = self.launch_only_f32out(hk, x_handle);
        let out_v = self.launch_only_f32out(hv, x_handle);
        let launch_time = t0.elapsed();

        let t0 = Instant::now();
        let results = self.client.read(vec![out_q, out_k, out_v]);
        let read_time = t0.elapsed();
        let rq = f32::from_bytes(&results[0]);
        let rk = f32::from_bytes(&results[1]);
        let rv = f32::from_bytes(&results[2]);
        q.copy_from_slice(&rq[..hq.out_dim]);
        k.copy_from_slice(&rk[..hk.out_dim]);
        v.copy_from_slice(&rv[..hv.out_dim]);

        if trace {
            eprintln!(
                "        qkv in_dim={}: launch(x3)={:.4}s read={:.4}s",
                hq.in_dim,
                launch_time.as_secs_f32(),
                read_time.as_secs_f32(),
            );
        }
        Ok(())
    }

    /// Fused N-way projection: every `hs[i]` reads the same input `x`, so
    /// this uploads `x` once, launches all kernels back-to-back, then
    /// reads every result with a single `client.read` — one blocking GPU
    /// round-trip instead of `hs.len()`. Generalizes `matmul_dequant_qkv`
    /// to an arbitrary number of same-input projections (e.g. Gated
    /// DeltaNet's wqkv/wgate/ssm_beta/ssm_alpha, all read from `xn`).
    pub fn matmul_dequant_multi(
        &self,
        hs: &[&GpuWeightHandle],
        x: &[f32],
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        let x_handle = self.upload_act(x);
        self.matmul_dequant_multi_from_handle(hs, &x_handle, outs)
    }

    /// Same as `matmul_dequant_multi`, but `x` is already a GPU handle (e.g.
    /// the output of `launch_rms_norm`) — no upload needed.
    pub(crate) fn matmul_dequant_multi_from_handle(
        &self,
        hs: &[&GpuWeightHandle],
        x_handle: &Handle,
        outs: &mut [&mut [f32]],
    ) -> Result<()> {
        use cubecl::prelude::*;
        use std::time::Instant;
        debug_assert_eq!(hs.len(), outs.len());
        let trace = trace_matmul_enabled();
        let _timer = crate::ops::trace::Timer::new("matmul_dequant_multi");

        let t0 = Instant::now();
        let handles: Vec<_> = hs.iter().map(|h| self.launch_only_f32out(h, &x_handle)).collect();
        let launch_time = t0.elapsed();

        let t0 = Instant::now();
        let results = self.client.read(handles);
        let read_time = t0.elapsed();
        for ((result, out), h) in results.iter().zip(outs.iter_mut()).zip(hs.iter()) {
            let vals = f32::from_bytes(result);
            out.copy_from_slice(&vals[..h.out_dim]);
        }

        if trace {
            eprintln!(
                "        multi({}) in_dim={}: launch={:.4}s read={:.4}s",
                hs.len(),
                hs.first().map(|h| h.in_dim).unwrap_or(0),
                launch_time.as_secs_f32(),
                read_time.as_secs_f32(),
            );
        }
        Ok(())
    }

    /// Fused FFN block: `gate = W_gate * x`, `up = W_up * x`, combined
    /// with SiLU, then `out = W_down * combined` — all chained through
    /// GPU handles with a single readback at the end, instead of two CPU
    /// matmuls (the old `ffn_gate_up`) plus a separate GPU sync for
    /// `ffn_down`.
    pub fn matmul_dequant_ffn(
        &self,
        h_gate: &GpuWeightHandle,
        h_up: &GpuWeightHandle,
        h_down: &GpuWeightHandle,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        use std::time::Instant;
        let t0 = Instant::now();
        let x_handle = self.upload_act(x);
        let upload_time = t0.elapsed();

        let t0 = Instant::now();
        let down_handle = self.ffn_chain_from_handle(h_gate, h_up, h_down, &x_handle);
        let launch_time = t0.elapsed();

        let t0 = Instant::now();
        // `ffn_chain_from_handle` returns an `Act` handle (shared with the
        // resident path); decode it back to f32 for the CPU-side result.
        let down_bytes = self.client.read_one_unchecked(down_handle);
        let actual = crate::ops::act_decode(&down_bytes, h_down.out_dim);
        let read_time = t0.elapsed();
        out.copy_from_slice(&actual);

        if trace_matmul_enabled() {
            eprintln!(
                "        ffn ffn_dim={}: upload={:.4}s launch(x4)={:.4}s read={:.4}s",
                h_gate.out_dim,
                upload_time.as_secs_f32(),
                launch_time.as_secs_f32(),
                read_time.as_secs_f32(),
            );
        }
        Ok(())
    }
}

/// Pack raw quantized bytes into u32 words, 4 bytes per word (WGSL has no u8).
/// Uses rayon parallelism for tensors with many rows (>= 16).
fn pack_bytes_to_u32(w: &[u8], row_bytes: usize, out_dim: usize, row_u32s: usize) -> Vec<u32> {
    let mut packed = vec![0u32; out_dim * row_u32s];
    
    // Parallel packing for tensors with enough rows to amortize overhead
    if out_dim >= 16 {
        use rayon::prelude::*;
        packed
            .par_chunks_mut(row_u32s)
            .enumerate()
            .for_each(|(row_idx, dst)| {
                let row_start = row_idx * row_bytes;
                let src = &w[row_start..row_start + row_bytes];
                for (i, chunk) in src.chunks(4).enumerate() {
                    let mut word: u32 = 0;
                    for (j, &b) in chunk.iter().enumerate() {
                        word |= (b as u32) << (8 * j);
                    }
                    dst[i] = word;
                }
            });
    } else {
        // Sequential for small tensors (low overhead path)
        for row_idx in 0..out_dim {
            let row_start = row_idx * row_bytes;
            let src = &w[row_start..row_start + row_bytes];
            let dst_base = row_idx * row_u32s;
            for (i, chunk) in src.chunks(4).enumerate() {
                let mut word: u32 = 0;
                for (j, &b) in chunk.iter().enumerate() {
                    word |= (b as u32) << (8 * j);
                }
                packed[dst_base + i] = word;
            }
        }
    }
    packed
}

/// Pre-uploaded quantized tensor on GPU.
pub struct GpuWeightHandle {
    handle: cubecl::server::Handle,
    dtype: GgmlType,
    in_dim: usize,
    out_dim: usize,
    row_u32s: usize,
}

impl GpuWeightHandle {
    /// Returns the underlying GPU handle for benchmarking.
    pub fn handle_ref(&self) -> &cubecl::server::Handle {
        &self.handle
    }

    /// The dtype and in_dim this handle was uploaded with — used by
    /// warmup to know which kernel variants to pre-compile.
    pub fn shape(&self) -> (GgmlType, usize) {
        (self.dtype, self.in_dim)
    }

    /// Number of output rows this weight tensor produces.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl Backend for WgpuBackend {
    type R = WgpuRuntime;

    fn client(&self) -> &ComputeClient<WgpuRuntime> {
        &self.client
    }

    fn name(&self) -> &str {
        "wgpu"
    }

    fn matmul_dequant(
        &self,
        dtype: GgmlType,
        w: &[u8],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        // Ad-hoc call: pack + upload + launch. Prefer pre-loaded tensors
        // via upload_weight + matmul_dequant_preloaded for repeated use.
        let in_dim = x.len();
        let row_bytes = dtype.type_size(in_dim);
        let out_dim = out.len();
        let row_u32s = (row_bytes + 3) / 4;

        let packed = pack_bytes_to_u32(w, row_bytes, out_dim, row_u32s);
        let h = GpuWeightHandle {
            handle: {
                use cubecl::prelude::*;
                self.client.create_from_slice(u32::as_bytes(&packed))
            },
            dtype,
            in_dim,
            out_dim,
            row_u32s,
        };
        self.matmul_dequant_preloaded(&h, x, out)
    }
}
