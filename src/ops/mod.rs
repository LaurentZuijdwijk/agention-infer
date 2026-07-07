pub mod kernels;
#[cfg(any(feature = "cpu", feature = "hip", feature = "cuda"))]
pub mod kernels_u8;
#[cfg(feature = "wgpu")]
pub mod kernels_wgpu;

use crate::error::Result;
use crate::types::GgmlType;
use cubecl::client::ComputeClient;
use cubecl::Runtime;

/// Quantization formats with a fused GPU dequant+matmul kernel.
/// Kept in one place so model code and warmup can agree on what's dispatchable.
pub const GPU_DEQUANT_DTYPES: [GgmlType; 3] = [GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K];

// ── Backend trait ────────────────────────────────────────────────────────

/// Backend abstraction — model code never knows if it's on CPU or GPU.
///
/// The trait wraps CubeCL kernel launches behind a clean interface.
/// All backends share the same `#[cube]` kernel source — only the
/// `ComputeClient` differs.
pub trait Backend: Send + Sync {
    type R: Runtime;

    fn client(&self) -> &ComputeClient<Self::R>;
    fn name(&self) -> &str;

    /// Fused dequant + matmul: `out[row] = sum_i dequant(w[row][i]) * x[i]`.
    ///
    /// `dtype` selects the quantization format (must be one of `GPU_DEQUANT_DTYPES`).
    /// `w` is the raw quantized weight bytes for one matrix row.
    /// `x` is the input vector (length = in_dim).
    /// `out` receives one scalar per row.
    fn matmul_dequant(&self, dtype: GgmlType, w: &[u8], x: &[f32], out: &mut [f32]) -> Result<()>;
}

// ── CPU backend ──────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
pub mod cpu {
    use super::*;
    use cubecl::cpu::{CpuDevice, CpuRuntime};

    pub struct CpuBackend {
        client: ComputeClient<CpuRuntime>,
    }

    impl CpuBackend {
        pub fn new() -> Self {
            let client = CpuRuntime::client(&CpuDevice::default());
            Self { client }
        }
    }

    impl Default for CpuBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Backend for CpuBackend {
        type R = CpuRuntime;

        fn client(&self) -> &ComputeClient<CpuRuntime> {
            &self.client
        }

        fn name(&self) -> &str {
            "cpu"
        }

        fn matmul_dequant(
            &self,
            dtype: crate::types::GgmlType,
            w: &[u8],
            x: &[f32],
            out: &mut [f32],
        ) -> Result<()> {
            // CPU path: use the existing pure-Rust fused dequant+dot kernels.
            // These are already correct and fast on CPU via rayon.
            let row_bytes = dtype.type_size(x.len());

            for (row_idx, o) in out.iter_mut().enumerate() {
                let row_data = &w[row_idx * row_bytes..];
                *o = match dtype {
                    crate::types::GgmlType::Q8_0 => crate::quant::q8_0::dot_q8_0(row_data, x)?,
                    crate::types::GgmlType::Q4_K => crate::quant::q4_k::dot_q4_k(row_data, x)?,
                    crate::types::GgmlType::Q6_K => crate::quant::q6_k::dot_q6_k(row_data, x)?,
                    other => {
                        return Err(crate::error::GgufError::BackendError(format!(
                            "matmul_dequant: unsupported dtype {other}"
                        )))
                    }
                };
            }
            Ok(())
        }
    }
}

// ── WGPU backend (Metal / Vulkan) ────────────────────────────────────────

#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    pub struct WgpuBackend {
        client: ComputeClient<WgpuRuntime>,
    }

    impl WgpuBackend {
        pub fn new() -> Self {
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);
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

        /// Launch the consolidated dequant matmul using a pre-uploaded weight tensor.
        pub fn matmul_dequant_preloaded(
            &self,
            h: &GpuWeightHandle,
            x: &[f32],
            out: &mut [f32],
        ) -> Result<()> {
            use cubecl::prelude::*;
            use std::time::Instant;
            let trace = std::env::var("GGUF_TRACE_MATMUL").is_ok();

            let t0 = Instant::now();
            let x_handle = self.client.create_from_slice(f32::as_bytes(x));
            let out_handle = self.client.empty(h.out_dim * core::mem::size_of::<f32>());
            let alloc_time = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                let threads = 64u32;
                let workgroups = (h.out_dim as u32 + threads - 1) / threads;
                crate::ops::kernels::wgpu::matmul_dequant_wgpu::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(workgroups, 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(h.handle.clone(), h.out_dim * h.row_u32s),
                    ArrayArg::from_raw_parts(x_handle.clone(), h.in_dim),
                    ArrayArg::from_raw_parts(out_handle.clone(), h.out_dim),
                    h.dtype as u32,
                    h.in_dim,
                    h.row_u32s,
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

        /// Launch a dequant matmul without reading the result back — caller
        /// batches multiple `launch_only` calls, then reads them all with one
        /// `client.read`, instead of paying a blocking round-trip per matmul.
        pub(crate) fn launch_only(
            &self,
            h: &GpuWeightHandle,
            x_handle: &cubecl::server::Handle,
        ) -> cubecl::server::Handle {
            use cubecl::prelude::*;
            let out_handle = self.client.empty(h.out_dim * core::mem::size_of::<f32>());
            unsafe {
                let threads = 64u32;
                let workgroups = (h.out_dim as u32 + threads - 1) / threads;
                crate::ops::kernels::wgpu::matmul_dequant_wgpu::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(workgroups, 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(h.handle.clone(), h.out_dim * h.row_u32s),
                    ArrayArg::from_raw_parts(x_handle.clone(), h.in_dim),
                    ArrayArg::from_raw_parts(out_handle.clone(), h.out_dim),
                    h.dtype as u32,
                    h.in_dim,
                    h.row_u32s,
                );
            }
            out_handle
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
            let x_handle = self.upload_activation(x);
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
            let trace = std::env::var("GGUF_TRACE_MATMUL").is_ok();

            let t0 = Instant::now();
            let out_q = self.launch_only(hq, x_handle);
            let out_k = self.launch_only(hk, x_handle);
            let out_v = self.launch_only(hv, x_handle);
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

        /// Launch the SwiGLU combine kernel reading two GPU-resident buffers
        /// directly (no CPU round-trip for `gate`/`up`).
        pub(crate) fn launch_silu_mul(
            &self,
            gate_handle: &cubecl::server::Handle,
            up_handle: &cubecl::server::Handle,
            len: usize,
        ) -> cubecl::server::Handle {
            use cubecl::prelude::*;
            let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
            unsafe {
                let threads = 64u32;
                let workgroups = (len as u32 + threads - 1) / threads;
                crate::ops::kernels::wgpu::silu_mul::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(workgroups, 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(gate_handle.clone(), len),
                    ArrayArg::from_raw_parts(up_handle.clone(), len),
                    ArrayArg::from_raw_parts(out_handle.clone(), len),
                );
            }
            out_handle
        }

        /// Launch the residual-add kernel: `out = a + b`, both already
        /// GPU-resident — no CPU round-trip.
        pub(crate) fn launch_residual_add(
            &self,
            a: &cubecl::server::Handle,
            b: &cubecl::server::Handle,
            len: usize,
        ) -> cubecl::server::Handle {
            use cubecl::prelude::*;
            let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
            unsafe {
                let threads = 64u32;
                let workgroups = (len as u32 + threads - 1) / threads;
                crate::ops::kernels::wgpu::residual_add::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(workgroups, 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(a.clone(), len),
                    ArrayArg::from_raw_parts(b.clone(), len),
                    ArrayArg::from_raw_parts(out_handle.clone(), len),
                );
            }
            out_handle
        }

        /// Launch RMSNorm: `out = (x / rms(x)) * weight`, reading `x` and
        /// `weight` directly from GPU handles — no CPU round-trip.
        pub(crate) fn launch_rms_norm(
            &self,
            x_handle: &cubecl::server::Handle,
            weight_handle: &cubecl::server::Handle,
            len: usize,
            eps: f32,
        ) -> cubecl::server::Handle {
            use cubecl::prelude::*;
            let out_handle = self.client.empty(len * core::mem::size_of::<f32>());
            unsafe {
                crate::ops::kernels::wgpu::rms_norm::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(x_handle.clone(), len),
                    ArrayArg::from_raw_parts(weight_handle.clone(), len),
                    ArrayArg::from_raw_parts(out_handle.clone(), len),
                    eps,
                );
            }
            out_handle
        }

        /// Upload a plain f32 activation/weight vector to GPU (no packing —
        /// used for embeddings and norm vectors, which are always f32).
        pub(crate) fn upload_activation(&self, x: &[f32]) -> cubecl::server::Handle {
            use cubecl::prelude::*;
            self.client.create_from_slice(f32::as_bytes(x))
        }

        /// Blocking read of a GPU handle back to a `Vec<f32>` of length `len`.
        pub(crate) fn read_handle(&self, h: cubecl::server::Handle, len: usize) -> Vec<f32> {
            use cubecl::prelude::*;
            let bytes = self.client.read_one_unchecked(h);
            f32::from_bytes(&bytes)[..len].to_vec()
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
            let x_handle = self.upload_activation(x);
            let upload_time = t0.elapsed();

            let t0 = Instant::now();
            let down_handle = self.ffn_chain_from_handle(h_gate, h_up, h_down, &x_handle);
            let launch_time = t0.elapsed();

            let t0 = Instant::now();
            let actual = self.read_handle(down_handle, h_down.out_dim);
            let read_time = t0.elapsed();
            out.copy_from_slice(&actual);

            if std::env::var("GGUF_TRACE_MATMUL").is_ok() {
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

        /// Same chain as `matmul_dequant_ffn` (`gate`→`up`→SiLU-combine→`down`)
        /// but takes and returns GPU handles with no readback at all — used by
        /// the fully GPU-resident forward pass to keep `down`'s output on GPU
        /// for a subsequent residual-add.
        pub(crate) fn ffn_chain_from_handle(
            &self,
            h_gate: &GpuWeightHandle,
            h_up: &GpuWeightHandle,
            h_down: &GpuWeightHandle,
            x_handle: &cubecl::server::Handle,
        ) -> cubecl::server::Handle {
            let gate_handle = self.launch_only(h_gate, x_handle);
            let up_handle = self.launch_only(h_up, x_handle);
            let act_handle = self.launch_silu_mul(&gate_handle, &up_handle, h_gate.out_dim);
            self.launch_only(h_down, &act_handle)
        }
    }

    /// Pack raw quantized bytes into u32 words, 4 bytes per word (WGSL has no u8).
    fn pack_bytes_to_u32(w: &[u8], row_bytes: usize, out_dim: usize, row_u32s: usize) -> Vec<u32> {
        let mut packed = vec![0u32; out_dim * row_u32s];
        for row_idx in 0..out_dim {
            // Bounded to exactly this row's bytes — an earlier unbounded slice
            // here (`&w[row_idx * row_bytes..]`, running to the end of `w`)
            // made every row repack the entire remaining tensor, i.e. O(rows^2).
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
}

// ── Backend selection ────────────────────────────────────────────────────

/// Backend selection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    Auto,
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "wgpu")]
    Metal,
    #[cfg(feature = "hip")]
    Hip,
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Enum over all possible backends so model code can dispatch without
/// knowing the concrete runtime type.
pub enum AnyBackend {
    #[cfg(feature = "cpu")]
    Cpu(cpu::CpuBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(wgpu_backend::WgpuBackend),
}

impl AnyBackend {
    pub fn name(&self) -> &str {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(b) => b.name(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(b) => b.name(),
        }
    }

    pub fn matmul_dequant(
        &self,
        dtype: GgmlType,
        w: &[u8],
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(b) => b.matmul_dequant(dtype, w, x, out),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(b) => b.matmul_dequant(dtype, w, x, out),
        }
    }
}

/// Create a backend based on preference. Falls back to CPU if the
/// preferred backend is unavailable.
pub fn create_backend(preference: BackendPreference) -> AnyBackend {
    match preference {
        #[cfg(feature = "cpu")]
        BackendPreference::Cpu | BackendPreference::Auto => AnyBackend::Cpu(cpu::CpuBackend::new()),
        #[cfg(feature = "wgpu")]
        BackendPreference::Wgpu => AnyBackend::Wgpu(wgpu_backend::WgpuBackend::new()),
        #[cfg(feature = "wgpu")]
        BackendPreference::Metal => AnyBackend::Wgpu(wgpu_backend::WgpuBackend::new_metal()),
        #[cfg(feature = "hip")]
        BackendPreference::Hip => {
            // TODO: HipBackend
            #[cfg(feature = "cpu")]
            {
                AnyBackend::Cpu(cpu::CpuBackend::new())
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("HIP backend not yet implemented")
            }
        }
        #[cfg(feature = "cuda")]
        BackendPreference::Cuda => {
            // TODO: CudaBackend
            #[cfg(feature = "cpu")]
            {
                AnyBackend::Cpu(cpu::CpuBackend::new())
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("CUDA backend not yet implemented")
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            #[cfg(feature = "cpu")]
            {
                AnyBackend::Cpu(cpu::CpuBackend::new())
            }
            #[cfg(not(feature = "cpu"))]
            {
                panic!("No backend available. Enable one of: cpu, wgpu, hip, cuda");
            }
        }
    }
}

#[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "hip", feature = "cuda")))]
compile_error!("No backend feature enabled. Enable one of: cpu, wgpu, hip, cuda");

// ── Kernel warmup ────────────────────────────────────────────────────────

/// Compile every GPU kernel variant a model will need, up front, with a
/// visible progress message — rather than silently stalling on the first
/// forward pass that hits a new dtype.
///
/// `shapes` should hold one `(dtype, in_dim)` pair per distinct dtype the
/// model's weights actually use. Since `dtype` and `in_dim` are now runtime
/// kernel parameters rather than `#[comptime]` ones, a single compile covers
/// every tensor shape sharing the same element type — so in practice this is
/// one shader compile per backend, not one per tensor.
pub fn warmup(backend: &AnyBackend, shapes: &[(GgmlType, usize)]) {
    if backend.name() == "cpu" || shapes.is_empty() {
        return;
    }

    use std::io::Write;
    use std::time::Instant;
    eprintln!("Compiling GPU kernels ({} dtype(s): {:?})...", shapes.len(), shapes);
    std::io::stderr().flush().ok();
    let start = Instant::now();

    for &(dtype, in_dim) in shapes {
        let t0 = Instant::now();
        let row_bytes = dtype.type_size(in_dim);
        let dummy_w = vec![0u8; row_bytes];
        let dummy_x = vec![0f32; in_dim];
        let mut dummy_out = vec![0f32; 1];
        let _ = backend.matmul_dequant(dtype, &dummy_w, &dummy_x, &mut dummy_out);
        eprintln!("  {dtype} compiled in {:.2}s", t0.elapsed().as_secs_f32());
        std::io::stderr().flush().ok();
    }

    eprintln!("Kernels ready in {:.1}s total", start.elapsed().as_secs_f32());
    std::io::stderr().flush().ok();
}
