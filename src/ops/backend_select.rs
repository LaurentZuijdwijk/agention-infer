//! Backend selection: `BackendPreference`, the `AnyBackend` dispatch enum,
//! `create_backend`, and up-front GPU kernel warmup.

use super::Backend;
use crate::error::Result;
use crate::types::GgmlType;

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
    Cpu(super::cpu::CpuBackend),
    #[cfg(feature = "wgpu")]
    Wgpu(super::wgpu_backend::WgpuBackend),
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
        BackendPreference::Cpu | BackendPreference::Auto => {
            AnyBackend::Cpu(super::cpu::CpuBackend::new())
        }
        #[cfg(feature = "wgpu")]
        BackendPreference::Wgpu => AnyBackend::Wgpu(super::wgpu_backend::WgpuBackend::new()),
        #[cfg(feature = "wgpu")]
        BackendPreference::Metal => {
            AnyBackend::Wgpu(super::wgpu_backend::WgpuBackend::new_metal())
        }
        #[cfg(feature = "hip")]
        BackendPreference::Hip => {
            // TODO: HipBackend
            #[cfg(feature = "cpu")]
            {
                AnyBackend::Cpu(super::cpu::CpuBackend::new())
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
                AnyBackend::Cpu(super::cpu::CpuBackend::new())
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
                AnyBackend::Cpu(super::cpu::CpuBackend::new())
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
