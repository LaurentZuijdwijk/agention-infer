pub mod kernels;
#[cfg(any(feature = "cpu", feature = "hip", feature = "cuda"))]
pub mod kernels_u8;
#[cfg(feature = "wgpu")]
pub mod kernels_wgpu;

mod backend_select;
#[cfg(feature = "cpu")]
pub mod cpu_backend;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

pub use backend_select::{create_backend, warmup, AnyBackend, BackendPreference};
#[cfg(feature = "cpu")]
pub use cpu_backend as cpu;

use crate::error::Result;
use crate::types::GgmlType;
use cubecl::client::ComputeClient;
use cubecl::Runtime;

/// Quantization formats with a fused GPU dequant+matmul kernel.
/// Kept in one place so model code and warmup can agree on what's dispatchable.
pub const GPU_DEQUANT_DTYPES: [GgmlType; 4] =
    [GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K];

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
