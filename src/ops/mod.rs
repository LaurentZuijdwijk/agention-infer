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

/// Activation element type for the GPU-resident forward path.
///
/// The intermediate activation buffers (residual stream, Q/K/V, FFN
/// intermediates, KV cache, attention scratch) that flow between resident
/// kernels are stored as this type. Reductions/accumulators inside the kernels
/// stay `f32` regardless — only the *storage* narrows. Weights (quantized) and
/// the final logits are unaffected.
///
/// `f16` by default (the fast path — halves activation bandwidth on
/// shader-f16-capable backends like Vulkan/RDNA). Build with the
/// `f32-activations` opt-out for `f32` storage where the backend lacks
/// shader-f16 (e.g. wgpu→Metal), which restores the original behaviour.
#[cfg(not(feature = "f32-activations"))]
pub type Act = half::f16;
#[cfg(feature = "f32-activations")]
pub type Act = f32;

/// Encode an `f32` activation slice into the storage bytes of [`Act`] for
/// upload to the GPU (identity byte-copy when `Act = f32`).
#[cfg(feature = "wgpu")]
pub fn act_encode(x: &[f32]) -> Vec<u8> {
    #[cfg(not(feature = "f32-activations"))]
    {
        let mut bytes = Vec::with_capacity(x.len() * 2);
        for &v in x {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        bytes
    }
    #[cfg(feature = "f32-activations")]
    {
        use cubecl::prelude::*;
        f32::as_bytes(x).to_vec()
    }
}

/// Decode `n` [`Act`] values from GPU-readback bytes back into `f32`.
#[cfg(feature = "wgpu")]
pub fn act_decode(bytes: &[u8], n: usize) -> Vec<f32> {
    #[cfg(not(feature = "f32-activations"))]
    {
        bytes
            .chunks_exact(2)
            .take(n)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }
    #[cfg(feature = "f32-activations")]
    {
        use cubecl::prelude::*;
        f32::from_bytes(bytes)[..n].to_vec()
    }
}

/// Size in bytes of one [`Act`] element.
#[cfg(feature = "wgpu")]
pub const ACT_SIZE: usize = core::mem::size_of::<Act>();

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
