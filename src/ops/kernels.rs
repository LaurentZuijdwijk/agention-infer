//! GPU compute kernels — re-exports from backend-specific modules.
//!
//! - `kernels_u8.rs`: native `Array<u8>` kernels for HIP, CUDA, CPU.
//! - `kernels_wgpu.rs`: `Array<u32>` packed kernels for WGPU (Metal, Vulkan).

#[cfg(any(feature = "cpu", feature = "hip", feature = "cuda"))]
pub mod u8 {
    pub use super::super::kernels_u8::*;
}

#[cfg(feature = "wgpu")]
pub mod wgpu {
    pub use super::super::kernels_wgpu::*;
}
