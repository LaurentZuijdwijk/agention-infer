//! CPU backend: dispatches straight to the pure-Rust fused dequant+dot
//! kernels in `crate::quant`, parallelized via rayon.

use super::Backend;
use crate::error::Result;
use cubecl::client::ComputeClient;
use cubecl::cpu::{CpuDevice, CpuRuntime};
use cubecl::Runtime;

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
        let _timer = crate::ops::trace::Timer::new("matmul_dequant_cpu");
        let row_bytes = dtype.type_size(x.len());

        for (row_idx, o) in out.iter_mut().enumerate() {
            let row_data = &w[row_idx * row_bytes..];
            *o = match dtype {
                crate::types::GgmlType::Q8_0 => crate::quant::q8_0::dot_q8_0(row_data, x)?,
                crate::types::GgmlType::Q4_K => crate::quant::q4_k::dot_q4_k(row_data, x)?,
                crate::types::GgmlType::Q5_K => crate::quant::q5_k::dot_q5_k(row_data, x)?,
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
