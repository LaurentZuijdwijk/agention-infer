//! Spike: does this platform's wgpu/Vulkan path support native `f16` compute?
//!
//! Metal (via wgpu) does not expose the shader-f16 extension, which is why the
//! quant kernels hand-decode f16 from u32 packing. But Strix Halo / RDNA3.5 via
//! Vulkan advertises `VK_KHR_shader_float16_int8`, and cubecl-wgpu enables the
//! WGSL `f16` type when `Features::SHADER_F16` is present. This probe uploads an
//! `Array<f16>`, doubles it in a kernel, and reads it back — if it runs and the
//! values are right, native f16 activations are viable here.
//!
//!   cargo run --release --features wgpu --bin f16_probe

#[cfg(feature = "wgpu")]
use cubecl::prelude::*;
#[cfg(feature = "wgpu")]
use half::f16;

#[cfg(feature = "wgpu")]
#[cube(launch)]
fn f16_double(input: &Array<f16>, output: &mut Array<f16>) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] + input[i];
    }
}

#[cfg(feature = "wgpu")]
fn main() {
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let input: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32 + 0.5)).collect();
    let mut bytes = Vec::with_capacity(input.len() * 2);
    for v in &input {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    let in_handle = client.create_from_slice(&bytes);
    let out_handle = client.empty(input.len() * core::mem::size_of::<f16>());

    let threads = 64u32;
    let workgroups = (input.len() as u32 + threads - 1) / threads;
    unsafe {
        f16_double::launch::<WgpuRuntime>(
            &client,
            CubeCount::Static(workgroups, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(in_handle.clone(), input.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), input.len()),
        );
    }

    let raw = client.read_one_unchecked(out_handle);
    let got: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let want: Vec<f32> = input.iter().map(|v| v.to_f32() * 2.0).collect();
    println!("in:   {:?}", input.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    println!("got:  {:?}", got);
    println!("want: {:?}", want);

    let ok = got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-2);
    if ok {
        println!("\nnative f16 compute WORKS on this wgpu device");
    } else {
        println!("\nnative f16 ran but produced WRONG values");
    }
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("f16_probe requires --features wgpu");
}
