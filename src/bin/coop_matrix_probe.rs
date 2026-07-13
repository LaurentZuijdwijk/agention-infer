//! Capability-only probe (feature "vulkan-poc"): does this GPU/driver actually
//! expose `VK_KHR_cooperative_matrix` (tensor-core-style matrix-multiply
//! hardware), and if so, what shapes/dtypes does it support?
//!
//! This does NOT dispatch any compute work — it only queries the physical
//! device, matching the discipline used earlier this session for the f16
//! capability check (`f16_probe.rs`): confirm the hardware/driver actually
//! supports a feature before investing in a kernel built on it. cubecl-wgpu's
//! own README states tensor-core acceleration "isn't supported on WebGPU yet"
//! (NVIDIA/CUDA-first) — this probe checks whether the *driver* has the
//! capability regardless, which would only matter if we later wrote a
//! hand-rolled Vulkan compute shader bypassing cubecl entirely.
//!
//!   cargo run --release --features vulkan-poc --bin coop_matrix_probe

use ash::vk;
use std::ffi::CStr;

fn main() {
    unsafe {
        let entry = ash::Entry::load().expect("failed to load Vulkan loader");

        let app_name = CStr::from_bytes_with_nul(b"coop_matrix_probe\0").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry
            .create_instance(&instance_info, None)
            .expect("failed to create instance");

        let physical_devices = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices");

        let mut chosen = None;
        for &pd in &physical_devices {
            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            println!("found device: {name} (type={:?})", props.device_type);
            if (props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                || props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                && chosen.is_none()
            {
                chosen = Some(pd);
            }
        }
        let physical_device = chosen.expect("no suitable GPU found");
        let props = instance.get_physical_device_properties(physical_device);
        println!(
            "\nusing device: {}",
            CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy()
        );

        // Does the device advertise the extension at all?
        let ext_props = instance
            .enumerate_device_extension_properties(physical_device)
            .expect("failed to enumerate device extensions");
        let has_ext = ext_props.iter().any(|e| {
            CStr::from_ptr(e.extension_name.as_ptr())
                .to_str()
                .map(|s| s == "VK_KHR_cooperative_matrix")
                .unwrap_or(false)
        });

        if !has_ext {
            println!("\n❌ VK_KHR_cooperative_matrix is NOT advertised by this device/driver.");
            println!("   No cooperative-matrix hardware path available via Vulkan here.");
            instance.destroy_instance(None);
            return;
        }
        println!("\n✅ VK_KHR_cooperative_matrix is advertised. Querying supported shapes...\n");

        let coop = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let matrix_props = coop
            .get_physical_device_cooperative_matrix_properties(physical_device)
            .expect("failed to query cooperative matrix properties");

        if matrix_props.is_empty() {
            println!("Extension present but reports zero supported configurations.");
        } else {
            println!(
                "{:>6} {:>6} {:>6}  {:>8} {:>8} {:>8} {:>10}  {:>10} {:>10}",
                "M", "N", "K", "A_type", "B_type", "C_type", "Result", "Scope", "Saturate"
            );
            for p in &matrix_props {
                println!(
                    "{:>6} {:>6} {:>6}  {:>8?} {:>8?} {:>8?} {:>10?}  {:>10?} {:>10}",
                    p.m_size,
                    p.n_size,
                    p.k_size,
                    p.a_type,
                    p.b_type,
                    p.c_type,
                    p.result_type,
                    p.scope,
                    p.saturating_accumulation == vk::TRUE,
                );
            }
        }

        println!(
            "\nRead: any row with A/B_type = F16 (or lower) and Result/C_type = F32 is a\n\
             usable tensor-core-style path for a hand-rolled matmul kernel — but note this\n\
             would mean bypassing cubecl-wgpu entirely (its own docs say tensor-core support\n\
             isn't wired up for the WebGPU/wgpu backend), i.e. a from-scratch Vulkan compute\n\
             shader, the largest engineering lift considered this session."
        );

        instance.destroy_instance(None);
    }
}
