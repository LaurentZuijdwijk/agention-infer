//! Raw-Vulkan dispatch-overhead prototype (feature "vulkan-poc").
//!
//! Not part of the engine. Answers one question: does bypassing wgpu's
//! per-dispatch bookkeeping (bind-group creation, resource-hazard tracking,
//! validation) actually lower the ~220us/dispatch cost measured in the real
//! engine, or is that cost inherent to dispatching *any* compute work on
//! this hardware/driver? Uses a simple matrix-vector-multiply shader
//! (src/bin/shaders/matmul.comp, precompiled to matmul.spv) — the shader's
//! own complexity doesn't matter for a dispatch-overhead measurement.
//!
//! Two timing modes:
//!   - "immediate": record + submit + wait a fence, once per dispatch —
//!     mirrors how the real engine's per-op sync currently behaves.
//!   - "batched": record N dispatches into one command buffer, submit once,
//!     wait once — mirrors cubecl-wgpu's command-buffer batching.

use ash::vk;
use std::ffi::CStr;
use std::time::Instant;

const IN_DIM: u32 = 4096;
const OUT_DIM: u32 = 4096;
const ITERS: u32 = 2000;

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..props.memory_type_count {
        let suitable = (type_bits & (1 << i)) != 0;
        let has_flags = props.memory_types[i as usize].property_flags.contains(flags);
        if suitable && has_flags {
            return i;
        }
    }
    panic!("no suitable memory type for flags {flags:?}");
}

fn main() {
    unsafe {
        let entry = ash::Entry::load().expect("failed to load Vulkan loader");

        let app_name = CStr::from_bytes_with_nul(b"vulkan_poc\0").unwrap();
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
            println!(
                "found device: {name} (type={:?})",
                props.device_type
            );
            if props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                || props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            {
                if chosen.is_none() {
                    chosen = Some(pd);
                }
            }
        }
        let physical_device = chosen.expect("no suitable GPU found");
        let props = instance.get_physical_device_properties(physical_device);
        println!(
            "using device: {}",
            CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy()
        );

        let queue_families =
            instance.get_physical_device_queue_family_properties(physical_device);
        let queue_family_index = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("no compute queue family") as u32;

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = instance
            .create_device(physical_device, &device_info, None)
            .expect("failed to create logical device");
        let queue = device.get_device_queue(queue_family_index, 0);

        let mem_props = instance.get_physical_device_memory_properties(physical_device);

        // Prefer the unified DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT type
        // (confirmed present on this Strix Halo APU) so uploads are direct
        // writes, not staged copies.
        let unified_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        let make_buffer = |size: vk::DeviceSize, usage: vk::BufferUsageFlags| {
            let buf_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.create_buffer(&buf_info, None).unwrap();
            let reqs = device.get_buffer_memory_requirements(buffer);
            let mem_type = find_memory_type(&mem_props, reqs.memory_type_bits, unified_flags);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(mem_type);
            let memory = device.allocate_memory(&alloc_info, None).unwrap();
            device.bind_buffer_memory(buffer, memory, 0).unwrap();
            (buffer, memory, reqs.size)
        };

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let (w_buf, w_mem, w_size) =
            make_buffer((IN_DIM as u64) * (OUT_DIM as u64) * 4, usage);
        let (x_buf, x_mem, x_size) = make_buffer((IN_DIM as u64) * 4, usage);
        let (o_buf, o_mem, o_size) = make_buffer((OUT_DIM as u64) * 4, usage);

        // Fill weight/input with deterministic data; leave output zeroed.
        let w_ptr = device
            .map_memory(w_mem, 0, w_size, vk::MemoryMapFlags::empty())
            .unwrap() as *mut f32;
        for i in 0..(IN_DIM as usize * OUT_DIM as usize) {
            *w_ptr.add(i) = ((i % 97) as f32) * 0.01;
        }
        device.unmap_memory(w_mem);

        let x_ptr = device
            .map_memory(x_mem, 0, x_size, vk::MemoryMapFlags::empty())
            .unwrap() as *mut f32;
        for i in 0..(IN_DIM as usize) {
            *x_ptr.add(i) = (i as f32) * 0.001;
        }
        device.unmap_memory(x_mem);
        let _ = o_size;

        // Shader module + pipeline.
        let spv_bytes = include_bytes!("shaders/matmul.spv");
        let spv_words: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        let shader_module = device.create_shader_module(&shader_info, None).unwrap();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = device.create_descriptor_set_layout(&dsl_info, None).unwrap();

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8); // two u32: in_dim, out_dim
        let set_layouts = [dsl];
        let push_ranges = [push_range];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .unwrap();

        let entry_point = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .unwrap()[0];

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let dpool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let dpool = device.create_descriptor_pool(&dpool_info, None).unwrap();
        let dsl_arr = [dsl];
        let dset_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(dpool)
            .set_layouts(&dsl_arr);
        let dset = device.allocate_descriptor_sets(&dset_alloc).unwrap()[0];

        let w_info = [vk::DescriptorBufferInfo::default().buffer(w_buf).offset(0).range(vk::WHOLE_SIZE)];
        let x_info = [vk::DescriptorBufferInfo::default().buffer(x_buf).offset(0).range(vk::WHOLE_SIZE)];
        let o_info = [vk::DescriptorBufferInfo::default().buffer(o_buf).offset(0).range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&w_info),
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&x_info),
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&o_info),
        ];
        device.update_descriptor_sets(&writes, &[]);

        let cpool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cpool = device.create_command_pool(&cpool_info, None).unwrap();

        let push_bytes = {
            let mut b = [0u8; 8];
            b[0..4].copy_from_slice(&IN_DIM.to_le_bytes());
            b[4..8].copy_from_slice(&OUT_DIM.to_le_bytes());
            b
        };

        let record_dispatch = |cmd: vk::CommandBuffer| {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[dset],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_bytes,
            );
            device.cmd_dispatch(cmd, OUT_DIM, 1, 1);
        };

        // ── Mode 1: immediate (record + submit + fence-wait per dispatch) ──
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cpool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = device.allocate_command_buffers(&cb_alloc).unwrap()[0];
        let fence_info = vk::FenceCreateInfo::default();
        let fence = device.create_fence(&fence_info, None).unwrap();

        // Warmup (pipeline caching, first-submit costs).
        for _ in 0..10 {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            record_dispatch(cmd);
            device.end_command_buffer(cmd).unwrap();
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.reset_fences(&[fence]).unwrap();
            device.queue_submit(queue, &[submit], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }

        let t0 = Instant::now();
        for _ in 0..ITERS {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            record_dispatch(cmd);
            device.end_command_buffer(cmd).unwrap();
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            device.reset_fences(&[fence]).unwrap();
            device.queue_submit(queue, &[submit], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }
        let immediate_elapsed = t0.elapsed();
        println!(
            "immediate: {ITERS} dispatches in {:.3}ms -> {:.1}us/dispatch",
            immediate_elapsed.as_secs_f64() * 1000.0,
            immediate_elapsed.as_secs_f64() * 1_000_000.0 / ITERS as f64
        );

        // ── Mode 2: batched (record N into one command buffer, submit once) ──
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        let begin_info = vk::CommandBufferBeginInfo::default();
        device.begin_command_buffer(cmd, &begin_info).unwrap();
        for _ in 0..ITERS {
            record_dispatch(cmd);
        }
        device.end_command_buffer(cmd).unwrap();

        let t0 = Instant::now();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        device.reset_fences(&[fence]).unwrap();
        device.queue_submit(queue, &[submit], fence).unwrap();
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        let batched_elapsed = t0.elapsed();
        println!(
            "batched:   {ITERS} dispatches in {:.3}ms -> {:.1}us/dispatch",
            batched_elapsed.as_secs_f64() * 1000.0,
            batched_elapsed.as_secs_f64() * 1_000_000.0 / ITERS as f64
        );

        // Sanity check output isn't all-zero (dispatch actually ran).
        let o_ptr = device
            .map_memory(o_mem, 0, (OUT_DIM as u64) * 4, vk::MemoryMapFlags::empty())
            .unwrap() as *const f32;
        let sample: Vec<f32> = (0..4).map(|i| *o_ptr.add(i)).collect();
        device.unmap_memory(o_mem);
        println!("output[0..4] = {sample:?} (non-zero confirms dispatch executed)");

        // Cleanup (best-effort; process exit would reclaim anyway).
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cpool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_pool(dpool, None);
        device.destroy_descriptor_set_layout(dsl, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_buffer(w_buf, None);
        device.free_memory(w_mem, None);
        device.destroy_buffer(x_buf, None);
        device.free_memory(x_mem, None);
        device.destroy_buffer(o_buf, None);
        device.free_memory(o_mem, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
