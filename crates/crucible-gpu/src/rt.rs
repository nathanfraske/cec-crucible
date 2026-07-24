// SPDX-License-Identifier: MIT
//! Ray-tracing-core load — hardware BVH traversal + triangle intersection.
//!
//! The FMA thrasher, tensor (cmma) and VRAM tests exercise the shader ALU, the
//! tensor cores and the memory system. None of them touch the **RT cores** — the
//! fixed-function units that walk the bounding-volume hierarchy and do
//! ray/triangle intersection. A dead or drifting RT core sails through every
//! other test in this suite and fails first in the customer's path-traced game
//! or GPU-render workload. This kernel drives those units directly.
//!
//! ## Why raw Vulkan (ash), not CubeCL or wgpu
//!
//! There is no portable, high-level, single-binary path to hardware ray
//! traversal in Rust today: CubeCL has no ray-query, wgpu's `EXPERIMENTAL_RAY_QUERY`
//! is Vulkan-only and gated behind an experimental flag, and OptiX is NVIDIA-only
//! with a driver-resident runtime. Raw Vulkan with `VK_KHR_ray_query` is the one
//! path that is stable, cross-vendor (NVIDIA RTX, AMD RDNA2+, Intel Arc) and ships
//! inside the single `cec-crucible` binary. `ash` is a thin FFI layer over the
//! Vulkan loader — the same loader the driver already installs — so nothing extra
//! is required on the target machine.
//!
//! ## No shader compiler required
//!
//! The ray-query compute shader is written in WGSL and compiled to SPIR-V *at
//! runtime* by `naga` (already in the wgpu dependency tree). So the build needs no
//! glslang/glslc/dxc and no Vulkan SDK, and there is no pre-compiled `.spv` blob to
//! keep in sync with the source.
//!
//! ## Verification (same rule as every other GPU test)
//!
//! The traversal is deterministic — a fixed camera fan against fixed geometry — so
//! every read-back must reproduce the first read-back's checksum bit-for-bit. Each
//! ray folds in its committed hit's **primitive index** (did the BVH find the right
//! triangle?) and **hit distance `t`** (is the intersection math right?), so an RT
//! core returning a wrong result changes the checksum and trips a FAIL. Liveness
//! (some rays must hit) catches a kernel that never ran.

use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use crate::GpuDevice;

/// Camera-fan resolution — 256x256 = 65_536 primary rays per dispatch.
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const N_RAYS: u32 = WIDTH * HEIGHT;
/// Compute workgroup size (must match the WGSL `@workgroup_size`).
const WG: u32 = 64;

/// RT-core load kernel.
#[derive(Debug, Clone)]
pub struct RtKernel {
    pub device: GpuDevice,
    /// Traversals per ray per dispatch — the RT-core stress + TDR granularity
    /// knob. ~192 keeps a dispatch well under the Windows 2 s watchdog on an
    /// RTX 3070 while fully feeding the traversal units.
    pub iters: u32,
    /// Read back + checksum every this many dispatches.
    pub verify_every: u64,
    /// Pop a live window showing the shaded ray-traced image (needs a
    /// `--features preview`, Windows build; ignored otherwise). Never affects the
    /// traversal or the checksum.
    pub preview: bool,
}

impl Default for RtKernel {
    fn default() -> Self {
        RtKernel {
            device: GpuDevice::Discrete(0),
            iters: 192,
            verify_every: 64,
            preview: false,
        }
    }
}

impl RtKernel {
    pub fn new(device: GpuDevice) -> Self {
        RtKernel {
            device,
            ..Default::default()
        }
    }
}

/// FNV-1a over bytes — content hash for self-consistency.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Uniform block handed to the shader. `repr(C)` so the layout matches WGSL's
/// `struct Params { iters, width, height, shade: u32 }`. `shade` is 1 only when a
/// preview window is open — then the shader also writes a lit colour image.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    iters: u32,
    width: u32,
    height: u32,
    shade: u32,
}

/// A displaced grid mesh — enough depth complexity that the rays hit many
/// different triangles, so the checksum is sensitive to BVH correctness. Returns
/// (positions, indices). Deterministic.
fn build_grid() -> (Vec<[f32; 3]>, Vec<u32>) {
    const GRID: usize = 64;
    let mut verts = Vec::with_capacity(GRID * GRID);
    for j in 0..GRID {
        for i in 0..GRID {
            let u = i as f32 / (GRID - 1) as f32 * 4.0 - 2.0;
            let v = j as f32 / (GRID - 1) as f32 * 4.0 - 2.0;
            let z = 0.3 * (3.0 * u).sin() * (3.0 * v).cos();
            verts.push([u, v, z]);
        }
    }
    let mut idx = Vec::with_capacity((GRID - 1) * (GRID - 1) * 6);
    for j in 0..GRID - 1 {
        for i in 0..GRID - 1 {
            let a = (j * GRID + i) as u32;
            let b = a + 1;
            let c = a + GRID as u32;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, idx)
}

/// Compile the bundled WGSL ray-query shader to SPIR-V with naga. No external
/// shader compiler is involved.
fn compile_spirv() -> Result<Vec<u32>, String> {
    let src = include_str!("rt.wgsl");
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| format!("rt shader parse: {}", e.emit_to_string(src)))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::RAY_QUERY,
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("rt shader validate: {e:?}"))?;
    // VK_KHR_ray_query executes in a SPIR-V 1.4 environment.
    let opts = naga::back::spv::Options {
        lang_version: (1, 4),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &opts, None)
        .map_err(|e| format!("rt shader spir-v emit: {e:?}"))
}

/// Round `v` up to a multiple of `align` (align must be a power of two, or 0/1).
fn align_up(v: u64, align: u64) -> u64 {
    if align <= 1 {
        v
    } else {
        (v + align - 1) & !(align - 1)
    }
}

/// A device buffer plus its backing allocation and cached device address.
struct Buf {
    buffer: vk::Buffer,
    address: vk::DeviceAddress,
    /// Host-mapped pointer, or null for GPU-only buffers.
    ptr: *mut u8,
}

/// Everything the RT test owns, so teardown is a single RAII `Drop` no matter
/// which setup step failed. Handles are null until created; the allocator is an
/// `Option` because it must be dropped (freeing `VkDeviceMemory`) *before* the
/// logical device is destroyed.
struct RtContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    as_dev: ash::khr::acceleration_structure::Device,
    allocator: Option<Allocator>,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    buffers: Vec<(vk::Buffer, Allocation)>,
    accels: Vec<vk::AccelerationStructureKHR>,
    dsl: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    scratch_align: u64,
    // runtime
    out_ptr: *mut u8,
    out_bytes: u64,
    // Host-visible RGBA8 colour image the shader writes for the preview window.
    // The buffer is always allocated + bound (so the descriptor layout is
    // constant), but its read-back pointer is only kept for a preview build.
    #[cfg(all(windows, feature = "preview"))]
    color_ptr: *mut u8,
    #[cfg(all(windows, feature = "preview"))]
    color_bytes: u64,
    groups: u32,
}

impl RtContext {
    /// Create a buffer, back it, bind it, and record it for teardown. Returns the
    /// buffer, its device address, and (for host-visible memory) a mapped pointer.
    fn make_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Buf, String> {
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&ci, None) }
            .map_err(|e| format!("create_buffer({name}): {e:?}"))?;
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alloc = self
            .allocator
            .as_mut()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements: reqs,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                // The buffer is not yet in `self.buffers`; free it so it does not leak.
                unsafe { self.device.destroy_buffer(buffer, None) };
                format!("allocate({name}): {e:?}")
            })?;
        unsafe {
            self.device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
        }
        .map_err(|e| format!("bind_buffer_memory({name}): {e:?}"))?;
        let address = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let ptr = alloc
            .mapped_ptr()
            .map(|p| p.as_ptr() as *mut u8)
            .unwrap_or(std::ptr::null_mut());
        self.buffers.push((buffer, alloc));
        Ok(Buf {
            buffer,
            address,
            ptr,
        })
    }

    /// Submit the current command buffer and block until the GPU signals the
    /// fence. `label` names the phase for error messages.
    fn submit_and_wait(&self, label: &str) -> Result<(), String> {
        let cmds = [self.cmd_buf];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let submits = [submit];
        let fences = [self.fence];
        unsafe {
            self.device
                .reset_fences(&fences)
                .map_err(|e| format!("{label}: reset_fences: {e:?}"))?;
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .map_err(|e| format!("{label}: queue_submit: {e:?}"))?;
            self.device
                .wait_for_fences(&fences, true, 5_000_000_000)
                .map_err(|e| format!("{label}: wait_for_fences: {e:?}"))?;
        }
        Ok(())
    }
}

impl Drop for RtContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.shader != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader, None);
            }
            if self.desc_pool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.desc_pool, None);
            }
            if self.dsl != vk::DescriptorSetLayout::null() {
                self.device.destroy_descriptor_set_layout(self.dsl, None);
            }
            for accel in self.accels.drain(..) {
                self.as_dev.destroy_acceleration_structure(accel, None);
            }
            for (buf, alloc) in self.buffers.drain(..) {
                self.device.destroy_buffer(buf, None);
                if let Some(a) = self.allocator.as_mut() {
                    let _ = a.free(alloc);
                }
            }
            if self.fence != vk::Fence::null() {
                self.device.destroy_fence(self.fence, None);
            }
            if self.cmd_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.cmd_pool, None);
            }
            // Free device memory (gpu-allocator) BEFORE destroying the device.
            self.allocator = None;
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Pick a physical device that supports ray-query, honoring the requested
/// discrete/integrated/default preference.
fn pick_physical(
    instance: &ash::Instance,
    want: GpuDevice,
) -> Result<vk::PhysicalDevice, String> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("enumerate_physical_devices: {e:?}"))?;

    // (device, is_discrete, is_integrated) for every ray-query-capable device.
    let mut capable: Vec<(vk::PhysicalDevice, bool, bool)> = Vec::new();
    for pd in devices {
        let mut rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut feats = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut rq)
            .push_next(&mut accel);
        unsafe { instance.get_physical_device_features2(pd, &mut feats) };
        if rq.ray_query == vk::TRUE && accel.acceleration_structure == vk::TRUE {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let disc = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
            let integ = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
            capable.push((pd, disc, integ));
        }
    }

    if capable.is_empty() {
        return Err("no ray-query capable GPU present".to_string());
    }

    let chosen = match want {
        GpuDevice::Discrete(i) => capable.iter().filter(|c| c.1).nth(i).map(|c| c.0),
        GpuDevice::Integrated(i) => capable.iter().filter(|c| c.2).nth(i).map(|c| c.0),
        GpuDevice::Default => capable
            .iter()
            .find(|c| c.1)
            .or_else(|| capable.first())
            .map(|c| c.0),
    };
    chosen.ok_or_else(|| format!("no ray-query capable GPU matching {}", want.label()))
}

/// Build the whole Vulkan context and both acceleration structures. Any failure
/// returns a message; partial state is cleaned up by `RtContext::drop`.
fn setup(kernel: &RtKernel) -> Result<RtContext, String> {
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| format!("Vulkan loader unavailable: {e:?}"))?;

    // Instance — API 1.2 (buffer_device_address is core there), no layers, no
    // extensions (headless).
    let app = vk::ApplicationInfo::default()
        .application_name(c"cec-crucible")
        .api_version(vk::make_api_version(0, 1, 2, 0));
    let inst_ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&inst_ci, None) }
        .map_err(|e| format!("create_instance: {e:?}"))?;

    // From here, clean up the instance on any early failure.
    let physical = match pick_physical(&instance, kernel.device) {
        Ok(p) => p,
        Err(e) => {
            unsafe { instance.destroy_instance(None) };
            return Err(e);
        }
    };

    let qf = {
        let props = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32)
    };
    let qf = match qf {
        Some(q) => q,
        None => {
            unsafe { instance.destroy_instance(None) };
            return Err("no compute queue family".to_string());
        }
    };

    // Logical device: enable ray-query, acceleration-structure and BDA features.
    let dev_exts = [
        vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ptr(),
        vk::KHR_RAY_QUERY_NAME.as_ptr(),
        vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ptr(),
        vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
    ];
    let priorities = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qf)
        .queue_priorities(&priorities)];
    let mut f_vk12 =
        vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
    let mut f_accel =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let dev_ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_extension_names(&dev_exts)
        .push_next(&mut f_vk12)
        .push_next(&mut f_accel)
        .push_next(&mut f_rq);
    let device = match unsafe { instance.create_device(physical, &dev_ci, None) } {
        Ok(d) => d,
        Err(e) => {
            unsafe { instance.destroy_instance(None) };
            return Err(format!("create_device: {e:?}"));
        }
    };

    let queue = unsafe { device.get_device_queue(qf, 0) };
    let as_dev = ash::khr::acceleration_structure::Device::new(&instance, &device);

    // Scratch buffers must be aligned to this.
    let scratch_align = {
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        as_props.min_acceleration_structure_scratch_offset_alignment as u64
    };

    let allocator = match Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: physical,
        debug_settings: Default::default(),
        buffer_device_address: true,
        allocation_sizes: Default::default(),
    }) {
        Ok(a) => a,
        Err(e) => {
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return Err(format!("gpu-allocator init: {e:?}"));
        }
    };

    // Context skeleton — from here, `?` + Drop handle all cleanup.
    let mut ctx = RtContext {
        _entry: entry,
        instance,
        device,
        as_dev,
        allocator: Some(allocator),
        queue,
        cmd_pool: vk::CommandPool::null(),
        cmd_buf: vk::CommandBuffer::null(),
        fence: vk::Fence::null(),
        buffers: Vec::new(),
        accels: Vec::new(),
        dsl: vk::DescriptorSetLayout::null(),
        desc_pool: vk::DescriptorPool::null(),
        desc_set: vk::DescriptorSet::null(),
        pipeline_layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
        shader: vk::ShaderModule::null(),
        scratch_align,
        out_ptr: std::ptr::null_mut(),
        out_bytes: 0,
        #[cfg(all(windows, feature = "preview"))]
        color_ptr: std::ptr::null_mut(),
        #[cfg(all(windows, feature = "preview"))]
        color_bytes: 0,
        groups: N_RAYS.div_ceil(WG),
    };

    // Command pool + buffer + fence.
    let pool_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qf)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    ctx.cmd_pool = unsafe { ctx.device.create_command_pool(&pool_ci, None) }
        .map_err(|e| format!("create_command_pool: {e:?}"))?;
    let cb_ai = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    ctx.cmd_buf = unsafe { ctx.device.allocate_command_buffers(&cb_ai) }
        .map_err(|e| format!("allocate_command_buffers: {e:?}"))?[0];
    ctx.fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .map_err(|e| format!("create_fence: {e:?}"))?;

    // ---- Geometry upload (host-visible; the driver reads it during the build) ----
    let (verts, indices) = build_grid();
    let tri_count = (indices.len() / 3) as u32;
    let vbytes = std::mem::size_of_val(verts.as_slice()) as u64;
    let ibytes = std::mem::size_of_val(indices.as_slice()) as u64;
    let as_input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

    let vbuf = ctx.make_buffer(vbytes, as_input, MemoryLocation::CpuToGpu, "rt-verts")?;
    let ibuf = ctx.make_buffer(ibytes, as_input, MemoryLocation::CpuToGpu, "rt-index")?;
    unsafe {
        std::ptr::copy_nonoverlapping(verts.as_ptr() as *const u8, vbuf.ptr, vbytes as usize);
        std::ptr::copy_nonoverlapping(indices.as_ptr() as *const u8, ibuf.ptr, ibytes as usize);
    }

    // ---- Bottom-level acceleration structure (the triangle mesh) ----
    let blas_addr = ctx.build_blas(&vbuf, &ibuf, verts.len() as u32, tri_count)?;

    // ---- Top-level acceleration structure (one instance of the BLAS) ----
    ctx.build_tlas(blas_addr)?;

    // ---- Output + colour + uniform buffers ----
    let out_bytes = (N_RAYS as u64) * 4;
    let obuf = ctx.make_buffer(
        out_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuToCpu,
        "rt-out",
    )?;
    ctx.out_ptr = obuf.ptr;
    ctx.out_bytes = out_bytes;

    // Colour image (RGBA8), host-visible so the preview can read it back. Always
    // allocated + bound (keeps one descriptor layout); the shader only writes it
    // when shading is on, and its pointer is only kept for a preview build.
    let color_bytes = (N_RAYS as u64) * 4;
    let cbuf = ctx.make_buffer(
        color_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuToCpu,
        "rt-color",
    )?;
    #[cfg(all(windows, feature = "preview"))]
    {
        ctx.color_ptr = cbuf.ptr;
        ctx.color_bytes = color_bytes;
    }

    // `shade` = 1 only for a real preview build with the window requested.
    #[cfg(all(windows, feature = "preview"))]
    let shade = if kernel.preview { 1u32 } else { 0u32 };
    #[cfg(not(all(windows, feature = "preview")))]
    let shade = 0u32;

    let params = Params {
        iters: kernel.iters.clamp(1, 1 << 20),
        width: WIDTH,
        height: HEIGHT,
        shade,
    };
    let ubuf = ctx.make_buffer(
        std::mem::size_of::<Params>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
        "rt-params",
    )?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            &params as *const Params as *const u8,
            ubuf.ptr,
            std::mem::size_of::<Params>(),
        );
    }

    // ---- Descriptors + pipeline ----
    ctx.build_pipeline()?;
    ctx.write_descriptors(obuf.buffer, ubuf.buffer, cbuf.buffer)?;
    ctx.record_dispatch();

    Ok(ctx)
}

impl RtContext {
    /// Build the bottom-level AS over the uploaded triangle mesh; returns its
    /// device address (for the TLAS instance to reference).
    fn build_blas(
        &mut self,
        vbuf: &Buf,
        ibuf: &Buf,
        vertex_count: u32,
        tri_count: u32,
    ) -> Result<vk::DeviceAddress, String> {
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vbuf.address,
            })
            .vertex_stride(12)
            .max_vertex(vertex_count - 1)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: ibuf.address,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        let geometries = [geometry];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.as_dev.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[tri_count],
                &mut sizes,
            )
        };

        let accel = self.create_accel(
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            "rt-blas",
        )?;
        let scratch_addr = self.make_scratch(sizes.build_scratch_size, "rt-blas-scratch")?;

        build_info = build_info
            .dst_acceleration_structure(accel)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(tri_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        let ranges = [range];
        let range_ptrs: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[&ranges];
        let infos = [build_info];

        self.begin_cmd()?;
        unsafe {
            self.as_dev
                .cmd_build_acceleration_structures(self.cmd_buf, &infos, range_ptrs);
        }
        self.end_cmd()?;
        self.submit_and_wait("blas build")?;

        Ok(unsafe {
            self.as_dev.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(accel),
            )
        })
    }

    /// Build the top-level AS holding a single instance of the BLAS.
    fn build_tlas(&mut self, blas_addr: vk::DeviceAddress) -> Result<(), String> {
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0,
                ],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_addr,
            },
        };
        let ibytes = std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64;
        let inst_buf = self.make_buffer(
            ibytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::CpuToGpu,
            "rt-instance",
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                &instance as *const _ as *const u8,
                inst_buf.ptr,
                ibytes as usize,
            );
        }

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: inst_buf.address,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        let geometries = [geometry];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.as_dev.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[1],
                &mut sizes,
            )
        };

        let accel = self.create_accel(
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            "rt-tlas",
        )?;
        let scratch_addr = self.make_scratch(sizes.build_scratch_size, "rt-tlas-scratch")?;

        build_info = build_info
            .dst_acceleration_structure(accel)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1);
        let ranges = [range];
        let range_ptrs: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[&ranges];
        let infos = [build_info];

        self.begin_cmd()?;
        unsafe {
            self.as_dev
                .cmd_build_acceleration_structures(self.cmd_buf, &infos, range_ptrs);
        }
        self.end_cmd()?;
        self.submit_and_wait("tlas build")?;

        // The TLAS handle must outlive the descriptor; it is stored in `accels`
        // (pushed by create_accel) so `desc_set` binding 0 stays valid.
        Ok(())
    }

    /// Allocate AS backing storage and create the acceleration structure object.
    fn create_accel(
        &mut self,
        size: u64,
        ty: vk::AccelerationStructureTypeKHR,
        name: &str,
    ) -> Result<vk::AccelerationStructureKHR, String> {
        let buf = self.make_buffer(
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            MemoryLocation::GpuOnly,
            name,
        )?;
        let ci = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buf.buffer)
            .offset(0)
            .size(size)
            .ty(ty);
        let accel = unsafe { self.as_dev.create_acceleration_structure(&ci, None) }
            .map_err(|e| format!("create_acceleration_structure({name}): {e:?}"))?;
        self.accels.push(accel);
        Ok(accel)
    }

    /// Allocate a scratch buffer and return a device address aligned to the AS
    /// scratch-offset requirement (the buffer is oversized to leave room).
    fn make_scratch(&mut self, size: u64, name: &str) -> Result<vk::DeviceAddress, String> {
        let align = self.scratch_align.max(1);
        let buf = self.make_buffer(
            size + align,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            name,
        )?;
        Ok(align_up(buf.address, align))
    }

    fn begin_cmd(&self) -> Result<(), String> {
        unsafe {
            self.device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset_command_buffer: {e:?}"))?;
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.cmd_buf, &bi)
                .map_err(|e| format!("begin_command_buffer: {e:?}"))
        }
    }

    fn end_cmd(&self) -> Result<(), String> {
        unsafe {
            self.device
                .end_command_buffer(self.cmd_buf)
                .map_err(|e| format!("end_command_buffer: {e:?}"))
        }
    }

    /// Compile the shader and build the compute pipeline + descriptor layout.
    fn build_pipeline(&mut self) -> Result<(), String> {
        let spirv = compile_spirv()?;
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&spirv);
        self.shader = unsafe { self.device.create_shader_module(&sm_ci, None) }
            .map_err(|e| format!("create_shader_module: {e:?}"))?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.dsl = unsafe { self.device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| format!("create_descriptor_set_layout: {e:?}"))?;

        let set_layouts = [self.dsl];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        self.pipeline_layout = unsafe { self.device.create_pipeline_layout(&pl_ci, None) }
            .map_err(|e| format!("create_pipeline_layout: {e:?}"))?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader)
            .name(c"main");
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None)
        }
        .map_err(|(_, e)| format!("create_compute_pipelines: {e:?}"))?;
        self.pipeline = pipelines[0];

        // Descriptor pool sized for one set: 1 AS, 2 storage buffers (out +
        // colour), 1 uniform.
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        self.desc_pool = unsafe { self.device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| format!("create_descriptor_pool: {e:?}"))?;
        let alloc_sets = [self.dsl];
        let ds_ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&alloc_sets);
        self.desc_set = unsafe { self.device.allocate_descriptor_sets(&ds_ai) }
            .map_err(|e| format!("allocate_descriptor_sets: {e:?}"))?[0];
        Ok(())
    }

    /// Point the descriptor set at the TLAS, the output/colour buffers and the
    /// uniforms.
    fn write_descriptors(
        &mut self,
        out_buf: vk::Buffer,
        uniform_buf: vk::Buffer,
        color_buf: vk::Buffer,
    ) -> Result<(), String> {
        // Binding 0: the TLAS (the last acceleration structure built).
        let tlas = *self
            .accels
            .last()
            .ok_or_else(|| "no TLAS to bind".to_string())?;
        let tlas_arr = [tlas];
        let mut as_write =
            vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlas_arr);
        let mut w0 = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut as_write);
        // descriptor_count is not set by push_next; set it explicitly to 1.
        w0.descriptor_count = 1;

        let out_info = [vk::DescriptorBufferInfo::default()
            .buffer(out_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let w1 = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&out_info);

        let uni_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let w2 = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&uni_info);

        let color_info = [vk::DescriptorBufferInfo::default()
            .buffer(color_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let w3 = vk::WriteDescriptorSet::default()
            .dst_set(self.desc_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&color_info);

        let writes = [w0, w1, w2, w3];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    /// Record the persistent dispatch command buffer (bind pipeline + set,
    /// dispatch, barrier so the host sees the writes).
    fn record_dispatch(&mut self) {
        // Recorded once; re-submitted every tick. Uses SIMULTANEOUS_USE so the
        // same recording can be resubmitted without a reset.
        unsafe {
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
            let _ = self.device.begin_command_buffer(self.cmd_buf, &bi);
            self.device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            let sets = [self.desc_set];
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );
            self.device.cmd_dispatch(self.cmd_buf, self.groups, 1, 1);
            // Make the shader's storage writes available to a host read.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            let _ = self.device.end_command_buffer(self.cmd_buf);
        }
    }

    /// Read the output buffer back into a Vec.
    fn readback(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.out_bytes as usize];
        if !self.out_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.out_ptr, out.as_mut_ptr(), out.len());
            }
        }
        out
    }

    /// Read the shaded colour image (RGBA8) back into a Vec, for the preview.
    #[cfg(all(windows, feature = "preview"))]
    fn readback_color(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.color_bytes as usize];
        if !self.color_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.color_ptr, out.as_mut_ptr(), out.len());
            }
        }
        out
    }
}

/// Live-preview holder for `rt`: a window + a wgpu device (separate from the ash
/// compute device) that upscales and shows the shaded colour image each frame.
///
/// Field order matters: `presenter` (which owns the surface created from the
/// window's raw handle) is declared before `window` so the surface is dropped
/// before the window it borrows.
#[cfg(all(windows, feature = "preview"))]
struct RtPreview {
    presenter: crate::preview::PixelPresenter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    window: crate::preview::PreviewWindow,
}

#[cfg(all(windows, feature = "preview"))]
impl RtPreview {
    /// Preview window edge (the traced image is upscaled to fill it).
    const WIN: u32 = 768;

    fn open(kernel: &RtKernel) -> Option<RtPreview> {
        if !kernel.preview {
            return None;
        }
        let title = format!("cec-crucible rt — {}", kernel.device.label());
        let window = match crate::preview::PreviewWindow::open(&title, Self::WIN, Self::WIN) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("note: --preview window unavailable ({e}); running headless");
                return None;
            }
        };
        let instance = wgpu::Instance::default();
        // Safety: the window outlives the surface (both owned by this RtPreview,
        // `presenter` — holding the surface — dropped before `window`).
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(window.raw_display_handle()),
            raw_window_handle: window.raw_window_handle(),
        };
        let surface = match unsafe { instance.create_surface_unsafe(target) } {
            Ok(s) => s,
            Err(e) => {
                eprintln!("note: --preview surface unavailable ({e}); running headless");
                return None;
            }
        };
        let adapter = cubecl::future::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            },
        ))
        .ok()?;
        let (device, queue) = cubecl::future::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rt-preview"),
                ..Default::default()
            },
        ))
        .ok()?;
        let presenter = crate::preview::PixelPresenter::new(
            &device,
            &adapter,
            surface,
            Self::WIN,
            Self::WIN,
            WIDTH,
            HEIGHT,
        )
        .ok()?;
        Some(RtPreview {
            window,
            device,
            queue,
            presenter,
        })
    }

    /// Pump the window and, when due (~60 Hz), read back the colour image and
    /// present it. Returns false once the window has been closed.
    fn show(&self, ctx: &RtContext) -> bool {
        if !self.window.pump() {
            return false;
        }
        if self.presenter.due() {
            let pixels = ctx.readback_color();
            self.presenter
                .present_rgba(&self.device, &self.queue, &pixels);
        }
        true
    }
}

impl LoadKernel for RtKernel {
    fn name(&self) -> &str {
        "rt"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let label = self.device.label();

        // All of Vulkan setup + AS build under catch_unwind: a missing loader,
        // an unsupported GPU or a driver fault becomes a clean setup failure
        // rather than a panic that takes down a whole cross-load run.
        let ctx = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| setup(self))) {
            Ok(Ok(c)) => c,
            Ok(Err(why)) => return LoadResult::setup_failure(format!("{label} rt: {why}")),
            Err(_) => {
                return LoadResult::setup_failure(format!("{label} rt: Vulkan setup panicked"))
            }
        };

        // Probe the first dispatch: submit, wait, and require liveness (some rays
        // must hit) before entering the timed loop.
        if let Err(why) = ctx.submit_and_wait("first dispatch") {
            return LoadResult::setup_failure(format!("{label} rt: {why}"));
        }
        let first = ctx.readback();
        // Liveness: a live traversal must leave some non-zero bytes (some rays
        // hit). A byte scan avoids any alignment assumption on the read-back Vec.
        if !first.iter().any(|&b| b != 0) {
            return LoadResult::setup_failure(format!(
                "{label} rt: output all-zero on probe — no rays hit (bad build?)"
            ));
        }
        let reference0 = fnv1a(&first);

        // Optional live window (windows+preview build). A failure here just runs
        // headless; the shader already wrote the colour image regardless.
        #[cfg(all(windows, feature = "preview"))]
        let preview = RtPreview::open(self);

        let backend = "vulkan-rt".to_string();
        let mut driver = ShapeDriver::start(budget, stop, markers, "rt", backend.clone());
        let start = Instant::now();
        let mut dispatches: u64 = 0;
        let mut verifications: u64 = 0;
        let mut errors: u64 = 0;
        let mut reference: Option<u64> = Some(reference0);

        loop {
            match driver.tick() {
                Tick::Work => {
                    if let Err(why) = ctx.submit_and_wait("dispatch") {
                        errors += 1;
                        return LoadResult::new(
                            true,
                            dispatches,
                            reference.unwrap_or(0),
                            errors,
                            format!("{label} rt: DEVICE LOST ({why})"),
                        );
                    }
                    dispatches += 1;

                    if dispatches.is_multiple_of(self.verify_every) {
                        let bytes = ctx.readback();
                        let h = fnv1a(&bytes);
                        verifications += 1;
                        match reference {
                            Some(r) if r != h => {
                                errors += 1;
                                return LoadResult::new(
                                    true,
                                    dispatches,
                                    h,
                                    errors,
                                    format!(
                                        "{label} rt: miscompare at dispatch {dispatches} \
                                         (got {h:#018x}, expected {r:#018x}) — RT-core soft error"
                                    ),
                                );
                            }
                            Some(_) => {}
                            None => reference = Some(h),
                        }
                    }

                    #[cfg(all(windows, feature = "preview"))]
                    if let Some(pv) = &preview {
                        if !pv.show(&ctx) {
                            stop.stop();
                        }
                    }
                }
                // Keep the preview responsive through idle phases of a burst shape.
                Tick::Idle => {
                    #[cfg(all(windows, feature = "preview"))]
                    if let Some(pv) = &preview {
                        if !pv.show(&ctx) {
                            stop.stop();
                        }
                    }
                }
                Tick::Stop => break,
            }
        }

        let secs = start.elapsed().as_secs_f64();
        // Every ray traces `iters` times per dispatch.
        let traversals =
            dispatches as f64 * N_RAYS as f64 * self.iters.max(1) as f64;
        let mrays = if secs > 0.0 {
            traversals / secs / 1.0e6
        } else {
            0.0
        };
        let detail = format!(
            "{label} {backend} {WIDTH}x{HEIGHT} rays x{} traces, {dispatches} dispatch(es), \
             {verifications} verified, ~{mrays:.0} Mray/s",
            self.iters
        );
        LoadResult::new(true, dispatches, reference.unwrap_or(0), errors, detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_watertight_indices() {
        let (verts, idx) = build_grid();
        assert_eq!(verts.len(), 64 * 64);
        assert_eq!(idx.len(), 63 * 63 * 6);
        // Every index is in range.
        assert!(idx.iter().all(|&i| (i as usize) < verts.len()));
    }

    #[test]
    fn shader_compiles_to_spirv_with_ray_query() {
        let words = compile_spirv().expect("naga must compile the rt shader");
        assert_eq!(words.first().copied(), Some(0x0723_0203)); // SPIR-V magic
        // OpTypeRayQueryKHR (4472) must be present.
        assert!(words.iter().any(|&w| (w & 0xffff) == 4472));
    }

    #[test]
    fn align_up_rounds() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(42, 1), 42);
    }
}
