// SPDX-License-Identifier: MIT
//! OptiX NVIDIA-native path-tracing test (Phase 2) — RT + SM stress on the
//! ray-tracing pipeline, verified by same-device self-consistency.
//!
//! The host side is a hand-rolled FFI (zero NVIDIA SDK source; the ABI is
//! re-declared from the public OptiX docs, with every struct size/offset
//! ground-truthed by a compiler probe — see the `layout_matches_ground_truth`
//! test). The OptiX runtime is driver-resident (`nvoptix.dll`), so nothing but
//! the driver is needed on the target. The device kernel is the committed
//! `path_tracer.ptx` (compiled from `path_tracer.cu` by nvcc at build).
//!
//! The CUDA context + device memory come from `cudarc` (driver-only, no toolkit
//! at runtime). Determinism is inherited from the kernel (fixed per-pixel RNG),
//! so every launch reproduces the checksum bit-for-bit.

#![allow(non_snake_case, non_camel_case_types)]
// cudarc 0.19 deprecated memcpy_stod/memcpy_dtov in favour of newer names; the
// deprecated ones still work and keep this NVIDIA-only module simple.
#![allow(deprecated)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;

use crate::GpuDevice;

// The device kernel, compiled to PTX at build (build_ptx.ps1). Committed so the
// target needs no toolkit; the driver JIT-links it.
const PATH_TRACER_PTX: &[u8] = include_bytes!("path_tracer.ptx");

// ---------------------------------------------------------------------------
// OptiX ABI (re-declared; sizes verified against the compiler in tests)
// ---------------------------------------------------------------------------

type OptixResult = c_uint; // 0 == OPTIX_SUCCESS
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuDeviceptr = u64;
type OptixDeviceContext = *mut c_void;
type OptixModule = *mut c_void;
type OptixProgramGroup = *mut c_void;
type OptixPipeline = *mut c_void;
type OptixTraversableHandle = u64;

const OPTIX_ABI_VERSION: c_uint = 118;
const OPTIX_SBT_RECORD_HEADER_SIZE: usize = 32;

// enums (values from optix_types.h)
const OPTIX_PROGRAM_GROUP_KIND_RAYGEN: c_uint = 0x2421;
const OPTIX_PROGRAM_GROUP_KIND_MISS: c_uint = 0x2422;
const OPTIX_PROGRAM_GROUP_KIND_HITGROUP: c_uint = 0x2424;
const OPTIX_BUILD_INPUT_TYPE_TRIANGLES: c_uint = 0x2141;
const OPTIX_VERTEX_FORMAT_FLOAT3: c_uint = 0x2121;
const OPTIX_INDICES_FORMAT_UNSIGNED_INT3: c_uint = 0x2103;
const OPTIX_BUILD_OPERATION_BUILD: c_uint = 0x2161;
const OPTIX_BUILD_FLAG_PREFER_FAST_TRACE: c_uint = 1 << 2;
const OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS: c_uint = 1 << 0;

type OptixLogCallback =
    extern "C" fn(level: c_uint, tag: *const c_char, message: *const c_char, cbdata: *mut c_void);

#[repr(C)]
struct OptixDeviceContextOptions {
    log_callback_function: Option<OptixLogCallback>,
    log_callback_data: *mut c_void,
    log_callback_level: c_int,
    validation_mode: c_uint,
}

#[repr(C)]
struct OptixModuleCompileOptions {
    max_register_count: c_int,
    opt_level: c_uint,
    debug_level: c_uint,
    bound_values: *const c_void,
    num_bound_values: c_uint,
    num_payload_types: c_uint,
    payload_types: *const c_void,
    base_module: OptixModule,
}

#[repr(C)]
struct OptixPipelineCompileOptions {
    uses_motion_blur: c_int,
    traversable_graph_flags: c_uint,
    num_payload_values: c_int,
    num_attribute_values: c_int,
    exception_flags: c_uint,
    pipeline_launch_params_variable_name: *const c_char,
    pipeline_launch_params_size_in_bytes: usize,
    uses_primitive_type_flags: c_uint,
    allow_opacity_micromaps: c_int,
    allow_clustered_geometry: c_int,
}

#[repr(C)]
struct OptixPipelineLinkOptions {
    max_trace_depth: c_uint,
    max_continuation_callable_depth: c_uint,
    max_direct_callable_depth_from_state: c_uint,
    max_direct_callable_depth_from_traversal: c_uint,
    max_traversable_graph_depth: c_uint,
}

// ProgramGroupDesc: { kind:u32, flags:u32, union[6 ptrs] } = 56 bytes.
#[repr(C)]
struct OptixProgramGroupDesc {
    kind: c_uint,
    flags: c_uint,
    // union payload, 6 pointers (largest variant = hitgroup: CH/AH/IS module+name)
    u: [*const c_void; 6],
}

#[repr(C)]
struct OptixProgramGroupOptions {
    payload_type: *const c_void,
}

#[repr(C)]
struct OptixShaderBindingTable {
    raygen_record: CuDeviceptr,
    exception_record: CuDeviceptr,
    miss_record_base: CuDeviceptr,
    miss_record_stride: c_uint,
    miss_record_count: c_uint,
    hitgroup_record_base: CuDeviceptr,
    hitgroup_record_stride: c_uint,
    hitgroup_record_count: c_uint,
    callables_record_base: CuDeviceptr,
    callables_record_stride: c_uint,
    callables_record_count: c_uint,
}

#[repr(C)]
struct OptixMotionOptions {
    num_keys: u16,
    flags: u16,
    time_begin: f32,
    time_end: f32,
}

#[repr(C)]
struct OptixAccelBuildOptions {
    build_flags: c_uint,
    operation: c_uint,
    motion_options: OptixMotionOptions,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct OptixAccelBufferSizes {
    output_size: usize,
    temp_size: usize,
    temp_update_size: usize,
}

// 144 bytes; fields at the verified offsets, trailing micromap padding zeroed.
#[repr(C)]
struct OptixBuildInputTriangleArray {
    vertex_buffers: *const CuDeviceptr, // @0
    num_vertices: c_uint,               // @8
    vertex_format: c_uint,              // @12
    vertex_stride: c_uint,              // @16
    index_buffer: CuDeviceptr,          // @24
    num_index_triplets: c_uint,         // @32
    index_format: c_uint,               // @36
    index_stride: c_uint,               // @40
    pre_transform: CuDeviceptr,         // @48
    flags: *const c_uint,               // @56
    num_sbt_records: c_uint,            // @64
    sbt_index_offset_buffer: CuDeviceptr, // @72
    sbt_index_offset_size: c_uint,      // @80
    sbt_index_offset_stride: c_uint,    // @84
    primitive_index_offset: c_uint,     // @88
    transform_format: c_uint,           // @92
    _tail: [u8; 48],                    // @96..144 (opacity/displacement micromap)
}

// 1032 bytes; type @0, triangle union @8, tail padding to full union size.
#[repr(C)]
struct OptixBuildInput {
    kind: c_uint, // @0 (padded to @8 for the 8-aligned union)
    triangles: OptixBuildInputTriangleArray, // @8
    _tail: [u8; 880], // @152..1032
}

// The launch parameters — MUST match `struct Params` in path_tracer.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    checksum: CuDeviceptr,
    image: CuDeviceptr,
    normals: CuDeviceptr,
    indices: CuDeviceptr,
    handle: OptixTraversableHandle,
    width: c_uint,
    height: c_uint,
    samples: c_uint,
    bounces: c_uint,
    seed: c_uint,
    shade: c_uint,
    cam_pos: [f32; 3],
    cam_fwd: [f32; 3],
    cam_right: [f32; 3],
    cam_up: [f32; 3],
    fov: f32,
}

// ---- the 60-entry function table; typed for the entries we call ----
type FnGetErrorString = extern "C" fn(OptixResult) -> *const c_char;
type FnDeviceContextCreate =
    extern "C" fn(CuContext, *const OptixDeviceContextOptions, *mut OptixDeviceContext) -> OptixResult;
type FnDeviceContextDestroy = extern "C" fn(OptixDeviceContext) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnModuleCreate = extern "C" fn(
    OptixDeviceContext,
    *const OptixModuleCompileOptions,
    *const OptixPipelineCompileOptions,
    *const c_char,
    usize,
    *mut c_char,
    *mut usize,
    *mut OptixModule,
) -> OptixResult;
type FnModuleDestroy = extern "C" fn(OptixModule) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnProgramGroupCreate = extern "C" fn(
    OptixDeviceContext,
    *const OptixProgramGroupDesc,
    c_uint,
    *const OptixProgramGroupOptions,
    *mut c_char,
    *mut usize,
    *mut OptixProgramGroup,
) -> OptixResult;
type FnProgramGroupDestroy = extern "C" fn(OptixProgramGroup) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnPipelineCreate = extern "C" fn(
    OptixDeviceContext,
    *const OptixPipelineCompileOptions,
    *const OptixPipelineLinkOptions,
    *const OptixProgramGroup,
    c_uint,
    *mut c_char,
    *mut usize,
    *mut OptixPipeline,
) -> OptixResult;
type FnPipelineDestroy = extern "C" fn(OptixPipeline) -> OptixResult;
type FnPipelineSetStackSize =
    extern "C" fn(OptixPipeline, c_uint, c_uint, c_uint, c_uint) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnAccelComputeMemoryUsage = extern "C" fn(
    OptixDeviceContext,
    *const OptixAccelBuildOptions,
    *const OptixBuildInput,
    c_uint,
    *mut OptixAccelBufferSizes,
) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnAccelBuild = extern "C" fn(
    OptixDeviceContext,
    CuStream,
    *const OptixAccelBuildOptions,
    *const OptixBuildInput,
    c_uint,
    CuDeviceptr,
    usize,
    CuDeviceptr,
    usize,
    *mut OptixTraversableHandle,
    *const c_void,
    c_uint,
) -> OptixResult;
type FnSbtRecordPackHeader = extern "C" fn(OptixProgramGroup, *mut c_void) -> OptixResult;
#[allow(clippy::type_complexity)]
type FnLaunch = extern "C" fn(
    OptixPipeline,
    CuStream,
    CuDeviceptr,
    usize,
    *const OptixShaderBindingTable,
    c_uint,
    c_uint,
    c_uint,
) -> OptixResult;

#[repr(C)]
struct OptixFunctionTable {
    get_error_name: *const c_void,                 // 0
    get_error_string: Option<FnGetErrorString>,    // 1
    device_context_create: Option<FnDeviceContextCreate>, // 2
    device_context_destroy: Option<FnDeviceContextDestroy>, // 3
    _p4_11: [*const c_void; 8],                    // 4..=11
    module_create: Option<FnModuleCreate>,         // 12
    _p13_17: [*const c_void; 5],                   // 13..=17
    module_destroy: Option<FnModuleDestroy>,       // 18
    _p19_23: [*const c_void; 5],                   // 19..=23
    program_group_create: Option<FnProgramGroupCreate>, // 24
    program_group_destroy: Option<FnProgramGroupDestroy>, // 25
    _p26: *const c_void,                           // 26
    pipeline_create: Option<FnPipelineCreate>,     // 27
    pipeline_destroy: Option<FnPipelineDestroy>,   // 28
    _p29: *const c_void,                           // 29
    pipeline_set_stack_size: Option<FnPipelineSetStackSize>, // 30
    _p31: *const c_void,                           // 31
    accel_compute_memory_usage: Option<FnAccelComputeMemoryUsage>, // 32
    accel_build: Option<FnAccelBuild>,             // 33
    _p34_47: [*const c_void; 14],                  // 34..=47
    sbt_record_pack_header: Option<FnSbtRecordPackHeader>, // 48
    launch: Option<FnLaunch>,                      // 49
    _p50_59: [*const c_void; 10],                  // 50..=59
}

type FnQueryFunctionTable = extern "C" fn(
    c_uint,
    c_int,
    *const *const c_char,
    *const *const c_char,
    *mut OptixFunctionTable,
    usize,
) -> OptixResult;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> *const c_void;
}

extern "C" fn log_cb(level: c_uint, tag: *const c_char, msg: *const c_char, _c: *mut c_void) {
    let t = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    eprintln!("  [optix {level}] {t}: {m}");
}

fn find_nvoptix() -> Option<String> {
    let base = r"C:\Windows\System32\DriverStore\FileRepository";
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for e in std::fs::read_dir(base).ok()?.flatten() {
        let dll = e.path().join("nvoptix.dll");
        if dll.exists() {
            let mt = dll
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            let s = dll.to_string_lossy().to_string();
            if best.as_ref().is_none_or(|(bt, _)| mt > *bt) {
                best = Some((mt, s));
            }
        }
    }
    best.map(|(_, s)| s)
}

/// Load the OptiX function table from the driver's nvoptix.dll.
fn load_optix_table() -> Result<OptixFunctionTable, String> {
    let path = find_nvoptix().ok_or("nvoptix.dll not found (driver too old / non-NVIDIA)")?;
    let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module == 0 {
        return Err("LoadLibrary(nvoptix.dll) failed".into());
    }
    let sym = unsafe { GetProcAddress(module, c"optixQueryFunctionTable".as_ptr() as *const u8) };
    if sym.is_null() {
        return Err("optixQueryFunctionTable symbol missing".into());
    }
    let query: FnQueryFunctionTable = unsafe { std::mem::transmute(sym) };
    let mut table: OptixFunctionTable = unsafe { std::mem::zeroed() };
    let r = query(
        OPTIX_ABI_VERSION,
        0,
        std::ptr::null(),
        std::ptr::null(),
        &mut table,
        std::mem::size_of::<OptixFunctionTable>(),
    );
    if r != 0 {
        return Err(format!(
            "optixQueryFunctionTable failed ({r}); driver may predate OptiX ABI {OPTIX_ABI_VERSION}"
        ));
    }
    Ok(table)
}

/// OptiX path-tracing load kernel.
#[derive(Debug, Clone)]
pub struct OptixKernel {
    pub device: GpuDevice,
    pub samples: u32,
    pub bounces: u32,
    pub verify_every: u64,
}

impl Default for OptixKernel {
    fn default() -> Self {
        OptixKernel {
            device: GpuDevice::Discrete(0),
            samples: 16,
            bounces: 8,
            verify_every: 16,
        }
    }
}

impl OptixKernel {
    pub fn new(device: GpuDevice) -> Self {
        OptixKernel {
            device,
            ..Default::default()
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;

impl LoadKernel for OptixKernel {
    fn name(&self) -> &str {
        "optix"
    }
    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let label = self.device.label();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_optix(self, budget, stop, markers)
        })) {
            Ok(Ok(res)) => res,
            Ok(Err(why)) => LoadResult::setup_failure(format!("{label} optix: {why}")),
            Err(_) => LoadResult::setup_failure(format!("{label} optix: setup panicked")),
        }
    }
}

// Kept in the module so its buffers live for the whole run.
struct OptixCtx {
    table: OptixFunctionTable,
    _cuda: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    context: OptixDeviceContext,
    module: OptixModule,
    groups: [OptixProgramGroup; 3],
    pipeline: OptixPipeline,
    sbt: OptixShaderBindingTable,
    handle: OptixTraversableHandle,
    // device buffers held alive for the whole run (type-erased; any CudaSlice<T>)
    _keepalive: Vec<Box<dyn std::any::Any>>,
    d_checksum: cudarc::driver::CudaSlice<u32>,
    d_params: cudarc::driver::CudaSlice<u8>,
    params: Params,
}

fn err(table: &OptixFunctionTable, r: OptixResult, what: &str) -> Result<(), String> {
    if r == 0 {
        return Ok(());
    }
    let msg = table
        .get_error_string
        .map(|f| unsafe { CStr::from_ptr(f(r)) }.to_string_lossy().into_owned())
        .unwrap_or_default();
    Err(format!("{what} failed: {r} ({msg})"))
}

fn setup(k: &OptixKernel) -> Result<OptixCtx, String> {
    use cudarc::driver::{CudaContext, DevicePtr};

    let table = load_optix_table()?;
    let cuda = CudaContext::new(0).map_err(|e| format!("CUDA context: {e:?}"))?;
    let stream = cuda.default_stream();
    let cu_ctx = cuda.cu_ctx() as CuContext;
    let cu_stream = stream.cu_stream() as CuStream;

    // --- device context ---
    let opts = OptixDeviceContextOptions {
        log_callback_function: Some(log_cb),
        log_callback_data: std::ptr::null_mut(),
        log_callback_level: 3,
        validation_mode: 0,
    };
    let mut context: OptixDeviceContext = std::ptr::null_mut();
    let create = table.device_context_create.ok_or("no create fn")?;
    err(&table, create(cu_ctx, &opts, &mut context), "deviceContextCreate")?;

    // --- module from the committed PTX ---
    let mco = OptixModuleCompileOptions {
        max_register_count: 0,
        opt_level: 0,
        debug_level: 0,
        bound_values: std::ptr::null(),
        num_bound_values: 0,
        num_payload_types: 0,
        payload_types: std::ptr::null(),
        base_module: std::ptr::null_mut(),
    };
    let pco = OptixPipelineCompileOptions {
        uses_motion_blur: 0,
        traversable_graph_flags: OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        num_payload_values: 2,
        num_attribute_values: 2,
        exception_flags: 0,
        pipeline_launch_params_variable_name: c"params".as_ptr(),
        pipeline_launch_params_size_in_bytes: std::mem::size_of::<Params>(),
        uses_primitive_type_flags: 0,
        allow_opacity_micromaps: 0,
        allow_clustered_geometry: 0,
    };
    let mut module: OptixModule = std::ptr::null_mut();
    let module_create = table.module_create.ok_or("no module fn")?;
    err(
        &table,
        module_create(
            context,
            &mco,
            &pco,
            PATH_TRACER_PTX.as_ptr() as *const c_char,
            PATH_TRACER_PTX.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut module,
        ),
        "moduleCreate",
    )?;

    // --- program groups: raygen, miss, hitgroup ---
    let pg_opts = OptixProgramGroupOptions {
        payload_type: std::ptr::null(),
    };
    let mk_desc = |kind: c_uint, m: OptixModule, name: *const c_char, hit_ch: bool| {
        let mut u = [std::ptr::null::<c_void>(); 6];
        if hit_ch {
            u[0] = m as *const c_void; // moduleCH
            u[1] = name as *const c_void; // entryFunctionNameCH
        } else {
            u[0] = m as *const c_void; // module
            u[1] = name as *const c_void; // entryFunctionName
        }
        OptixProgramGroupDesc { kind, flags: 0, u }
    };
    let descs = [
        mk_desc(OPTIX_PROGRAM_GROUP_KIND_RAYGEN, module, c"__raygen__pt".as_ptr(), false),
        mk_desc(OPTIX_PROGRAM_GROUP_KIND_MISS, module, c"__miss__pt".as_ptr(), false),
        mk_desc(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, module, c"__closesthit__pt".as_ptr(), true),
    ];
    let mut groups = [std::ptr::null_mut::<c_void>(); 3];
    let pg_create = table.program_group_create.ok_or("no pg fn")?;
    err(
        &table,
        pg_create(
            context,
            descs.as_ptr(),
            3,
            &pg_opts,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            groups.as_mut_ptr(),
        ),
        "programGroupCreate",
    )?;

    // --- pipeline ---
    let plo = OptixPipelineLinkOptions {
        max_trace_depth: 2,
        max_continuation_callable_depth: 0,
        max_direct_callable_depth_from_state: 0,
        max_direct_callable_depth_from_traversal: 0,
        max_traversable_graph_depth: 1,
    };
    let mut pipeline: OptixPipeline = std::ptr::null_mut();
    let pipe_create = table.pipeline_create.ok_or("no pipeline fn")?;
    err(
        &table,
        pipe_create(
            context,
            &pco,
            &plo,
            groups.as_ptr(),
            3,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut pipeline,
        ),
        "pipelineCreate",
    )?;
    if let Some(set_stack) = table.pipeline_set_stack_size {
        err(&table, set_stack(pipeline, 0, 0, 4096, 1), "pipelineSetStackSize")?;
    }

    let mut buffers: Vec<Box<dyn std::any::Any>> = Vec::new();

    // --- SBT: one 32-byte record per group, no record data ---
    let mut make_record = |group: OptixProgramGroup| -> Result<CuDeviceptr, String> {
        let mut hdr = [0u8; OPTIX_SBT_RECORD_HEADER_SIZE];
        let pack = table.sbt_record_pack_header.ok_or("no sbt fn")?;
        err(&table, pack(group, hdr.as_mut_ptr() as *mut c_void), "sbtRecordPackHeader")?;
        let d = stream
            .memcpy_stod(&hdr)
            .map_err(|e| format!("sbt upload: {e:?}"))?;
        let ptr = d.device_ptr(&stream).0;
        buffers.push(Box::new(d));
        Ok(ptr)
    };
    let raygen_rec = make_record(groups[0])?;
    let miss_rec = make_record(groups[1])?;
    let hit_rec = make_record(groups[2])?;
    let sbt = OptixShaderBindingTable {
        raygen_record: raygen_rec,
        exception_record: 0,
        miss_record_base: miss_rec,
        miss_record_stride: OPTIX_SBT_RECORD_HEADER_SIZE as c_uint,
        miss_record_count: 1,
        hitgroup_record_base: hit_rec,
        hitgroup_record_stride: OPTIX_SBT_RECORD_HEADER_SIZE as c_uint,
        hitgroup_record_count: 1,
        callables_record_base: 0,
        callables_record_stride: 0,
        callables_record_count: 0,
    };

    // --- geometry: upload the torus knot, build a GAS ---
    let (verts, indices, normals) = crate::geom::build_torus_knot();
    let verts_flat: Vec<f32> = verts.iter().flat_map(|v| *v).collect();
    let normals_flat: Vec<f32> = normals.iter().flat_map(|v| *v).collect();
    let tri_count = (indices.len() / 3) as u32;

    let d_verts = stream.memcpy_stod(&verts_flat).map_err(|e| format!("verts: {e:?}"))?;
    let d_indices = stream.memcpy_stod(&indices).map_err(|e| format!("indices: {e:?}"))?;
    let d_normals = stream.memcpy_stod(&normals_flat).map_err(|e| format!("normals: {e:?}"))?;
    let d_verts_ptr = d_verts.device_ptr(&stream).0;
    let d_indices_ptr = d_indices.device_ptr(&stream).0;
    let d_normals_ptr = d_normals.device_ptr(&stream).0;

    let vbuf_ptrs = [d_verts_ptr];
    let geo_flags = [0u32]; // OPTIX_GEOMETRY_FLAG_NONE
    let tri = OptixBuildInputTriangleArray {
        vertex_buffers: vbuf_ptrs.as_ptr(),
        num_vertices: verts.len() as c_uint,
        vertex_format: OPTIX_VERTEX_FORMAT_FLOAT3,
        vertex_stride: 12,
        index_buffer: d_indices_ptr,
        num_index_triplets: tri_count,
        index_format: OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
        index_stride: 0,
        pre_transform: 0,
        flags: geo_flags.as_ptr(),
        num_sbt_records: 1,
        sbt_index_offset_buffer: 0,
        sbt_index_offset_size: 0,
        sbt_index_offset_stride: 0,
        primitive_index_offset: 0,
        transform_format: 0,
        _tail: [0u8; 48],
    };
    let build_input = OptixBuildInput {
        kind: OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        triangles: tri,
        _tail: [0u8; 880],
    };
    let accel_opts = OptixAccelBuildOptions {
        build_flags: OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        operation: OPTIX_BUILD_OPERATION_BUILD,
        motion_options: OptixMotionOptions {
            num_keys: 1,
            flags: 0,
            time_begin: 0.0,
            time_end: 0.0,
        },
    };
    let mut sizes = OptixAccelBufferSizes::default();
    let acc_mem = table.accel_compute_memory_usage.ok_or("no accel mem fn")?;
    err(
        &table,
        acc_mem(context, &accel_opts, &build_input, 1, &mut sizes),
        "accelComputeMemoryUsage",
    )?;
    let d_temp = stream
        .alloc_zeros::<u8>(sizes.temp_size.max(1))
        .map_err(|e| format!("accel temp: {e:?}"))?;
    let d_output = stream
        .alloc_zeros::<u8>(sizes.output_size.max(1))
        .map_err(|e| format!("accel out: {e:?}"))?;
    let d_temp_ptr = d_temp.device_ptr(&stream).0;
    let d_output_ptr = d_output.device_ptr(&stream).0;

    let mut handle: OptixTraversableHandle = 0;
    let acc_build = table.accel_build.ok_or("no accel build fn")?;
    err(
        &table,
        acc_build(
            context,
            cu_stream,
            &accel_opts,
            &build_input,
            1,
            d_temp_ptr,
            sizes.temp_size,
            d_output_ptr,
            sizes.output_size,
            &mut handle,
            std::ptr::null(),
            0,
        ),
        "accelBuild",
    )?;
    stream.synchronize().map_err(|e| format!("accel sync: {e:?}"))?;

    // Keep geometry + accel buffers alive.
    buffers.push(Box::new(d_verts));
    buffers.push(Box::new(d_indices));
    buffers.push(Box::new(d_normals));
    buffers.push(Box::new(d_temp));
    buffers.push(Box::new(d_output));

    // --- output + params buffers ---
    let n = (WIDTH * HEIGHT) as usize;
    let d_checksum = stream.alloc_zeros::<u32>(n).map_err(|e| format!("checksum buf: {e:?}"))?;
    let d_image = stream.alloc_zeros::<u32>(n).map_err(|e| format!("image buf: {e:?}"))?;
    let d_checksum_ptr = d_checksum.device_ptr(&stream).0;
    let d_image_ptr = d_image.device_ptr(&stream).0;
    buffers.push(Box::new(d_image)); // hold the image buffer alive

    // Fixed camera (deterministic), same framing as pathtrace.wgsl.
    let cam_pos = [2.6f32, 1.9, -3.4];
    let len = (cam_pos[0] * cam_pos[0] + cam_pos[1] * cam_pos[1] + cam_pos[2] * cam_pos[2]).sqrt();
    let fwd = [-cam_pos[0] / len, -cam_pos[1] / len, -cam_pos[2] / len];
    // right = normalize(cross(fwd, up_world))
    let up_world = [0.0f32, 1.0, 0.0];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |a: [f32; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-8);
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let right = norm(cross(fwd, up_world));
    let up = cross(right, fwd);

    let params = Params {
        checksum: d_checksum_ptr,
        image: d_image_ptr,
        normals: d_normals_ptr,
        indices: d_indices_ptr,
        handle,
        width: WIDTH,
        height: HEIGHT,
        samples: k.samples.clamp(1, 1 << 16),
        bounces: k.bounces.clamp(1, 64),
        seed: 0x9e37_79b9,
        shade: 0,
        cam_pos,
        cam_fwd: fwd,
        cam_right: right,
        cam_up: up,
        fov: 0.52,
    };
    let params_bytes: [u8; std::mem::size_of::<Params>()] =
        unsafe { std::mem::transmute_copy(&params) };
    let d_params = stream.memcpy_stod(&params_bytes).map_err(|e| format!("params: {e:?}"))?;

    Ok(OptixCtx {
        table,
        _cuda: cuda,
        stream,
        context,
        module,
        groups,
        pipeline,
        sbt,
        handle,
        _keepalive: buffers,
        d_checksum,
        d_params,
        params,
    })
}

impl OptixCtx {
    fn launch(&self) -> Result<(), String> {
        use cudarc::driver::DevicePtr;
        let launch = self.table.launch.ok_or("no launch fn")?;
        let d_params_ptr = self.d_params.device_ptr(&self.stream).0;
        let cu_stream = self.stream.cu_stream() as CuStream;
        let r = launch(
            self.pipeline,
            cu_stream,
            d_params_ptr,
            std::mem::size_of::<Params>(),
            &self.sbt,
            self.params.width,
            self.params.height,
            1,
        );
        err(&self.table, r, "optixLaunch")?;
        self.stream.synchronize().map_err(|e| format!("launch sync: {e:?}"))
    }

    fn readback_checksum(&self) -> Result<Vec<u32>, String> {
        self.stream
            .memcpy_dtov(&self.d_checksum)
            .map_err(|e| format!("checksum readback: {e:?}"))
    }
}

impl Drop for OptixCtx {
    fn drop(&mut self) {
        // The destroy entries are plain `extern "C" fn` pointers — calling them
        // is safe. Order: pipeline, program groups, module, context.
        if let Some(f) = self.table.pipeline_destroy {
            if !self.pipeline.is_null() {
                f(self.pipeline);
            }
        }
        if let Some(f) = self.table.program_group_destroy {
            for g in self.groups {
                if !g.is_null() {
                    f(g);
                }
            }
        }
        if let Some(f) = self.table.module_destroy {
            if !self.module.is_null() {
                f(self.module);
            }
        }
        if let Some(f) = self.table.device_context_destroy {
            if !self.context.is_null() {
                f(self.context);
            }
        }
        let _ = self.handle;
    }
}

fn run_optix(
    k: &OptixKernel,
    budget: &Budget,
    stop: &StopFlag,
    markers: &MarkerLog,
) -> Result<LoadResult, String> {
    let label = k.device.label();
    let ctx = setup(k)?;

    // Probe + liveness.
    ctx.launch()?;
    let first = ctx.readback_checksum()?;
    if !first.iter().any(|&w| w != 0) {
        return Err("output all-zero on probe (no rays traced?)".into());
    }
    let first_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(first.as_ptr() as *const u8, first.len() * 4) };
    let reference = fnv1a(first_bytes);

    let mut driver = ShapeDriver::start(budget, stop, markers, "optix", "optix".to_string());
    let start = Instant::now();
    let mut launches: u64 = 0;
    let mut verifications: u64 = 0;
    let mut errors: u64 = 0;

    loop {
        match driver.tick() {
            Tick::Work => {
                if let Err(why) = ctx.launch() {
                    errors += 1;
                    return Ok(LoadResult::new(
                        true,
                        launches,
                        reference,
                        errors,
                        format!("{label} optix: DEVICE LOST ({why})"),
                    ));
                }
                launches += 1;
                if launches.is_multiple_of(k.verify_every) {
                    let bytes = ctx.readback_checksum()?;
                    let b: &[u8] = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr() as *const u8, bytes.len() * 4)
                    };
                    let h = fnv1a(b);
                    verifications += 1;
                    if h != reference {
                        errors += 1;
                        return Ok(LoadResult::new(
                            true,
                            launches,
                            h,
                            errors,
                            format!(
                                "{label} optix: miscompare at launch {launches} \
                                 (got {h:#018x}, expected {reference:#018x}) — RT/SM soft error"
                            ),
                        ));
                    }
                }
            }
            Tick::Idle => {}
            Tick::Stop => break,
        }
    }

    let secs = start.elapsed().as_secs_f64();
    let rays = launches as f64 * (WIDTH * HEIGHT) as f64 * k.samples.max(1) as f64 * k.bounces.max(1) as f64;
    let mrays = if secs > 0.0 { rays / secs / 1.0e6 } else { 0.0 };
    let detail = format!(
        "{label} optix {WIDTH}x{HEIGHT} {}spp x{}bounce, {launches} launch(es), \
         {verifications} verified, ~{mrays:.0} Mray/s",
        k.samples, k.bounces
    );
    Ok(LoadResult::new(true, launches, reference, errors, detail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_ground_truth() {
        assert_eq!(std::mem::size_of::<OptixModuleCompileOptions>(), 48);
        assert_eq!(std::mem::size_of::<OptixPipelineCompileOptions>(), 56);
        assert_eq!(std::mem::size_of::<OptixPipelineLinkOptions>(), 20);
        assert_eq!(std::mem::size_of::<OptixProgramGroupDesc>(), 56);
        assert_eq!(std::mem::size_of::<OptixProgramGroupOptions>(), 8);
        assert_eq!(std::mem::size_of::<OptixShaderBindingTable>(), 64);
        assert_eq!(std::mem::size_of::<OptixBuildInput>(), 1032);
        assert_eq!(std::mem::size_of::<OptixBuildInputTriangleArray>(), 144);
        assert_eq!(std::mem::size_of::<OptixAccelBuildOptions>(), 20);
        assert_eq!(std::mem::size_of::<OptixAccelBufferSizes>(), 24);
        assert_eq!(std::mem::size_of::<OptixDeviceContextOptions>(), 24);
        assert_eq!(std::mem::size_of::<OptixFunctionTable>(), 60 * 8);
        // BuildInput.triangles at offset 8.
        assert_eq!(std::mem::offset_of!(OptixBuildInput, triangles), 8);
        // TriangleArray field offsets.
        assert_eq!(std::mem::offset_of!(OptixBuildInputTriangleArray, num_vertices), 8);
        assert_eq!(std::mem::offset_of!(OptixBuildInputTriangleArray, index_buffer), 24);
        assert_eq!(std::mem::offset_of!(OptixBuildInputTriangleArray, flags), 56);
        assert_eq!(std::mem::offset_of!(OptixBuildInputTriangleArray, num_sbt_records), 64);
    }
}
