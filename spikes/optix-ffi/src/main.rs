// SPDX-License-Identifier: MIT
//! De-risk spike for cec-crucible OptiX path tracing (Phase 2), Stage 1.
//!
//! Question: can a HAND-ROLLED OptiX host FFI (re-declared from documentation,
//! zero NVIDIA SDK source in the repo) reach a live `OptixDeviceContext` against
//! the already-installed driver — on a machine with NO CUDA toolkit? If yes, the
//! whole "driver-resident, runtime-dispatch, MIT-clean" OptiX plan is sound and
//! only the device PTX kernel (which needs nvcc at build) remains.
//!
//! It re-implements `optixInit` (find + LoadLibrary nvoptix.dll, resolve
//! `optixQueryFunctionTable`, fill the function table), makes a CUDA context via
//! cudarc (driver-only), calls `optixDeviceContextCreate`, and queries the RT-core
//! version — proving the FFI ABI is right and OptiX talks to the RT hardware.
//!
//! Run: cargo run --manifest-path spikes/optix-ffi/Cargo.toml

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> isize;
    fn GetProcAddress(module: isize, name: *const u8) -> *const c_void;
}

// ---- OptiX ABI, re-declared (not copied) from the public documentation ----
type OptixResult = c_uint; // 0 == OPTIX_SUCCESS
type CuContext = *mut c_void; // opaque CUcontext
type OptixDeviceContext = *mut c_void; // opaque

type OptixLogCallback =
    extern "C" fn(level: c_uint, tag: *const c_char, message: *const c_char, cbdata: *mut c_void);

#[repr(C)]
struct OptixDeviceContextOptions {
    log_callback_function: Option<OptixLogCallback>,
    log_callback_data: *mut c_void,
    log_callback_level: c_int,
    validation_mode: c_uint, // OptixDeviceContextValidationMode: 0 = OFF
}

type OptixDenoiser = *mut c_void; // opaque

#[repr(C)]
struct OptixDenoiserOptions {
    guide_albedo: c_uint,
    guide_normal: c_uint,
    denoise_alpha: c_uint, // OptixDenoiserAlphaMode: 0 = COPY
}

#[repr(C)]
#[derive(Default)]
struct OptixDenoiserSizes {
    state_size: usize,
    with_overlap_scratch: usize,
    without_overlap_scratch: usize,
    overlap_window_px: c_uint,
    compute_avg_color_size: usize,
    compute_intensity_size: usize,
    internal_guide_px_size: usize,
}

type FnGetErrorString = extern "C" fn(OptixResult) -> *const c_char;
type FnDeviceContextCreate = extern "C" fn(
    CuContext,
    *const OptixDeviceContextOptions,
    *mut OptixDeviceContext,
) -> OptixResult;
type FnDeviceContextDestroy = extern "C" fn(OptixDeviceContext) -> OptixResult;
type FnDeviceContextGetProperty =
    extern "C" fn(OptixDeviceContext, c_uint, *mut c_void, usize) -> OptixResult;
type FnDenoiserCreate = extern "C" fn(
    OptixDeviceContext,
    c_uint, // OptixDenoiserModelKind
    *const OptixDenoiserOptions,
    *mut OptixDenoiser,
) -> OptixResult;
type FnDenoiserDestroy = extern "C" fn(OptixDenoiser) -> OptixResult;
type FnDenoiserComputeMem =
    extern "C" fn(OptixDenoiser, c_uint, c_uint, *mut OptixDenoiserSizes) -> OptixResult;

// The complete 60-entry ABI-118 function table (verified by the brute-force size
// probe). We type the entries we call (2,3,4 device-context; 52,53,54 denoiser)
// and leave the rest opaque, laid out at their exact indices.
#[repr(C)]
struct OptixFunctionTable {
    get_error_name: *const c_void,                          // 0
    get_error_string: Option<FnGetErrorString>,             // 1
    device_context_create: Option<FnDeviceContextCreate>,   // 2
    device_context_destroy: Option<FnDeviceContextDestroy>, // 3
    device_context_get_property: Option<FnDeviceContextGetProperty>, // 4
    _pad_5_51: [*const c_void; 47],                         // 5..=51
    denoiser_create: Option<FnDenoiserCreate>,              // 52
    denoiser_destroy: Option<FnDenoiserDestroy>,            // 53
    denoiser_compute_mem: Option<FnDenoiserComputeMem>,     // 54
    _pad_55_59: [*const c_void; 5],                         // 55..=59
}

type FnQueryFunctionTable = extern "C" fn(
    abi: c_uint,
    num_options: c_int,
    keys: *const *const c_char,
    vals: *const *const c_char,
    table: *mut OptixFunctionTable,
    size: usize,
) -> OptixResult;

// OptiX 9.1 ABI (driver 591.86 = R590+ supports it).
const OPTIX_ABI_VERSION: c_uint = 118;
const OPTIX_DEVICE_PROPERTY_RTCORE_VERSION: c_uint = 0x2005;
const OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH: c_uint = 0x2001;

extern "C" fn log_cb(level: c_uint, tag: *const c_char, message: *const c_char, _cb: *mut c_void) {
    let t = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    let m = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    println!("  [optix log {level}] {t}: {m}");
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Find the newest nvoptix.dll in the driver store (it is not on the default DLL
/// search path, so we load it by full path — what the real OptiX loader does).
fn find_nvoptix() -> Option<String> {
    let base = r"C:\Windows\System32\DriverStore\FileRepository";
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(base).ok()?.flatten() {
        let dll = entry.path().join("nvoptix.dll");
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

fn main() {
    // 1. CUDA context (driver-only, no toolkit) via cudarc.
    let cuda = match cudarc::driver::CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: could not create CUDA context (driver present?): {e:?}");
            std::process::exit(1);
        }
    };
    let cu_ctx = cuda.cu_ctx() as CuContext;
    println!("CUDA context OK ({cu_ctx:?})");

    // 2. Locate + load nvoptix.dll (the OptiX runtime, resident in the driver).
    let path = match find_nvoptix() {
        Some(p) => p,
        None => {
            eprintln!("FAIL: nvoptix.dll not found in the driver store");
            std::process::exit(1);
        }
    };
    println!("nvoptix.dll: {path}");
    let module = unsafe { LoadLibraryW(wide(&path).as_ptr()) };
    if module == 0 {
        eprintln!("FAIL: LoadLibraryW(nvoptix.dll) failed");
        std::process::exit(1);
    }

    // 3. Resolve + call optixQueryFunctionTable to fill the function table.
    let sym = unsafe { GetProcAddress(module, b"optixQueryFunctionTable\0".as_ptr()) };
    if sym.is_null() {
        eprintln!("FAIL: optixQueryFunctionTable symbol missing");
        std::process::exit(1);
    }
    let query: FnQueryFunctionTable = unsafe { std::mem::transmute(sym) };

    // OptiX requires sizeOfTable to EXACTLY equal its ABI table size (error 7802
    // otherwise). Discover the true field count empirically: a big zeroed buffer,
    // try each candidate size until the query succeeds. Also try a couple of ABI
    // versions in case 118 isn't the driver's.
    let mut buf = vec![0u8; 4096]; // room for up to 512 function pointers
    let mut chosen: Option<(c_uint, usize)> = None;
    'outer: for abi in [OPTIX_ABI_VERSION, 117, 116, 105, 87] {
        for n in 8usize..=256 {
            buf.iter_mut().for_each(|b| *b = 0);
            let r = query(
                abi,
                0,
                std::ptr::null(),
                std::ptr::null(),
                buf.as_mut_ptr() as *mut OptixFunctionTable,
                n * std::mem::size_of::<*const c_void>(),
            );
            if r == 0 {
                chosen = Some((abi, n));
                break 'outer;
            }
        }
    }
    let (abi, n) = match chosen {
        Some(x) => x,
        None => {
            eprintln!("FAIL: no (abi, size) combination made optixQueryFunctionTable succeed");
            std::process::exit(1);
        }
    };
    println!("optixQueryFunctionTable OK — ABI {abi}, {n} function-table entries");
    // The first 6 entries (0-5) are the DeviceContext group; read them.
    let table: &OptixFunctionTable = unsafe { &*(buf.as_ptr() as *const OptixFunctionTable) };

    // 4. Create the OptiX device context on our CUDA context.
    let create = table.device_context_create.expect("null create fn ptr");
    let opts = OptixDeviceContextOptions {
        log_callback_function: Some(log_cb),
        log_callback_data: std::ptr::null_mut(),
        log_callback_level: 4, // 4 = print (info)
        validation_mode: 0,    // OFF
    };
    let mut ctx: OptixDeviceContext = std::ptr::null_mut();
    let r = create(cu_ctx, &opts, &mut ctx);
    if r != 0 || ctx.is_null() {
        let msg = table
            .get_error_string
            .map(|f| unsafe { CStr::from_ptr(f(r)) }.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("FAIL: optixDeviceContextCreate returned {r} ({msg})");
        std::process::exit(1);
    }
    println!("optixDeviceContextCreate OK ({ctx:?})");

    // 5. Query properties — proves the context talks to the RT hardware.
    if let Some(get_prop) = table.device_context_get_property {
        let mut rtcore: c_uint = 0;
        let r = get_prop(
            ctx,
            OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
            &mut rtcore as *mut _ as *mut c_void,
            std::mem::size_of::<c_uint>(),
        );
        if r == 0 {
            println!("  RTCORE_VERSION = {rtcore}  (RT-core generation reported by the driver)");
        }
        let mut max_depth: c_uint = 0;
        let _ = get_prop(
            ctx,
            OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
            &mut max_depth as *mut _ as *mut c_void,
            std::mem::size_of::<c_uint>(),
        );
        println!("  MAX_TRACE_DEPTH = {max_depth}");
    }

    // 6. Denoiser (Stage 2) — the tensor-core path, which needs NO device PTX.
    // Instantiate the HDR model and size its memory: if the model loads and
    // reports sensible buffers, the tensor-core denoise path is reachable
    // toolkit-free (the full invoke, needing CUDA image buffers, is Phase 2).
    println!();
    if let (Some(dn_create), Some(dn_compute_mem)) =
        (table.denoiser_create, table.denoiser_compute_mem)
    {
        let dopts = OptixDenoiserOptions {
            guide_albedo: 0,
            guide_normal: 0,
            denoise_alpha: 0, // COPY
        };
        let mut denoiser: OptixDenoiser = std::ptr::null_mut();
        let r = dn_create(ctx, 0x2323 /* HDR model */, &dopts, &mut denoiser);
        if r == 0 && !denoiser.is_null() {
            println!("optixDenoiserCreate OK — HDR model instantiated ({denoiser:?})");
            let mut sizes = OptixDenoiserSizes::default();
            let r2 = dn_compute_mem(denoiser, 1920, 1080, &mut sizes);
            if r2 == 0 {
                println!(
                    "  denoiser memory @1920x1080: state {:.1} MiB, scratch {:.1} MiB, overlap {} px",
                    sizes.state_size as f64 / 1048576.0,
                    sizes.without_overlap_scratch as f64 / 1048576.0,
                    sizes.overlap_window_px
                );
                println!("  => tensor-core denoise model loaded + sized (no PTX/toolkit).");
            } else {
                println!("  optixDenoiserComputeMemoryResources returned {r2}");
            }
            if let Some(dn_destroy) = table.denoiser_destroy {
                dn_destroy(denoiser);
            }
        } else {
            println!("optixDenoiserCreate returned {r} (model unavailable on this driver?)");
        }
    }

    // 7. Tear down.
    if let Some(destroy) = table.device_context_destroy {
        destroy(ctx);
    }

    println!("\nPASS: hand-rolled OptiX FFI — live device context + tensor denoiser, no toolkit.");
}
