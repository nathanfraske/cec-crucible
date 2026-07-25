// SPDX-License-Identifier: MIT
//! # cec-crucible
//!
//! The orchestrator binary. It is both the per-test tool the PowerShell QC
//! harness invokes (`cec-crucible cpu --seconds 60 --device-id <uuid>`) and a
//! standalone runner (`cec-crucible run cross`).
//!
//! Responsibilities: parse args, resolve device identity, build the requested
//! kernel(s) + budget, run them under one [`StopFlag`] and one marker timeline,
//! then write a device-ID'd JSON report and the JSONL marker feed.

mod args;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, Shape, StopFlag};
use crucible_core::markers::{telemetry_csv_header, telemetry_csv_rows, Event, MarkerLog};
use crucible_core::report::{Report, StageReport, Verdict};
use crucible_core::{sysinfo, Clock, DeviceId};

use crucible_cpu::{CoreSel, CpuKernel};
use crucible_mem::{MemKernel, MemSize};
use crucible_storage::{StorageConfig, StorageKernel, StorageStats};

// GPU support is behind the `gpu` cargo feature so the default build stays
// zero-dependency and offline-buildable. The shipped binary enables it.
#[cfg(feature = "gpu")]
use crucible_gpu::{GpuDevice, GpuKernel};

use args::Parsed;

#[cfg(feature = "tui")]
mod theme;

#[cfg(feature = "tui")]
mod tui;

#[cfg(feature = "tui")]
mod menu;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Options accepted by (nearly) every command.
const COMMON_BOOLS: &[&str] = &[
    "json",
    "no-report",
    "keep",
    "help",
    "buffered",
    "unbuffered",
    "all-drives",
    "gpu-alu-only",
    "link-cuda",
    "per-core",
    "ui",
    "preview",
    "csv",
    "telemetry-csv",
];

/// Options recognized for the GPU kernel (accepted even in non-GPU builds so
/// the error message can be "rebuild with --features gpu" rather than
/// "unknown option").
const GPU_OPTS: &[&str] = &[
    "gpu-device",
    "gpu-threads",
    "gpu-iters",
    "gpu-mb",
    "gpu-alu-only",
    "vram-mb",
    "vram-chunk-mb",
    "link-mb",
    "link-dir",
    "link-cuda",
    "render-w",
    "render-h",
    "render-inst",
    "scene",
    "tensor-tiles",
    "tensor-iters",
    "rt-iters",
    "pt-samples",
    "pt-bounces",
    "optix-samples",
    "optix-bounces",
];

const USAGE: &str = "\
cec-crucible — CEC in-house PC-build stress/validation suite

USAGE:
    cec-crucible <command> [options]

COMMANDS:
    menu                 Interactive launcher for every test/profile. [--features tui]
                         (Also opens when run with no command.)
    info                 Print device id, CPU, memory and QPC info, then exit.
    drives               List fixed physical drives (NVMe/SATA), then exit.
    gpu-info             List usable GPUs, then exit.            [--features gpu]
    cpu                  Run the CPU FMA/AVX burn kernel.
    mem                  Run the RAM pattern kernel.
    storage              Run the storage scratch-file kernel.
    gpu                  Run the GPU thrasher kernel (watts).    [--features gpu]
    vram                 Run the VRAM integrity test.            [--features gpu]
    link                 PCIe host<->device transfer + verify.   [--features gpu]
    render               Graphics pipeline: raster/TMU/ROP + verify. [--features gpu]
    tensor               Tensor/matrix-core cmma stress + verify.  [--features tensor]
    rt                   Ray-tracing-core (BVH traversal) + verify. [--features rt]
    pathtrace            Multi-bounce path tracer (deep/divergent RT+SM). [--features rt]
    optix                NVIDIA-native OptiX path tracer (RT+SM).   [--features optix]
    run <profile>        See PROFILES below.
    version              Print version.
    help                 Print this help.

PROFILES:
    quick | soak         Sequential CPU/RAM/storage QC.
    cross                All domains concurrently (GPU included if built in).
    power                CPU burst with dense markers for the power rig.
    storage-cross        Multi-SSD: solo baseline vs concurrent per drive.
    in-phase             CPU+GPU burst together   -> peak draw, PSU/OCP.  [gpu]
    anti-phase           CPU and GPU alternate    -> VRM/PSU chase load.  [gpu]
    beat                 Slightly different periods -> sweeps all phases. [gpu]
    worst-case           Everything at once: CPU transients anti-phase to the
                         GPU, under RAM + storage + VRAM-integrity + PCIe load.
    chaos                Randomized never-settle: CPU+GPU on independent seeded
                         jitter over steady RAM/storage/VRAM/PCIe.        [gpu]
    game-load            Frame-paced CPU->GPU handoff at moderate power — the
                         game electrical/thermal signature (not graphics). [gpu]
    core-cycle           Single-core steady boost, rotated over all cores — the
                         weak-core-at-max-boost hunt (CoreCycler-style).
    c-states             Single-core pulse + deep idle, rotated — the idle /
                         C-state / low-load-voltage class (needs BIOS C-states).

COMMON OPTIONS:
    --seconds <N>        Run duration in seconds.
    --device-id <ID>     Machine id from the harness (else auto-detected).
    --out <DIR>          Output directory for report + markers (+ CSVs).
    --no-report          Do not write report/marker files.
    --csv                Also write a per-stage results CSV (Excel/Sheets-ready).
    --telemetry-csv      Also log a time-series telemetry CSV (per lane, ~4/s)
                         for graphing a run afterward. Both land in --out.
    --json               Emit the report as JSON on stdout (pipe-friendly).
    --ui                 Live terminal dashboard (per-core + per-domain activity).
                         Needs a build with `--features tui`; q / Ctrl-C to stop.

CPU OPTIONS:
    --core <all|N>       All cores (default) or a single logical core index.
    --shape <steady|burst|pulse|jitter>
    --burst-on <MS> --burst-off <MS>   Burst duty cycle (default 20/20 ms).

TRANSIENT / LIGHT-LOAD OPTIONS (jitter & pulse shapes; chaos/game-load/rotation):
    --shape jitter       Randomized never-settle burst (spike/floor).
      --jit-on-min/-max <MS>    Spike (ON) bounds (default 5/50).
      --jit-off-min/-max <MS>   Floor (OFF) bounds (default 3/40).
      --floor <PCT>             Trickle % during the floor (default 12; 0 = idle).
      --per-core                CPU only: decorrelate each core -> CPU-VRM chaos
                                instead of one synchronized system-level step.
    --shape pulse        Short work pulse + deep idle (lets a core reach C6).
      --pulse-ms <MS>           Work pulse (default 5).
      --idle-ms <MS>            Deep idle (default 300).
    --seed <N|0xHEX>     Seed the jitter/chaos PRNG (default: a logged random seed;
                         the report prints it so any run can be replayed).
    --dwell <SEC>        Per-core seconds for core-cycle / c-states (default 30 / 120).
    --passes <N>         Rotation passes for core-cycle / c-states (default 2 / 1).
    --fps <N>            game-load frame rate (default 120).
    --bound <gpu|cpu|balanced>   game-load duty split (default gpu-bound).
    --handoff-ms <MS>    game-load GPU-start delay after CPU submit (default: cpu-on).

MEM OPTIONS:
    --mb <N>             Buffer size in MiB (default: 50% of free RAM).

STORAGE OPTIONS:
    --path <DIR>         Target directory for the scratch file (default: .).
    --size-mb <N>        Scratch file size in MiB (default 1024).
    --block-kb <N>       Block size in KiB (default 1024).
    --keep               Keep the scratch file after the run.
    --unbuffered         Force device-direct I/O (FILE_FLAG_NO_BUFFERING).
    --buffered           Force buffered I/O (default off Windows).
    --all-drives         Cross-load every fixed physical drive: solo baseline
                         then concurrent, reporting per-drive slowdown.
                         (Same as `run storage-cross`.)

GPU OPTIONS (builds with --features gpu):
    --gpu-device <discrete|integrated|default>
    --gpu-threads <N>    Total GPU threads (default 1048576).
    --gpu-iters <N>      Work per dispatch. Default depends on the load shape:
                         4096 for steady (max sustained power) and 256 for burst
                         (~6 ms, fits inside the ON window so edges stay sharp).
                         Also the load-edge granularity; any value keeps a
                         dispatch far under the ~2000 ms Windows TDR watchdog.
    --gpu-mb <N>         VRAM stream buffer in MiB (default 1024).
    --gpu-alu-only       Disable the VRAM stream. Pure ALU reaches only ~75% of
                         the board power limit; the mix reaches ~92%.
    --vram-mb <N>        VRAM to place under integrity test (default 2048).
    --vram-chunk-mb <N>  Allocation chunk size (default 64; storage-buffer
                         binding limits cap how large one allocation can be).
    --link-mb <N>        PCIe transfer buffer size in MiB (default 256).
    --link-dir <up|down|bidir>   Transfer direction (default bidir; only bidir
                         and down verify data — up-only has no read-back).
    --link-cuda          Use the CUDA transfer path: pinned memory + two streams
                         for true full-duplex (both copy engines at once), which
                         wgpu's single queue can't do. NVIDIA + `--features cuda`
                         build; probed at runtime, falls back to wgpu if absent.
    --render-w <N> --render-h <N>   Framebuffer size (default 1280x720).
    --render-inst <N>    Mesh instances drawn per frame — the geometry/overdraw
                         knob (default 48). `render` exercises the rasterizer,
                         texture units and ROP the compute thrasher never touches.
    --scene <FILE>       Render a glTF/.glb scene (real game geometry + texture)
                         instead of the procedural mesh. Needs `--features gpu-gltf`.
    --preview            (render/rt) Pop a live window showing the work as it runs:
                         render mirrors its framebuffer; rt shows the shaded, self-
                         shadowed ray-traced image. Needs a `--features preview`
                         (Windows) build; never changes what is verified. Close the
                         window to stop.
    --tensor-tiles <N>   Tensor: 16x16 output tiles / warps (default 4096).
    --tensor-iters <N>   Tensor: cmma accumulations per warp per dispatch (default
                         256). `tensor` drives the tensor cores via cooperative
                         matrix (f16->f32) — needs a `--features tensor` build.
    --rt-iters <N>       RT: ray re-traces per pixel per dispatch (default 192).
                         `rt` drives the ray-tracing cores (BVH traversal +
                         triangle intersection) via VK_KHR_ray_query, verified by
                         self-consistency — needs a `--features rt` build.
    --pt-samples <N>     Path-trace: paths per pixel per dispatch (default 16).
    --pt-bounces <N>     Path-trace: max path depth / bounces (default 8).
    --material <name>    Path-trace surface: metal (default), matte, plastic,
                         mirror, glass, velvet, marble. Each is a distinct (still
                         deterministic, self-verified) BSDF workload.
                         `pathtrace` is a deterministic multi-bounce Monte-Carlo
                         path tracer — the deep, divergent RT-core + SM stress
                         beyond `rt`. `--preview` shows the live GI render.
                         Needs a `--features rt` build.
    --optix-samples <N>  OptiX: paths per pixel per launch (default 16).
    --optix-bounces <N>  OptiX: max path depth / bounces (default 8).
                         `optix` is the NVIDIA-native path tracer on the OptiX
                         ray-tracing pipeline (driver-resident, NVIDIA-only),
                         deterministic + self-consistency verified. Needs a
                         `--features optix` build.

EXIT CODES:
    0 PASS/PARTIAL   1 FAIL   2 usage error

Zero external dependencies. Emits QPC-timestamped markers for correlation with
an external power-monitoring rig, and a device-ID'd JSON report.
";

// Set by the console-control handler; polled by a bridge thread that trips the
// run's StopFlag so a Ctrl-C ends the current stage cleanly with a report.
static CTRLC: AtomicBool = AtomicBool::new(false);

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => ExitCode::from(code),
        Err(msg) => {
            eprintln!("error: {msg}");
            eprintln!("\nrun `cec-crucible help` for usage.");
            ExitCode::from(2)
        }
    }
}

fn run(argv: &[String]) -> Result<u8, String> {
    let command = match argv.first() {
        Some(c) => c.as_str(),
        None => {
            // Bare `cec-crucible` opens the interactive menu when built with the
            // `tui` feature; otherwise it prints usage.
            #[cfg(feature = "tui")]
            {
                return menu::run_menu();
            }
            #[cfg(not(feature = "tui"))]
            {
                print!("{USAGE}");
                return Ok(2);
            }
        }
    };
    let rest = &argv[1..];

    match command {
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(0)
        }
        "menu" => {
            #[cfg(feature = "tui")]
            {
                let _ = rest;
                menu::run_menu()
            }
            #[cfg(not(feature = "tui"))]
            {
                let _ = rest;
                Err("the interactive menu needs a `--features tui` build".to_string())
            }
        }
        "version" | "-V" | "--version" => {
            println!("cec-crucible {VERSION}");
            Ok(0)
        }
        "info" => cmd_info(rest),
        "drives" => cmd_drives(rest),
        "gpu-info" => cmd_gpu_info(rest),
        "cpu" => cmd_cpu(rest),
        "mem" => cmd_mem(rest),
        "storage" => cmd_storage(rest),
        "gpu" => cmd_gpu(rest),
        "vram" => cmd_vram(rest),
        "link" => cmd_link(rest),
        "render" => cmd_render(rest),
        "tensor" => cmd_tensor(rest),
        "rt" => cmd_rt(rest),
        "pathtrace" => cmd_pathtrace(rest),
        "optix" => cmd_optix(rest),
        "run" => cmd_run(rest),
        other => Err(format!("unknown command '{other}'")),
    }
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&["json", "device-id", "help"])?;

    let clock = Clock::new();
    let device = resolve_device(&p);
    let mem = sysinfo::memory();
    let cpus = sysinfo::logical_cpus();

    if p.has("json") {
        let mut o = crucible_core::Json::object();
        o.push("tool", "cec-crucible")
            .push("version", VERSION)
            .push("device", device.to_json())
            .push("logical_cpus", cpus)
            .push("cpu_backend", CpuKernel::backend())
            .push("qpc_frequency", clock.frequency())
            .push("qpc_backed", clock.is_qpc());
        if let Some(m) = mem {
            let mut mj = crucible_core::Json::object();
            mj.push("total_bytes", m.total_bytes)
                .push("avail_bytes", m.avail_bytes);
            o.push("memory", mj);
        } else {
            o.push("memory", crucible_core::Json::Null);
        }
        println!("{}", o.to_pretty());
        return Ok(0);
    }

    println!("cec-crucible {VERSION}");
    println!("device:");
    println!("  short-id : {}", device.short_id);
    println!("  uuid     : {}", device.uuid);
    println!(
        "  board    : {} (serial {})",
        device.board, device.board_serial
    );
    println!("  system   : {}", device.system);
    println!("  host     : {}", device.host);
    println!("  id-source: {}", device.source);
    println!("cpu:");
    println!("  logical  : {cpus}");
    println!("  backend  : {}", CpuKernel::backend());
    if let Some(m) = mem {
        println!("memory:");
        println!("  total    : {:.1} GiB", gib(m.total_bytes));
        println!("  available: {:.1} GiB", gib(m.avail_bytes));
    } else {
        println!("memory: (unavailable on this platform)");
    }
    println!("timing:");
    println!(
        "  qpc      : {:.3} MHz{}",
        clock.frequency() as f64 / 1.0e6,
        if clock.is_qpc() {
            ""
        } else {
            " (fallback, not QPC)"
        }
    );
    Ok(0)
}

// ---------------------------------------------------------------------------
// drives
// ---------------------------------------------------------------------------

fn cmd_drives(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&["json", "help"])?;
    let drives = crucible_storage::drives::discover();

    if p.has("json") {
        let arr: Vec<crucible_core::Json> = drives
            .iter()
            .map(|d| {
                let mut o = crucible_core::Json::object();
                o.push("number", d.number)
                    .push("bus", d.bus.as_str())
                    .push("primary_root", d.primary_root.as_str())
                    .push("roots", d.roots.clone());
                o
            })
            .collect();
        println!("{}", crucible_core::Json::Array(arr).to_pretty());
        return Ok(0);
    }

    if drives.is_empty() {
        println!("no fixed physical drives detected (or unsupported platform)");
        return Ok(0);
    }
    println!("fixed physical drives ({}):", drives.len());
    for d in &drives {
        println!(
            "  disk {:<2} [{:<5}]  scratch target {}  (volumes: {})",
            d.number,
            d.bus.as_str(),
            d.primary_root,
            d.roots.join(", "),
        );
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// single-kernel commands
// ---------------------------------------------------------------------------

fn cmd_cpu(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    let mut allowed = vec![
        "seconds",
        "core",
        "shape",
        "burst-on",
        "burst-off",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ];
    allowed.extend_from_slice(SHAPE_OPTS);
    p.reject_unknown(&allowed)?;
    let seconds = seconds_arg(&p, 60)?;
    let shape = shape_from(&p)?;
    let cores = core_from(&p)?;
    let mut kernel = CpuKernel::new(cores);
    kernel.per_core_jitter = p.has("per-core");
    let per_core = if kernel.per_core_jitter && matches!(shape, Shape::Jitter { .. }) {
        " per-core"
    } else {
        ""
    };
    let mode = format!("{} {}{}", shape_label(&shape), cores_label(cores), per_core);
    let budget = Budget {
        duration: Duration::from_secs(seconds),
        shape,
        target_watts: None,
        phase_epoch: None,
        phase_offset: Duration::ZERO,
    };

    let mut runner = Runner::new(&p)?;
    runner.single_stage(&kernel, &budget, &mode)
}

fn cmd_mem(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "mb",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let seconds = seconds_arg(&p, 60)?;
    let size = mem_size_from(&p, None)?;
    let kernel = MemKernel::new(size);
    let budget = Budget::steady(Duration::from_secs(seconds));

    let mut runner = Runner::new(&p)?;
    runner.single_stage(&kernel, &budget, "steady")
}

fn cmd_storage(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "path",
        "size-mb",
        "block-kb",
        "keep",
        "buffered",
        "unbuffered",
        "all-drives",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let seconds = seconds_arg(&p, 60)?;

    let mut runner = Runner::new(&p)?;
    if p.has("all-drives") {
        return runner.all_drives_storage(&p, seconds);
    }

    let cfg = storage_cfg_from(&p, 1024)?;
    let kernel = StorageKernel::new(cfg);
    let mode = if unbuffered_from(&p) {
        "unbuffered"
    } else {
        "buffered"
    };
    let budget = Budget::steady(Duration::from_secs(seconds));
    runner.single_stage(&kernel, &budget, mode)
}

// ---------------------------------------------------------------------------
// gpu
// ---------------------------------------------------------------------------

/// Message shown when a GPU command is used in a build without the feature.
#[cfg(not(feature = "gpu"))]
const NO_GPU: &str = "this binary was built without GPU support; rebuild with:\n    \
     cargo build --release -p crucible-cli --features gpu";

/// Build the GPU kernel, defaulting work-per-dispatch to suit the load shape.
///
/// Steady and burst want opposite tuning and it is worth ~40 W: steady wants
/// large dispatches so the GPU never drains (max sustained power), burst wants
/// small ones so each dispatch fits inside the ON window and the edge stays
/// sharp instead of smearing past it.
#[cfg(feature = "gpu")]
fn gpu_kernel_from(p: &Parsed, shape: Shape) -> Result<GpuKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = GpuKernel::new(device);
    if let Some(v) = p.get_u64("gpu-threads")? {
        k.threads = v.clamp(1024, 1 << 26) as usize;
    }
    k.iters = match p.get_u64("gpu-iters")? {
        // Keep dispatches far under the ~2000 ms TDR watchdog.
        Some(v) => v.clamp(1, 65536) as u32,
        None => match shape {
            // ~50 ms/dispatch on an RTX 3070: maximum sustained power.
            Shape::Steady => 4096,
            // ~6 ms/dispatch: fits inside a typical 20 ms ON window. Pulse and
            // jitter want the same sharp-edged granularity as burst.
            Shape::Burst { .. } | Shape::Pulse { .. } | Shape::Jitter { .. } => 256,
        },
    };
    if let Some(v) = p.get_u64("gpu-mb")? {
        k.data_mb = v.clamp(16, 32768) as usize;
    }
    if p.has("gpu-alu-only") {
        // Pure ALU reaches only ~75% of the power limit; opt-in for comparison.
        k.mix = false;
    }
    Ok(k)
}

fn cmd_gpu(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rest;
        Err(NO_GPU.to_string())
    }
    #[cfg(feature = "gpu")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = gpu_kernel_from(&p, shape)?;
        let mode = format!("{} {}", shape.mode_str(), kernel.device.label());
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "gpu")]
fn vram_kernel_from(p: &Parsed) -> Result<crucible_gpu::vram::VramKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::vram::VramKernel::new(device);
    if let Some(v) = p.get_u64("vram-mb")? {
        k.vram_mb = v.clamp(16, 65536) as usize;
    }
    if let Some(v) = p.get_u64("vram-chunk-mb")? {
        // Storage-buffer binding limits cap how big one allocation can be.
        k.chunk_mb = v.clamp(1, 128) as usize;
    }
    Ok(k)
}

#[cfg(feature = "gpu")]
fn link_kernel_from(p: &Parsed) -> Result<crucible_gpu::link::LinkKernel, String> {
    use crucible_gpu::link::LinkDir;
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::link::LinkKernel::new(device);
    if let Some(v) = p.get_u64("link-mb")? {
        // Bounded: large enough to amortize, small enough to stay well under TDR.
        k.buf_mb = v.clamp(1, 2048) as usize;
    }
    k.dir = match p.get("link-dir").unwrap_or("bidir") {
        "up" => LinkDir::Up,
        "down" => LinkDir::Down,
        "bidir" | "both" => LinkDir::Both,
        other => return Err(format!("--link-dir expects up|down|bidir, got '{other}'")),
    };
    // CUDA path: pinned + dual-stream full-duplex. Probed at runtime; falls back
    // to wgpu if no NVIDIA driver. Only compiled in a `--features cuda` build.
    k.cuda = p.has("link-cuda");
    Ok(k)
}

/// PCIe link load — sustained verified host<->device transfers. Reports achieved
/// H2D/D2H bandwidth; catches *uncorrected* corruption across the link. (A
/// marginal riser that retries is caught by the error plane, not here — see
/// docs/pcie-plan.md.)
fn cmd_link(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rest;
        Err(NO_GPU.to_string())
    }
    #[cfg(feature = "gpu")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> =
            vec!["seconds", "device-id", "out", "no-report", "json", "help"];
        allowed.extend_from_slice(GPU_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let kernel = link_kernel_from(&p)?;
        let mode = format!("{} {}", kernel.dir.as_str(), kernel.device.label());
        let budget = Budget::steady(Duration::from_secs(seconds));
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "gpu")]
fn render_kernel_from(p: &Parsed) -> Result<crucible_gpu::render::RenderKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::render::RenderKernel::new(device);
    if let Some(v) = p.get_u64("render-w")? {
        k.width = v.clamp(64, 7680) as u32;
    }
    if let Some(v) = p.get_u64("render-h")? {
        k.height = v.clamp(64, 4320) as u32;
    }
    if let Some(v) = p.get_u64("render-inst")? {
        k.instances = v.clamp(1, 4096) as u32;
    }
    if let Some(path) = p.get("scene") {
        // A glTF/.glb scene (needs a --features gpu-gltf build; load errors at run).
        k.scene = crucible_gpu::render::SceneSource::File(path.into());
    }
    // Live preview window (needs a --features preview, Windows build; otherwise a
    // no-op with a note). Never changes what gets verified.
    k.preview = p.has("preview");
    Ok(k)
}

/// Graphics-pipeline load — drives a real headless render (rasterizer / TMU /
/// ROP, the fixed-function units the compute thrasher cannot reach) and verifies
/// the framebuffer bit-for-bit against the first frame (same-device
/// self-consistency; a miscompare is a raster/TMU/ROP/VRAM soft error).
fn cmd_render(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rest;
        Err(NO_GPU.to_string())
    }
    #[cfg(feature = "gpu")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
            "preview",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        allowed.extend_from_slice(SHAPE_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = render_kernel_from(&p)?;
        let scene_tag = match &kernel.scene {
            crucible_gpu::render::SceneSource::File(path) => format!(
                " scene={}",
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("glb")
            ),
            crucible_gpu::render::SceneSource::Procedural => String::new(),
        };
        let mode = format!(
            "{} render {}x{} x{}{scene_tag}",
            shape.mode_str(),
            kernel.width,
            kernel.height,
            kernel.instances
        );
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "tensor")]
fn tensor_kernel_from(p: &Parsed) -> Result<crucible_gpu::tensor::TensorKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::tensor::TensorKernel::new(device);
    if let Some(v) = p.get_u64("tensor-tiles")? {
        k.tiles = v.clamp(1, 1 << 20) as u32;
    }
    if let Some(v) = p.get_u64("tensor-iters")? {
        k.iters = v.clamp(1, 1 << 20) as u32;
    }
    Ok(k)
}

/// Tensor / matrix-core load — a sustained cooperative-matrix (cmma) chain on the
/// tensor cores (the one GPU unit the FMA thrasher can't reach), verified by
/// same-device self-consistency. Forces the Vulkan/SPIR-V backend.
fn cmd_tensor(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "tensor"))]
    {
        let _ = rest;
        Err("the tensor test needs a build with `--features tensor`".to_string())
    }
    #[cfg(feature = "tensor")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        allowed.extend_from_slice(SHAPE_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = tensor_kernel_from(&p)?;
        let mode = format!(
            "{} tensor {}tiles x{}",
            shape.mode_str(),
            kernel.tiles,
            kernel.iters
        );
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "rt")]
fn rt_kernel_from(p: &Parsed) -> Result<crucible_gpu::rt::RtKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::rt::RtKernel::new(device);
    if let Some(v) = p.get_u64("rt-iters")? {
        k.iters = v.clamp(1, 1 << 20) as u32;
    }
    // Live window showing the shaded ray-traced image (needs a --features preview,
    // Windows build; otherwise a no-op with a note). Never changes verification.
    k.preview = p.has("preview");
    Ok(k)
}

/// Ray-tracing-core load — sustained hardware BVH traversal + triangle
/// intersection via `VK_KHR_ray_query` (raw Vulkan), the one GPU unit neither the
/// FMA thrasher nor the tensor test can reach. Verified by same-device
/// self-consistency; the WGSL shader is compiled to SPIR-V at runtime by naga.
fn cmd_rt(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "rt"))]
    {
        let _ = rest;
        Err("the rt test needs a build with `--features rt`".to_string())
    }
    #[cfg(feature = "rt")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
            "preview",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        allowed.extend_from_slice(SHAPE_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = rt_kernel_from(&p)?;
        let mode = format!("{} rt x{} traces", shape.mode_str(), kernel.iters);
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "rt")]
fn pathtrace_kernel_from(p: &Parsed) -> Result<crucible_gpu::rt::RtKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::rt::RtKernel::path_tracer(device);
    if let Some(v) = p.get_u64("pt-samples")? {
        k.samples = v.clamp(1, 1 << 16) as u32;
    }
    if let Some(v) = p.get_u64("pt-bounces")? {
        k.bounces = v.clamp(1, 64) as u32;
    }
    if let Some(name) = p.get("material") {
        k.material = match name {
            "metal" => 0,
            "matte" => 1,
            "plastic" => 2,
            "mirror" => 3,
            "glass" => 4,
            "velvet" => 5,
            "marble" => 6,
            other => {
                return Err(format!(
                    "--material expects metal|matte|plastic|mirror|glass|velvet|marble, got '{other}'"
                ))
            }
        };
    }
    k.preview = p.has("preview");
    Ok(k)
}

/// Path-tracing load — a multi-bounce Monte-Carlo path tracer (inline ray-query),
/// the deep/divergent RT-core + SM stress beyond `rt`'s coherent single-bounce
/// fan. Deterministic megakernel (fixed per-pixel RNG), verified by same-device
/// self-consistency; the WGSL shader is compiled to SPIR-V at runtime by naga.
fn cmd_pathtrace(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "rt"))]
    {
        let _ = rest;
        Err("the pathtrace test needs a build with `--features rt`".to_string())
    }
    #[cfg(feature = "rt")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
            "preview",
            "material",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        allowed.extend_from_slice(SHAPE_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = pathtrace_kernel_from(&p)?;
        let mat_name = match kernel.material {
            1 => "matte",
            2 => "plastic",
            3 => "mirror",
            4 => "glass",
            5 => "velvet",
            6 => "marble",
            _ => "metal",
        };
        let mode = format!(
            "{} pathtrace {} {}spp x{}bounce",
            shape.mode_str(),
            mat_name,
            kernel.samples,
            kernel.bounces
        );
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

#[cfg(feature = "optix")]
fn optix_kernel_from(p: &Parsed) -> Result<crucible_gpu::optix::OptixKernel, String> {
    let device = match p.get("gpu-device").unwrap_or("discrete") {
        "discrete" => GpuDevice::Discrete(0),
        "integrated" | "igpu" => GpuDevice::Integrated(0),
        "default" => GpuDevice::Default,
        other => {
            return Err(format!(
                "--gpu-device expects discrete|integrated|default, got '{other}'"
            ))
        }
    };
    let mut k = crucible_gpu::optix::OptixKernel::new(device);
    if let Some(v) = p.get_u64("optix-samples")? {
        k.samples = v.clamp(1, 1 << 16) as u32;
    }
    if let Some(v) = p.get_u64("optix-bounces")? {
        k.bounces = v.clamp(1, 64) as u32;
    }
    Ok(k)
}

/// OptiX NVIDIA-native path tracer — a multi-bounce Monte-Carlo path tracer on the
/// OptiX ray-tracing pipeline (driver-resident; NVIDIA-only). Deterministic,
/// verified by same-device self-consistency. The device kernel ships as committed
/// PTX (JIT-linked by the driver), so the target needs only the NVIDIA driver.
fn cmd_optix(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "optix"))]
    {
        let _ = rest;
        Err("the optix test needs a build with `--features optix` (NVIDIA only)".to_string())
    }
    #[cfg(feature = "optix")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> = vec![
            "seconds",
            "shape",
            "burst-on",
            "burst-off",
            "device-id",
            "out",
            "no-report",
            "json",
            "help",
        ];
        allowed.extend_from_slice(GPU_OPTS);
        allowed.extend_from_slice(SHAPE_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let shape = shape_from(&p)?;
        let kernel = optix_kernel_from(&p)?;
        let mode = format!(
            "{} optix {}spp x{}bounce",
            shape.mode_str(),
            kernel.samples,
            kernel.bounces
        );
        let budget = Budget {
            duration: Duration::from_secs(seconds),
            shape,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        };
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

/// VRAM **integrity** test — a different test from `gpu` (the wattage thrasher).
/// This one hunts bad video memory; watts are irrelevant to it.
fn cmd_vram(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rest;
        Err(NO_GPU.to_string())
    }
    #[cfg(feature = "gpu")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        let mut allowed: Vec<&str> =
            vec!["seconds", "device-id", "out", "no-report", "json", "help"];
        allowed.extend_from_slice(GPU_OPTS);
        p.reject_unknown(&allowed)?;

        let seconds = seconds_arg(&p, 60)?;
        let kernel = vram_kernel_from(&p)?;
        let mode = format!("integrity {}", kernel.device.label());
        let budget = Budget::steady(Duration::from_secs(seconds));
        let mut runner = Runner::new(&p)?;
        runner.single_stage(&kernel, &budget, &mode)
    }
}

fn cmd_gpu_info(rest: &[String]) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rest;
        Err(NO_GPU.to_string())
    }
    #[cfg(feature = "gpu")]
    {
        let p = Parsed::parse(rest, COMMON_BOOLS)?;
        p.reject_unknown(&["help"])?;
        println!("GPU devices:");
        for d in [
            GpuDevice::Discrete(0),
            GpuDevice::Integrated(0),
            GpuDevice::Default,
        ] {
            match crucible_gpu::probe(d) {
                Ok(s) | Err(s) => println!("  {s}"),
            }
        }
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// run <profile>
// ---------------------------------------------------------------------------

fn cmd_run(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    let mut allowed: Vec<&str> = vec![
        "seconds",
        "core",
        "shape",
        "burst-on",
        "burst-off",
        "mb",
        "path",
        "size-mb",
        "block-kb",
        "keep",
        "buffered",
        "unbuffered",
        "all-drives",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
        // core-cycle / c-states rotation, game-load cadence.
        "dwell",
        "passes",
        "fps",
        "bound",
        "handoff-ms",
    ];
    allowed.extend_from_slice(GPU_OPTS);
    allowed.extend_from_slice(SHAPE_OPTS);
    p.reject_unknown(&allowed)?;
    let profile = p
        .positional
        .first()
        .cloned()
        .ok_or("run needs a profile: quick | soak | cross | power | storage-cross")?;

    let mut runner = Runner::new(&p)?;

    match profile.as_str() {
        "quick" => {
            let dur = Duration::from_secs(seconds_arg(&p, 15)?);
            let stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, Some(1024))?)),
                    "steady".into(),
                ),
                (
                    Box::new(StorageKernel::new(storage_cfg_from(&p, 256)?)),
                    "steady".into(),
                ),
            ];
            runner.sequential(stages, |_| Budget::steady(dur))
        }
        "soak" => {
            let dur = Duration::from_secs(seconds_arg(&p, 600)?);
            let stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, None)?)),
                    "steady".into(),
                ),
            ];
            runner.sequential(stages, |_| Budget::steady(dur))
        }
        "cross" => {
            // Concurrent CPU + RAM + storage under one stop and one timeline —
            // the worst-case transient mix that steady single-domain tests miss.
            let budget = Budget::steady(Duration::from_secs(seconds_arg(&p, 60)?));
            // `mut` is only needed when the gpu feature pushes an extra stage.
            #[allow(unused_mut)]
            let mut stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, Some(1024))?)),
                    "steady".into(),
                ),
                (
                    Box::new(StorageKernel::new(storage_cfg_from(&p, 512)?)),
                    "steady".into(),
                ),
            ];
            // GPU joins the full-system cross-load when built with `--features gpu`.
            #[cfg(feature = "gpu")]
            {
                let gpu = gpu_kernel_from(&p, Shape::Steady)?;
                let label = format!("steady {}", gpu.device.label());
                stages.push((Box::new(gpu), label));
            }
            runner.concurrent(stages, &budget)
        }
        "power" => {
            // Dense-marker CPU burst for the 1kHz power rig to profile the rails.
            let shape = shape_from_burst(&p)?;
            let budget = Budget {
                duration: Duration::from_secs(seconds_arg(&p, 60)?),
                shape,
                target_watts: None,
                phase_epoch: None,
                phase_offset: Duration::ZERO,
            };
            runner.note(
                "power profile: CPU burst only in Phase 1; GPU wattage sweeps arrive in Phase 3",
            );
            let kernel = CpuKernel::new(core_from(&p)?);
            let mode = format!("burst {}", cores_label(core_from(&p)?));
            runner.single_stage_budget(&kernel, &budget, &mode)
        }
        "storage-cross" => {
            // Multi-SSD cross-load: solo baseline vs concurrent, to expose
            // shared-lane / chipset-uplink slowdowns across physical drives.
            let seconds = seconds_arg(&p, 60)?;
            runner.all_drives_storage(&p, seconds)
        }
        // CPU <-> GPU transient scenarios. The choreography is the test: these
        // are the worst-case VRM/PSU patterns a steady-state run cannot produce.
        "in-phase" | "anti-phase" | "beat" => run_transient_scenario(&mut runner, &p, &profile),
        // Everything at once: CPU transients under RAM + storage + GPU + VRAM.
        "worst-case" => run_worst_case(&mut runner, &p),
        // Randomized never-settle transient cross-load.
        "chaos" => run_chaos(&mut runner, &p),
        // Frame-paced CPU->GPU handoff at a moderate power point (game signature).
        "game-load" => run_game_load(&mut runner, &p),
        // Single-core rotations: max-boost weak-core hunt / C-state idle test.
        "core-cycle" => run_core_cycle(&mut runner, &p),
        "c-states" => run_c_states(&mut runner, &p),
        other => Err(format!(
            "unknown profile '{other}' (expected: quick | soak | cross | power | \
             storage-cross | in-phase | anti-phase | beat | worst-case | chaos | \
             game-load | core-cycle | c-states)"
        )),
    }
}

fn ms(v: u64) -> Duration {
    Duration::from_millis(v)
}

/// CPU ↔ GPU transient scenarios — the choreography *is* the test.
///
/// A steady-state run cannot produce any of these; they are the patterns that
/// kill marginal VRMs and PSUs. The GPU is the strongest transient source in the
/// box (measured on an RTX 3070: ~43 W idle to ~221 W loaded, a ~180 W step).
fn run_transient_scenario(runner: &mut Runner, p: &Parsed, profile: &str) -> Result<u8, String> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (runner, p);
        Err(format!("profile '{profile}' needs a GPU; {NO_GPU}"))
    }
    #[cfg(feature = "gpu")]
    {
        let dur = Duration::from_secs(seconds_arg(p, 60)?);
        let on = p.get_u64("burst-on")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
        let off = p.get_u64("burst-off")?.unwrap_or(20).clamp(1, MAX_BURST_MS);

        let gpu = gpu_kernel_from(
            p,
            Shape::Burst {
                on: ms(on),
                off: ms(off),
            },
        )?;
        let gpu_label = gpu.device.label();
        let cores = core_from(p)?;

        // (gpu_on, gpu_off, gpu_start_delay, explanation)
        let (gpu_on, gpu_off, delay_ms, note) = match profile {
            "in-phase" => (
                on,
                off,
                0,
                "in-phase: CPU and GPU burst together — peak total system draw, the case \
                 that exposes PSU headroom and trips OCP",
            ),
            "anti-phase" => (
                on,
                off,
                // Offset by the ON time so the GPU drives while the CPU idles.
                on,
                "anti-phase: GPU bursts during CPU idle and vice versa — VRMs and PSU chase \
                 load back and forth; often nastier than simultaneous max",
            ),
            _ => (
                // Slightly longer GPU period: the phase relationship drifts
                // through every alignment on its own, no enumeration needed.
                on + 3,
                off + 3,
                0,
                "beat: CPU and GPU run at slightly different periods, sweeping the entire \
                 phase relationship over the run",
            ),
        };

        runner.note(note);

        let stages = vec![
            PhasedStage {
                kernel: Box::new(CpuKernel::new(cores)),
                mode: format!("burst {} {on}/{off}ms", cores_label(cores)),
                budget: Budget::burst(dur, ms(on), ms(off)),
                phase_offset: Duration::ZERO,
            },
            PhasedStage {
                kernel: Box::new(gpu),
                mode: format!("burst {gpu_label} {gpu_on}/{gpu_off}ms"),
                budget: Budget::burst(dur, ms(gpu_on), ms(gpu_off)),
                phase_offset: ms(delay_ms),
            },
        ];
        runner.concurrent_phased(stages, profile)
    }
}

/// Everything at once — the whole-platform worst case.
///
/// CPU transients running **anti-phase** to the GPU (so the VRMs and PSU never
/// settle) while RAM, storage and the VRAM integrity test all run underneath.
/// The value is not any single domain's number; it is that every domain
/// *verifies its own data* while the platform is maximally contended, so a
/// corruption that only appears under full load has nowhere to hide.
///
/// PCIe is loaded too: a verified host↔device transfer runs underneath, so the
/// link is exercised *while* the RAM controller and DMA engines are already
/// saturated — the condition under which a marginal riser or slot is most
/// likely to surface (fault-mode gap G1). It respects `--link-cuda` /
/// `--link-mb` / `--link-dir`; without them it uses the always-available wgpu
/// path.
fn run_worst_case(runner: &mut Runner, p: &Parsed) -> Result<u8, String> {
    let dur = Duration::from_secs(seconds_arg(p, 120)?);
    let on = p.get_u64("burst-on")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
    let off = p.get_u64("burst-off")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
    let cores = core_from(p)?;

    #[allow(unused_mut)]
    let mut stages: Vec<PhasedStage> = vec![
        PhasedStage {
            kernel: Box::new(CpuKernel::new(cores)),
            mode: format!("burst {} {on}/{off}ms", cores_label(cores)),
            budget: Budget::burst(dur, ms(on), ms(off)),
            phase_offset: Duration::ZERO,
        },
        PhasedStage {
            kernel: Box::new(MemKernel::new(mem_size_from(p, Some(2048))?)),
            mode: "steady".into(),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        },
        PhasedStage {
            kernel: Box::new(StorageKernel::new(storage_cfg_from(p, 512)?)),
            mode: "steady".into(),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        },
    ];

    #[cfg(feature = "gpu")]
    {
        // GPU bursts opposite the CPU: the load hand-off is the stressor.
        let gpu = gpu_kernel_from(
            p,
            Shape::Burst {
                on: ms(on),
                off: ms(off),
            },
        )?;
        let gpu_label = gpu.device.label();
        stages.push(PhasedStage {
            kernel: Box::new(gpu),
            mode: format!("burst {gpu_label} {on}/{off}ms"),
            budget: Budget::burst(dur, ms(on), ms(off)),
            phase_offset: ms(on),
        });

        // VRAM integrity underneath it all — a miscompare here, under full
        // platform contention, is the finding this whole profile exists for.
        let vram = vram_kernel_from(p)?;
        let vram_label = vram.device.label();
        stages.push(PhasedStage {
            kernel: Box::new(vram),
            mode: format!("integrity {vram_label}"),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        });

        // PCIe under contention (gap G1): a verified host<->device transfer
        // running while the RAM controller and DMA engines are already
        // saturated by everything above. Steady, so the link is never idle.
        let link = link_kernel_from(p)?;
        let link_label = link.device.label();
        let link_via = if link.cuda { "cuda" } else { "wgpu" };
        stages.push(PhasedStage {
            kernel: Box::new(link),
            mode: format!("transfer {link_label} {link_via}"),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        });
    }

    runner.note(
        "worst-case: CPU transients anti-phase to the GPU, under simultaneous RAM, \
         storage and VRAM-integrity load; every domain verifies its own data",
    );
    #[cfg(not(feature = "gpu"))]
    runner.note("built without GPU support - running CPU + RAM + storage only");

    runner.concurrent_phased(stages, "worst-case")
}

/// Randomized never-settle cross-load. CPU and GPU each run an independent
/// seeded jitter (spike/floor), so at any instant one may be slamming while the
/// other floors, and the alignment swaps continuously — a stochastic superset of
/// beat that also traverses the coincident-spike (OCP) and hand-off (VRM-chase)
/// worst cases. RAM/storage/VRAM/PCIe run steady underneath and verify their own
/// data while the power environment thrashes. `--per-core` decorrelates each CPU
/// core for CPU-VRM chaos instead of one synchronized system step.
fn run_chaos(runner: &mut Runner, p: &Parsed) -> Result<u8, String> {
    let dur = Duration::from_secs(seconds_arg(p, 120)?);
    let base = seed_from(p)?;
    let per_core = p.has("per-core");

    let cpu_jit = jitter_shape_from(p, crucible_core::rng::hash2(base, 1))?;
    let mut cpu_k = CpuKernel::new(core_from(p)?);
    cpu_k.per_core_jitter = per_core;

    #[allow(unused_mut)]
    let mut stages: Vec<PhasedStage> = vec![
        PhasedStage {
            kernel: Box::new(cpu_k),
            mode: format!(
                "{}{}",
                shape_label(&cpu_jit),
                if per_core { " per-core" } else { "" }
            ),
            budget: budget_with(dur, cpu_jit),
            phase_offset: Duration::ZERO,
        },
        PhasedStage {
            kernel: Box::new(MemKernel::new(mem_size_from(p, Some(2048))?)),
            mode: "steady".into(),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        },
        PhasedStage {
            kernel: Box::new(StorageKernel::new(storage_cfg_from(p, 512)?)),
            mode: "steady".into(),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        },
    ];

    #[cfg(feature = "gpu")]
    {
        let gpu_jit = jitter_shape_from(p, crucible_core::rng::hash2(base, 2))?;
        let gpu = gpu_kernel_from(p, gpu_jit)?;
        let gpu_label = format!("{} {}", shape_label(&gpu_jit), gpu.device.label());
        stages.push(PhasedStage {
            kernel: Box::new(gpu),
            mode: gpu_label,
            budget: budget_with(dur, gpu_jit),
            phase_offset: Duration::ZERO,
        });

        let vram = vram_kernel_from(p)?;
        let vram_label = format!("integrity {}", vram.device.label());
        stages.push(PhasedStage {
            kernel: Box::new(vram),
            mode: vram_label,
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        });

        let link = link_kernel_from(p)?;
        let link_via = if link.cuda { "cuda" } else { "wgpu" };
        let link_label = link.device.label();
        stages.push(PhasedStage {
            kernel: Box::new(link),
            mode: format!("transfer {link_label} {link_via}"),
            budget: Budget::steady(dur),
            phase_offset: Duration::ZERO,
        });
    }

    runner.note(&format!(
        "chaos: CPU + GPU on independent seeded jitter (base seed=0x{base:x}{}), never-settle, \
         under steady RAM/storage/VRAM-integrity/PCIe; every domain verifies its own data. \
         Re-run with --seed 0x{base:x} to re-command the same pattern.",
        if per_core { ", CPU per-core" } else { "" }
    ));
    #[cfg(not(feature = "gpu"))]
    runner.note("built without GPU support - chaos running CPU + RAM + storage only");

    runner.concurrent_phased(stages, "chaos")
}

/// Frame-paced CPU->GPU handoff at a moderate power point — the game *signature*
/// (frame cadence, CPU-leads-GPU, moderate power, VRAM streaming). It reproduces
/// the electrical/thermal/transient load a game puts on the PSU/VRM/boost, NOT
/// the graphics stack (no draw calls/shaders/present) — a graphics-path in-game
/// crash can still PASS here. Uniquely catches moderate-load boost instability
/// that a max-load test throttles away.
fn run_game_load(runner: &mut Runner, p: &Parsed) -> Result<u8, String> {
    let dur = Duration::from_secs(seconds_arg(p, 120)?);
    let fps = p.get_u64("fps")?.unwrap_or(120).clamp(10, 1000);
    let frame_us = 1_000_000 / fps;
    // Duty split: what fraction of the frame the CPU is busy (early — sim +
    // render-thread submission). The GPU's fraction is computed with its stage
    // (gpu-feature-only) so a non-GPU build carries no unused values.
    let bound = p.get("bound").unwrap_or("gpu").to_string();
    let cpu_frac = match bound.as_str() {
        "cpu" => 0.70,
        "balanced" => 0.55,
        _ => 0.40, // gpu-bound (typical AAA)
    };
    let frame = Duration::from_micros(frame_us);
    let cpu_on = Duration::from_micros((frame_us as f64 * cpu_frac) as u64);
    let cpu_off = frame.saturating_sub(cpu_on);

    #[allow(unused_mut)]
    let mut stages: Vec<PhasedStage> = vec![PhasedStage {
        kernel: Box::new(CpuKernel::new(core_from(p)?)),
        mode: format!("frame {fps}fps cpu-on {}us", cpu_on.as_micros()),
        budget: budget_with(
            dur,
            Shape::Burst {
                on: cpu_on,
                off: cpu_off,
            },
        ),
        phase_offset: Duration::ZERO,
    }];

    #[cfg(feature = "gpu")]
    {
        let gpu_frac = match bound.as_str() {
            "cpu" => 0.45,
            "balanced" => 0.60,
            _ => 0.70, // gpu-bound
        };
        let gpu_on = Duration::from_micros((frame_us as f64 * gpu_frac) as u64);
        let gpu_off = frame.saturating_sub(gpu_on);
        // GPU render starts after the CPU has "submitted" the frame.
        let handoff = match p.get_u64("handoff-ms")? {
            Some(v) => ms(v),
            None => cpu_on,
        };
        let gpu_shape = Shape::Burst {
            on: gpu_on,
            off: gpu_off,
        };
        let mut gpu = gpu_kernel_from(p, gpu_shape)?;
        // Size one dispatch to fit inside the render window (a 6 ms burst
        // default would overrun a 120 fps frame). Explicit --gpu-iters wins.
        if p.get_u64("gpu-iters")?.is_none() {
            gpu.iters = ((gpu_on.as_micros() / 50).clamp(32, 512)) as u32;
        }
        let gpu_label = format!(
            "frame {fps}fps gpu-on {}us handoff {}us {}",
            gpu_on.as_micros(),
            handoff.as_micros(),
            gpu.device.label()
        );
        stages.push(PhasedStage {
            kernel: Box::new(gpu),
            mode: gpu_label,
            budget: budget_with(dur, gpu_shape),
            phase_offset: handoff,
        });
    }

    runner.note(&format!(
        "game-load: {fps}fps CPU->GPU per-frame handoff ({bound}-bound), moderate power. \
         Reproduces the game's electrical/thermal/transient signature, NOT the graphics stack \
         (draw calls/shaders/present) — a graphics-path in-game crash can still PASS here."
    ));
    #[cfg(not(feature = "gpu"))]
    runner.note("built without GPU support - game-load runs the CPU frame cadence only");

    runner.concurrent_phased(stages, "game-load")
}

/// CoreCycler-style single-core boost rotation: load one logical core at a time
/// at steady (= its top single-core boost bin, unreachable in an all-core run
/// where the package clocks down), rotate through every core, N passes. The
/// weak-core-at-max-boost catcher; a miscompare is attributed to the pinned core.
fn run_core_cycle(runner: &mut Runner, p: &Parsed) -> Result<u8, String> {
    let dwell = p.get_u64("dwell")?.unwrap_or(30).clamp(1, 86_400);
    let passes = p.get_u64("passes")?.unwrap_or(2).clamp(1, 100);
    let total = crucible_core::sysinfo::logical_cpus().max(1);
    let mut stages: Vec<(Box<dyn LoadKernel>, String)> = Vec::new();
    for pass in 0..passes {
        for core in 0..total {
            stages.push((
                Box::new(CpuKernel::new(CoreSel::One(core))),
                format!("steady core={core} (pass {}/{passes})", pass + 1),
            ));
        }
    }
    runner.note(&format!(
        "core-cycle: single-core steady boost, {total} core(s) x {passes} pass(es), {dwell}s each \
         — places each core at its top boost bin to catch a weak core the all-core run masks. \
         Needs boost (CPB/Turbo) enabled in BIOS."
    ));
    let d = Duration::from_secs(dwell);
    runner.sequential(stages, move |_| Budget::steady(d))
}

/// Light-load / C-state test: a short work pulse + deep idle, one core at a
/// time, rotated — forcing C0<->C6 cycling, the idle->boost step, and low idle
/// voltage (the "crashes at idle, passes Prime95" class). Requires BIOS C-states
/// and boost enabled plus a power plan that permits deep idle, or it silently
/// tests nothing (the tool has no telemetry to confirm the state was reached).
/// An idle-only fault shows only via WHEA / reboot — the in-kernel check is dark
/// while the core idles.
fn run_c_states(runner: &mut Runner, p: &Parsed) -> Result<u8, String> {
    let dwell = p.get_u64("dwell")?.unwrap_or(120).clamp(1, 86_400);
    let passes = p.get_u64("passes")?.unwrap_or(1).clamp(1, 100);
    let shape = pulse_shape_from(p)?;
    let total = crucible_core::sysinfo::logical_cpus().max(1);
    let mut stages: Vec<(Box<dyn LoadKernel>, String)> = Vec::new();
    for pass in 0..passes {
        for core in 0..total {
            stages.push((
                Box::new(CpuKernel::new(CoreSel::One(core))),
                format!(
                    "{} core={core} (pass {}/{passes})",
                    shape_label(&shape),
                    pass + 1
                ),
            ));
        }
    }
    runner.note(
        "c-states: single-core pulse + deep-idle rotation. REQUIRES BIOS C-states + boost enabled \
         and a deep-idle power plan (Balanced, not High-Performance), else it silently no-ops. \
         Idle-only faults surface via WHEA/reboot only — the in-kernel check is dark at idle.",
    );
    let d = Duration::from_secs(dwell);
    runner.sequential(stages, move |_| Budget {
        duration: d,
        shape,
        target_watts: None,
        phase_epoch: None,
        phase_offset: Duration::ZERO,
    })
}

/// A duration+shape budget with no phase pinning (the orchestrator supplies the
/// epoch). Keeps the phased-stage builders terse.
fn budget_with(duration: Duration, shape: Shape) -> Budget {
    Budget {
        duration,
        shape,
        target_watts: None,
        phase_epoch: None,
        phase_offset: Duration::ZERO,
    }
}

// ---------------------------------------------------------------------------
// Runner: shared orchestration state + output
// ---------------------------------------------------------------------------

struct Runner {
    clock: Clock,
    device: DeviceId,
    markers: Arc<MarkerLog>,
    stop: StopFlag,
    out_dir: Option<PathBuf>,
    json: bool,
    stamp: String,
    report: Report,
    bridge_done: Arc<AtomicBool>,
    bridge: Option<JoinHandle<()>>,
    ui: bool,
    ui_stop: Option<Arc<AtomicBool>>,
    ui_handle: Option<JoinHandle<()>>,
    /// Write a per-stage results CSV alongside the JSON report.
    csv: bool,
    /// Log a periodic time-series telemetry CSV for the whole run.
    telemetry: bool,
    telemetry_stop: Option<Arc<AtomicBool>>,
    telemetry_handle: Option<JoinHandle<()>>,
}

impl Runner {
    fn new(p: &Parsed) -> Result<Runner, String> {
        let clock = Clock::new();
        let device = resolve_device(p);
        let markers = Arc::new(MarkerLog::new(clock));
        let stop = StopFlag::new();

        let out_dir = if p.has("no-report") {
            None
        } else {
            Some(match p.get("out") {
                Some(dir) => PathBuf::from(dir),
                None => default_out_dir(),
            })
        };

        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stamp = format!("{}-{}-{}", device.short_id, unix_secs, std::process::id());

        let report = Report::new(VERSION, device.clone(), &clock);

        // Register the console-control handler and start the bridge that trips
        // the run's StopFlag when Ctrl-C is seen, so a soak ends with a report
        // rather than a hard kill.
        install_ctrlc();
        let bridge_done = Arc::new(AtomicBool::new(false));
        let bridge = spawn_ctrlc_bridge(stop.clone(), bridge_done.clone());

        Ok(Runner {
            clock,
            device,
            markers,
            stop,
            out_dir,
            json: p.has("json"),
            stamp,
            report,
            bridge_done,
            bridge: Some(bridge),
            ui: p.has("ui"),
            ui_stop: None,
            ui_handle: None,
            csv: p.has("csv"),
            telemetry: p.has("telemetry-csv"),
            telemetry_stop: None,
            telemetry_handle: None,
        })
    }

    fn note(&mut self, msg: &str) {
        self.report.note(msg);
        eprintln!("note: {msg}");
    }

    /// Run one kernel as a whole run (steady default budget already built).
    fn single_stage(
        &mut self,
        kernel: &dyn LoadKernel,
        budget: &Budget,
        mode: &str,
    ) -> Result<u8, String> {
        self.single_stage_budget(kernel, budget, mode)
    }

    fn single_stage_budget(
        &mut self,
        kernel: &dyn LoadKernel,
        budget: &Budget,
        mode: &str,
    ) -> Result<u8, String> {
        self.begin();
        self.run_one(kernel, budget, mode);
        self.finish()
    }

    /// Run kernels back to back; a Ctrl-C between stages ends the run.
    fn sequential(
        &mut self,
        stages: Vec<(Box<dyn LoadKernel>, String)>,
        budget_for: impl Fn(usize) -> Budget,
    ) -> Result<u8, String> {
        self.begin();
        for (i, (kernel, mode)) in stages.iter().enumerate() {
            if self.stop.stopped() {
                self.note("stop requested — skipping remaining stages");
                break;
            }
            let budget = budget_for(i);
            self.run_one(kernel.as_ref(), &budget, mode);
        }
        self.finish()
    }

    /// Run kernels concurrently under one stop flag and one timeline.
    fn concurrent(
        &mut self,
        stages: Vec<(Box<dyn LoadKernel>, String)>,
        budget: &Budget,
    ) -> Result<u8, String> {
        self.begin();
        self.markers
            .stamp(Event::StageStart, "cross", "concurrent", "");
        eprintln!("running {} kernels concurrently…", stages.len());
        let t0 = Instant::now();

        let stop = &self.stop;
        let markers = &self.markers;
        let results: Vec<LoadResult> = std::thread::scope(|scope| {
            let handles: Vec<_> = stages
                .iter()
                .map(|(kernel, _)| scope.spawn(move || kernel.run(budget, stop, markers)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| LoadResult::setup_failure("kernel thread panicked"))
                })
                .collect()
        });

        let secs = t0.elapsed().as_secs_f64();
        self.markers
            .stamp(Event::StageStop, "cross", "concurrent", "");

        for ((kernel, mode), result) in stages.iter().zip(results) {
            print_stage(kernel.name(), mode, secs, &result);
            self.report.add_stage(StageReport::new(
                kernel.name(),
                kernel.kind(),
                mode.clone(),
                secs,
                result,
            ));
        }
        self.finish()
    }

    /// Cross-load with per-kernel load shapes and phase offsets.
    ///
    /// This is where the worst-case transients live. Unlike [`Runner::concurrent`]
    /// (one shared budget), each kernel gets its own [`Budget`] and its own start
    /// delay, which is what lets us run:
    ///
    /// * **in-phase** — every domain bursts ON together: peak total system draw,
    ///   the case that trips a marginal PSU's OCP.
    /// * **anti-phase** — CPU ON while GPU is OFF and vice versa: makes the VRMs
    ///   and PSU chase load back and forth continuously. Often nastier than
    ///   simultaneous max, and unreachable from any steady-state test.
    /// * **beat** — slightly different periods per domain, so the phase
    ///   relationship drifts through every alignment on its own.
    fn concurrent_phased(&mut self, stages: Vec<PhasedStage>, label: &str) -> Result<u8, String> {
        self.begin();
        self.markers.stamp(
            Event::StageStart,
            "cross",
            label,
            &format!("{} kernel(s)", stages.len()),
        );
        eprintln!(
            "cross-load [{label}]: {} kernel(s) concurrently",
            stages.len()
        );
        for st in &stages {
            eprintln!(
                "  {} <- {} (phase +{}ms)",
                st.kernel.name(),
                st.mode,
                st.phase_offset.as_millis()
            );
        }

        // One shared phase origin for every kernel in the run. Each kernel's
        // burst phase is measured from this, so the commanded offsets hold
        // exactly regardless of how long each kernel takes to initialize.
        let epoch = Instant::now();
        let phased: Vec<Budget> = stages
            .iter()
            .map(|st| st.budget.clone().phased(epoch, st.phase_offset))
            .collect();

        let t0 = Instant::now();
        let stop = &self.stop;
        let markers = &self.markers;
        let results: Vec<LoadResult> = std::thread::scope(|scope| {
            let handles: Vec<_> = stages
                .iter()
                .zip(phased.iter())
                .map(|(st, budget)| scope.spawn(move || st.kernel.run(budget, stop, markers)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| LoadResult::setup_failure("kernel thread panicked"))
                })
                .collect()
        });

        let secs = t0.elapsed().as_secs_f64();
        self.markers.stamp(Event::StageStop, "cross", label, "");

        for (st, result) in stages.iter().zip(results) {
            print_stage(st.kernel.name(), &st.mode, secs, &result);
            self.report.add_stage(StageReport::new(
                st.kernel.name(),
                st.kernel.kind(),
                st.mode.clone(),
                secs,
                result,
            ));
        }
        self.finish()
    }

    /// Multi-SSD cross-load: measure each physical drive's storage throughput
    /// solo, then all drives concurrently, and report the per-drive slowdown —
    /// the signature of shared PCIe/DMI lanes or a saturated chipset uplink.
    fn all_drives_storage(&mut self, p: &Parsed, seconds: u64) -> Result<u8, String> {
        let drives = crucible_storage::drives::discover();
        if drives.is_empty() {
            // No enumeration (non-Windows, or nothing fixed) — degrade to a
            // single-drive run against --path so the command still does work.
            self.note("no fixed physical drives detected; running single-drive storage on --path");
            let cfg = storage_cfg_from(p, 1024)?;
            let kernel = StorageKernel::new(cfg);
            let budget = Budget::steady(Duration::from_secs(seconds));
            let mode = if unbuffered_from(p) {
                "unbuffered"
            } else {
                "buffered"
            };
            return self.single_stage(&kernel, &budget, mode);
        }

        // Split the budget between the solo baseline and the concurrent phase.
        let solo_secs = (seconds * 2 / 5).max(1); // ~40%
        let conc_secs = seconds.saturating_sub(solo_secs).max(1);

        self.begin();
        self.markers.stamp(
            Event::StageStart,
            "storage-cross",
            "multi-drive",
            &format!("{} drive(s)", drives.len()),
        );
        eprintln!(
            "multi-SSD cross-load: {} physical drive(s), {solo_secs}s solo baseline then {conc_secs}s concurrent",
            drives.len()
        );

        // ---- Phase A: solo baseline (sequential) ----
        let mut runs: Vec<DriveRun> = Vec::new();
        for d in &drives {
            if self.stop.stopped() {
                break;
            }
            let cfg = storage_cfg_for(p, PathBuf::from(&d.primary_root), 1024)?;
            let kernel = StorageKernel::new(cfg);
            let budget = Budget::steady(Duration::from_secs(solo_secs));
            let mode = format!("solo {} [{}]", d.primary_root, d.bus.as_str());
            self.markers.stamp(Event::StageStart, "storage", &mode, "");
            let (result, stats) = kernel.run_measured(&budget, &self.stop, &self.markers);
            self.markers
                .stamp(Event::StageStop, "storage", &mode, &result.detail);
            // A drive we can't write to (e.g. a locked system root without
            // elevation) is skipped, not failed — it must not poison the verdict.
            if !result.ok && result.iterations == 0 {
                eprintln!("  [storage] SKIP  {} — {}", d.primary_root, result.detail);
                self.note(&format!("skipped {} — {}", d.primary_root, result.detail));
                continue;
            }
            print_stage("storage", &mode, solo_secs as f64, &result);
            self.report.add_stage(StageReport::new(
                "storage",
                Kind::Storage,
                mode,
                solo_secs as f64,
                result,
            ));
            runs.push(DriveRun {
                drive: d.clone(),
                solo: stats,
                conc: None,
            });
        }

        // ---- Phase B: concurrent (all writable drives at once) ----
        if runs.len() < 2 {
            self.note("fewer than 2 writable drives — concurrent phase has no cross-load signal");
        }
        if !runs.is_empty() && !self.stop.stopped() {
            let mut kernels: Vec<(StorageKernel, String)> = Vec::new();
            for dr in &runs {
                let cfg = storage_cfg_for(p, PathBuf::from(&dr.drive.primary_root), 1024)?;
                let mode = format!(
                    "concurrent {} [{}]",
                    dr.drive.primary_root,
                    dr.drive.bus.as_str()
                );
                kernels.push((StorageKernel::new(cfg), mode));
            }
            let budget = Budget::steady(Duration::from_secs(conc_secs));
            let budget_ref = &budget;
            let stop = &self.stop;
            let markers = &self.markers;
            eprintln!("running {} drive(s) concurrently…", kernels.len());
            let results: Vec<(LoadResult, StorageStats)> = std::thread::scope(|scope| {
                let handles: Vec<_> = kernels
                    .iter()
                    .map(|(k, _)| scope.spawn(move || k.run_measured(budget_ref, stop, markers)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            (
                                LoadResult::setup_failure("kernel thread panicked"),
                                StorageStats::default(),
                            )
                        })
                    })
                    .collect()
            });
            for (i, ((_, mode), (result, stats))) in kernels.iter().zip(results).enumerate() {
                print_stage("storage", mode, conc_secs as f64, &result);
                runs[i].conc = Some(stats);
                self.report.add_stage(StageReport::new(
                    "storage",
                    Kind::Storage,
                    mode.clone(),
                    conc_secs as f64,
                    result,
                ));
            }
        }

        self.markers
            .stamp(Event::StageStop, "storage-cross", "multi-drive", "");

        // ---- Comparison: solo vs concurrent per drive ----
        eprintln!("\nmulti-SSD cross-load summary (solo -> concurrent):");
        for dr in &runs {
            let Some(c) = dr.conc else { continue };
            let dw = pct_delta(dr.solo.write_mibps, c.write_mibps);
            let dr_ = pct_delta(dr.solo.read_mibps, c.read_mibps);
            let line = format!(
                "disk {} [{}] {}: write {:.0}->{:.0} MiB/s ({dw:+.0}%), read {:.0}->{:.0} MiB/s ({dr_:+.0}%)",
                dr.drive.number,
                dr.drive.bus.as_str(),
                dr.drive.primary_root,
                dr.solo.write_mibps,
                c.write_mibps,
                dr.solo.read_mibps,
                c.read_mibps,
            );
            eprintln!("  {line}");
            self.report.note(line);
            // A large drop under concurrency is the shared-lane / uplink signal.
            if dw <= -15.0 || dr_ <= -15.0 {
                self.report.note(format!(
                    "drive {} shows >15% slowdown under concurrent load — possible shared-lane/uplink contention",
                    dr.drive.primary_root
                ));
            }
        }

        self.finish()
    }

    fn begin(&mut self) {
        eprintln!(
            "cec-crucible {VERSION}  device {} ({})  qpc {:.1} MHz{}",
            self.device.short_id,
            self.device.board,
            self.clock.frequency() as f64 / 1.0e6,
            if self.clock.is_qpc() {
                ""
            } else {
                " (fallback)"
            },
        );
        let ts = self.markers.stamp(Event::RunStart, "run", "", &self.stamp);
        self.report.started = Some(ts);

        if self.ui {
            #[cfg(feature = "tui")]
            {
                self.markers.enable_live();
                let ui_stop = Arc::new(AtomicBool::new(false));
                let markers = Arc::clone(&self.markers);
                let stop_flag = Arc::clone(&ui_stop);
                let run_stop = self.stop.clone();
                let title = self.device.board.clone();
                self.ui_handle = Some(std::thread::spawn(move || {
                    tui::render_loop(markers, stop_flag, run_stop, title);
                }));
                self.ui_stop = Some(ui_stop);
            }
            #[cfg(not(feature = "tui"))]
            eprintln!("note: --ui needs a build with `--features tui`; continuing without it");
        }

        // Time-series telemetry logger: sample every lane ~4×/s into a CSV for
        // post-run graphing. Needs an output dir (skipped under --no-report). Uses
        // the same live lanes as the UI, so it costs nothing when not requested.
        if self.telemetry {
            match self.out_dir.clone() {
                Some(dir) => {
                    // The out dir is otherwise created lazily in finish(); the
                    // logger writes from the start of the run, so ensure it now.
                    if let Err(e) = std::fs::create_dir_all(&dir) {
                        eprintln!("warning: telemetry out dir {}: {e}", dir.display());
                    } else {
                        self.markers.enable_live();
                        let path = dir.join(format!("crucible-{}.telemetry.csv", self.stamp));
                        let stop = Arc::new(AtomicBool::new(false));
                        let stop_c = Arc::clone(&stop);
                        let markers = Arc::clone(&self.markers);
                        let start = Instant::now();
                        self.telemetry_handle = Some(std::thread::spawn(move || {
                            telemetry_loop(&path, markers, stop_c, start)
                        }));
                        self.telemetry_stop = Some(stop);
                    }
                }
                None => eprintln!(
                    "note: --telemetry-csv needs an output dir; ignored with --no-report"
                ),
            }
        }
    }

    fn run_one(&mut self, kernel: &dyn LoadKernel, budget: &Budget, mode: &str) {
        self.markers
            .stamp(Event::StageStart, kernel.name(), mode, "");
        eprintln!(
            "running {} for {}s ({mode})…",
            kernel.name(),
            budget.duration.as_secs()
        );
        let t0 = Instant::now();
        let result = kernel.run(budget, &self.stop, &self.markers);
        let secs = t0.elapsed().as_secs_f64();
        self.markers
            .stamp(Event::StageStop, kernel.name(), mode, &result.detail);
        print_stage(kernel.name(), mode, secs, &result);
        self.report.add_stage(StageReport::new(
            kernel.name(),
            kernel.kind(),
            mode,
            secs,
            result,
        ));
    }

    fn finish(&mut self) -> Result<u8, String> {
        // Stop and join the live UI first, so the terminal is restored before the
        // summary prints below.
        if let Some(ui_stop) = self.ui_stop.take() {
            ui_stop.store(true, Ordering::SeqCst);
        }
        if let Some(h) = self.ui_handle.take() {
            let _ = h.join();
        }
        // Stop + join the telemetry logger so its file is flushed and complete.
        if let Some(stop) = self.telemetry_stop.take() {
            stop.store(true, Ordering::SeqCst);
        }
        if let Some(h) = self.telemetry_handle.take() {
            let _ = h.join();
        }

        let ts = self.markers.stamp(Event::RunStop, "run", "", "");
        self.report.ended = Some(ts);
        self.report.aborted = self.stop.stopped();

        // Retire the Ctrl-C bridge thread now that the run is over.
        self.bridge_done.store(true, Ordering::SeqCst);
        if let Some(h) = self.bridge.take() {
            let _ = h.join();
        }

        // Write outputs.
        let mut report_path: Option<PathBuf> = None;
        let mut markers_path: Option<PathBuf> = None;
        let mut csv_path: Option<PathBuf> = None;
        let mut telemetry_path: Option<PathBuf> = None;
        if let Some(dir) = self.out_dir.clone() {
            match std::fs::create_dir_all(&dir) {
                Ok(()) => {
                    let rp = dir.join(format!("crucible-{}.report.json", self.stamp));
                    let mp = dir.join(format!("crucible-{}.markers.jsonl", self.stamp));
                    self.report.markers_file =
                        Some(mp.file_name().unwrap().to_string_lossy().into_owned());
                    if let Err(e) = self.report.write_json(&rp) {
                        eprintln!("warning: could not write report to {}: {e}", rp.display());
                    } else {
                        report_path = Some(rp);
                    }
                    if let Err(e) = self.markers.write_jsonl(&mp) {
                        eprintln!("warning: could not write markers to {}: {e}", mp.display());
                    } else {
                        markers_path = Some(mp);
                    }
                    if self.csv {
                        let cp = dir.join(format!("crucible-{}.report.csv", self.stamp));
                        match self.report.write_csv(&cp) {
                            Ok(()) => csv_path = Some(cp),
                            Err(e) => {
                                eprintln!("warning: could not write csv to {}: {e}", cp.display())
                            }
                        }
                    }
                    // The telemetry file was streamed by its logger thread; just
                    // surface its path in the summary.
                    if self.telemetry {
                        telemetry_path =
                            Some(dir.join(format!("crucible-{}.telemetry.csv", self.stamp)));
                    }
                }
                Err(e) => eprintln!("warning: could not create out dir {}: {e}", dir.display()),
            }
        }

        let verdict = self.report.verdict();

        if self.json {
            // Machine-readable: report JSON is the sole stdout content.
            println!("{}", self.report.to_pretty_json());
        } else {
            println!(
                "verdict: {}   errors: {}   markers: {}",
                verdict.as_str(),
                self.report.error_count(),
                self.markers.len(),
            );
            if let Some(rp) = &report_path {
                println!("report:  {}", rp.display());
            }
            if let Some(mp) = &markers_path {
                println!("markers: {}", mp.display());
            }
            if let Some(cp) = &csv_path {
                println!("csv:     {}", cp.display());
            }
            if let Some(tp) = &telemetry_path {
                println!("telem:   {}", tp.display());
            }
        }

        Ok(match verdict {
            Verdict::Pass | Verdict::Partial => 0,
            Verdict::Fail => 1,
        })
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Upper bound on a run duration, so an absurd `--seconds` can't overflow the
/// `Instant + Duration` deadline math (7 days is far beyond any real soak).
const MAX_SECONDS: u64 = 7 * 24 * 3600;
/// Upper bound on a burst half-period in milliseconds.
const MAX_BURST_MS: u64 = 60_000;

/// Read `--seconds`, clamped to a sane range.
fn seconds_arg(p: &Parsed, default: u64) -> Result<u64, String> {
    Ok(p.get_u64("seconds")?
        .unwrap_or(default)
        .clamp(1, MAX_SECONDS))
}

fn resolve_device(p: &Parsed) -> DeviceId {
    match p.get("device-id") {
        Some(id) => DeviceId::from_override(id),
        None => DeviceId::detect(),
    }
}

fn shape_from(p: &Parsed) -> Result<Shape, String> {
    match p.get("shape") {
        None | Some("steady") => Ok(Shape::Steady),
        Some("burst") => shape_from_burst(p),
        Some("pulse") => pulse_shape_from(p),
        Some("jitter") => jitter_shape_from(p, seed_from(p)?),
        Some(other) => Err(format!(
            "--shape expects steady|burst|pulse|jitter, got '{other}'"
        )),
    }
}

/// Options that only apply to the jitter/pulse shapes — listed so single-kernel
/// commands can accept them without a "unknown option" error.
const SHAPE_OPTS: &[&str] = &[
    "jit-on-min",
    "jit-on-max",
    "jit-off-min",
    "jit-off-max",
    "floor",
    "seed",
    "pulse-ms",
    "idle-ms",
    "per-core",
];

/// Parse `--seed` (decimal or `0x`-hex). Absent → a logged, entropy-derived
/// seed so an unseeded run is still reproducible from its report.
fn seed_from(p: &Parsed) -> Result<u64, String> {
    match p.get("seed") {
        Some(s) => {
            let s = s.trim();
            let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => s.parse::<u64>(),
            };
            parsed.map_err(|_| format!("--seed expects a decimal or 0x-hex integer, got '{s}'"))
        }
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            Ok(nanos ^ ((std::process::id() as u64) << 17) ^ 0x9E37_79B9_7F4A_7C15)
        }
    }
}

/// A randomized (jitter) shape from `--jit-*` / `--floor` (all clamped).
fn jitter_shape_from(p: &Parsed, seed: u64) -> Result<Shape, String> {
    let on_min = p.get_u64("jit-on-min")?.unwrap_or(5).clamp(1, MAX_BURST_MS);
    let on_max = p
        .get_u64("jit-on-max")?
        .unwrap_or(50)
        .clamp(on_min, MAX_BURST_MS);
    let off_min = p
        .get_u64("jit-off-min")?
        .unwrap_or(3)
        .clamp(1, MAX_BURST_MS);
    let off_max = p
        .get_u64("jit-off-max")?
        .unwrap_or(40)
        .clamp(off_min, MAX_BURST_MS);
    let floor = p.get_u64("floor")?.unwrap_or(12).min(100) as u8;
    Ok(Shape::Jitter {
        on_min: ms(on_min),
        on_max: ms(on_max),
        off_min: ms(off_min),
        off_max: ms(off_max),
        floor_pct: floor,
        seed,
    })
}

/// A deep-idle pulse shape from `--pulse-ms` (work) / `--idle-ms` (idle).
fn pulse_shape_from(p: &Parsed) -> Result<Shape, String> {
    let work = p.get_u64("pulse-ms")?.unwrap_or(5).clamp(1, 1000);
    let idle = p.get_u64("idle-ms")?.unwrap_or(300).clamp(1, MAX_BURST_MS);
    Ok(Shape::Pulse {
        work: ms(work),
        idle: ms(idle),
    })
}

/// A human/report label for a shape — includes the jitter seed so a failing run
/// can be replayed with `--seed`.
fn shape_label(shape: &Shape) -> String {
    match shape {
        Shape::Jitter {
            on_min,
            on_max,
            off_min,
            off_max,
            floor_pct,
            seed,
        } => format!(
            "jitter on[{},{}] off[{},{}]ms floor{}% seed=0x{:x}",
            on_min.as_millis(),
            on_max.as_millis(),
            off_min.as_millis(),
            off_max.as_millis(),
            floor_pct,
            seed
        ),
        Shape::Pulse { work, idle } => {
            format!("pulse {}/{}ms", work.as_millis(), idle.as_millis())
        }
        other => other.mode_str().to_string(),
    }
}

/// A burst shape from `--burst-on`/`--burst-off` (clamped), always bursty.
fn shape_from_burst(p: &Parsed) -> Result<Shape, String> {
    let on = p.get_u64("burst-on")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
    let off = p.get_u64("burst-off")?.unwrap_or(20).clamp(0, MAX_BURST_MS);
    Ok(Shape::Burst {
        on: Duration::from_millis(on),
        off: Duration::from_millis(off),
    })
}

fn core_from(p: &Parsed) -> Result<CoreSel, String> {
    match p.get("core") {
        None | Some("all") => Ok(CoreSel::All),
        Some(n) => n
            .parse::<usize>()
            .map(CoreSel::One)
            .map_err(|_| format!("--core expects 'all' or an index, got '{n}'")),
    }
}

fn cores_label(sel: CoreSel) -> String {
    match sel {
        CoreSel::All => "core=all".to_string(),
        CoreSel::One(i) => format!("core={i}"),
    }
}

fn mem_size_from(p: &Parsed, default_mb: Option<u64>) -> Result<MemSize, String> {
    match p.get_u64("mb")? {
        // saturating_mul so an absurd --mb can't overflow/panic (the kernel
        // caps to available RAM via try_reserve anyway).
        Some(mb) => Ok(MemSize::Bytes(mb.saturating_mul(1024 * 1024))),
        None => match default_mb {
            Some(mb) => Ok(MemSize::Bytes(mb.saturating_mul(1024 * 1024))),
            None => Ok(MemSize::default()),
        },
    }
}

/// Resolve buffering: explicit `--buffered`/`--unbuffered` win, else default to
/// unbuffered (true device load) on Windows.
fn unbuffered_from(p: &Parsed) -> bool {
    // Explicit `--unbuffered` wins; else `--buffered` opts out; else default to
    // device-direct I/O on Windows. (Single expression so the `cfg!` literal
    // doesn't fold into a lintable bool-literal if/else.)
    p.has("unbuffered") || (!p.has("buffered") && cfg!(windows))
}

fn storage_cfg_from(p: &Parsed, default_size_mb: u64) -> Result<StorageConfig, String> {
    storage_cfg_for(
        p,
        PathBuf::from(p.get("path").unwrap_or(".")),
        default_size_mb,
    )
}

/// Storage config for a specific directory (used per-drive by `--all-drives`).
fn storage_cfg_for(
    p: &Parsed,
    dir: PathBuf,
    default_size_mb: u64,
) -> Result<StorageConfig, String> {
    let size_mb = p.get_u64("size-mb")?.unwrap_or(default_size_mb);
    let block_kb = p.get_u64("block-kb")?.unwrap_or(1024);
    Ok(StorageConfig {
        dir,
        file_bytes: size_mb.saturating_mul(1024 * 1024),
        block_bytes: block_kb.saturating_mul(1024).min(usize::MAX as u64) as usize,
        keep: p.has("keep"),
        unbuffered: unbuffered_from(p),
    })
}

/// One kernel in a phased cross-load: its own load shape and its own start
/// offset, so domains can be run in-phase, anti-phase, or at beat frequencies.
struct PhasedStage {
    kernel: Box<dyn LoadKernel>,
    mode: String,
    budget: Budget,
    /// Phase offset from the run's shared epoch. Applied to the burst phase
    /// itself, not as a thread start delay — a start delay is swamped by
    /// per-kernel setup time (GPU init is ~100 ms, dwarfing a 20 ms offset).
    phase_offset: Duration,
}

/// One physical drive's solo baseline and concurrent throughput, for the
/// multi-SSD cross-load comparison.
struct DriveRun {
    drive: crucible_storage::drives::PhysicalDrive,
    solo: StorageStats,
    conc: Option<StorageStats>,
}

/// Percent change from `base` to `val` (negative = slowdown).
fn pct_delta(base: f64, val: f64) -> f64 {
    if base > 0.0 {
        (val - base) / base * 100.0
    } else {
        0.0
    }
}

fn print_stage(name: &str, mode: &str, secs: f64, result: &LoadResult) {
    let verdict = if result.passed() { "PASS" } else { "FAIL" };
    eprintln!(
        "  [{name:<7}] {verdict}  {secs:6.1}s  ({mode})  {}",
        result.detail
    );
}

/// Default output directory: the harness's collection dir if it already exists,
/// otherwise the current directory (never silently create ProgramData paths).
/// The telemetry-CSV logger thread body: write the header, then append one
/// sample (every live lane) roughly 4×/s until stopped, capturing one final
/// sample after the stop so the end state (phases DONE) is recorded. Best-effort
/// — a write error just ends logging, never the run.
fn telemetry_loop(path: &Path, markers: Arc<MarkerLog>, stop: Arc<AtomicBool>, start: Instant) {
    use std::io::Write;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("warning: telemetry csv {}: {e}", path.display());
            return;
        }
    };
    let mut w = std::io::BufWriter::new(file);
    if w.write_all(telemetry_csv_header().as_bytes()).is_err() {
        return;
    }
    loop {
        let done = stop.load(Ordering::SeqCst);
        let el = start.elapsed().as_secs_f64();
        if w
            .write_all(telemetry_csv_rows(el, &markers.live_snapshot()).as_bytes())
            .is_err()
        {
            return;
        }
        if done {
            break; // the final sample (post-stop) has been written
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let _ = w.flush();
}

fn default_out_dir() -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(pd) = std::env::var("ProgramData") {
            let p = PathBuf::from(pd).join("firstboot").join("logs");
            if p.is_dir() {
                return p;
            }
        }
    }
    PathBuf::from(".")
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

// --- Ctrl-C handling ---

fn install_ctrlc() {
    #[cfg(windows)]
    {
        win_ctrlc::install();
    }
}

/// Poll the console-handler flag and trip the run's stop when Ctrl-C is seen.
/// Exits promptly once `done` is set at end of run.
fn spawn_ctrlc_bridge(stop: StopFlag, done: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || loop {
        if done.load(Ordering::SeqCst) {
            break;
        }
        if CTRLC.load(Ordering::SeqCst) {
            eprintln!("\nstop requested (Ctrl-C) — finishing current stage and writing report…");
            stop.stop();
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    })
}

#[cfg(windows)]
mod win_ctrlc {
    use super::CTRLC;
    use std::sync::atomic::Ordering;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }

    unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
        CTRLC.store(true, Ordering::SeqCst);
        1 // TRUE: handled — suppress the default terminate so we can report.
    }

    pub fn install() {
        // SAFETY: registering a 'static handler fn; idempotent enough for our use.
        unsafe {
            SetConsoleCtrlHandler(Some(handler), 1);
        }
    }
}
