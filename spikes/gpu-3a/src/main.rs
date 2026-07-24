// SPDX-License-Identifier: MIT
//! cec-crucible GPU spike (milestone 3a).
//!
//! Questions this answers:
//!  1. Can a CubeCL kernel pin a GPU at its power limit? (the *wattage* test)
//!  2. Does the same kernel run unmodified on more than one vendor?
//!  3. Does it stay under the Windows TDR watchdog?
//!  4. Does the CUDA runtime work without a CUDA toolkit installed?
//!
//! Note: this is only the **wattage/thrasher** test. The VRAM *integrity* test
//! (pattern write/verify to find bad memory) is a separate test, not built here.
//! Memory traffic appears below purely as a power-tuning knob, because on a
//! modern board the memory controller + GDDR are a large share of total watts.

use std::time::{Duration, Instant};

use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// Pure-ALU thrasher. Eight independent FMA accumulators per thread keep the
/// FMA pipeline fed; `a = a*c + d` converges slowly and stays bounded so values
/// never reach inf/NaN. The final store makes results observable so nothing is
/// optimized away.
#[cube(launch)]
fn thrash<F: Float>(output: &mut Array<F>, iters: u32) {
    if ABSOLUTE_POS < output.len() {
        let seed = F::cast_from(ABSOLUTE_POS);
        let scale = F::new(0.0000001);
        let c = F::new(0.9999999);
        let d = F::new(0.0000001);

        let mut a0 = seed * scale + F::new(0.10);
        let mut a1 = seed * scale + F::new(0.11);
        let mut a2 = seed * scale + F::new(0.12);
        let mut a3 = seed * scale + F::new(0.13);
        let mut a4 = seed * scale + F::new(0.14);
        let mut a5 = seed * scale + F::new(0.15);
        let mut a6 = seed * scale + F::new(0.16);
        let mut a7 = seed * scale + F::new(0.17);

        for _i in 0..iters {
            a0 = a0 * c + d;
            a1 = a1 * c + d;
            a2 = a2 * c + d;
            a3 = a3 * c + d;
            a4 = a4 * c + d;
            a5 = a5 * c + d;
            a6 = a6 * c + d;
            a7 = a7 * c + d;
        }

        output[ABSOLUTE_POS] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
}

/// Mixed ALU + VRAM-bandwidth thrasher — the *wattage* knob. Same FMA core plus
/// one streaming load per iteration. `ABSOLUTE_POS` and `Array::len()` are
/// usize in CubeCL 0.10, so all index math stays usize.
#[cube(launch)]
fn thrash_mix<F: Float>(data: &Array<F>, output: &mut Array<F>, iters: u32, stride: usize) {
    if ABSOLUTE_POS < output.len() {
        let n = data.len();
        let seed = F::cast_from(ABSOLUTE_POS);
        let scale = F::new(0.0000001);
        let c = F::new(0.9999999);
        let d = F::new(0.0000001);

        let mut a0 = seed * scale + F::new(0.10);
        let mut a1 = seed * scale + F::new(0.11);
        let mut a2 = seed * scale + F::new(0.12);
        let mut a3 = seed * scale + F::new(0.13);
        let mut a4 = seed * scale + F::new(0.14);
        let mut a5 = seed * scale + F::new(0.15);
        let mut a6 = seed * scale + F::new(0.16);
        let mut a7 = seed * scale + F::new(0.17);

        let mut idx = ABSOLUTE_POS;
        for _i in 0..iters {
            idx = (idx + stride) % n;
            let v = data[idx];
            a0 = a0 * c + v * d;
            a1 = a1 * c + d;
            a2 = a2 * c + d;
            a3 = a3 * c + d;
            a4 = a4 * c + d;
            a5 = a5 * c + d;
            a6 = a6 * c + d;
            a7 = a7 * c + d;
        }

        output[ABSOLUTE_POS] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
}

/// Accumulators per thread — must match the kernel bodies above.
const ACCUMULATORS: f64 = 8.0;

struct Cfg {
    threads: usize,
    iters: u32,
    workgroup: u32,
    seconds: u64,
    mix: bool,
    data_mb: usize,
    stride: usize,
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Runtime-agnostic benchmark: identical kernel, identical schedule, whichever
/// backend the caller supplies.
fn run_bench<R: Runtime>(client: ComputeClient<R>, cfg: &Cfg) {
    let handle = client.empty(cfg.threads * std::mem::size_of::<f32>());
    let cube_dim = CubeDim::new_1d(cfg.workgroup);
    let groups = cfg.threads.div_ceil(cfg.workgroup as usize) as u32;

    let data_elems = cfg.data_mb * 1024 * 1024 / std::mem::size_of::<f32>();
    let data_handle = if cfg.mix {
        println!(
            "mix mode: {} MiB VRAM buffer, element stride {}",
            cfg.data_mb, cfg.stride
        );
        Some(client.empty(data_elems * std::mem::size_of::<f32>()))
    } else {
        None
    };

    let launch_once = || match &data_handle {
        Some(dh) => thrash_mix::launch::<f32, R>(
            &client,
            CubeCount::Static(groups, 1, 1),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(dh.clone(), data_elems) },
            unsafe { ArrayArg::from_raw_parts(handle.clone(), cfg.threads) },
            cfg.iters,
            cfg.stride,
        ),
        None => thrash::launch::<f32, R>(
            &client,
            CubeCount::Static(groups, 1, 1),
            cube_dim,
            unsafe { ArrayArg::from_raw_parts(handle.clone(), cfg.threads) },
            cfg.iters,
        ),
    };

    // Warm-up: forces shader/PTX compilation so the timed window is steady state.
    let warm = Instant::now();
    launch_once();
    cubecl::future::block_on(client.sync()).expect("warm-up sync failed");
    println!(
        "warm-up (incl. kernel compile): {:.0} ms",
        warm.elapsed().as_secs_f64() * 1000.0
    );

    // One dispatch alone, to show the margin against the ~2000 ms TDR watchdog.
    let one = Instant::now();
    launch_once();
    cubecl::future::block_on(client.sync()).expect("single-dispatch sync failed");
    println!(
        "single dispatch: {:.1} ms  (TDR watchdog is ~2000 ms)",
        one.elapsed().as_secs_f64() * 1000.0
    );

    println!("thrashing for {}s ...", cfg.seconds);
    let start = Instant::now();
    let deadline = start + Duration::from_secs(cfg.seconds);
    let mut dispatches: u64 = 0;

    while Instant::now() < deadline {
        for _ in 0..4 {
            launch_once();
            dispatches += 1;
        }
        if cubecl::future::block_on(client.sync()).is_err() {
            eprintln!("DEVICE LOST / sync error - likely TDR reset");
            std::process::exit(3);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Read back: a thrasher that silently no-ops would otherwise look like a pass.
    let bytes = client.read_one(handle.clone()).expect("read-back failed");
    let out: &[f32] = f32::from_bytes(&bytes);
    let finite = out.iter().take(4096).all(|v| v.is_finite());
    let nonzero = out.iter().take(4096).any(|v| *v != 0.0);

    let flops =
        dispatches as f64 * cfg.threads as f64 * cfg.iters as f64 * ACCUMULATORS * 2.0;

    println!();
    println!("dispatches      : {dispatches}");
    println!("elapsed         : {elapsed:.2} s");
    println!(
        "throughput      : {:.2} TFLOP/s (fp32 FMA)",
        flops / elapsed / 1.0e12
    );
    println!("sample out[0]   : {}", out[0]);
    println!("finite / nonzero: {finite} / {nonzero}");
    if !finite || !nonzero {
        eprintln!("WARNING: kernel output looks wrong - thrasher may not be doing real work");
        std::process::exit(2);
    }
    println!("OK");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let runtime = arg(&args, "--runtime").unwrap_or_else(|| "wgpu".to_string());
    let which = arg(&args, "--device").unwrap_or_else(|| "discrete".to_string());
    let threads: usize = arg(&args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20);

    let cfg = Cfg {
        threads,
        iters: arg(&args, "--iters")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192),
        workgroup: arg(&args, "--workgroup")
            .and_then(|s| s.parse().ok())
            .unwrap_or(256),
        seconds: arg(&args, "--seconds")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
        mix: arg(&args, "--mode").unwrap_or_default() == "mix",
        data_mb: arg(&args, "--data-mb")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024),
        // Default stride = thread count, so adjacent threads read adjacent
        // elements (coalesced) while the loop sweeps the buffer. Sustained
        // bandwidth is what drives memory-side power, not stall count.
        stride: arg(&args, "--stride")
            .and_then(|s| s.parse().ok())
            .unwrap_or(threads),
    };

    println!("cec-crucible GPU spike (CubeCL 0.10)");
    println!(
        "runtime={runtime} device={which} threads={} iters={} workgroup={} mode={} seconds={}",
        cfg.threads,
        cfg.iters,
        cfg.workgroup,
        if cfg.mix { "mix" } else { "fma" },
        cfg.seconds
    );

    match runtime.as_str() {
        "cuda" => {
            // No CUDA toolkit is required: cubecl-cuda uses cudarc with
            // fallback-dynamic-loading, so the driver is resolved at runtime.
            let device = CudaDevice { index: 0 };
            println!("acquiring CUDA client (driver loaded dynamically) ...");
            run_bench(CudaRuntime::client(&device), &cfg);
        }
        _ => {
            let device = match which.as_str() {
                "integrated" | "igpu" => WgpuDevice::IntegratedGpu(0),
                "default" => WgpuDevice::DefaultDevice,
                _ => WgpuDevice::DiscreteGpu(0),
            };
            println!("acquiring wgpu client ...");
            run_bench(WgpuRuntime::client(&device), &cfg);
        }
    }
}
