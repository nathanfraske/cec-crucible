// SPDX-License-Identifier: MIT
//! # crucible-gpu
//!
//! GPU load kernel — the Phase 3 thrasher, built on CubeCL (wgpu runtime by
//! default). It implements [`crucible_core::kernel::LoadKernel`] like every
//! other domain, so the orchestrator drives it with the same `StopFlag`,
//! `ShapeDriver` and QPC `MarkerLog` as CPU/RAM/storage — which is what makes
//! CPU↔GPU cross-load possible.
//!
//! ## What this is (and is not)
//!
//! This is the **wattage / thrasher** test: maximize board power and hammer VRM
//! transient response with load shapes. The **VRAM integrity test** (pattern
//! write/verify to find bad memory) is a *separate* test and is not implemented
//! here. Memory traffic appears below only as a power-tuning knob, because GDDR
//! and the memory controller are a large share of board watts — measured on an
//! RTX 3070, pure ALU reached only ~75% of the power limit while ALU + a
//! coalesced VRAM stream reached ~92%.
//!
//! ## Verification is not optional
//!
//! Spike 3a caught the failure mode that matters: when kernel compilation failed
//! on a CubeCL worker thread, the host loop happily reported 772 dispatches and
//! a plausible 1.65 TFLOP/s for work that never executed, while the GPU idled at
//! 43 W. Timings and dispatch counts can report a confident lie. So every run
//! here reads results back and checks them:
//!
//! * **liveness** — output must be finite and non-zero (catches "never ran"),
//! * **self-consistency** — the kernel is deterministic, so every verification
//!   must reproduce the first verification's checksum bit-for-bit; a mismatch is
//!   a soft error on the GPU.
//!
//! Either failure sets `error_count > 0`, which the report rolls up to FAIL.

pub(crate) mod geom;
pub mod link;
#[cfg(feature = "optix")]
pub mod optix;
#[cfg(all(windows, feature = "preview"))]
pub mod preview;
pub mod render;
#[cfg(feature = "rt")]
pub mod rt;
#[cfg(feature = "tensor")]
pub mod tensor;
pub mod vram;

use std::time::Instant;

use crucible_core::kernel::{
    Budget, Kind, LoadKernel, LoadResult, Shape, ShapeDriver, StopFlag, Tick,
};
use crucible_core::markers::{Event, MarkerLog};

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// Pure-ALU thrasher. Eight independent FMA accumulators per thread keep the
/// FMA pipeline fed; `a = a*c + d` converges slowly and stays bounded so values
/// never reach inf/NaN. The final store makes the result observable.
#[cube(launch)]
fn thrash<F: Float>(output: &mut Array<F>, iters: u32) {
    if ABSOLUTE_POS < output.len() {
        let seed = F::cast_from(ABSOLUTE_POS);
        let scale = F::new(0.0000001f32);
        let c = F::new(0.9999999f32);
        let d = F::new(0.0000001f32);

        let mut a0 = seed * scale + F::new(0.10f32);
        let mut a1 = seed * scale + F::new(0.11f32);
        let mut a2 = seed * scale + F::new(0.12f32);
        let mut a3 = seed * scale + F::new(0.13f32);
        let mut a4 = seed * scale + F::new(0.14f32);
        let mut a5 = seed * scale + F::new(0.15f32);
        let mut a6 = seed * scale + F::new(0.16f32);
        let mut a7 = seed * scale + F::new(0.17f32);

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

/// Mixed ALU + VRAM-bandwidth thrasher — the wattage knob. Same FMA core plus a
/// streaming load per iteration. `ABSOLUTE_POS` and `Array::len()` are `usize`
/// in CubeCL 0.10, so all index math stays `usize`.
#[cube(launch)]
fn thrash_mix<F: Float>(data: &Array<F>, output: &mut Array<F>, iters: u32, stride: usize) {
    if ABSOLUTE_POS < output.len() {
        let n = data.len();
        let seed = F::cast_from(ABSOLUTE_POS);
        let scale = F::new(0.0000001f32);
        let c = F::new(0.9999999f32);
        let d = F::new(0.0000001f32);

        let mut a0 = seed * scale + F::new(0.10f32);
        let mut a1 = seed * scale + F::new(0.11f32);
        let mut a2 = seed * scale + F::new(0.12f32);
        let mut a3 = seed * scale + F::new(0.13f32);
        let mut a4 = seed * scale + F::new(0.14f32);
        let mut a5 = seed * scale + F::new(0.15f32);
        let mut a6 = seed * scale + F::new(0.16f32);
        let mut a7 = seed * scale + F::new(0.17f32);

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

/// Which GPU to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDevice {
    /// Nth discrete GPU (the usual QC target).
    Discrete(usize),
    /// Nth integrated GPU.
    Integrated(usize),
    /// Whatever wgpu considers best / highest-power.
    Default,
}

impl GpuDevice {
    fn to_wgpu(self) -> WgpuDevice {
        match self {
            GpuDevice::Discrete(i) => WgpuDevice::DiscreteGpu(i),
            GpuDevice::Integrated(i) => WgpuDevice::IntegratedGpu(i),
            GpuDevice::Default => WgpuDevice::DefaultDevice,
        }
    }

    pub fn label(self) -> String {
        match self {
            GpuDevice::Discrete(i) => format!("discrete{i}"),
            GpuDevice::Integrated(i) => format!("integrated{i}"),
            GpuDevice::Default => "default".to_string(),
        }
    }
}

/// The GPU load kernel. Plain data only, so it is `Send + Sync`; the CubeCL
/// client is created inside [`LoadKernel::run`] on the worker thread.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub device: GpuDevice,
    /// Total GPU threads (output elements).
    pub threads: usize,
    /// Inner-loop iterations per dispatch. Runtime scalar, never comptime — a
    /// comptime bound risks unrolling a long loop into a shader-compiler
    /// explosion. Also sets the load-edge granularity: ~8192 ≈ 30 ms per
    /// dispatch on an RTX 3070, ~128 ≈ 3 ms (the practical floor).
    pub iters: u32,
    pub workgroup: u32,
    /// Mix in VRAM streaming. Strongly recommended — pure ALU tops out near 75%
    /// of the power limit.
    pub mix: bool,
    /// VRAM buffer size for `mix`, in MiB.
    pub data_mb: usize,
    /// Verify results every N dispatches (read-back costs PCIe bandwidth).
    pub verify_every: u64,
}

impl Default for GpuKernel {
    fn default() -> Self {
        GpuKernel {
            device: GpuDevice::Discrete(0),
            threads: 1 << 20,
            // ~3 ms per dispatch: fine-grained enough for burst transients and
            // ~600x under the Windows TDR watchdog.
            iters: 256,
            workgroup: 256,
            mix: true,
            data_mb: 1024,
            verify_every: 16,
        }
    }
}

impl GpuKernel {
    pub fn new(device: GpuDevice) -> Self {
        GpuKernel {
            device,
            ..Default::default()
        }
    }

    fn mode_detail(&self) -> String {
        format!(
            "{} {} threads={} iters={}",
            self.device.label(),
            if self.mix { "alu+vram" } else { "alu" },
            self.threads,
            self.iters
        )
    }
}

/// Is a device usable? Used by `gpu-info` to report what is present without
/// committing to a full run.
pub fn probe(device: GpuDevice) -> Result<String, String> {
    let dev = device.to_wgpu();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let client = WgpuRuntime::client(&dev);
        // Touch the device so a lazily-initialized failure surfaces here.
        let h = client.empty(1024);
        let _ = client.read_one(h);
    }));
    match res {
        Ok(()) => Ok(format!("{} : available", device.label())),
        Err(_) => Err(format!("{} : unavailable", device.label())),
    }
}

/// Deterministic checksum over a prefix of the output, for self-consistency.
fn checksum(out: &[f32], n: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in out.iter().take(n) {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// How many elements to fold into the verification checksum.
const VERIFY_PREFIX: usize = 4096;

impl LoadKernel for GpuKernel {
    fn name(&self) -> &str {
        "gpu"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let dev = self.device.to_wgpu();

        // A missing/failed adapter must be a clean setup failure, not a panic
        // that takes down the whole cross-load run.
        let client = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&dev)
        })) {
            Ok(c) => c,
            Err(_) => {
                return LoadResult::setup_failure(format!(
                    "no usable GPU for {} (adapter init failed)",
                    self.device.label()
                ))
            }
        };

        let threads = self.threads.max(1);
        let workgroup = self.workgroup.clamp(1, 1024);
        let groups = threads.div_ceil(workgroup as usize) as u32;
        let cube_dim = CubeDim::new_1d(workgroup);

        let handle = client.empty(threads * std::mem::size_of::<f32>());
        let data_elems = (self.data_mb * 1024 * 1024) / std::mem::size_of::<f32>();
        let data_handle = if self.mix {
            Some(client.empty(data_elems * std::mem::size_of::<f32>()))
        } else {
            None
        };
        // Adjacent threads read adjacent elements (coalesced) while the loop
        // sweeps the buffer; sustained bandwidth is what drives memory power.
        let stride = threads;

        markers.stamp(
            Event::Mark,
            "gpu",
            budget.shape.mode_str(),
            &self.mode_detail(),
        );

        let launch_once = || match &data_handle {
            Some(dh) => thrash_mix::launch::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                cube_dim,
                unsafe { ArrayArg::from_raw_parts(dh.clone(), data_elems) },
                unsafe { ArrayArg::from_raw_parts(handle.clone(), threads) },
                self.iters,
                stride,
            ),
            None => thrash::launch::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                cube_dim,
                unsafe { ArrayArg::from_raw_parts(handle.clone(), threads) },
                self.iters,
            ),
        };

        // Dispatches queued before each sync. Steady load wants a few in flight
        // so the GPU never drains between syncs — measured on an RTX 3070,
        // syncing every dispatch cost ~40 W of sustained board power. Burst
        // wants one, so the on/off edges actually track the shape driver instead
        // of being smeared by queued work.
        let batch = match budget.shape {
            Shape::Steady => 4,
            // Burst, pulse and jitter all want one dispatch per sync so the
            // on/off edges track the shape driver instead of being smeared by
            // queued work.
            Shape::Burst { .. } | Shape::Pulse { .. } | Shape::Jitter { .. } => 1,
        };

        let mut driver = ShapeDriver::start(budget, stop, markers, "gpu", self.mode_detail());
        let start = Instant::now();

        let mut dispatches: u64 = 0;
        let mut errors: u64 = 0;
        let mut verifications: u64 = 0;
        let mut reference: Option<u64> = None;
        let mut last_checksum: u64 = 0;
        let mut detail_extra = String::new();
        let mut device_lost = false;

        loop {
            match driver.tick() {
                Tick::Work => {
                    for _ in 0..batch {
                        launch_once();
                        dispatches += 1;
                    }

                    // Bounded queue: never let submissions run away from the GPU.
                    if cubecl::future::block_on(client.sync()).is_err() {
                        device_lost = true;
                        errors += 1;
                        detail_extra.push_str("; DEVICE LOST / sync error (possible TDR reset)");
                        break;
                    }

                    if self.verify_every > 0 && dispatches.is_multiple_of(self.verify_every) {
                        match verify(&client, &handle, &mut reference) {
                            Ok(sum) => {
                                last_checksum = sum;
                                verifications += 1;
                            }
                            Err(why) => {
                                errors += 1;
                                verifications += 1;
                                if !detail_extra.contains("VERIFY") {
                                    detail_extra.push_str(&format!("; VERIFY FAIL: {why}"));
                                }
                            }
                        }
                    }
                }
                Tick::Idle => {}
                Tick::Stop => break,
            }
        }

        let seconds = start.elapsed().as_secs_f64();

        // Always verify at least once, even for a very short run — otherwise a
        // kernel that never executed would report a clean pass.
        if verifications == 0 && !device_lost {
            match verify(&client, &handle, &mut reference) {
                Ok(sum) => {
                    last_checksum = sum;
                    verifications += 1;
                }
                Err(why) => {
                    errors += 1;
                    detail_extra.push_str(&format!("; VERIFY FAIL: {why}"));
                }
            }
        }

        // FMA lane-ops: 8 accumulators, 1 FMA each per iteration, 2 flops per FMA.
        let flops = dispatches as f64 * threads as f64 * self.iters as f64 * 8.0 * 2.0;
        let tflops = if seconds > 0.0 {
            flops / seconds / 1.0e12
        } else {
            0.0
        };

        let detail = format!(
            "{}, {dispatches} dispatch(es), {verifications} verify, ~{tflops:.2} TFLOP/s{}",
            self.mode_detail(),
            detail_extra
        );

        LoadResult::new(true, dispatches, last_checksum, errors, detail)
    }
}

/// Read the output back and check liveness + self-consistency.
///
/// The kernel is deterministic, so the checksum must be identical on every
/// verification. The first successful verification establishes the reference.
fn verify(
    client: &ComputeClient<WgpuRuntime>,
    handle: &cubecl::server::Handle,
    reference: &mut Option<u64>,
) -> Result<u64, String> {
    let bytes = client
        .read_one(handle.clone())
        .map_err(|_| "read-back failed".to_string())?;
    let out: &[f32] = f32::from_bytes(&bytes);
    if out.is_empty() {
        return Err("empty read-back".to_string());
    }

    let n = VERIFY_PREFIX.min(out.len());
    // Liveness: a kernel that never executed leaves the buffer zeroed.
    if !out.iter().take(n).all(|v| v.is_finite()) {
        return Err("non-finite output".to_string());
    }
    if !out.iter().take(n).any(|v| *v != 0.0) {
        return Err("output all zero - kernel did not run".to_string());
    }

    let sum = checksum(out, n);
    match *reference {
        None => {
            *reference = Some(sum);
            Ok(sum)
        }
        Some(expected) if expected == sum => Ok(sum),
        Some(expected) => Err(format!(
            "checksum mismatch (expected 0x{expected:016x}, got 0x{sum:016x}) - GPU compute error"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_deterministic_and_order_sensitive() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 2.0, 4.0, 3.0];
        assert_eq!(checksum(&a, 4), checksum(&a, 4));
        assert_ne!(checksum(&a, 4), checksum(&b, 4));
    }

    #[test]
    fn device_labels() {
        assert_eq!(GpuDevice::Discrete(0).label(), "discrete0");
        assert_eq!(GpuDevice::Integrated(1).label(), "integrated1");
        assert_eq!(GpuDevice::Default.label(), "default");
    }

    #[test]
    fn default_kernel_is_tdr_safe_and_mixed() {
        let k = GpuKernel::default();
        // ~3 ms per dispatch at 256 iters — far under the ~2000 ms watchdog.
        assert!(k.iters <= 512, "default dispatch must stay fine-grained");
        assert!(k.mix, "pure ALU only reaches ~75% of the power limit");
        assert_eq!(k.kind(), Kind::Gpu);
        assert_eq!(k.name(), "gpu");
    }
}
