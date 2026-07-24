// SPDX-License-Identifier: MIT
//! Tensor / matrix-core load — the cooperative-matrix (`cmma`) stress.
//!
//! The FMA thrasher, VRAM test and PCIe link all run on the general shader ALU,
//! memory or copy engines — none touch the **tensor cores**. This kernel does: a
//! sustained chain of 16×16×16 cooperative-matrix multiply-accumulates, which map
//! to the tensor cores. A dead tensor MAC lane sails through every other test and
//! fails first in the customer's DLSS / AI / ray-reconstruction workload.
//!
//! ## Why the Vulkan/SPIR-V backend
//!
//! CubeCL reaches tensor cores only through its SPIR-V compiler (`cubecl-spirv`)
//! or CUDA — **never** through WGSL (naga has no cooperative-matrix path). So this
//! kernel forces the Vulkan backend (`init_setup::<Vulkan>`), which switches
//! CubeCL to `wgpu<spirv>`. Requires the crate `tensor` feature (adds `ash`).
//!
//! ## dtype: f16 -> f32 (verified on the RTX 3070)
//!
//! Spike result: f16->f32 cmma is bit-exact vs a CPU reference on the 3070, so
//! the tensor cores are genuinely exercised. int8->int32 (which would be a
//! bit-exact cross-vendor golden) returned zeros on the 3070's Vulkan
//! cooperative-matrix — not exposed there — so this ships f16->f32 with
//! **same-device self-consistency** verification, exactly like the fp32 thrasher
//! and the render kernel: the computation is deterministic, so every verification
//! must reproduce the first's checksum bit-for-bit; a mismatch is a tensor-core
//! soft error. Liveness (non-zero output) catches a kernel that never ran.

use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;

use cubecl::prelude::*;
use cubecl::wgpu::{init_device, init_setup, RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime};
use half::f16;

use crate::GpuDevice;

/// Each cube (warp) computes `C(16x16 f32) = A @ B.T`, accumulating `iters` times
/// into one accumulator — a pure tensor-core throughput chain — then stores its
/// 16x16 tile. All cubes share A,B, so the result is deterministic
/// (`tile = iters * (A @ B.T)`), which the host self-consistency check relies on.
#[cube(launch)]
fn tensor_gemm(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>, iters: u32) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    for _ in 0..iters {
        cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);
    }
    let base = CUBE_POS * 256;
    cmma::store(
        &mut out.slice_mut(base, base + 256),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

/// Tensor-core load kernel.
#[derive(Debug, Clone)]
pub struct TensorKernel {
    pub device: GpuDevice,
    /// Number of 16x16 output tiles (one cube/warp each) — the occupancy knob.
    pub tiles: u32,
    /// cmma accumulations per cube per dispatch — the stress + TDR knob.
    pub iters: u32,
    /// Verify (read back + checksum) every this many dispatches.
    pub verify_every: u64,
}

impl Default for TensorKernel {
    fn default() -> Self {
        TensorKernel {
            device: GpuDevice::Discrete(0),
            tiles: 4096,
            iters: 256,
            verify_every: 64,
        }
    }
}

impl TensorKernel {
    pub fn new(device: GpuDevice) -> Self {
        TensorKernel {
            device,
            ..Default::default()
        }
    }

    fn wgpu_device(&self) -> WgpuDevice {
        match self.device {
            GpuDevice::Integrated(i) => WgpuDevice::IntegratedGpu(i),
            GpuDevice::Discrete(i) => WgpuDevice::DiscreteGpu(i),
            GpuDevice::Default => WgpuDevice::DefaultDevice,
        }
    }
}

/// FNV-1a over bytes — exact content hash for self-consistency.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl LoadKernel for TensorKernel {
    fn name(&self) -> &str {
        "tensor"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let tiles = self.tiles.clamp(1, 1 << 20);
        let iters = self.iters.clamp(1, 1 << 20);
        let label = self.device.label();

        // Force the Vulkan/SPIR-V backend, under catch_unwind: no Vulkan adapter
        // (or no cooperative-matrix support) becomes a clean setup failure.
        let client = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let base = self.wgpu_device();
            let setup = init_setup::<Vulkan>(&base, RuntimeOptions::default());
            let device = init_device(setup, RuntimeOptions::default());
            WgpuRuntime::client(&device)
        })) {
            Ok(c) => c,
            Err(_) => {
                return LoadResult::setup_failure(format!(
                    "{label} tensor: no Vulkan/SPIR-V backend (cooperative-matrix unavailable)"
                ))
            }
        };
        let backend = WgpuRuntime::name(&client);

        // Small shared A,B inputs (16x16 f16). Values kept small so the f32
        // accumulation over `iters` stays well within range.
        let a: Vec<f16> = (0..256)
            .map(|i| f16::from_f32(((i % 16) as f32 - 7.5) * 0.01))
            .collect();
        let b: Vec<f16> = (0..256)
            .map(|i| f16::from_f32(((i / 16) as f32 - 7.5) * 0.01))
            .collect();
        let a_h = client.create_from_slice(f16::as_bytes(&a));
        let b_h = client.create_from_slice(f16::as_bytes(&b));
        let out_len = (tiles as usize) * 256;
        let out_h = client.empty(out_len * std::mem::size_of::<f32>());

        let launch = || {
            tensor_gemm::launch::<WgpuRuntime>(
                &client,
                CubeCount::Static(tiles, 1, 1),
                CubeDim::new_1d(32),
                unsafe { ArrayArg::from_raw_parts(a_h.clone(), 256) },
                unsafe { ArrayArg::from_raw_parts(b_h.clone(), 256) },
                unsafe { ArrayArg::from_raw_parts(out_h.clone(), out_len) },
                iters,
            );
        };

        // Probe the first dispatch under catch_unwind (shader compile can fail).
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(launch)).is_err() {
            return LoadResult::setup_failure(format!(
                "{label} tensor: cmma kernel failed to build on the SPIR-V backend"
            ));
        }
        if cubecl::future::block_on(client.sync()).is_err() {
            return LoadResult::setup_failure(format!("{label} tensor: device lost on first sync"));
        }

        let mut driver = ShapeDriver::start(budget, stop, markers, "tensor", backend);
        let start = Instant::now();
        let mut dispatches: u64 = 0;
        let mut verifications: u64 = 0;
        let mut errors: u64 = 0;
        let mut reference: Option<u64> = None;

        loop {
            match driver.tick() {
                Tick::Work => {
                    launch();
                    dispatches += 1;
                    if dispatches.is_multiple_of(self.verify_every) {
                        if cubecl::future::block_on(client.sync()).is_err() {
                            errors += 1;
                            return LoadResult::new(
                                true,
                                dispatches,
                                reference.unwrap_or(0),
                                errors,
                                format!("{label} tensor: DEVICE LOST (possible TDR reset)"),
                            );
                        }
                        let bytes = match client.read_one(out_h.clone()) {
                            Ok(b) => b,
                            Err(_) => {
                                errors += 1;
                                return LoadResult::new(
                                    true,
                                    dispatches,
                                    0,
                                    errors,
                                    format!("{label} tensor: read-back failed"),
                                );
                            }
                        };
                        let h = fnv1a(&bytes);
                        verifications += 1;
                        match reference {
                            None => {
                                // Liveness: a live cmma chain must produce non-zero output.
                                let nonzero =
                                    f32::from_bytes(&bytes).iter().take(256).any(|v| *v != 0.0);
                                if !nonzero {
                                    errors += 1;
                                    return LoadResult::new(
                                        true,
                                        dispatches,
                                        h,
                                        errors,
                                        format!(
                                            "{label} tensor: output all-zero — cmma did not run"
                                        ),
                                    );
                                }
                                reference = Some(h);
                            }
                            Some(r) if r != h => {
                                errors += 1;
                                return LoadResult::new(
                                    true,
                                    dispatches,
                                    h,
                                    errors,
                                    format!(
                                        "{label} tensor: miscompare at dispatch {dispatches} \
                                         (got {h:#018x}, expected {r:#018x})"
                                    ),
                                );
                            }
                            _ => {}
                        }
                    }
                }
                Tick::Idle => {}
                Tick::Stop => break,
            }
        }

        let secs = start.elapsed().as_secs_f64();
        if verifications == 0 {
            return LoadResult::new(
                false,
                dispatches,
                0,
                0,
                format!("{label} tensor: nothing verified ({dispatches} dispatch(es), {secs:.1}s)"),
            );
        }
        // cmma MACs: tiles * iters * (16*16*16 MAC) * 2 flop.
        let tflops = dispatches as f64 * tiles as f64 * iters as f64 * 4096.0 * 2.0 / secs / 1.0e12;
        let detail = format!(
            "{label} {backend} {tiles} tiles x {iters} cmma/dispatch, {dispatches} dispatch(es), \
             {verifications} verified, ~{tflops:.1} TFLOP/s (f16 tensor)"
        );
        LoadResult::new(true, dispatches, reference.unwrap_or(0), errors, detail)
    }
}
