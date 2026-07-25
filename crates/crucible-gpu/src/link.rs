// SPDX-License-Identifier: MIT
//! PCIe link load — sustained, verified host↔device transfers.
//!
//! Unlike the thrasher and the VRAM test (both entirely on-card), this moves
//! real bytes across the PCIe link and verifies them. On a discrete GPU an
//! upload copies a CPU-visible staging buffer → device VRAM (a PCIe crossing)
//! and a read-back copies VRAM → a CPU-visible buffer (the other direction).
//!
//! ## Why raw wgpu, not the CubeCL client
//!
//! CubeCL's `create_from_slice` / `read_one` allocate a fresh buffer on *every*
//! call, so timing them measures allocation overhead, not the link — on the
//! bench they reported ~0.6 GB/s, which would make a fast Gen5 link and a bad
//! riser look identical. This kernel instead allocates a staging/device/readback
//! buffer pool **once** and reuses it, so each iteration is a pure
//! `copy_buffer_to_buffer` DMA. The number reflects the link.
//!
//! ## What this measures, and what it does not
//!
//! Achieved **H2D / D2H throughput** plus end-to-end integrity. It is
//! deliberately **not** a bad-riser detector: a marginal link *retries* and
//! still delivers correct data, so throughput barely moves and the checksum
//! still matches. Catching that needs the WHEA/AER error plane in
//! `docs/pcie-plan.md`; this is the *stimulus* plus a check for **uncorrected**
//! corruption. A host-RAM copy baseline is reported alongside, because host RAM
//! bandwidth — not the link — is usually the real ceiling, and a RAM-limited
//! number must not be mistaken for a bad link.
//!
//! Safety: buffer size is bounded and clamped to the adapter's max buffer size;
//! each iteration checks the stop flag and deadline; adapter/device init and the
//! first transfer are probed under `catch_unwind`; a lost device (poll/map
//! error) is caught and reported; and a run that moved nothing reports FAIL.

use std::time::{Duration, Instant};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, StopFlag};
use crucible_core::markers::{Event, MarkerLog, PHASE_DONE, PHASE_WORK};

use cubecl::future::block_on;

use crate::GpuDevice;

/// Transfer direction(s) to exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkDir {
    /// Host → device only (upload). No verification (nothing is read back).
    Up,
    /// Device → host only (download), verified against the known pattern.
    Down,
    /// Verified round trip: upload then read back and check. Exercises both
    /// directions and is the only mode that verifies integrity end to end.
    Both,
}

impl LinkDir {
    pub fn as_str(self) -> &'static str {
        match self {
            LinkDir::Up => "up",
            LinkDir::Down => "down",
            LinkDir::Both => "bidir",
        }
    }
}

/// PCIe transfer/verify load kernel.
#[derive(Debug, Clone)]
pub struct LinkKernel {
    pub device: GpuDevice,
    /// Size of each transfer, in MiB. Bounded well under any TDR concern (a
    /// 256 MiB DMA is milliseconds) and clamped to the adapter's max buffer.
    pub buf_mb: usize,
    pub dir: LinkDir,
    /// Use the CUDA transfer path (pinned memory + separate streams for true
    /// full-duplex on the GPU's copy engines). NVIDIA only; requires the `cuda`
    /// build feature. Ignored on the wgpu path.
    pub cuda: bool,
}

impl Default for LinkKernel {
    fn default() -> Self {
        LinkKernel {
            device: GpuDevice::Discrete(0),
            buf_mb: 256,
            dir: LinkDir::Both,
            cuda: false,
        }
    }
}

impl LinkKernel {
    pub fn new(device: GpuDevice) -> Self {
        LinkKernel {
            device,
            ..Default::default()
        }
    }

    fn power_preference(&self) -> wgpu::PowerPreference {
        match self.device {
            GpuDevice::Integrated(_) => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::HighPerformance,
        }
    }
}

/// FNV-1a over a byte slice — a cheap content hash for the pattern.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// High-entropy verifiable pattern. High entropy matches what PCIe's per-lane
/// scrambler puts on the wire anyway, and doubles as the verification payload.
fn make_pattern(bytes: usize, seed: u64) -> Vec<u8> {
    let mut buf = vec![0u8; bytes];
    let mut state = seed;
    for chunk in buf.chunks_mut(8) {
        let v = splitmix64(&mut state).to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&v[..n]);
    }
    buf
}

fn gbps(bytes: u64, secs: f64) -> f64 {
    if secs > 0.0 {
        (bytes as f64 / 1.0e9) / secs
    } else {
        0.0
    }
}

/// A reused buffer pool + the wgpu device/queue driving it.
struct LinkGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging: wgpu::Buffer, // MAP_WRITE | COPY_SRC, pre-filled with the pattern
    device_buf: wgpu::Buffer, // COPY_DST | COPY_SRC (VRAM)
    readback: wgpu::Buffer, // MAP_READ | COPY_DST
    bytes: u64,
}

impl LinkGpu {
    /// Initialise the adapter/device and allocate the reusable pool. Returns the
    /// pool and the actual buffer size (possibly clamped to the adapter limit).
    fn init(link: &LinkKernel, want_bytes: u64, pattern: &[u8]) -> Result<LinkGpu, String> {
        // Default instance enables all available backends (DX12/Vulkan on
        // Windows); the adapter request picks the high-performance one.
        let instance = wgpu::Instance::default();

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: link.power_preference(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no usable GPU adapter: {e}"))?;

        let limits = adapter.limits();
        // Clamp to what the adapter can bind in one buffer.
        let bytes = want_bytes.min(limits.max_buffer_size).max(4096) & !7;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("crucible-link"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("could not create device: {e}"))?;

        // Staging buffer, filled once with the pattern (mapped at creation).
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("link-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(&pattern[..bytes as usize]);
        }
        staging.unmap();

        let device_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("link-device"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("link-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(LinkGpu {
            device,
            queue,
            staging,
            device_buf,
            readback,
            bytes,
        })
    }

    /// Host → device: copy staging → VRAM and wait. Returns false on device loss.
    fn upload(&self) -> bool {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.staging, 0, &self.device_buf, 0, self.bytes);
        self.queue.submit(Some(enc.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_ok()
    }

    /// Device → host: copy VRAM → readback, map it, and run `check` on the bytes.
    /// Returns false on device loss / map failure.
    fn download(&self, check: impl FnOnce(&[u8])) -> bool {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.device_buf, 0, &self.readback, 0, self.bytes);
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r.is_ok());
            });
        if self
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return false;
        }
        let mapped_ok = rx.recv().unwrap_or(false);
        if !mapped_ok {
            return false;
        }
        {
            let view = self.readback.slice(..).get_mapped_range();
            check(&view);
        }
        self.readback.unmap();
        true
    }
}

impl LoadKernel for LinkKernel {
    fn name(&self) -> &str {
        "pcie"
    }

    fn kind(&self) -> Kind {
        Kind::Pcie
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let want_bytes = (self.buf_mb.max(1) * 1024 * 1024) as u64 & !7;
        let pattern = make_pattern(want_bytes as usize, 0x50C1_E571_A7A5_0001);
        let want_hash = fnv1a(&pattern);

        // Host-RAM copy baseline (the ceiling a PCIe number is read against).
        // Multi-threaded so it reflects real memory bandwidth, not a single
        // core; a single-threaded memcpy under-reports and can read *below* the
        // DMA rate, which would be nonsense. `dst` is pre-touched so first-write
        // page faults don't skew it.
        let host_gbps = {
            let threads = crucible_core::sysinfo::logical_cpus().clamp(1, 16);
            let mut dst = pattern.clone(); // allocate + first-touch all pages
            let reps = 8u64;
            let chunk = (want_bytes as usize).div_ceil(threads).max(1);
            let t = Instant::now();
            for _ in 0..reps {
                std::thread::scope(|s| {
                    for (d, src) in dst.chunks_mut(chunk).zip(pattern.chunks(chunk)) {
                        s.spawn(move || d.copy_from_slice(src));
                    }
                });
            }
            std::hint::black_box(&dst);
            gbps(want_bytes * reps, t.elapsed().as_secs_f64())
        };

        // CUDA transfer path (pinned + dual-stream full-duplex). Only compiled
        // with the `cuda` feature and only when requested. `run_cuda` probes the
        // driver at runtime and returns None if CUDA isn't actually available,
        // in which case we fall through to the portable wgpu path.
        #[cfg(feature = "cuda")]
        if self.cuda {
            if let Some(result) = self.run_cuda(budget, stop, markers, host_gbps, &pattern) {
                return result;
            }
            markers.stamp(
                Event::Mark,
                "pcie",
                "cuda-fallback",
                "CUDA unavailable -> wgpu",
            );
        }

        // Init adapter/device + pool under catch_unwind so a driver/validation
        // panic becomes a clean setup failure instead of taking down the run.
        let gpu = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            LinkGpu::init(self, want_bytes, &pattern)
        })) {
            Ok(Ok(g)) => g,
            Ok(Err(e)) => return LoadResult::setup_failure(e),
            Err(_) => {
                return LoadResult::setup_failure(format!(
                    "GPU init panicked for {}",
                    self.device.label()
                ))
            }
        };

        let buf_bytes = gpu.bytes;
        let real_mb = buf_bytes / (1024 * 1024);
        let mode = format!(
            "{} {} {}MiB",
            self.device.label(),
            self.dir.as_str(),
            real_mb
        );
        markers.stamp(Event::Mark, "pcie", self.dir.as_str(), &mode);

        // Seed the device buffer for download-only mode.
        if self.dir == LinkDir::Down && !gpu.upload() {
            return LoadResult::setup_failure("device lost seeding VRAM for download test");
        }

        let deadline = Instant::now() + budget.duration;
        let start = Instant::now();

        let mut bytes_up: u64 = 0;
        let mut bytes_down: u64 = 0;
        let mut secs_up = 0.0f64;
        let mut secs_down = 0.0f64;
        let mut transfers: u64 = 0;
        let mut verifies: u64 = 0;
        let mut errors: u64 = 0;
        let mut first_fail: Option<usize> = None;
        let mut device_lost = false;

        // Live-UI lane (None unless a UI is attached); fed verbosely each iteration.
        let lane = markers.register_lane("pcie");
        let mut last_note = Instant::now() - Duration::from_secs(1);

        while !stop.stopped() && Instant::now() < deadline {
            if matches!(self.dir, LinkDir::Up | LinkDir::Both) {
                let t = Instant::now();
                if !gpu.upload() {
                    device_lost = true;
                    break;
                }
                secs_up += t.elapsed().as_secs_f64();
                bytes_up += buf_bytes;
            }
            if matches!(self.dir, LinkDir::Down | LinkDir::Both) {
                let t = Instant::now();
                let mut local_err: Option<usize> = None;
                let ok = gpu.download(|bytes| {
                    if bytes.len() != pattern.len() || bytes != pattern.as_slice() {
                        local_err = Some(
                            bytes
                                .iter()
                                .zip(pattern.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0),
                        );
                    }
                });
                if !ok {
                    device_lost = true;
                    break;
                }
                secs_down += t.elapsed().as_secs_f64();
                bytes_down += buf_bytes;
                verifies += 1;
                if let Some(i) = local_err {
                    errors += 1;
                    first_fail.get_or_insert(i);
                }
            }
            transfers += 1;

            // Verbose live readout (throttled to ~90 ms; a no-op when headless).
            if let Some(l) = &lane {
                l.set_phase(PHASE_WORK);
                l.bump_work();
                let now = Instant::now();
                if now.duration_since(last_note) >= Duration::from_millis(90) {
                    last_note = now;
                    let up = gbps(bytes_up, secs_up);
                    let down = gbps(bytes_down, secs_down);
                    let moved = (bytes_up + bytes_down) as f64 / (1024.0 * 1024.0 * 1024.0);
                    let dir = match self.dir {
                        LinkDir::Up => "H2D (upload)",
                        LinkDir::Down => "D2H (readback)",
                        LinkDir::Both => "H2D+D2H (full-duplex)",
                    };
                    l.set_hash(want_hash);
                    l.set_detail(&format!(
                        "dir: {dir}\n\
                         H2D: {up:.2} GB/s\n\
                         D2H: {down:.2} GB/s\n\
                         moved: {moved:.2} GiB\n\
                         verified: {verifies}\n\
                         errors: {errors}"
                    ));
                }
            }
        }

        if let Some(l) = &lane {
            l.set_hash(want_hash);
            l.set_phase(PHASE_DONE);
        }

        let up = gbps(bytes_up, secs_up);
        let down = gbps(bytes_down, secs_down);
        let total_gib = (bytes_up + bytes_down) as f64 / (1024.0 * 1024.0 * 1024.0);
        let _ = start;

        let mut detail = format!(
            "{mode}, {transfers} transfer(s), {verifies} verified, {total_gib:.1} GiB moved, \
             H2D ~{up:.1} GB/s, D2H ~{down:.1} GB/s, host-RAM ~{host_gbps:.1} GB/s"
        );
        if device_lost {
            detail.push_str("; DEVICE LOST / transfer error");
            errors += 1;
        }
        if let Some(i) = first_fail {
            errors = errors.max(1);
            detail.push_str(&format!("; VERIFY FAIL: first mismatch at byte {i}"));
        }
        if transfers == 0 && !device_lost {
            detail.push_str("; NO TRANSFERS COMPLETED");
            errors += 1;
        }
        if self.dir == LinkDir::Up {
            detail.push_str("; (up-only: not verified — no read-back)");
        }

        LoadResult::new(true, transfers, want_hash, errors, detail)
    }
}

#[cfg(feature = "cuda")]
impl LinkKernel {
    /// CUDA transfer path. Pinned host buffers + two streams let H2D and D2H run
    /// on the GPU's separate copy engines *at the same time* — true full-duplex,
    /// which wgpu's single queue cannot do. Measures H2D solo, D2H solo, and the
    /// simultaneous aggregate, splitting the budget three ways. Driver-only: no
    /// NVRTC, so it runs with just the NVIDIA driver, no CUDA toolkit.
    ///
    /// Uses `cudarc` directly rather than CubeCL's CUDA runtime: CubeCL's
    /// transfer primitives allocate a fresh buffer per call (which is what made
    /// the wgpu-via-CubeCL numbers meaningless), and this path needs a reused
    /// pool + explicit streams for full-duplex. It still *detects* CUDA the same
    /// way — `CudaContext::new` probes the driver at runtime (loading nvcuda.dll
    /// dynamically); if there is no NVIDIA driver/device this returns `None` and
    /// the caller falls back to wgpu. It is never assumed present.
    fn run_cuda(
        &self,
        budget: &Budget,
        stop: &StopFlag,
        markers: &MarkerLog,
        host_gbps: f64,
        pattern: &[u8],
    ) -> Option<LoadResult> {
        use cudarc::driver::CudaContext;
        use std::panic::AssertUnwindSafe;

        let bytes = pattern.len();

        // Runtime capability probe: this actually initializes the CUDA driver on
        // device 0. Absent driver / non-NVIDIA GPU -> Err/panic -> None (fall
        // back to wgpu). This is the detection, not an assumption.
        let setup = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<_, String> {
            let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
            let up = ctx.new_stream().map_err(|e| format!("{e:?}"))?;
            let dn = ctx.new_stream().map_err(|e| format!("{e:?}"))?;
            Ok((ctx, up, dn))
        }));
        let (ctx, up, dn) = match setup {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                eprintln!("note: CUDA not available ({e}); falling back to the wgpu link path");
                return None;
            }
            Err(_) => {
                eprintln!("note: CUDA not available (no NVIDIA driver?); falling back to wgpu");
                return None;
            }
        };

        let alloc = (|| -> Result<_, String> {
            let dev_up = up
                .alloc_zeros::<u8>(bytes)
                .map_err(|e| format!("device alloc: {e:?}"))?;
            let dev_dn = dn
                .alloc_zeros::<u8>(bytes)
                .map_err(|e| format!("device alloc: {e:?}"))?;
            // SAFETY: memory is uninitialised; we fill pin_up before reading it
            // and only ever write pin_dn from the GPU before reading it back.
            let pin_up = unsafe { ctx.alloc_pinned::<u8>(bytes) }
                .map_err(|e| format!("pinned alloc: {e:?}"))?;
            let pin_dn = unsafe { ctx.alloc_pinned::<u8>(bytes) }
                .map_err(|e| format!("pinned alloc: {e:?}"))?;
            Ok((dev_up, dev_dn, pin_up, pin_dn))
        })();
        let (mut dev_up, mut dev_dn, mut pin_up, mut pin_dn) = match alloc {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "note: CUDA alloc failed ({e}); falling back to wgpu (try a smaller --link-mb)"
                );
                return None;
            }
        };

        // Fill the pinned upload source, and seed the download source device
        // buffer with the pattern (one H2D) so the read-back can be verified.
        if let Ok(s) = pin_up.as_mut_slice() {
            s.copy_from_slice(pattern);
        }
        let src: &[u8] = match pin_up.as_slice() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("note: CUDA pinned read failed ({e:?}); falling back to wgpu");
                return None;
            }
        };
        if dn.memcpy_htod(src, &mut dev_dn).is_err() || dn.synchronize().is_err() {
            eprintln!("note: CUDA seed transfer failed; falling back to wgpu");
            return None;
        }

        markers.stamp(
            Event::Mark,
            "pcie",
            "cuda",
            &format!(
                "{} cuda {}MiB dual-stream",
                self.device.label(),
                bytes / (1024 * 1024)
            ),
        );

        let third = budget.duration / 3;
        let mut lost = false;

        // Live-UI lane (None unless a UI is attached); fed verbosely each phase.
        let lane = markers.register_lane("pcie");
        let cuda_hash = fnv1a(pattern);
        let mut last_note = Instant::now() - Duration::from_secs(1);

        // Phase 1 — H2D solo.
        let (mut h2d_bytes, mut h2d_secs) = (0u64, 0.0f64);
        let end = Instant::now() + third;
        while !stop.stopped() && Instant::now() < end && !lost {
            let t = Instant::now();
            if up.memcpy_htod(src, &mut dev_up).is_err() || up.synchronize().is_err() {
                lost = true;
                break;
            }
            h2d_secs += t.elapsed().as_secs_f64();
            h2d_bytes += bytes as u64;
            if let Some(l) = &lane {
                l.set_phase(PHASE_WORK);
                l.bump_work();
                let now = Instant::now();
                if now.duration_since(last_note) >= Duration::from_millis(90) {
                    last_note = now;
                    let r = gbps(h2d_bytes, h2d_secs);
                    let moved = h2d_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    l.set_hash(cuda_hash);
                    l.set_detail(&format!(
                        "phase: 1/3 H2D solo (upload)\n\
                         H2D: {r:.2} GB/s\n\
                         host-RAM: {host_gbps:.1} GB/s\n\
                         moved: {moved:.2} GiB\n\
                         path: CUDA dual-stream (pinned)"
                    ));
                }
            }
        }

        // Phase 2 — D2H solo.
        let (mut d2h_bytes, mut d2h_secs) = (0u64, 0.0f64);
        let end = Instant::now() + third;
        while !stop.stopped() && Instant::now() < end && !lost {
            let t = Instant::now();
            let ok = match pin_dn.as_mut_slice() {
                Ok(dst) => dn.memcpy_dtoh(&dev_dn, dst).is_ok(),
                Err(_) => false,
            };
            if !ok || dn.synchronize().is_err() {
                lost = true;
                break;
            }
            d2h_secs += t.elapsed().as_secs_f64();
            d2h_bytes += bytes as u64;
            if let Some(l) = &lane {
                l.set_phase(PHASE_WORK);
                l.bump_work();
                let now = Instant::now();
                if now.duration_since(last_note) >= Duration::from_millis(90) {
                    last_note = now;
                    let r = gbps(d2h_bytes, d2h_secs);
                    let moved = d2h_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    l.set_hash(cuda_hash);
                    l.set_detail(&format!(
                        "phase: 2/3 D2H solo (readback)\n\
                         D2H: {r:.2} GB/s\n\
                         host-RAM: {host_gbps:.1} GB/s\n\
                         moved: {moved:.2} GiB\n\
                         path: CUDA dual-stream (pinned)"
                    ));
                }
            }
        }

        // Phase 3 — full-duplex: both directions enqueued before syncing, on
        // separate streams, so they run on separate copy engines at once.
        let (mut dup_bytes, mut dup_secs) = (0u64, 0.0f64);
        let end = Instant::now() + third;
        while !stop.stopped() && Instant::now() < end && !lost {
            let t = Instant::now();
            let enq_up = up.memcpy_htod(src, &mut dev_up).is_ok();
            let enq_dn = match pin_dn.as_mut_slice() {
                Ok(dst) => dn.memcpy_dtoh(&dev_dn, dst).is_ok(),
                Err(_) => false,
            };
            if !enq_up || !enq_dn || up.synchronize().is_err() || dn.synchronize().is_err() {
                lost = true;
                break;
            }
            dup_secs += t.elapsed().as_secs_f64();
            dup_bytes += 2 * bytes as u64;
            if let Some(l) = &lane {
                l.set_phase(PHASE_WORK);
                l.bump_work();
                let now = Instant::now();
                if now.duration_since(last_note) >= Duration::from_millis(90) {
                    last_note = now;
                    let r = gbps(dup_bytes, dup_secs);
                    let solo = gbps(h2d_bytes, h2d_secs).max(gbps(d2h_bytes, d2h_secs));
                    let overlap = if solo > 0.0 { r / solo } else { 0.0 };
                    let moved = dup_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    l.set_hash(cuda_hash);
                    l.set_detail(&format!(
                        "phase: 3/3 full-duplex (both engines)\n\
                         duplex: {r:.2} GB/s\n\
                         vs solo: {overlap:.2}x\n\
                         host-RAM: {host_gbps:.1} GB/s\n\
                         moved: {moved:.2} GiB\n\
                         path: CUDA dual-stream (pinned)"
                    ));
                }
            }
        }

        if let Some(l) = &lane {
            l.set_hash(cuda_hash);
            l.set_phase(PHASE_DONE);
        }

        // Integrity: the read-back must equal the pattern we seeded.
        let mut errors = 0u64;
        let mut verify_note = String::new();
        match pin_dn.as_slice() {
            Ok(s) if s == pattern => {}
            Ok(s) => {
                errors += 1;
                let i = s.iter().zip(pattern).position(|(a, b)| a != b).unwrap_or(0);
                verify_note = format!("; VERIFY FAIL: first mismatch at byte {i}");
            }
            Err(_) => {
                errors += 1;
                verify_note = "; VERIFY FAIL: could not read pinned buffer".to_string();
            }
        }

        let h2d = gbps(h2d_bytes, h2d_secs);
        let d2h = gbps(d2h_bytes, d2h_secs);
        let dup = gbps(dup_bytes, dup_secs);
        let total_gib = (h2d_bytes + d2h_bytes + dup_bytes) as f64 / (1024.0 * 1024.0 * 1024.0);
        let full_duplex = dup > h2d.max(d2h) * 1.15;

        let mut detail = format!(
            "{} cuda {}MiB dual-stream, {:.1} GiB moved, H2D ~{h2d:.1} GB/s, D2H ~{d2h:.1} GB/s, \
             full-duplex ~{dup:.1} GB/s ({}), host-RAM ~{host_gbps:.1} GB/s",
            self.device.label(),
            bytes / (1024 * 1024),
            total_gib,
            if full_duplex {
                "both engines overlapping"
            } else {
                "no overlap gain"
            },
        );
        if lost {
            detail.push_str("; DEVICE LOST / transfer error");
            errors += 1;
        }
        if dup_bytes == 0 && !lost {
            detail.push_str("; NO TRANSFERS COMPLETED");
            errors += 1;
        }
        detail.push_str(&verify_note);

        // Keep the context alive until here.
        drop(ctx);
        Some(LoadResult::new(true, 1, 0, errors, detail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_is_deterministic_and_high_entropy() {
        let a = make_pattern(4096, 1);
        assert_eq!(a, make_pattern(4096, 1));
        assert_ne!(a, make_pattern(4096, 2));
        assert!(a.iter().any(|&x| x != a[0]));
    }

    #[test]
    fn link_defaults_are_bounded_and_bidir() {
        let k = LinkKernel::default();
        assert!(k.buf_mb <= 1024);
        assert_eq!(k.dir, LinkDir::Both);
        assert_eq!(k.kind(), Kind::Pcie);
        assert_eq!(k.name(), "pcie");
    }
}
