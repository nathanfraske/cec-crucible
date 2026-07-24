// SPDX-License-Identifier: MIT
//! The load-kernel abstraction shared by every domain (CPU/RAM/storage/GPU).
//!
//! A kernel generates load; the *shape* of that load — steady vs bursty duty
//! cycles — is the differentiator (steady 100% misses real transient bugs).
//! [`ShapeDriver`] centralizes the shape logic and burst-edge marker stamping
//! so each kernel only writes its inner work chunk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::markers::{Event, MarkerLog};

/// Which hardware domain a kernel exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Cpu,
    Mem,
    Storage,
    Gpu,
    /// Host↔device transfer / PCIe-link load.
    Pcie,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Cpu => "cpu",
            Kind::Mem => "mem",
            Kind::Storage => "storage",
            Kind::Gpu => "gpu",
            Kind::Pcie => "pcie",
        }
    }
}

/// A shared stop switch. Cloning yields another handle to the *same* flag, so
/// the orchestrator can end N concurrent kernels together — one flag, one
/// timer, one timeline. This is the mechanism behind cross-load.
#[derive(Debug, Clone, Default)]
pub struct StopFlag(Arc<AtomicBool>);

impl StopFlag {
    pub fn new() -> StopFlag {
        StopFlag(Arc::new(AtomicBool::new(false)))
    }

    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn stopped(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// How the load moves over time.
///
/// `Steady` and `Burst` are implemented in Phase 1. `Ramp`/`Sweep` (for the
/// closed-loop wattage servo) are Phase 3 and intentionally absent here rather
/// than stubbed — add them with the servo that drives them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Continuous full load.
    Steady,
    /// Duty cycle: `on` under load, `off` idle, repeating. Hammers VRM
    /// transient response.
    Burst { on: Duration, off: Duration },
}

impl Shape {
    pub fn mode_str(&self) -> &'static str {
        match self {
            Shape::Steady => "steady",
            Shape::Burst { .. } => "burst",
        }
    }
}

/// A kernel's run budget: how long, in what shape, toward what optional power
/// target. `target_watts` is carried for the Phase 3 servo and ignored today.
#[derive(Debug, Clone)]
pub struct Budget {
    pub duration: Duration,
    pub shape: Shape,
    pub target_watts: Option<f64>,
    /// Shared phase origin for burst shapes.
    ///
    /// Without this, every [`ShapeDriver`] derives its phase from whenever *it*
    /// happened to start, so load edges never line up: the CPU kernel's worker
    /// threads smear across the period instead of stepping together, and a
    /// cross-load offset is swamped by per-kernel setup time (GPU client init +
    /// shader compile is ~100 ms, dwarfing a 20 ms intended offset). Giving
    /// every kernel one common origin makes in-phase / anti-phase exact and
    /// independent of setup jitter. `None` = each driver uses its own start.
    pub phase_epoch: Option<Instant>,
    /// Offset from `phase_epoch` for this kernel — how the anti-phase and beat
    /// scenarios shift one domain against another.
    pub phase_offset: Duration,
}

impl Budget {
    pub fn steady(duration: Duration) -> Budget {
        Budget {
            duration,
            shape: Shape::Steady,
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        }
    }

    pub fn burst(duration: Duration, on: Duration, off: Duration) -> Budget {
        Budget {
            duration,
            shape: Shape::Burst { on, off },
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        }
    }

    /// Pin this budget's burst phase to a shared origin plus an offset.
    pub fn phased(mut self, epoch: Instant, offset: Duration) -> Budget {
        self.phase_epoch = Some(epoch);
        self.phase_offset = offset;
        self
    }

    /// Adopt `epoch` only if no phase origin was set, preserving any offset an
    /// orchestrator already chose.
    pub fn phased_if_unset(mut self, epoch: Instant) -> Budget {
        if self.phase_epoch.is_none() {
            self.phase_epoch = Some(epoch);
        }
        self
    }
}

/// The outcome of one kernel run.
#[derive(Debug, Clone)]
pub struct LoadResult {
    /// `false` if the kernel could not run (setup failure) — distinct from
    /// running successfully but detecting hardware errors.
    pub ok: bool,
    /// Work units completed (kernel-defined: FMA blocks, buffer passes, etc.).
    pub iterations: u64,
    /// A checksum of the work, for cross-run / cross-core comparison.
    pub checksum: u64,
    /// Count of detected miscompares / soft errors. `> 0` means FAIL even if
    /// the run did not crash.
    pub error_count: u64,
    /// Human-readable summary (first-fail address, throughput, etc.).
    pub detail: String,
}

impl LoadResult {
    pub fn new(ok: bool, iterations: u64, checksum: u64, error_count: u64, detail: String) -> Self {
        LoadResult {
            ok,
            iterations,
            checksum,
            error_count,
            detail,
        }
    }

    /// A clean, error-free result.
    pub fn clean(iterations: u64, checksum: u64, detail: impl Into<String>) -> Self {
        LoadResult::new(true, iterations, checksum, 0, detail.into())
    }

    /// A setup failure that never produced load.
    pub fn setup_failure(detail: impl Into<String>) -> Self {
        LoadResult::new(false, 0, 0, 0, detail.into())
    }

    /// Did this result pass? (Ran, and detected no errors.)
    pub fn passed(&self) -> bool {
        self.ok && self.error_count == 0
    }
}

/// The trait every domain kernel implements. `Send + Sync` so the orchestrator
/// can run several concurrently under one [`StopFlag`].
pub trait LoadKernel: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> Kind;
    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult;
}

/// What a kernel should do this instant, per [`ShapeDriver::tick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tick {
    /// Do one chunk of load work.
    Work,
    /// Off-phase of a burst — the driver already slept a slice; do nothing.
    Idle,
    /// Deadline reached or stop requested — exit the loop.
    Stop,
}

/// Drives a kernel's inner loop according to its [`Shape`], handling the burst
/// duty cycle and stamping `burst_on` / `burst_off` edges into the marker log.
///
/// Typical use:
/// ```ignore
/// let mut driver = ShapeDriver::start(budget, stop, markers, "cpu");
/// loop {
///     match driver.tick() {
///         Tick::Work => { /* one chunk of load */ }
///         Tick::Idle => {}
///         Tick::Stop => break,
///     }
/// }
/// driver.finish();
/// ```
pub struct ShapeDriver<'a> {
    shape: Shape,
    mode: &'static str,
    start: Instant,
    /// Origin the burst phase is measured from — shared across kernels so their
    /// edges line up deterministically (see [`Budget::phase_epoch`]).
    phase_origin: Instant,
    phase_offset: Duration,
    deadline: Instant,
    stop: &'a StopFlag,
    markers: &'a MarkerLog,
    kernel: &'a str,
    /// Current burst phase (only meaningful for `Burst`).
    on_phase: bool,
    detail: String,
}

impl<'a> ShapeDriver<'a> {
    /// Begin driving. Records the start instant and deadline; does not stamp a
    /// stage marker (the orchestrator brackets stages).
    pub fn start(
        budget: &Budget,
        stop: &'a StopFlag,
        markers: &'a MarkerLog,
        kernel: &'a str,
        detail: impl Into<String>,
    ) -> ShapeDriver<'a> {
        let now = Instant::now();
        ShapeDriver {
            shape: budget.shape,
            mode: budget.shape.mode_str(),
            start: now,
            // Phase comes from the shared origin when one is supplied, so setup
            // time before this driver started does not shift the load edges.
            phase_origin: budget.phase_epoch.unwrap_or(now),
            phase_offset: budget.phase_offset,
            deadline: now + budget.duration,
            stop,
            markers,
            kernel,
            on_phase: false,
            detail: detail.into(),
        }
    }

    /// Cap on how long an off-phase nap runs, so the loop stays responsive to
    /// the stop flag even with long burst `off` periods.
    const MAX_NAP: Duration = Duration::from_millis(2);

    pub fn tick(&mut self) -> Tick {
        if self.stop.stopped() || Instant::now() >= self.deadline {
            self.close_burst();
            return Tick::Stop;
        }
        match self.shape {
            Shape::Steady => Tick::Work,
            Shape::Burst { on, off } => {
                let period = (on + off).as_nanos().max(1);
                let pos = (self.phase_origin.elapsed() + self.phase_offset).as_nanos() % period;
                let on_ns = on.as_nanos();
                if pos < on_ns {
                    if !self.on_phase {
                        self.markers
                            .stamp(Event::BurstOn, self.kernel, self.mode, &self.detail);
                        self.on_phase = true;
                    }
                    Tick::Work
                } else {
                    if self.on_phase {
                        self.markers
                            .stamp(Event::BurstOff, self.kernel, self.mode, &self.detail);
                        self.on_phase = false;
                    }
                    let remaining = (period - pos) as u64;
                    let nap = Duration::from_nanos(remaining).min(Self::MAX_NAP);
                    std::thread::sleep(nap);
                    Tick::Idle
                }
            }
        }
    }

    /// Stamp a trailing `burst_off` if we ended mid-burst. Idempotent.
    fn close_burst(&mut self) {
        if self.on_phase {
            self.markers
                .stamp(Event::BurstOff, self.kernel, self.mode, &self.detail);
            self.on_phase = false;
        }
    }

    /// Seconds elapsed since start (wall, monotonic).
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;

    #[test]
    fn stop_flag_shares_state() {
        let a = StopFlag::new();
        let b = a.clone();
        assert!(!b.stopped());
        a.stop();
        assert!(b.stopped());
    }

    #[test]
    fn steady_driver_runs_until_deadline() {
        let stop = StopFlag::new();
        let markers = MarkerLog::new(Clock::new());
        let budget = Budget::steady(Duration::from_millis(30));
        let mut driver = ShapeDriver::start(&budget, &stop, &markers, "test", "");
        let mut work = 0u64;
        loop {
            match driver.tick() {
                Tick::Work => work += 1,
                Tick::Idle => {}
                Tick::Stop => break,
            }
        }
        assert!(work > 0);
        // Steady load emits no burst edges.
        assert_eq!(markers.len(), 0);
    }

    #[test]
    fn stop_flag_ends_driver_promptly() {
        let stop = StopFlag::new();
        let markers = MarkerLog::new(Clock::new());
        let budget = Budget::steady(Duration::from_secs(3600));
        let mut driver = ShapeDriver::start(&budget, &stop, &markers, "test", "");
        stop.stop();
        assert_eq!(driver.tick(), Tick::Stop);
    }

    #[test]
    fn burst_driver_emits_edges() {
        let stop = StopFlag::new();
        let markers = MarkerLog::new(Clock::new());
        let budget = Budget::burst(
            Duration::from_millis(80),
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        let mut driver = ShapeDriver::start(&budget, &stop, &markers, "cpu", "core=all");
        let mut saw_work = false;
        let mut saw_idle = false;
        loop {
            match driver.tick() {
                Tick::Work => saw_work = true,
                Tick::Idle => saw_idle = true,
                Tick::Stop => break,
            }
        }
        assert!(saw_work, "burst should have on-phase work");
        assert!(saw_idle, "burst should have off-phase idle");
        // At ~20ms period over ~80ms we expect several on/off edges.
        assert!(
            markers.len() >= 2,
            "expected burst edges, got {}",
            markers.len()
        );
    }
}
