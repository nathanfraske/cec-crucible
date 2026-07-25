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

use crate::markers::{Event, LiveLane, MarkerLog, PHASE_DONE, PHASE_IDLE, PHASE_WORK};

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
/// `Ramp`/`Sweep` (for the closed-loop wattage servo) are Phase 3 and
/// intentionally absent here rather than stubbed — add them with the servo that
/// drives them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Continuous full load.
    Steady,
    /// Duty cycle: `on` under load, `off` idle, repeating. Hammers VRM
    /// transient response with a *fixed* period. The off-phase parks only
    /// briefly (2 ms slices) so the core stays responsive but shallow — it does
    /// *not* reach deep C-states. For that, use `Pulse`.
    Burst { on: Duration, off: Duration },
    /// A burst with a *deep* idle: a short `work` pulse followed by a long,
    /// genuinely-parked `idle` (slept in ~50 ms slices, not 2 ms), so the core
    /// demotes into C6 between pulses. This is the only shape that exercises
    /// C-state entry/exit, the idle→boost voltage step, and low-load/idle
    /// undervolt instability — the "crashes at idle, passes Prime95" class.
    /// Pair with single-core pinning so one core cycles C0↔C6 while the rest of
    /// the package idles.
    Pulse { work: Duration, idle: Duration },
    /// Randomized burst — the "never-settle" transient. Each on (spike) and off
    /// (floor) segment length is drawn from a seeded PRNG within
    /// `[on_min, on_max]` / `[off_min, off_max]`, keyed on the segment *index* so
    /// the schedule is a pure function of `(seed, phase_origin)` — every worker
    /// thread and every kernel sharing the seed agrees on it regardless of tick
    /// rate, exactly as `Burst` stays aligned via `elapsed % period`. Because the
    /// period never repeats and each kernel is seeded independently, the domains
    /// swap at uncorrelated offsets and no VRM/PSU control loop ever settles.
    ///
    /// `floor_pct` is the "barely loaded, then slam" trickle: during an off
    /// segment the driver runs a light micro-duty of `floor_pct`% instead of
    /// going fully idle, so the rail stays out of its deep-idle/skip regulation
    /// mode and every spike lands on an already-perturbed baseline. `0` = a
    /// burst-style dead idle. (It is approximate — the work chunk quantizes it
    /// upward.) The `seed` makes the *commanded* pattern reproducible (subject to
    /// OS-scheduler jitter on the exact wall-clock edges); log it to re-run.
    Jitter {
        on_min: Duration,
        on_max: Duration,
        off_min: Duration,
        off_max: Duration,
        floor_pct: u8,
        seed: u64,
    },
}

impl Shape {
    pub fn mode_str(&self) -> &'static str {
        match self {
            Shape::Steady => "steady",
            Shape::Burst { .. } => "burst",
            Shape::Pulse { .. } => "pulse",
            Shape::Jitter { .. } => "jitter",
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

    /// A deep-idle pulse budget (for the C-state / idle-boost test).
    pub fn pulse(duration: Duration, work: Duration, idle: Duration) -> Budget {
        Budget {
            duration,
            shape: Shape::Pulse { work, idle },
            target_watts: None,
            phase_epoch: None,
            phase_offset: Duration::ZERO,
        }
    }

    /// A randomized-burst budget. `seed` seeds this kernel's PRNG — give each
    /// concurrent kernel a distinct seed so their streams decorrelate.
    #[allow(clippy::too_many_arguments)]
    pub fn jitter(
        duration: Duration,
        on_min: Duration,
        on_max: Duration,
        off_min: Duration,
        off_max: Duration,
        floor_pct: u8,
        seed: u64,
    ) -> Budget {
        Budget {
            duration,
            shape: Shape::Jitter {
                on_min,
                on_max,
                off_min,
                off_max,
                floor_pct,
                seed,
            },
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
    /// Current on/off phase (`Burst`, `Pulse`, `Jitter`).
    on_phase: bool,
    /// `Jitter` segment cursor: the active segment's index and its phase-relative
    /// `[start, end)`. Boundaries are a deterministic prefix-sum of per-index
    /// draws, so every thread/kernel sharing the seed lands on the same segment
    /// for a given elapsed time — no shared mutable PRNG state to desync.
    seg_idx: u64,
    seg_start: Duration,
    seg_end: Duration,
    detail: String,
    /// Live-UI lane for this kernel; `None` unless a UI enabled tracking.
    lane: Option<Arc<LiveLane>>,
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
            // `seg_idx = MAX` so the first advance wraps to segment 0 (a spike);
            // `seg_end = 0` makes the first tick cross into it.
            seg_idx: u64::MAX,
            seg_start: Duration::ZERO,
            seg_end: Duration::ZERO,
            detail: detail.into(),
            lane: markers.register_lane(kernel),
        }
    }

    /// Point this driver's live lane at a different label (the CPU kernel uses
    /// this to report per-core lanes while keeping the marker kernel name "cpu").
    pub fn relabel_lane(&mut self, label: &str) {
        self.lane = self.markers.register_lane(label);
    }

    /// Is a live UI attached to this lane? Kernels guard the (allocating) status
    /// formatting with this so the headless/no-UI path stays zero-overhead.
    #[inline]
    pub fn live(&self) -> bool {
        self.lane.is_some()
    }

    /// Publish the latest verification checksum to the live lane (no-op if no UI).
    #[inline]
    pub fn set_hash(&self, h: u64) {
        if let Some(l) = &self.lane {
            l.set_hash(h);
        }
    }

    /// Publish a live status detail (multi-line "field: value") to the lane so the
    /// UI can show what this kernel is doing. No-op if no UI; call it throttled,
    /// guarded by [`live`](Self::live), never on the hot inner loop.
    #[inline]
    pub fn set_status(&self, s: &str) {
        if let Some(l) = &self.lane {
            l.set_detail(s);
        }
    }

    /// Cap on how long an off-phase nap runs, so the loop stays responsive to
    /// the stop flag even with long burst `off` periods.
    const MAX_NAP: Duration = Duration::from_millis(2);

    /// Off-phase nap cap for `Pulse` — long enough for the core to actually
    /// demote into C6 (a 2 ms burst nap never does), short enough to keep stop
    /// latency and package-idle disruption bounded.
    const PULSE_IDLE_CHUNK: Duration = Duration::from_millis(50);

    /// Floor micro-duty slot for `Jitter` — the trickle repeats on this period.
    const FLOOR_SLOT: Duration = Duration::from_millis(10);

    pub fn tick(&mut self) -> Tick {
        if self.stop.stopped() || Instant::now() >= self.deadline {
            self.close_burst();
            if let Some(l) = &self.lane {
                l.set_phase(PHASE_DONE);
            }
            return Tick::Stop;
        }
        let t = match self.shape {
            Shape::Steady => Tick::Work,
            Shape::Burst { on, off } => self.duty_tick(on, off, Self::MAX_NAP),
            // Pulse is a burst with a deep idle so the core reaches C6.
            Shape::Pulse { work, idle } => self.duty_tick(work, idle, Self::PULSE_IDLE_CHUNK),
            Shape::Jitter {
                on_min,
                on_max,
                off_min,
                off_max,
                floor_pct,
                seed,
            } => self.jitter_tick(on_min, on_max, off_min, off_max, floor_pct, seed),
        };
        // Feed the live UI lane (no-op when no UI enabled tracking).
        if let Some(l) = &self.lane {
            match t {
                Tick::Work => {
                    l.set_phase(PHASE_WORK);
                    l.bump_work();
                }
                Tick::Idle => l.set_phase(PHASE_IDLE),
                Tick::Stop => {}
            }
        }
        t
    }

    /// Fixed-period duty cycle (Burst and Pulse): Work during the on-window,
    /// Idle during the off-window. `nap_cap` is the only difference between a
    /// shallow burst (2 ms, stays responsive) and a deep pulse (50 ms, lets the
    /// core demote into C6). Position is taken from the shared `phase_origin`, so
    /// all workers step together regardless of when each started.
    fn duty_tick(&mut self, on: Duration, off: Duration, nap_cap: Duration) -> Tick {
        let period = (on + off).as_nanos().max(1);
        let pos = (self.phase_origin.elapsed() + self.phase_offset).as_nanos() % period;
        if pos < on.as_nanos() {
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
            let nap = Duration::from_nanos(remaining).min(nap_cap);
            std::thread::sleep(nap);
            Tick::Idle
        }
    }

    /// Randomized duty cycle. Segment boundaries are a deterministic prefix-sum
    /// of per-index draws measured from `phase_origin`, so this is a pure
    /// function of `(seed, elapsed)` — identical across every thread and kernel
    /// that shares the seed, independent of start time or tick rate. That is what
    /// keeps a multi-threaded kernel producing *one* system-level step instead of
    /// N desynchronized ones.
    #[allow(clippy::too_many_arguments)]
    fn jitter_tick(
        &mut self,
        on_min: Duration,
        on_max: Duration,
        off_min: Duration,
        off_max: Duration,
        floor_pct: u8,
        seed: u64,
    ) -> Tick {
        let elapsed = self.phase_origin.elapsed() + self.phase_offset;
        // Advance to the segment containing `elapsed`. `while` (not `if`) lets a
        // late starter catch up to the right segment in a single tick.
        while elapsed >= self.seg_end {
            self.seg_idx = self.seg_idx.wrapping_add(1); // MAX -> 0 on first tick
            self.seg_start = self.seg_end;
            let spike = self.seg_idx % 2 == 0; // even = spike, odd = floor
            let (min, max) = if spike {
                (on_min, on_max)
            } else {
                (off_min, off_max)
            };
            self.seg_end = self.seg_start + jitter_interval(seed, self.seg_idx, min, max);
            let event = if spike {
                Event::BurstOn
            } else {
                Event::BurstOff
            };
            self.markers
                .stamp(event, self.kernel, self.mode, &self.detail);
            self.on_phase = spike;
        }
        if self.on_phase {
            return Tick::Work; // spike ("slam")
        }
        // Floor segment: a light `floor_pct` micro-duty keeps the rail out of its
        // deep-idle regulation mode so the next spike lands on a perturbed
        // baseline. `floor_pct == 0` degenerates to a dead idle.
        if floor_pct > 0 {
            let into = elapsed.saturating_sub(self.seg_start).as_nanos();
            let slot = Self::FLOOR_SLOT.as_nanos();
            if into % slot < slot * floor_pct.min(100) as u128 / 100 {
                return Tick::Work; // the trickle
            }
        }
        let nap = self.seg_end.saturating_sub(elapsed).min(Self::MAX_NAP);
        std::thread::sleep(nap);
        Tick::Idle
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

/// One jitter segment's length, a pure function of `(seed, index)` — no mutable
/// stream, so every thread agrees on the schedule for a given index.
fn jitter_interval(seed: u64, index: u64, min: Duration, max: Duration) -> Duration {
    let lo = min.as_nanos() as u64;
    let hi = (max.as_nanos() as u64).max(lo);
    if hi == lo {
        return Duration::from_nanos(lo);
    }
    Duration::from_nanos(lo + crate::rng::hash2(seed, index) % (hi - lo))
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
    fn jitter_interval_is_pure_and_bounded() {
        // A pure function of (seed, index): same inputs -> same length, always.
        // This is what makes the whole jitter schedule thread-independent.
        let (lo, hi) = (Duration::from_millis(5), Duration::from_millis(50));
        let a = jitter_interval(0xABCD, 7, lo, hi);
        assert_eq!(a, jitter_interval(0xABCD, 7, lo, hi));
        assert!(a >= lo && a < hi, "must stay within [min, max)");
        assert_ne!(
            jitter_interval(0xABCD, 7, lo, hi),
            jitter_interval(0xABCD, 8, lo, hi),
            "adjacent indices decorrelate"
        );
        // Degenerate range collapses to the bound (no modulo-by-zero).
        assert_eq!(jitter_interval(1, 1, lo, lo), lo);
    }

    #[test]
    fn jitter_driver_emits_edges_and_naps() {
        let stop = StopFlag::new();
        let markers = MarkerLog::new(Clock::new());
        let budget = Budget::jitter(
            Duration::from_millis(150),
            Duration::from_millis(4),
            Duration::from_millis(12),
            Duration::from_millis(4),
            Duration::from_millis(12),
            12,
            0xABCD,
        );
        let mut driver = ShapeDriver::start(&budget, &stop, &markers, "cpu", "jitter");
        let mut saw_work = false;
        let mut saw_idle = false;
        loop {
            match driver.tick() {
                Tick::Work => saw_work = true,
                Tick::Idle => saw_idle = true,
                Tick::Stop => break,
            }
        }
        assert!(saw_work, "jitter should have on-phase work");
        assert!(saw_idle, "jitter should have off-phase idle");
        assert!(
            markers.len() >= 2,
            "expected several jitter edges, got {}",
            markers.len()
        );
    }

    #[test]
    fn pulse_driver_pulses_and_deep_idles() {
        let stop = StopFlag::new();
        let markers = MarkerLog::new(Clock::new());
        // Short work pulse, long deep idle.
        let budget = Budget::pulse(
            Duration::from_millis(120),
            Duration::from_millis(3),
            Duration::from_millis(40),
        );
        let mut driver = ShapeDriver::start(&budget, &stop, &markers, "cpu", "pulse");
        let mut saw_work = false;
        let mut saw_idle = false;
        loop {
            match driver.tick() {
                Tick::Work => saw_work = true,
                Tick::Idle => saw_idle = true,
                Tick::Stop => break,
            }
        }
        assert!(saw_work, "pulse should have a work phase");
        assert!(saw_idle, "pulse should have a deep-idle phase");
        assert!(markers.len() >= 2, "pulse should emit on/off edges");
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
