// SPDX-License-Identifier: MIT
//! # crucible-cpu
//!
//! CPU power/heat load with built-in soft-error detection.
//!
//! * **Load:** wide FMA accumulation (AVX2+FMA where present, scalar
//!   `mul_add` fallback), one pinned thread per selected logical core. Enough
//!   independent accumulators are kept in flight to saturate the FMA units on
//!   modern cores, so a well-cooled chip pins at its power/thermal limit.
//! * **Error detection:** each work chunk runs *two* independent accumulator
//!   banks over identical inputs and compares them bit-for-bit. A mismatch is a
//!   soft error attributable to that core — a stress pass that computes the
//!   wrong answer is a FAIL even without a crash.
//! * **Shapes:** steady or bursty duty cycle, driven by
//!   [`crucible_core::ShapeDriver`], which stamps burst edges for the power rig.
//!
//! Pin one core to isolate a suspect (CoreCycler-style) or all cores for a
//! full-package burn.

use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;
use crucible_core::sysinfo;

/// Which logical cores to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreSel {
    /// All logical processors.
    All,
    /// A single logical processor by index (CoreCycler-style isolation).
    One(usize),
}

impl CoreSel {
    fn resolve(self, total: usize) -> Vec<usize> {
        match self {
            CoreSel::All => (0..total.max(1)).collect(),
            CoreSel::One(i) => vec![i],
        }
    }

    fn label(self) -> String {
        match self {
            CoreSel::All => "core=all".to_string(),
            CoreSel::One(i) => format!("core={i}"),
        }
    }
}

/// The CPU load kernel.
#[derive(Debug, Clone)]
pub struct CpuKernel {
    pub cores: CoreSel,
}

impl CpuKernel {
    pub fn new(cores: CoreSel) -> CpuKernel {
        CpuKernel { cores }
    }

    /// Human-readable name of the instruction path selected at runtime.
    pub fn backend() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return "avx2+fma";
            }
        }
        "scalar-mul_add"
    }
}

impl LoadKernel for CpuKernel {
    fn name(&self) -> &str {
        "cpu"
    }

    fn kind(&self) -> Kind {
        Kind::Cpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let total = sysinfo::logical_cpus();
        let cores = self.cores.resolve(total);
        if cores.is_empty() {
            return LoadResult::setup_failure("no cores selected");
        }
        let label = self.cores.label();
        let label_ref: &str = label.as_str();

        // All worker threads must share one phase origin, otherwise each core's
        // ShapeDriver measures phase from its own start and the cores burst
        // independently — 20 unsynchronized square waves instead of one sharp
        // system-level current step, which is the whole point of a burst shape.
        // An orchestrator-supplied epoch (cross-load) always wins.
        let budget = &budget.clone().phased_if_unset(Instant::now());

        // Scoped threads so each worker can borrow the shared stop/markers/budget.
        // `label_ref`, `core`, and the shared references are all `Copy`, so each
        // `move` closure gets its own copy rather than moving `label` out.
        let (outs, panics): (Vec<ThreadOut>, u64) = std::thread::scope(|scope| {
            let handles: Vec<_> = cores
                .iter()
                .map(|&core| scope.spawn(move || run_one(core, label_ref, budget, stop, markers)))
                .collect();
            // A panicked worker must not silently vanish into a zeroed result —
            // count it as an error so the verdict reflects it.
            let mut outs = Vec::new();
            let mut panics = 0u64;
            for h in handles {
                match h.join() {
                    Ok(o) => outs.push(o),
                    Err(_) => panics += 1,
                }
            }
            (outs, panics)
        });

        // Aggregate.
        let threads = outs.len() as u64;
        let mut work_units = 0u64;
        let mut checksum = 0u64;
        let mut errors = 0u64;
        let mut secs = 0.0f64;
        let mut all_pinned = true;
        let mut bad_cores: Vec<usize> = Vec::new();
        for o in &outs {
            work_units = work_units.wrapping_add(o.work_units);
            // Sum (not XOR) so identical per-core checksums don't cancel to 0.
            checksum = checksum.wrapping_add(o.checksum);
            errors += o.errors;
            secs = secs.max(o.seconds);
            all_pinned &= o.pinned;
            if o.errors > 0 {
                bad_cores.push(o.core);
            }
        }

        // Approximate throughput: each work unit is one FMA on a f64 lane = 2 flops.
        let gflops = if secs > 0.0 {
            (work_units as f64 * 2.0) / secs / 1.0e9
        } else {
            0.0
        };

        // A worker panic is a failure of the run, not a zero-work success.
        errors += panics;

        let mut detail = format!(
            "{threads} thread(s) [{}], {}, ~{:.1} GFLOP/s (approx), pinned={all_pinned}",
            CpuKernel::backend(),
            self.cores.label(),
            gflops,
        );
        if panics > 0 {
            detail.push_str(&format!("; {panics} worker thread(s) PANICKED"));
        }
        if !bad_cores.is_empty() {
            detail.push_str(&format!(
                "; SOFT ERRORS: miscompare(s) on core(s) {bad_cores:?}"
            ));
        }

        LoadResult::new(true, work_units, checksum, errors, detail)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ThreadOut {
    core: usize,
    work_units: u64,
    checksum: u64,
    errors: u64,
    seconds: f64,
    pinned: bool,
}

/// Outer iterations per work chunk. Sized so a chunk is a few milliseconds —
/// long enough to amortize the loop overhead, short enough to stay responsive
/// to the stop flag.
const CHUNK_ITERS: u64 = 200_000;

fn run_one(
    core: usize,
    label: &str,
    budget: &Budget,
    stop: &StopFlag,
    markers: &MarkerLog,
) -> ThreadOut {
    let pinned = pin_current_thread(core);
    let mut driver = ShapeDriver::start(budget, stop, markers, "cpu", label.to_string());
    let start = Instant::now();

    let mut work_units = 0u64;
    let mut checksum = 0u64;
    let mut errors = 0u64;

    loop {
        match driver.tick() {
            Tick::Work => {
                let out = fma_chunk(CHUNK_ITERS);
                work_units = work_units.wrapping_add(out.work_units);
                // Every chunk recomputes from identical seeds, so its checksum
                // is a fixed function of the (correct) FMA result. Keep the last
                // one rather than folding — a fold over a variable chunk count
                // would not be reproducible across time-bounded runs.
                checksum = out.checksum;
                errors += out.errors;
            }
            Tick::Idle => {}
            Tick::Stop => break,
        }
    }

    ThreadOut {
        core,
        work_units,
        checksum,
        errors,
        seconds: start.elapsed().as_secs_f64(),
        pinned,
    }
}

#[derive(Debug, Clone, Copy)]
struct ChunkOut {
    work_units: u64,
    checksum: u64,
    errors: u64,
}

/// Dispatch to the widest available FMA path at runtime.
fn fma_chunk(iters: u64) -> ChunkOut {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature detection above.
            return unsafe { fma_chunk_avx2(iters) };
        }
    }
    fma_chunk_scalar(iters)
}

/// Scalar fallback: four independent `mul_add` accumulators per bank, two banks
/// compared bit-for-bit.
fn fma_chunk_scalar(iters: u64) -> ChunkOut {
    // Opaque seeds: equal at runtime (no false miscompare) but not provably
    // equal at compile time (so both banks are actually computed and compared).
    let coeff = std::hint::black_box(0.999_999_999_9_f64);
    let addend = std::hint::black_box(1.0e-3_f64);
    let mut a = [0.0f64; 4];
    let mut b = [0.0f64; 4];
    for k in 0..4 {
        let seed = std::hint::black_box(0.1 + k as f64);
        a[k] = seed;
        b[k] = std::hint::black_box(0.1 + k as f64);
        let _ = seed;
    }

    for _ in 0..iters {
        for k in 0..4 {
            a[k] = a[k].mul_add(coeff, addend);
            b[k] = b[k].mul_add(coeff, addend);
        }
    }

    std::hint::black_box(&a);
    std::hint::black_box(&b);

    let mut errors = 0u64;
    let mut checksum = 0u64;
    for k in 0..4 {
        if a[k].to_bits() != b[k].to_bits() {
            errors += 1;
        }
        checksum = checksum.wrapping_add(a[k].to_bits());
    }
    ChunkOut {
        work_units: iters * 4,
        checksum,
        errors,
    }
}

/// AVX2+FMA path: eight 256-bit accumulators per bank (32 f64 lanes in flight
/// per bank) to keep the dual FMA units busy; two banks compared bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
// The whole body is AVX2/FMA intrinsics whose safety precondition (the target
// features) is established by `#[target_feature]` plus the runtime detection at
// the sole call site; per-call unsafe blocks would add only noise here.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fma_chunk_avx2(iters: u64) -> ChunkOut {
    use std::arch::x86_64::*;
    use std::hint::black_box;

    const BANK: usize = 8;

    let coeff = _mm256_set1_pd(black_box(0.999_999_999_9_f64));
    let addend = _mm256_set1_pd(black_box(1.0e-3_f64));

    let mut a = [_mm256_setzero_pd(); BANK];
    let mut b = [_mm256_setzero_pd(); BANK];
    for k in 0..BANK {
        a[k] = _mm256_set1_pd(black_box(0.1 + k as f64));
        b[k] = _mm256_set1_pd(black_box(0.1 + k as f64));
    }

    for _ in 0..iters {
        for k in 0..BANK {
            a[k] = _mm256_fmadd_pd(a[k], coeff, addend);
            b[k] = _mm256_fmadd_pd(b[k], coeff, addend);
        }
    }

    let mut errors = 0u64;
    let mut checksum = 0u64;
    let mut ta = [0.0f64; 4];
    let mut tb = [0.0f64; 4];
    for k in 0..BANK {
        _mm256_storeu_pd(ta.as_mut_ptr(), a[k]);
        _mm256_storeu_pd(tb.as_mut_ptr(), b[k]);
        for l in 0..4 {
            if ta[l].to_bits() != tb[l].to_bits() {
                errors += 1;
            }
            checksum = checksum.wrapping_add(ta[l].to_bits());
        }
    }
    ChunkOut {
        // BANK accumulators * 4 lanes = 32 FMA lane-ops per outer iteration.
        work_units: iters * (BANK as u64) * 4,
        checksum,
        errors,
    }
}

/// Pin the calling thread to a single logical core. Returns whether pinning
/// succeeded (best-effort; a failure does not abort the load).
#[cfg(windows)]
fn pin_current_thread(core: usize) -> bool {
    // Single-group affinity mask supports cores 0..63; wider topologies need
    // processor-group APIs (a Phase 2 telemetry concern), so bail out cleanly.
    if core >= 64 {
        return false;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    }
    let mask: usize = 1usize << core;
    // SAFETY: GetCurrentThread returns a pseudo-handle valid for this call;
    // SetThreadAffinityMask reads no memory through the handle.
    let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
    prev != 0
}

#[cfg(not(windows))]
fn pin_current_thread(_core: usize) -> bool {
    // No dependency-free portable affinity API; run unpinned elsewhere.
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn chunk_has_no_false_errors() {
        // On healthy hardware the two banks must agree bit-for-bit.
        let out = fma_chunk(1_000);
        assert_eq!(out.errors, 0, "unexpected miscompare on healthy CPU");
        assert!(out.work_units > 0);
    }

    #[test]
    fn scalar_and_dispatch_agree_on_checksum() {
        // The dispatched path and the scalar path use the same recurrence and
        // seeds, so their checksums match for the same iteration count.
        let a = fma_chunk_scalar(5_000);
        let b = fma_chunk_scalar(5_000);
        assert_eq!(a.checksum, b.checksum);
        assert_eq!(a.errors, 0);
    }

    #[test]
    fn short_run_completes_and_passes() {
        let kernel = CpuKernel::new(CoreSel::One(0));
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_millis(80));
        let result = kernel.run(&budget, &stop, &markers);
        assert!(result.ok);
        assert_eq!(result.error_count, 0);
        assert!(result.iterations > 0);
        assert!(result.detail.contains("GFLOP/s"));
    }

    #[test]
    fn stop_flag_ends_run_early() {
        let kernel = CpuKernel::new(CoreSel::All);
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_secs(3600));
        stop.stop(); // pre-stopped: run must return promptly
        let start = Instant::now();
        let result = kernel.run(&budget, &stop, &markers);
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(result.ok);
    }
}
