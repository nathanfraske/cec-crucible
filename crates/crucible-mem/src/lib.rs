// SPDX-License-Identifier: MIT
//! # crucible-mem
//!
//! RAM instability testing — the fast-catch, XMP/EXPO-margin class of check
//! (TestMem5 / Karhu in spirit), not a boot-time full-coverage sweep.
//!
//! A large buffer (a configurable fraction of free RAM) is split into disjoint
//! per-thread chunks and run through a pattern battery:
//!
//! * **own-address** — every cell written with its own address, then verified
//!   (catches address-decode faults);
//! * **moving inversions** over `0x00`, `0xFF`, `0xAA`, `0x55` fills — write a
//!   pattern, verify + write its complement ascending, verify + rewrite
//!   descending (catches stuck/coupled bits);
//! * **checkerboard** and its inverse (adjacent-cell complement stress);
//! * **walking ones / walking zeros** — an isolated set/clear bit marched
//!   through every position (isolated-bit drive strength, data-line coupling);
//! * **March C- (10N)** — the classic ordered ascending/descending
//!   read-modify-write march, which *isolates* transition faults from coupling
//!   faults in a way the parallel-friendly patterns above cannot;
//! * **modulo-20** — one cell in twenty holds `P` amid `!P` neighbours (isolated
//!   coupling + cache-masking defeat);
//! * **seeded pseudo-random** — fill from a SplitMix64 stream, then re-seed and
//!   verify (catches data-dependent instability).
//!
//! On the first miscompare the failing virtual address, word index, expected
//! and observed values, and pattern are captured; the run is a FAIL.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, StopFlag};
use crucible_core::markers::{Event, LiveLane, MarkerLog, PHASE_DONE, PHASE_WORK};
use crucible_core::sysinfo;

/// Fraction of *available* physical memory to test when no explicit size given.
pub const DEFAULT_FRACTION: f64 = 0.5;
/// Sentinel fraction meaning "as much as this machine can hold resident".
///
/// Distinct from any ordinary fraction because it is not a fraction at all: the
/// budget comes from [`sysinfo::safe_test_budget_bytes`], which subtracts a
/// reserve that scales with the machine. A flat 90%-of-available is what pushes
/// a box into paging, and a memory test that pages is measuring the disk.
pub const FRACTION_MAX: f64 = -1.0;
/// Fallback buffer size when available memory can't be queried.
const FALLBACK_BYTES: u64 = 256 * 1024 * 1024;
/// Stop responsiveness: check the stop flag / deadline every this many words.
const CHECK_STRIDE: usize = 1 << 18;

/// How to size the test buffer.
#[derive(Debug, Clone, Copy)]
pub enum MemSize {
    /// Explicit byte count.
    Bytes(u64),
    /// Fraction (0.0–1.0) of available physical memory.
    Fraction(f64),
}

impl Default for MemSize {
    /// Test [`DEFAULT_FRACTION`] of available memory.
    fn default() -> Self {
        MemSize::Fraction(DEFAULT_FRACTION)
    }
}

/// The RAM load kernel.
#[derive(Debug, Clone)]
pub struct MemKernel {
    pub size: MemSize,
    /// Worker threads; `None` = one per logical CPU.
    pub threads: Option<usize>,
}

impl MemKernel {
    pub fn new(size: MemSize) -> MemKernel {
        MemKernel {
            size,
            threads: None,
        }
    }

    /// Resolve the requested size to a word count, honoring the safety margin.
    fn resolve_words(&self) -> u64 {
        let avail = sysinfo::memory().map(|m| m.avail_bytes);
        let bytes = match self.size {
            MemSize::Bytes(b) => b,
            // `max`: everything the machine can hold resident, and not one page
            // more. Sized from available memory minus a reserve that scales with
            // total RAM, because "available" is not "spare" — Windows is holding
            // a file cache, the GPU driver has pinned pages, and the operator's
            // session needs a working set.
            MemSize::Fraction(f) if f == FRACTION_MAX => {
                sysinfo::safe_test_budget_bytes().unwrap_or(FALLBACK_BYTES)
            }
            MemSize::Fraction(f) => {
                let f = f.clamp(0.01, 0.95);
                match avail {
                    Some(a) => (a as f64 * f) as u64,
                    None => FALLBACK_BYTES,
                }
            }
        };
        // Whatever was asked for, never commit past what the machine can hold
        // resident. This applies to an explicit `--mb 60000` too: the operator
        // asking for more than exists should get the largest honest test, not a
        // frozen desktop.
        let capped = match sysinfo::safe_test_budget_bytes() {
            Some(budget) => bytes.min(budget),
            None => bytes,
        };
        capped.max(1) / 8 // bytes -> u64 words
    }
}

impl LoadKernel for MemKernel {
    fn name(&self) -> &str {
        "mem"
    }

    fn kind(&self) -> Kind {
        Kind::Mem
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let words = self.resolve_words();
        if words == 0 {
            return LoadResult::setup_failure("resolved buffer size is zero");
        }

        let threads = self.threads.unwrap_or_else(sysinfo::logical_cpus).max(1);

        // Allocate and touch the buffer. `try_reserve` avoids an abort if the
        // requested size is unsatisfiable; fall back to a smaller buffer once.
        let mut buf: Vec<u64> = Vec::new();
        if buf.try_reserve_exact(words as usize).is_err() {
            let smaller = (words / 4).max(1) as usize;
            if buf.try_reserve_exact(smaller).is_err() {
                return LoadResult::setup_failure("could not allocate test buffer");
            }
            buf.resize(smaller, 0);
        } else {
            buf.resize(words as usize, 0);
        }

        let total_words = buf.len();
        let mib = (total_words as f64 * 8.0) / (1024.0 * 1024.0);
        markers.stamp(
            Event::Mark,
            "mem",
            "steady",
            &format!("buffer={:.0}MiB words={}", mib, total_words),
        );

        let deadline = Instant::now() + budget.duration;
        let start = Instant::now();

        // One live lane for the UI, driven by thread 0 (a representative chunk) so
        // the panel does not thrash from all threads writing it. `None` when no UI.
        let lane = markers.register_lane("mem");

        // Disjoint mutable chunks, one per thread — no aliasing, no locks.
        let chunk_len = total_words.div_ceil(threads).max(1);
        let (outs, panics): (Vec<ThreadMemOut>, u64) = std::thread::scope(|scope| {
            let handles: Vec<_> = buf
                .chunks_mut(chunk_len)
                .enumerate()
                .map(|(idx, chunk)| {
                    let base = idx * chunk_len;
                    let tlane = if idx == 0 { lane.clone() } else { None };
                    scope.spawn(move || run_battery(base, chunk, stop, deadline, tlane))
                })
                .collect();
            // A panicked worker must not silently vanish into a zeroed result.
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

        let seconds = start.elapsed().as_secs_f64();

        // Aggregate: sum errors, min cycles (slowest thread), earliest fail.
        let mut errors = 0u64;
        let mut cycles = u64::MAX;
        let mut checksum = 0u64;
        let mut total_bytes = 0u64;
        let mut first: Option<Fail> = None;
        for o in &outs {
            errors += o.errors;
            cycles = cycles.min(o.cycles);
            checksum = checksum.wrapping_add(o.checksum);
            total_bytes += o.bytes;
            if let Some(f) = o.first {
                first = Some(match first {
                    Some(cur) if cur.vaddr <= f.vaddr => cur,
                    _ => f,
                });
            }
        }
        if cycles == u64::MAX {
            cycles = 0;
        }

        // Throughput from bytes actually touched (counts partial cycles too).
        let gbps = if seconds > 0.0 {
            total_bytes as f64 / seconds / 1.0e9
        } else {
            0.0
        };

        // A worker panic is a failure of the run, not a zero-work success.
        errors += panics;

        let mut detail = format!(
            "{:.0} MiB buffer, {} thread(s), {} battery cycle(s), ~{:.1} GB/s (approx)",
            mib, threads, cycles, gbps,
        );
        if panics > 0 {
            detail.push_str(&format!("; {panics} worker thread(s) PANICKED"));
        }
        if let Some(f) = first {
            detail.push_str(&format!(
                "; FIRST FAIL @ vaddr 0x{:x} (word {}) [{}]: expected 0x{:016x} got 0x{:016x}",
                f.vaddr, f.index, f.pattern, f.expected, f.got
            ));
        }

        LoadResult::new(true, cycles, checksum, errors, detail)
    }
}

/// Details of the first observed miscompare.
#[derive(Debug, Clone, Copy)]
struct Fail {
    vaddr: usize,
    index: usize,
    expected: u64,
    got: u64,
    pattern: &'static str,
}

#[derive(Debug, Default, Clone, Copy)]
struct ThreadMemOut {
    cycles: u64,
    errors: u64,
    checksum: u64,
    /// Bytes actually read+written, including a partial (incomplete) cycle, so
    /// throughput is meaningful even when no full battery cycle finished.
    bytes: u64,
    first: Option<Fail>,
}

/// SplitMix64 step — a fast, well-distributed deterministic stream for the
/// random-pattern pass. Not cryptographic; that is not the goal.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Runs the pattern battery over one chunk until stop/deadline. `base` is the
/// chunk's absolute word offset within the whole buffer (for own-address).
fn run_battery(
    base: usize,
    buf: &mut [u64],
    stop: &StopFlag,
    deadline: Instant,
    lane: Option<Arc<LiveLane>>,
) -> ThreadMemOut {
    let mut scan = Scan::new(base, stop, deadline, lane);

    let mut cycles = 0u64;
    'outer: loop {
        if scan.interrupted() {
            break;
        }
        if !scan.own_address(buf) {
            break 'outer;
        }
        for pattern in [
            0x0000_0000_0000_0000u64,
            u64::MAX,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ] {
            if !scan.moving_inversions(buf, pattern) {
                break 'outer;
            }
        }
        if !scan.checkerboard(buf) {
            break 'outer;
        }
        if !scan.walking(buf) {
            break 'outer;
        }
        // March C- and modulo-20 add ordered / isolated-cell coupling coverage
        // the parallel patterns above cannot.
        if !scan.march_c_minus(buf) {
            break 'outer;
        }
        if !scan.modulo20(buf) {
            break 'outer;
        }
        let seed = 0xC0FF_EE00_u64
            .wrapping_add(cycles.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(base as u64);
        if !scan.random(buf, seed) {
            break 'outer;
        }
        cycles += 1;
    }

    if let Some(l) = &scan.lane {
        l.set_phase(PHASE_DONE);
    }
    ThreadMemOut {
        cycles,
        errors: scan.errors,
        checksum: scan.checksum,
        bytes: scan.bytes,
        first: scan.first,
    }
}

struct Scan<'a> {
    base: usize,
    stop: &'a StopFlag,
    deadline: Instant,
    errors: u64,
    checksum: u64,
    bytes: u64,
    first: Option<Fail>,
    /// Live-UI lane (only the reporting thread has one; `None` otherwise or when
    /// no UI is attached). Fed from `interrupted()` so every pattern ticks it.
    lane: Option<Arc<LiveLane>>,
    /// Throttle for the (locking) status-detail push.
    last_note: Instant,
}

impl<'a> Scan<'a> {
    fn new(
        base: usize,
        stop: &'a StopFlag,
        deadline: Instant,
        lane: Option<Arc<LiveLane>>,
    ) -> Scan<'a> {
        Scan {
            base,
            stop,
            deadline,
            errors: 0,
            checksum: 0,
            bytes: 0,
            first: None,
            lane,
            last_note: Instant::now() - Duration::from_secs(1),
        }
    }

    #[inline]
    fn interrupted(&self) -> bool {
        // The per-block hook every pattern calls — so the live rate stays smooth
        // across all of them, not just the ones that publish detail.
        if let Some(l) = &self.lane {
            l.bump_work();
            l.set_phase(PHASE_WORK);
        }
        self.stop.stopped() || Instant::now() >= self.deadline
    }

    /// Publish the current pattern + read/expected sample to the live UI, throttled
    /// so the mutex is touched ~10×/s, not per block. No-op without a lane.
    fn progress(&mut self, pattern: &str, word: usize, got: u64, expected: u64) {
        let Some(lane) = &self.lane else { return };
        let now = Instant::now();
        if now.duration_since(self.last_note) < Duration::from_millis(90) {
            return;
        }
        self.last_note = now;
        lane.set_hash(self.checksum);
        let gib = self.bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let status = if got == expected { "OK" } else { "MISMATCH" };
        lane.set_detail(&format!(
            "pattern: {pattern}\nread:   {got:#018x}\nexpect: {expected:#018x}  {status}\nword:   {}\nverified: {gib:.2} GiB",
            self.base + word,
        ));
    }

    #[inline]
    fn record(&mut self, buf: &[u64], i: usize, expected: u64, got: u64, pattern: &'static str) {
        self.errors += 1;
        if self.first.is_none() {
            self.first = Some(Fail {
                vaddr: buf.as_ptr().wrapping_add(i) as usize,
                index: self.base + i,
                expected,
                got,
                pattern,
            });
        }
    }

    /// Write each cell = its absolute address, then verify. Returns `false` if
    /// interrupted mid-pass.
    fn own_address(&mut self, buf: &mut [u64]) -> bool {
        let n = buf.len();
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for (j, cell) in buf[i..end].iter_mut().enumerate() {
                *cell = (self.base + i + j) as u64;
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for j in i..end {
                let want = (self.base + j) as u64;
                let got = buf[j];
                self.checksum = self.checksum.wrapping_add(got);
                if got != want {
                    self.record(buf, j, want, got, "own-address");
                }
            }
            self.bytes += ((end - i) as u64) * 8;
            self.progress("own-address", end - 1, buf[end - 1], (self.base + end - 1) as u64);
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        true
    }

    /// Moving inversions: fill `p`, verify `p` + write `!p` ascending, then
    /// verify `!p` + rewrite `p` descending.
    fn moving_inversions(&mut self, buf: &mut [u64], p: u64) -> bool {
        let n = buf.len();
        let np = !p;
        let name = format!("moving-inv {p:#018x}");

        // Fill p.
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            buf[i..end].fill(p);
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }

        // Ascending: expect p, write !p.
        i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            let mut last = p;
            for j in i..end {
                let got = buf[j];
                last = got;
                if got != p {
                    self.record(buf, j, p, got, "moving-inv");
                }
                buf[j] = np;
            }
            self.bytes += ((end - i) as u64) * 8;
            self.progress(&name, end - 1, last, p);
            if self.interrupted() {
                return false;
            }
            i = end;
        }

        // Descending: expect !p, rewrite p.
        let mut done = n;
        while done > 0 {
            let start = done.saturating_sub(CHECK_STRIDE);
            for j in (start..done).rev() {
                let got = buf[j];
                if got != np {
                    self.record(buf, j, np, got, "moving-inv-rev");
                }
                buf[j] = p;
            }
            self.bytes += ((done - start) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            done = start;
        }
        true
    }

    /// Seeded pseudo-random fill then re-seeded verify.
    fn random(&mut self, buf: &mut [u64], seed: u64) -> bool {
        let n = buf.len();
        let mut state = seed;
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for cell in &mut buf[i..end] {
                *cell = splitmix64(&mut state);
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        state = seed;
        i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for j in i..end {
                let want = splitmix64(&mut state);
                let got = buf[j];
                self.checksum = self.checksum.wrapping_add(got);
                if got != want {
                    self.record(buf, j, want, got, "random");
                }
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        true
    }

    /// Fill every cell from `f`, then read back and verify against `f`. Two full
    /// passes so the verify actually reaches DRAM rather than cache. `f` takes
    /// the cell's *absolute* word index.
    fn fill_verify(
        &mut self,
        buf: &mut [u64],
        label: &'static str,
        f: impl Fn(usize) -> u64,
    ) -> bool {
        let n = buf.len();
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for (off, cell) in buf[i..end].iter_mut().enumerate() {
                *cell = f(self.base + i + off);
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for j in i..end {
                let want = f(self.base + j);
                let got = buf[j];
                self.checksum = self.checksum.wrapping_add(got);
                if got != want {
                    self.record(buf, j, want, got, label);
                }
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        true
    }

    /// Checkerboard and its inverse: physically-adjacent cells held at opposite
    /// values at once (best-effort — logical index adjacency only). Complement
    /// pair, so it is also a moving inversion.
    fn checkerboard(&mut self, buf: &mut [u64]) -> bool {
        const A: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        const B: u64 = 0x5555_5555_5555_5555;
        self.fill_verify(buf, "checker", |idx| if idx & 1 == 0 { A } else { B })
            && self.fill_verify(buf, "checker-inv", |idx| if idx & 1 == 0 { B } else { A })
    }

    /// Walking ones then walking zeros: an isolated set (then clear) bit marches
    /// through every bit position. The `+ step` offset makes neighbouring cells
    /// carry the bit in adjacent positions (a diagonal), so a single pass covers
    /// all 64 positions across the buffer; two steps add coupling variety.
    fn walking(&mut self, buf: &mut [u64]) -> bool {
        for step in [0usize, 32] {
            if !self.fill_verify(buf, "walk-one", |idx| 1u64 << ((idx + step) % 64)) {
                return false;
            }
        }
        for step in [0usize, 32] {
            if !self.fill_verify(buf, "walk-zero", |idx| !(1u64 << ((idx + step) % 64))) {
                return false;
            }
        }
        true
    }

    /// Modulo-20: one cell in twenty holds `P` while the other nineteen hold
    /// `!P`, cycling which residue is the odd one out. Isolated-cell coupling
    /// plus cache/write-buffer masking defeat. A subset of residues is sampled
    /// to keep it bounded (full 0..20 would be 20 passes each).
    fn modulo20(&mut self, buf: &mut [u64]) -> bool {
        for p in [0u64, u64::MAX] {
            for r in [0usize, 5, 10, 15] {
                if !self.fill_verify(buf, "modulo20", |idx| if idx % 20 == r { p } else { !p }) {
                    return false;
                }
            }
        }
        true
    }

    /// March C- (10N): the classic ordered march. Its ascending-then-descending
    /// read-modify-write elements *isolate* transition faults (verified in place)
    /// from coupling faults in a way the parallel-friendly moving-inversions
    /// cannot. `0` = all-zero words, `1` = all-one words. Block-stride preserves
    /// global address order, which the march depends on.
    fn march_c_minus(&mut self, buf: &mut [u64]) -> bool {
        const P0: u64 = 0x0000_0000_0000_0000;
        const P1: u64 = u64::MAX;
        self.fill(buf, P0)
            && self.march(buf, true, P0, P1, "march-up-r0w1")
            && self.march(buf, true, P1, P0, "march-up-r1w0")
            && self.march(buf, false, P0, P1, "march-dn-r0w1")
            && self.march(buf, false, P1, P0, "march-dn-r1w0")
            && self.verify_all(buf, P0, "march-r0")
    }

    /// Block-stride fill (write pass only).
    fn fill(&mut self, buf: &mut [u64], p: u64) -> bool {
        let n = buf.len();
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            buf[i..end].fill(p);
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        true
    }

    /// One march element: in strict `ascending`/descending order, verify each
    /// cell holds `expect` then immediately write `write`.
    fn march(
        &mut self,
        buf: &mut [u64],
        ascending: bool,
        expect: u64,
        write: u64,
        label: &'static str,
    ) -> bool {
        let n = buf.len();
        if ascending {
            let mut i = 0;
            while i < n {
                let end = (i + CHECK_STRIDE).min(n);
                for j in i..end {
                    let got = buf[j];
                    if got != expect {
                        self.record(buf, j, expect, got, label);
                    }
                    buf[j] = write;
                }
                self.bytes += ((end - i) as u64) * 8;
                if self.interrupted() {
                    return false;
                }
                i = end;
            }
        } else {
            let mut done = n;
            while done > 0 {
                let start = done.saturating_sub(CHECK_STRIDE);
                for j in (start..done).rev() {
                    let got = buf[j];
                    if got != expect {
                        self.record(buf, j, expect, got, label);
                    }
                    buf[j] = write;
                }
                self.bytes += ((done - start) as u64) * 8;
                if self.interrupted() {
                    return false;
                }
                done = start;
            }
        }
        true
    }

    /// Read pass: verify every cell holds `p`.
    fn verify_all(&mut self, buf: &mut [u64], p: u64, label: &'static str) -> bool {
        let n = buf.len();
        let mut i = 0;
        while i < n {
            let end = (i + CHECK_STRIDE).min(n);
            for j in i..end {
                let got = buf[j];
                self.checksum = self.checksum.wrapping_add(got);
                if got != p {
                    self.record(buf, j, p, got, label);
                }
            }
            self.bytes += ((end - i) as u64) * 8;
            if self.interrupted() {
                return false;
            }
            i = end;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn healthy_buffer_reports_no_errors() {
        let kernel = MemKernel {
            size: MemSize::Bytes(4 * 1024 * 1024),
            threads: Some(2),
        };
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_millis(120));
        let r = kernel.run(&budget, &stop, &markers);
        assert!(r.ok, "kernel failed to run: {}", r.detail);
        assert_eq!(r.error_count, 0, "false miscompare: {}", r.detail);
        assert!(r.detail.contains("MiB buffer"));
    }

    #[test]
    fn own_address_writes_absolute_addresses() {
        let mut buf = vec![0u64; 4096];
        let stop = StopFlag::new();
        let base = 1000;
        let mut scan = Scan::new(base, &stop, Instant::now() + Duration::from_secs(60), None);
        assert!(scan.own_address(&mut buf));
        assert_eq!(scan.errors, 0);
        // After the pass every cell holds its absolute word address — this is
        // exactly what the verify half compares against.
        for (j, &cell) in buf.iter().enumerate() {
            assert_eq!(cell, (base + j) as u64);
        }
    }

    #[test]
    fn moving_inversions_detects_stuck_bit() {
        // Simulate a stuck bit: after we fill a pattern, force one word to a
        // wrong value before the verify sees it by using a custom sequence.
        let mut buf = vec![0u64; 1024];
        let stop = StopFlag::new();
        let mut scan = Scan::new(0, &stop, Instant::now() + Duration::from_secs(60), None);
        // Pre-corrupt: fill with 0xFF, but the kernel will fill with p=0 first,
        // so instead we test the verify math directly: fill p, poke, verify.
        buf.fill(0);
        buf[500] = 0xDEAD_BEEF; // will mismatch p=0 on the ascending verify
                                // Emulate the ascending verify half of moving_inversions for p=0:
        let p = 0u64;
        for (j, &got) in buf.iter().enumerate() {
            if got != p {
                scan.record(&buf, j, p, got, "moving-inv");
            }
        }
        assert_eq!(scan.errors, 1);
        let f = scan.first.expect("first fail recorded");
        assert_eq!(f.index, 500);
        assert_eq!(f.got, 0xDEAD_BEEF);
        assert_eq!(f.pattern, "moving-inv");
    }

    #[test]
    fn random_pass_is_deterministic_and_clean() {
        let mut buf = vec![0u64; 8192];
        let stop = StopFlag::new();
        let mut scan = Scan::new(0, &stop, Instant::now() + Duration::from_secs(60), None);
        assert!(scan.random(&mut buf, 0xABCD));
        assert_eq!(scan.errors, 0);
    }

    #[test]
    fn new_batteries_are_clean_on_healthy_memory() {
        // fill_verify uses one closure for both fill and check, so the new
        // walking / checkerboard / modulo-20 patterns must never self-mismatch.
        let mut buf = vec![0u64; 4096];
        let stop = StopFlag::new();
        let mut scan = Scan::new(100, &stop, Instant::now() + Duration::from_secs(60), None);
        assert!(scan.checkerboard(&mut buf));
        assert!(scan.walking(&mut buf));
        assert!(scan.modulo20(&mut buf));
        assert!(scan.march_c_minus(&mut buf));
        assert_eq!(scan.errors, 0, "new batteries produced a false miscompare");
    }

    #[test]
    fn march_element_catches_mismatch() {
        // A march element reading `expect` from cells that hold something else
        // must record every mismatch — this guards the ordered read/write path.
        let mut buf = vec![0u64; 512]; // all zero
        let stop = StopFlag::new();
        let mut scan = Scan::new(0, &stop, Instant::now() + Duration::from_secs(60), None);
        // Expect all-ones but the buffer is all-zero: every cell mismatches.
        assert!(scan.march(&mut buf, true, u64::MAX, 0, "test"));
        assert_eq!(scan.errors, 512);
        // And it wrote 0 (the `write` value) — buffer unchanged here, all zero.
        assert!(buf.iter().all(|&w| w == 0));
    }

    #[test]
    fn stop_flag_ends_run_early() {
        let kernel = MemKernel::new(MemSize::Bytes(64 * 1024 * 1024));
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_secs(3600));
        stop.stop();
        let start = Instant::now();
        let _ = kernel.run(&budget, &stop, &markers);
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
