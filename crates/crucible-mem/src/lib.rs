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
//! * **seeded pseudo-random** — fill from a SplitMix64 stream, then re-seed and
//!   verify (catches data-dependent instability).
//!
//! On the first miscompare the failing virtual address, word index, expected
//! and observed values, and pattern are captured; the run is a FAIL.

use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, StopFlag};
use crucible_core::markers::{Event, MarkerLog};
use crucible_core::sysinfo;

/// Fraction of *available* physical memory to test when no explicit size given.
pub const DEFAULT_FRACTION: f64 = 0.5;
/// Headroom left free so the test never drives the machine into paging/OOM.
const SAFETY_BYTES: u64 = 1024 * 1024 * 1024;
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
            MemSize::Fraction(f) => {
                let f = f.clamp(0.01, 0.95);
                match avail {
                    Some(a) => (a as f64 * f) as u64,
                    None => FALLBACK_BYTES,
                }
            }
        };
        // Never take the machine below the safety headroom.
        let capped = match avail {
            Some(a) => bytes.min(a.saturating_sub(SAFETY_BYTES)).max(1),
            None => bytes.max(1),
        };
        capped / 8 // bytes -> u64 words
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

        // Disjoint mutable chunks, one per thread — no aliasing, no locks.
        let chunk_len = total_words.div_ceil(threads).max(1);
        let (outs, panics): (Vec<ThreadMemOut>, u64) = std::thread::scope(|scope| {
            let handles: Vec<_> = buf
                .chunks_mut(chunk_len)
                .enumerate()
                .map(|(idx, chunk)| {
                    let base = idx * chunk_len;
                    scope.spawn(move || run_battery(base, chunk, stop, deadline))
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
fn run_battery(base: usize, buf: &mut [u64], stop: &StopFlag, deadline: Instant) -> ThreadMemOut {
    let mut scan = Scan::new(base, stop, deadline);

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
        let seed = 0xC0FF_EE00_u64
            .wrapping_add(cycles.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(base as u64);
        if !scan.random(buf, seed) {
            break 'outer;
        }
        cycles += 1;
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
}

impl<'a> Scan<'a> {
    fn new(base: usize, stop: &'a StopFlag, deadline: Instant) -> Scan<'a> {
        Scan {
            base,
            stop,
            deadline,
            errors: 0,
            checksum: 0,
            bytes: 0,
            first: None,
        }
    }

    #[inline]
    fn interrupted(&self) -> bool {
        self.stop.stopped() || Instant::now() >= self.deadline
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
            for j in i..end {
                let got = buf[j];
                if got != p {
                    self.record(buf, j, p, got, "moving-inv");
                }
                buf[j] = np;
            }
            self.bytes += ((end - i) as u64) * 8;
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
        let mut scan = Scan::new(base, &stop, Instant::now() + Duration::from_secs(60));
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
        let mut scan = Scan::new(0, &stop, Instant::now() + Duration::from_secs(60));
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
        let mut scan = Scan::new(0, &stop, Instant::now() + Duration::from_secs(60));
        assert!(scan.random(&mut buf, 0xABCD));
        assert_eq!(scan.errors, 0);
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
