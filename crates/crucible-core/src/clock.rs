// SPDX-License-Identifier: MIT
//! High-resolution timing.
//!
//! On Windows the suite timestamps every load transition with
//! `QueryPerformanceCounter` (QPC) — the *same* clock the external 1kHz+ power
//! rig and the PowerShell harness align to. Emitting raw QPC ticks plus the
//! counter frequency lets the analog capture line up to the exact load edge
//! with sub-microsecond precision; no sampling or inference on our side.
//!
//! On non-Windows targets (CI, dev laptops) we fall back to a monotonic
//! `Instant` expressed as nanosecond "ticks" so the whole workspace still
//! builds and `cargo test` runs everywhere. The fallback is clearly marked in
//! output via [`Clock::is_qpc`].

use std::time::{SystemTime, UNIX_EPOCH};

/// A single point in time captured from [`Clock::now`].
///
/// `qpc_ticks` / `qpc_frequency` are the correlation key for the power rig;
/// `unix_nanos` is a human/wall-clock anchor (not monotonic, do not diff it
/// for durations — use [`Timestamp::seconds_since`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timestamp {
    pub qpc_ticks: u64,
    pub qpc_frequency: u64,
    pub unix_nanos: u128,
}

impl Timestamp {
    /// Elapsed seconds from `earlier` to `self`, computed from the monotonic
    /// QPC ticks (never from wall-clock `unix_nanos`).
    pub fn seconds_since(&self, earlier: Timestamp) -> f64 {
        let dt = self.qpc_ticks.saturating_sub(earlier.qpc_ticks);
        dt as f64 / self.qpc_frequency as f64
    }
}

/// A cheap, copyable handle to the platform high-resolution counter.
///
/// Construct once with [`Clock::new`] and share freely across threads; reading
/// [`Clock::now`] is a single syscall/counter read with no allocation.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    frequency: u64,
    is_qpc: bool,
    #[cfg(not(windows))]
    base: std::time::Instant,
}

impl Clock {
    pub fn new() -> Clock {
        #[cfg(windows)]
        {
            Clock {
                frequency: win::frequency(),
                is_qpc: true,
            }
        }
        #[cfg(not(windows))]
        {
            Clock {
                // 1 tick == 1 nanosecond on the fallback path.
                frequency: 1_000_000_000,
                is_qpc: false,
                base: std::time::Instant::now(),
            }
        }
    }

    /// Ticks per second of the underlying counter.
    pub fn frequency(&self) -> u64 {
        self.frequency
    }

    /// `true` when backed by a real `QueryPerformanceCounter`, `false` on the
    /// `Instant` fallback. Reports surface this so a run's markers are never
    /// mistaken for rig-grade timestamps when they are not.
    pub fn is_qpc(&self) -> bool {
        self.is_qpc
    }

    pub fn now(&self) -> Timestamp {
        let unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        #[cfg(windows)]
        let qpc_ticks = win::counter();
        #[cfg(not(windows))]
        let qpc_ticks = self.base.elapsed().as_nanos() as u64;

        Timestamp {
            qpc_ticks,
            qpc_frequency: self.frequency,
            unix_nanos,
        }
    }
}

impl Default for Clock {
    fn default() -> Clock {
        Clock::new()
    }
}

#[cfg(windows)]
mod win {
    // Hand-declared FFI — zero external crates. These two calls are documented
    // to never fail on systems that boot Windows XP or later, so the BOOL
    // return is treated as infallible and we fall back to 0/1 defensively.
    #[link(name = "kernel32")]
    extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    }

    pub(super) fn frequency() -> u64 {
        let mut f: i64 = 0;
        // SAFETY: `f` is a valid, aligned i64 the kernel writes exactly 8 bytes
        // into; the call has no other effects.
        let ok = unsafe { QueryPerformanceFrequency(&mut f) };
        if ok != 0 && f > 0 {
            f as u64
        } else {
            1
        }
    }

    pub(super) fn counter() -> u64 {
        let mut c: i64 = 0;
        // SAFETY: as above; `c` is a valid out-param.
        let ok = unsafe { QueryPerformanceCounter(&mut c) };
        if ok != 0 {
            c as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_is_positive() {
        assert!(Clock::new().frequency() >= 1);
    }

    #[test]
    fn now_is_monotonic_nondecreasing() {
        let clock = Clock::new();
        let a = clock.now();
        // Busy a touch so at least one tick elapses on any real counter.
        let mut x = 0u64;
        for i in 0..100_000 {
            x = x.wrapping_add(i);
        }
        std::hint::black_box(x);
        let b = clock.now();
        assert!(b.qpc_ticks >= a.qpc_ticks);
        assert!(b.seconds_since(a) >= 0.0);
    }
}
