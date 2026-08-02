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

/// Format a Unix-nanosecond instant as ISO-8601 UTC, e.g.
/// `2026-08-02T02:07:22.000Z`.
///
/// Needed because the Windows event log's reliable time filter is an **absolute**
/// comparison — `TimeCreated[@SystemTime>='…']` — not the relative `timediff()`
/// function. `timediff` is accepted by `EvtQuery` and then silently ignored on
/// many channels: a field capture came back with seven channels returning their
/// entire contents, 99% of every archive being events from twelve hours outside
/// the run. An absolute timestamp is honoured everywhere.
///
/// ISO-8601 UTC also sorts lexicographically, which is what lets the archive
/// double-check each record with a string comparison rather than a date parser.
pub fn iso8601_utc(unix_nanos: u128) -> String {
    let secs = (unix_nanos / 1_000_000_000) as i64;
    let millis = ((unix_nanos % 1_000_000_000) / 1_000_000) as u32;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Days since the Unix epoch to a civil (year, month, day).
///
/// Howard Hinnant's `civil_from_days`, which is exact for the proleptic
/// Gregorian calendar and needs no table — worth carrying rather than a date
/// crate for the one place this suite formats a wall-clock time.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The current wall-clock instant, `ms` milliseconds ago, as ISO-8601 UTC.
pub fn iso8601_utc_ago(ms: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    iso8601_utc(now.saturating_sub(ms as u128 * 1_000_000))
}

#[cfg(test)]
mod iso_tests {
    use super::*;

    #[test]
    fn known_epochs_format_exactly() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00.000Z");
        // 2026-08-02T02:07:22Z — the field capture whose archives exposed the
        // `timediff` bug, so the one instant worth pinning by name.
        assert_eq!(
            iso8601_utc(1_785_636_442_000_000_000),
            "2026-08-02T02:07:22.000Z"
        );
        // A leap day, which is where a hand-rolled calendar goes wrong.
        assert_eq!(
            iso8601_utc(1_709_164_800_000_000_000),
            "2024-02-29T00:00:00.000Z"
        );
        // Y2K, and a millisecond that must not be truncated away.
        assert_eq!(iso8601_utc(946_684_800_500_000_000), "2000-01-01T00:00:00.500Z");
    }

    #[test]
    fn the_format_sorts_chronologically_as_a_string() {
        // The archive relies on this: it rejects stale records with a string
        // comparison rather than by parsing every timestamp.
        let a = iso8601_utc(1_785_636_442_000_000_000);
        let b = iso8601_utc(1_785_636_522_000_000_000);
        let c = iso8601_utc(1_700_000_000_000_000_000);
        assert!(a < b, "{a} !< {b}");
        assert!(c < a, "{c} !< {a}");
    }

    #[test]
    fn ago_moves_backwards_and_stays_well_formed() {
        let now = iso8601_utc_ago(0);
        let then = iso8601_utc_ago(60_000);
        assert!(then < now, "{then} !< {now}");
        assert_eq!(now.len(), 24, "{now}");
        assert!(now.ends_with('Z'));
    }
}
