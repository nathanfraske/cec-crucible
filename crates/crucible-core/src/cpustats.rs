// SPDX-License-Identifier: MIT
//! Per-core CPU telemetry — effective clock and utilization — with no kernel
//! driver and no admin rights.
//!
//! The suite refuses WinRing0-class MSR drivers (an HVCI/CVE liability), so we
//! never read `APERF`/`MPERF` ourselves. Instead we take the OS's *own*
//! derivation of those ratios through **PDH** (Performance Data Helper)
//! performance counters, which any non-elevated process may open:
//!
//! * `\Processor Information(*)\% Processor Performance` — the core's current
//!   speed as a percentage of its nominal (base) clock; reads above 100 while
//!   the part is boosting.
//! * `\Processor Information(*)\% Processor Time` — the core's busy percentage.
//!
//! Effective clock is then `base_MHz * (% Processor Performance / 100)`, where
//! the base (nominal) MHz is read once from the registry value `~MHz` (with the
//! `Processor Frequency` PDH counter as a fallback).
//!
//! Both `%` counters are *rate* counters: PDH needs two [`PdhCollectQueryData`]
//! calls spread across time before it can format a value, so [`CpuStats::new`]
//! primes one collect and every [`CpuStats::sample`] does another before it
//! reads. The first sample taken right after `new()` may therefore read low —
//! callers are expected to poll periodically.
//!
//! Zero external dependencies: the PDH / registry FFI is hand-declared against
//! `pdh` and `advapi32`, matching the rest of the crate. Non-Windows targets get
//! a stub ([`CpuStats::new`] returns `None`) so the workspace still builds and
//! `cargo test` runs everywhere.

/// One logical core's telemetry from a single [`CpuStats::sample`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreStat {
    /// Ascending logical-core index (0-based). For the common single processor
    /// group this equals the instance's own core number.
    pub core: u32,
    /// Effective clock in MHz (`base_MHz * % Processor Performance / 100`); may
    /// exceed the base clock while the core is boosting.
    pub effective_mhz: u32,
    /// Busy percentage for this core, clamped to `0.0..=100.0`.
    pub util_pct: f32,
}

/// An open per-core telemetry source.
///
/// Opaque by design: it owns the PDH query, its rate-counter handles, and the
/// cached base MHz. Construct once with [`CpuStats::new`] and poll
/// [`CpuStats::sample`] periodically; the query is closed on drop.
pub struct CpuStats {
    #[cfg(windows)]
    inner: win::Query,
}

impl CpuStats {
    /// Open the PDH query and prime the rate counters. Returns `None` when PDH
    /// is unavailable — non-Windows, or any PDH failure — so callers degrade
    /// gracefully rather than losing the whole run. Never panics.
    pub fn new() -> Option<CpuStats> {
        #[cfg(windows)]
        {
            win::Query::open().map(|inner| CpuStats { inner })
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Collect and read one snapshot: one [`CoreStat`] per logical core, sorted
    /// ascending by core index. Returns an empty vector if this round's collect
    /// or read failed (the caller simply polls again). Never panics.
    pub fn sample(&mut self) -> Vec<CoreStat> {
        #[cfg(windows)]
        {
            self.inner.sample()
        }
        #[cfg(not(windows))]
        {
            Vec::new()
        }
    }
}

/// Summarize a slice of [`CoreStat`] as `(avg_mhz, min_mhz, max_mhz, avg_util)`.
/// Returns `None` for an empty slice.
pub fn aggregate(stats: &[CoreStat]) -> Option<(u32, u32, u32, f32)> {
    if stats.is_empty() {
        return None;
    }
    let n = stats.len() as u64;
    let mut sum_mhz: u64 = 0;
    let mut min_mhz = u32::MAX;
    let mut max_mhz = 0u32;
    let mut sum_util = 0.0f64;
    for s in stats {
        sum_mhz += s.effective_mhz as u64;
        min_mhz = min_mhz.min(s.effective_mhz);
        max_mhz = max_mhz.max(s.effective_mhz);
        sum_util += s.util_pct as f64;
    }
    let avg_mhz = (sum_mhz / n) as u32;
    let avg_util = (sum_util / n as f64) as f32;
    Some((avg_mhz, min_mhz, max_mhz, avg_util))
}

#[cfg(windows)]
mod win {
    use super::CoreStat;
    use core::ffi::c_void;
    use std::collections::BTreeMap;
    use std::ptr::{null, null_mut};

    // --- PDH status / format constants ------------------------------------
    const ERROR_SUCCESS: u32 = 0;
    /// `PdhGetFormattedCounterArrayW`'s "buffer too small" sizing return.
    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;
    /// Do not cap percentage counters at 100 — `% Processor Performance` legit-
    /// imately exceeds 100 while boosting, and that surplus is the boost clock.
    const PDH_FMT_NOCAP100: u32 = 0x0000_8000;
    const PDH_CSTATUS_VALID_DATA: u32 = 0x0000_0000;
    const PDH_CSTATUS_NEW_DATA: u32 = 0x0000_0001;

    // Predefined registry root. Per `winreg.h` this is `(LONG)0x80000002` widened
    // to a HANDLE — i.e. sign-extended on 64-bit — which `as i32 as isize`
    // reproduces exactly.
    const HKEY_LOCAL_MACHINE: isize = 0x8000_0002u32 as i32 as isize;
    const RRF_RT_REG_DWORD: u32 = 0x0000_0010;

    // English counter paths. `PdhAddEnglishCounterW` maps these to the localized
    // names internally, so the code works on non-English Windows. The `(*)`
    // wildcard enumerates every core instance ("0,0", "0,1", …).
    const PERF_PATH: &str = r"\Processor Information(*)\% Processor Performance";
    const TIME_PATH: &str = r"\Processor Information(*)\% Processor Time";
    // Fallback base-clock source used only if the registry `~MHz` read fails; a
    // specific instance (not a wildcard) so it yields a single value.
    const FREQ_PATH: &str = r"\Processor Information(0,0)\Processor Frequency";

    #[link(name = "pdh")]
    extern "system" {
        fn PdhOpenQueryW(data_source: *const u16, user_data: usize, query: *mut isize) -> u32;
        fn PdhAddEnglishCounterW(
            query: isize,
            counter_path: *const u16,
            user_data: usize,
            counter: *mut isize,
        ) -> u32;
        fn PdhCollectQueryData(query: isize) -> u32;
        fn PdhGetFormattedCounterArrayW(
            counter: isize,
            format: u32,
            buffer_size: *mut u32,
            item_count: *mut u32,
            item_buffer: *mut PdhFmtCountervalueItemW,
        ) -> u32;
        fn PdhCloseQuery(query: isize) -> u32;
    }

    #[link(name = "advapi32")]
    extern "system" {
        fn RegGetValueW(
            key: isize,
            subkey: *const u16,
            value: *const u16,
            flags: u32,
            out_type: *mut u32,
            data: *mut c_void,
            data_size: *mut u32,
        ) -> u32;
    }

    // Mirror of `PDH_FMT_COUNTERVALUE` for the double case: a `DWORD CStatus`
    // then (after the natural 4-byte pad) the 8-byte value union. `repr(C)`
    // inserts that pad, placing `double_value` at offset 8 to match the C layout.
    #[repr(C)]
    struct PdhFmtCountervalueDouble {
        c_status: u32,
        double_value: f64,
    }

    // Mirror of `PDH_FMT_COUNTERVALUE_ITEM_W`: an instance-name pointer followed
    // by the formatted value. On x64 this is a 24-byte, 8-byte-aligned struct.
    #[repr(C)]
    struct PdhFmtCountervalueItemW {
        sz_name: *mut u16,
        fmt_value: PdhFmtCountervalueDouble,
    }

    /// The live query plus the cached base MHz.
    pub(super) struct Query {
        query: isize,
        h_perf: isize,
        h_time: isize,
        base_mhz: u32,
    }

    impl Query {
        pub(super) fn open() -> Option<Query> {
            let mut base_mhz = base_mhz_from_registry();

            let mut query: isize = 0;
            // SAFETY: a null data source selects live data; `query` is a valid
            // out-param the call writes the query handle into.
            let st = unsafe { PdhOpenQueryW(null(), 0, &mut query) };
            if st != ERROR_SUCCESS {
                return None;
            }

            let h_perf = match add_counter(query, PERF_PATH) {
                Some(h) => h,
                None => return close_and_none(query),
            };
            let h_time = match add_counter(query, TIME_PATH) {
                Some(h) => h,
                None => return close_and_none(query),
            };
            // Only worth adding when the registry base is missing.
            let h_freq = if base_mhz == 0 {
                add_counter(query, FREQ_PATH).unwrap_or(0)
            } else {
                0
            };

            // Prime the rate counters. Their formatted values only exist after a
            // second collect, which the first sample() performs.
            // SAFETY: `query` is a valid open handle.
            let st = unsafe { PdhCollectQueryData(query) };
            if st != ERROR_SUCCESS {
                return close_and_none(query);
            }

            // Processor Frequency is instantaneous (valid after one collect), so
            // the fallback can read it right here.
            if base_mhz == 0 && h_freq != 0 {
                if let Some(items) = read_array(h_freq, PDH_FMT_DOUBLE) {
                    if let Some((_, v)) = items.into_iter().next() {
                        base_mhz = v.round().max(0.0) as u32;
                    }
                }
            }

            Some(Query {
                query,
                h_perf,
                h_time,
                base_mhz,
            })
        }

        pub(super) fn sample(&mut self) -> Vec<CoreStat> {
            // Second/subsequent collect: yields the rate values accumulated since
            // the previous collect.
            // SAFETY: `self.query` is a valid open handle for this borrow.
            let st = unsafe { PdhCollectQueryData(self.query) };
            if st != ERROR_SUCCESS {
                return Vec::new();
            }

            // Read Performance uncapped so boost (> 100%) survives; Time stays
            // capped by our own clamp below.
            let perf = match read_array(self.h_perf, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100) {
                Some(v) => v,
                None => return Vec::new(),
            };
            let time = read_array(self.h_time, PDH_FMT_DOUBLE).unwrap_or_default();

            // The two counters' array orders are not guaranteed to match, so
            // correlate by instance name. Keying on the parsed (group, core) also
            // sorts ascending and drops "_Total"-style instances, which fail the
            // integer parse in `parse_instance`.
            let mut time_by: BTreeMap<(u32, u32), f64> = BTreeMap::new();
            for (name, v) in &time {
                if let Some(key) = parse_instance(name) {
                    time_by.insert(key, *v);
                }
            }
            let mut rows: BTreeMap<(u32, u32), (f64, f64)> = BTreeMap::new();
            for (name, v) in &perf {
                if let Some(key) = parse_instance(name) {
                    let t = time_by.get(&key).copied().unwrap_or(0.0);
                    rows.insert(key, (*v, t));
                }
            }

            let base = self.base_mhz as f64;
            rows.into_iter()
                .enumerate()
                .map(|(idx, (_key, (perf_pct, time_pct)))| CoreStat {
                    core: idx as u32,
                    // f64->u32 casts saturate, so a stray negative or huge value
                    // can't wrap; `max(0.0)` documents the intent.
                    effective_mhz: (base * perf_pct / 100.0).round().max(0.0) as u32,
                    util_pct: time_pct.clamp(0.0, 100.0) as f32,
                })
                .collect()
        }
    }

    impl Drop for Query {
        fn drop(&mut self) {
            // SAFETY: `query` came from PdhOpenQueryW and is closed exactly once
            // here; closing it also frees its counters. Teardown status is moot.
            let _ = unsafe { PdhCloseQuery(self.query) };
        }
    }

    /// Close a partially-built query and return `None` (open()'s error path).
    fn close_and_none(query: isize) -> Option<Query> {
        // SAFETY: `query` is a valid open handle not yet owned by a `Query`.
        let _ = unsafe { PdhCloseQuery(query) };
        None
    }

    fn add_counter(query: isize, path: &str) -> Option<isize> {
        let w = wide(path);
        let mut h: isize = 0;
        // SAFETY: `query` is open; `w` is NUL-terminated; `h` is a valid
        // out-param the call writes the counter handle into.
        let st = unsafe { PdhAddEnglishCounterW(query, w.as_ptr(), 0, &mut h) };
        if st == ERROR_SUCCESS {
            Some(h)
        } else {
            None
        }
    }

    /// Read a formatted counter into `(instance_name, value)` pairs via PDH's
    /// two-call sizing protocol. Returns `None` on any failure (including data
    /// that is not yet valid — the caller retries on the next poll).
    fn read_array(counter: isize, format: u32) -> Option<Vec<(String, f64)>> {
        let mut size: u32 = 0;
        let mut count: u32 = 0;
        // First call: a null buffer with zero size asks PDH for the required
        // byte size and item count.
        // SAFETY: the null item buffer is the documented sizing form; `size` and
        // `count` are valid out-params.
        let st = unsafe {
            PdhGetFormattedCounterArrayW(counter, format, &mut size, &mut count, null_mut())
        };
        if st != PDH_MORE_DATA || size == 0 || count == 0 {
            return None;
        }

        // Back the byte buffer with u64 storage so the item structs land 8-byte
        // aligned. PDH also packs the instance-name strings that each `szName`
        // points at into this same buffer, so it must be the *full* byte size
        // PDH asked for — not merely `count * size_of::<item>()`.
        let words = (size as usize).div_ceil(8).max(1);
        let mut buf = vec![0u64; words];
        let mut size2 = (buf.len() * 8) as u32;
        let mut count2 = count;
        // SAFETY: `buf` provides >= `size` bytes at 8-byte alignment; the pointer
        // is cast to the item type PDH fills; `size2`/`count2` are valid in/out
        // params.
        let st = unsafe {
            PdhGetFormattedCounterArrayW(
                counter,
                format,
                &mut size2,
                &mut count2,
                buf.as_mut_ptr() as *mut PdhFmtCountervalueItemW,
            )
        };
        if st != ERROR_SUCCESS {
            return None;
        }

        let n = count2 as usize;
        let items = buf.as_ptr() as *const PdhFmtCountervalueItemW;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: PDH wrote `count2` contiguous items into `buf`; `i < n`.
            let item = unsafe { &*items.add(i) };
            let status = item.fmt_value.c_status;
            if status != PDH_CSTATUS_VALID_DATA && status != PDH_CSTATUS_NEW_DATA {
                continue;
            }
            // SAFETY: `sz_name` points at a NUL-terminated UTF-16 string inside
            // `buf` that PDH wrote alongside the items.
            let name = unsafe { wide_to_string(item.sz_name) };
            out.push((name, item.fmt_value.double_value));
        }
        Some(out)
    }

    /// Parse a `"group,core"` instance name into `(group, core)`. Totals such as
    /// `"_Total"` / `"0,_Total"` fail the integer parse and yield `None`, which
    /// is how they get filtered out.
    fn parse_instance(name: &str) -> Option<(u32, u32)> {
        let (g, c) = name.split_once(',')?;
        let group: u32 = g.trim().parse().ok()?;
        let core: u32 = c.trim().parse().ok()?;
        Some((group, core))
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Read a NUL-terminated UTF-16 string into a `String`.
    ///
    /// # Safety
    /// `p` must be null, or point to a NUL-terminated UTF-16 string that stays
    /// valid for the duration of the read.
    unsafe fn wide_to_string(p: *const u16) -> String {
        if p.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        // SAFETY: the caller guarantees a NUL terminator; we stop at it.
        while unsafe { *p.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `p..p+len` is the string body preceding that NUL terminator.
        let slice = unsafe { std::slice::from_raw_parts(p, len) };
        String::from_utf16_lossy(slice)
    }

    /// Nominal (base) core clock in MHz from
    /// `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\~MHz`, or 0 if absent.
    fn base_mhz_from_registry() -> u32 {
        let subkey = wide(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
        let value = wide("~MHz");
        let mut data: u32 = 0;
        let mut size = std::mem::size_of::<u32>() as u32;
        // SAFETY: HKLM is a predefined key; subkey/value are NUL-terminated; the
        // REG_DWORD flag matches the u32 out-buffer, whose byte size is passed in
        // and updated through `size`.
        let st = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                subkey.as_ptr(),
                value.as_ptr(),
                RRF_RT_REG_DWORD,
                null_mut(),
                &mut data as *mut u32 as *mut c_void,
                &mut size,
            )
        };
        if st == ERROR_SUCCESS {
            data
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_is_none_on_empty() {
        assert_eq!(aggregate(&[]), None);
    }

    #[test]
    fn aggregate_min_avg_max_util() {
        let stats = [
            CoreStat {
                core: 0,
                effective_mhz: 1000,
                util_pct: 10.0,
            },
            CoreStat {
                core: 1,
                effective_mhz: 3000,
                util_pct: 50.0,
            },
            CoreStat {
                core: 2,
                effective_mhz: 5000,
                util_pct: 90.0,
            },
        ];
        let (avg, min, max, avg_util) = aggregate(&stats).expect("non-empty");
        assert_eq!((avg, min, max), (3000, 1000, 5000));
        assert!((avg_util - 50.0).abs() < 1e-3);
    }

    // Live end-to-end check. No-ops cleanly (and passes) where PDH is
    // unavailable, e.g. non-Windows CI. Run with:
    //   cargo test -p crucible-core cpustats -- --nocapture
    #[test]
    fn sample_reports_sane_values_when_available() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let mut cpu = match CpuStats::new() {
            Some(c) => c,
            None => {
                eprintln!("cpustats: PDH unavailable on this platform; skipping live check");
                return;
            }
        };

        // Put a light, time-boxed load on every core so the effective-clock and
        // utilization counters have something non-trivial to report across the
        // sample window.
        let stop = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::new();
        for _ in 0..crate::sysinfo::logical_cpus() {
            let stop = Arc::clone(&stop);
            workers.push(std::thread::spawn(move || {
                let mut x = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    std::hint::black_box(x);
                }
            }));
        }

        // Rate counters are computed between collects; wait between new()'s prime
        // collect and this read so the values cover a real interval.
        std::thread::sleep(Duration::from_millis(700));
        let stats = cpu.sample();

        stop.store(true, Ordering::Relaxed);
        for w in workers {
            let _ = w.join();
        }

        assert!(!stats.is_empty(), "expected at least one per-core row");
        for s in &stats {
            println!(
                "core {:>3}: {:>6} MHz   util {:>5.1}%",
                s.core, s.effective_mhz, s.util_pct
            );
            assert!(
                (0.0..=100.0).contains(&s.util_pct),
                "util out of range: {}",
                s.util_pct
            );
            assert!(
                s.effective_mhz < 20_000,
                "implausible effective MHz: {}",
                s.effective_mhz
            );
        }

        let (avg, min, max, avg_util) = aggregate(&stats).expect("aggregate over non-empty");
        println!(
            "aggregate: avg {avg} MHz   min {min} MHz   max {max} MHz   avg util {avg_util:.1}%"
        );
        assert!(min <= avg && avg <= max, "avg must lie within [min, max]");
        assert!((0.0..=100.0).contains(&avg_util));
    }
}
