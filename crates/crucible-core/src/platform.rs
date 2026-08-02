// SPDX-License-Identifier: MIT
//! Platform sensors: ACPI thermal zones and, where the firmware exposes one, a
//! system power meter.
//!
//! ## What is deliberately *not* here, and why
//!
//! **CPU die temperature and package power are not obtainable.** Both live in
//! model-specific registers (Intel RAPL / AMD equivalents) that only ring 0 can
//! read. Every tool that reports them — HWiNFO, LibreHardwareMonitor, Core Temp
//! — ships a signed kernel driver to do it, and those drivers are a recurring
//! source of privilege-escalation CVEs precisely because they expose arbitrary
//! MSR and physical-memory access to user mode. This suite will not ship one:
//! the whole point is to be a QC tool a technician can run on a customer's
//! machine without leaving an attack surface behind.
//!
//! **DIMM temperature and power are not obtainable either.** DDR4 carries a TS
//! sensor and DDR5 a PMIC, but both are reached over the SMBus, which again means
//! a driver. `Win32_PhysicalMemory` exposes voltage *ratings* — configured, min
//! and max — which are firmware constants, not measurements, and reporting them
//! as telemetry would be inventing data.
//!
//! What *is* available without a driver:
//!
//! * **ACPI thermal zones** via the `Thermal Zone Information` PDH counter set.
//!   Read honestly, this is a *board* zone, not the CPU die — on this bench it
//!   sits near ambient while the CPU is loaded. Useful as a chassis/ambient
//!   trend, misleading if labelled "CPU temp", so it is not.
//! * **A system power meter** via the `Power Meter` counter set, which is backed
//!   by the ACPI Energy Metering Interface. Present on most laptops and tablets,
//!   absent on most desktops (no instances on this bench). Where it exists it is
//!   whole-system input power, not a CPU figure.
//!
//! Both come through PDH, the same mechanism as the per-core clocks, so they
//! need no elevation.

use crate::json::Json;

/// One reading of the platform sensor plane.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlatformSample {
    /// Warmest ACPI thermal zone, °C. `None` when no zone reports.
    pub zone_c: Option<f64>,
    /// How many zones the machine exposes — one on a typical desktop, several on
    /// a laptop.
    pub zones: usize,
    /// System input power in watts from the ACPI Energy Metering Interface.
    /// `None` on hardware without one, which is most desktops.
    pub power_w: Option<f64>,
}

/// Run-long summary of the platform plane.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlatformSummary {
    pub samples: u64,
    pub zone_peak_c: f64,
    pub zone_avg_c: f64,
    pub zones: usize,
    pub power_peak_w: f64,
    pub power_avg_w: f64,
    pub saw_power: bool,
}

impl PlatformSummary {
    pub fn accumulate(&mut self, s: &PlatformSample) {
        self.samples += 1;
        self.zones = self.zones.max(s.zones);
        if let Some(c) = s.zone_c {
            self.zone_peak_c = self.zone_peak_c.max(c);
            // Running mean so a long soak cannot overflow a sum.
            self.zone_avg_c += (c - self.zone_avg_c) / self.samples as f64;
        }
        if let Some(w) = s.power_w {
            self.saw_power = true;
            self.power_peak_w = self.power_peak_w.max(w);
            self.power_avg_w += (w - self.power_avg_w) / self.samples as f64;
        }
    }

    /// One-line console form, or `None` when the machine reported nothing worth
    /// printing — a line that says "0 °C" is worse than no line.
    pub fn line(&self) -> Option<String> {
        if self.samples == 0 || (self.zone_peak_c <= 0.0 && !self.saw_power) {
            return None;
        }
        let mut s = String::from("board:   ");
        if self.zone_peak_c > 0.0 {
            s.push_str(&format!(
                "ACPI zone peak {:.1} °C, avg {:.1} °C ({} zone{})",
                self.zone_peak_c,
                self.zone_avg_c,
                self.zones,
                if self.zones == 1 { "" } else { "s" }
            ));
            // Say what it is not, once, where somebody will read it.
            s.push_str(" — chassis/board zone, NOT the CPU die");
        }
        if self.saw_power {
            if self.zone_peak_c > 0.0 {
                s.push_str("; ");
            }
            s.push_str(&format!(
                "system power avg {:.1} W, peak {:.1} W (EMI)",
                self.power_avg_w, self.power_peak_w
            ));
        }
        Some(s)
    }

    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("samples", Json::U64(self.samples))
            .push("acpi_zones", Json::U64(self.zones as u64))
            .push(
                "zone_peak_c",
                if self.zone_peak_c > 0.0 {
                    Json::F64((self.zone_peak_c * 10.0).round() / 10.0)
                } else {
                    Json::Null
                },
            )
            .push(
                "zone_avg_c",
                if self.zone_peak_c > 0.0 {
                    Json::F64((self.zone_avg_c * 10.0).round() / 10.0)
                } else {
                    Json::Null
                },
            )
            .push(
                "system_power_peak_w",
                if self.saw_power {
                    Json::F64((self.power_peak_w * 10.0).round() / 10.0)
                } else {
                    Json::Null
                },
            )
            .push(
                "system_power_avg_w",
                if self.saw_power {
                    Json::F64((self.power_avg_w * 10.0).round() / 10.0)
                } else {
                    Json::Null
                },
            )
            .push(
                "note",
                Json::str(
                    "ACPI thermal zones are board/chassis sensors, not the CPU die. CPU die \
                     temperature and package power require ring-0 MSR access (a kernel driver), \
                     which this tool deliberately does not ship.",
                ),
            );
        o
    }
}

/// An open handle on the platform sensor plane.
pub struct Platform {
    #[cfg(windows)]
    inner: win::Pdh,
}

impl Platform {
    /// Open the counters. `None` when PDH is unavailable or neither counter set
    /// has a single instance — which is a legitimate answer on a machine that
    /// exposes no zones at all, not an error.
    pub fn open() -> Option<Platform> {
        #[cfg(windows)]
        {
            win::Pdh::open().map(|inner| Platform { inner })
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    pub fn sample(&self) -> PlatformSample {
        #[cfg(windows)]
        {
            self.inner.sample()
        }
        #[cfg(not(windows))]
        {
            PlatformSample::default()
        }
    }
}

#[cfg(windows)]
mod win {
    use super::PlatformSample;
    use core::ffi::c_void;
    use std::ptr::{null, null_mut};

    const ERROR_SUCCESS: u32 = 0;
    const PDH_MORE_DATA: u32 = 0x800007D2;
    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;

    /// The ACPI thermal zone plane. `High Precision Temperature` is reported in
    /// **tenths of a Kelvin**, which is why a raw 3010 is 27.9 °C and not 3010
    /// of anything.
    const ZONE_PATH: &str = r"\Thermal Zone Information(*)\High Precision Temperature";
    /// ACPI Energy Metering Interface. Absent on most desktops.
    const POWER_PATH: &str = r"\Power Meter(*)\Power";

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
            item_buffer: *mut Item,
        ) -> u32;
        fn PdhCloseQuery(query: isize) -> u32;
    }

    #[repr(C)]
    struct Value {
        status: u32,
        double_value: f64,
    }

    #[repr(C)]
    struct Item {
        name: *mut u16,
        value: Value,
    }

    pub(super) struct Pdh {
        query: isize,
        zone: isize,
        power: isize,
    }

    // SAFETY: the handles are only read from, and PDH queries are documented as
    // usable from any thread provided a single query is not used concurrently —
    // which the sampler thread guarantees by owning it.
    unsafe impl Send for Pdh {}
    unsafe impl Sync for Pdh {}

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn add(query: isize, path: &str) -> isize {
        let p = wide(path);
        let mut h: isize = 0;
        // SAFETY: NUL-terminated path; `h` is a valid out-param.
        let st = unsafe { PdhAddEnglishCounterW(query, p.as_ptr(), 0, &mut h) };
        if st == ERROR_SUCCESS {
            h
        } else {
            0
        }
    }

    /// Read every instance of a counter as `(name, value)`.
    fn read(counter: isize) -> Vec<f64> {
        if counter == 0 {
            return Vec::new();
        }
        let mut size: u32 = 0;
        let mut count: u32 = 0;
        // SAFETY: the documented two-call sizing protocol — a null buffer asks
        // for the required size.
        let st = unsafe {
            PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, null_mut())
        };
        if st != PDH_MORE_DATA || size == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; size as usize];
        // SAFETY: buffer is `size` bytes as PDH just asked for.
        let st = unsafe {
            PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_DOUBLE,
                &mut size,
                &mut count,
                buf.as_mut_ptr() as *mut Item,
            )
        };
        if st != ERROR_SUCCESS {
            return Vec::new();
        }
        let mut out = Vec::new();
        for i in 0..count as usize {
            // SAFETY: PDH wrote `count` items at the head of the buffer.
            let item = unsafe { &*(buf.as_ptr() as *const Item).add(i) };
            if item.value.status == ERROR_SUCCESS {
                out.push(item.value.double_value);
            }
        }
        out
    }

    impl Pdh {
        pub(super) fn open() -> Option<Pdh> {
            let mut query: isize = 0;
            // SAFETY: null data source = live data; valid out-param.
            if unsafe { PdhOpenQueryW(null(), 0, &mut query) } != ERROR_SUCCESS {
                return None;
            }
            let zone = add(query, ZONE_PATH);
            let power = add(query, POWER_PATH);
            if zone == 0 && power == 0 {
                // SAFETY: opened above, closed exactly once.
                unsafe { PdhCloseQuery(query) };
                return None;
            }
            // Both counters are instantaneous, so one collect is enough to make
            // the first sample valid — unlike the rate counters in `cpustats`.
            // SAFETY: valid open query.
            if unsafe { PdhCollectQueryData(query) } != ERROR_SUCCESS {
                // SAFETY: as above.
                unsafe { PdhCloseQuery(query) };
                return None;
            }
            Some(Pdh { query, zone, power })
        }

        pub(super) fn sample(&self) -> PlatformSample {
            // SAFETY: valid open query.
            if unsafe { PdhCollectQueryData(self.query) } != ERROR_SUCCESS {
                return PlatformSample::default();
            }
            let zones = read(self.zone);
            // Tenths of a Kelvin -> °C. A zone reporting absolute zero is a
            // firmware stub rather than a reading, so it is dropped.
            let temps: Vec<f64> = zones
                .iter()
                .map(|k| k / 10.0 - 273.15)
                .filter(|c| *c > -50.0 && *c < 200.0)
                .collect();
            let zone_c = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // EMI reports milliwatts.
            let power = read(self.power);
            let power_w = power
                .iter()
                .cloned()
                .filter(|v| *v > 0.0)
                .map(|mw| mw / 1000.0)
                .fold(0.0, f64::max);

            PlatformSample {
                zone_c: if temps.is_empty() { None } else { Some(zone_c) },
                zones: temps.len(),
                power_w: if power_w > 0.0 { Some(power_w) } else { None },
            }
        }
    }

    impl Drop for Pdh {
        fn drop(&mut self) {
            // SAFETY: opened in `open`, closed exactly once here.
            unsafe { PdhCloseQuery(self.query) };
        }
    }

    // Keeps the unused-import lint quiet on the c_void the Item layout implies.
    const _: Option<*const c_void> = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_summary_with_nothing_in_it_prints_nothing() {
        // A "0 °C" line would read as a measurement. Silence is the honest
        // output when the machine exposes no zone and no meter.
        assert_eq!(PlatformSummary::default().line(), None);
    }

    #[test]
    fn the_line_says_what_the_zone_is_not() {
        let mut s = PlatformSummary::default();
        s.accumulate(&PlatformSample {
            zone_c: Some(31.5),
            zones: 1,
            power_w: None,
        });
        let line = s.line().expect("a zone reading produces a line");
        assert!(line.contains("31.5"));
        // The single most likely misreading of this number is "CPU temperature".
        assert!(
            line.contains("NOT the CPU die"),
            "the disclaimer is the point of the line: {line}"
        );
        assert!(!line.contains("W"), "no power was measured: {line}");
    }

    #[test]
    fn peaks_and_means_track_separately() {
        let mut s = PlatformSummary::default();
        for c in [30.0, 40.0, 35.0] {
            s.accumulate(&PlatformSample {
                zone_c: Some(c),
                zones: 1,
                power_w: Some(50.0),
            });
        }
        assert_eq!(s.samples, 3);
        assert_eq!(s.zone_peak_c, 40.0);
        assert!((s.zone_avg_c - 35.0).abs() < 1e-9, "avg was {}", s.zone_avg_c);
        assert!(s.saw_power);
        assert_eq!(s.power_peak_w, 50.0);
    }

    #[test]
    fn an_absent_meter_is_null_in_json_not_zero() {
        let mut s = PlatformSummary::default();
        s.accumulate(&PlatformSample {
            zone_c: Some(28.0),
            zones: 1,
            power_w: None,
        });
        let j = s.to_json().to_compact();
        // A 0 W system draw is not a reading anyone should be able to plot.
        assert!(j.contains("\"system_power_peak_w\":null"), "{j}");
        assert!(j.contains("\"zone_peak_c\":28"), "{j}");
        // And the JSON carries the same caveat as the console line, because the
        // report outlives the terminal it was printed in.
        assert!(j.contains("not the CPU die"), "{j}");
    }

    #[test]
    fn the_live_machine_reports_something_plausible_or_nothing() {
        // Whatever this bench exposes, a zone reading must be a temperature and
        // not a raw Kelvin count — the tenths-of-a-Kelvin conversion is the one
        // thing here that silently produces a number like 3010.
        let Some(p) = Platform::open() else { return };
        let s = p.sample();
        if let Some(c) = s.zone_c {
            assert!(
                (0.0..120.0).contains(&c),
                "implausible ACPI zone temperature {c} °C — check the K/10 conversion"
            );
        }
        if let Some(w) = s.power_w {
            assert!((0.1..2000.0).contains(&w), "implausible system power {w} W");
        }
    }
}
