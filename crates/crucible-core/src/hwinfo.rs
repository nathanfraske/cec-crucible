// SPDX-License-Identifier: MIT
//! CPU package power and die temperature, via HWiNFO's shared-memory sensor
//! table.
//!
//! ## Why a bridge instead of reading the hardware ourselves
//!
//! CPU package power and die temperature live in model-specific registers —
//! Intel RAPL, AMD's equivalents — which only ring 0 can read. There is no
//! user-mode path, no WMI class, and no performance counter that exposes them.
//! Every tool that reports these numbers ships a kernel driver to do it.
//!
//! **This suite will not ship one.** The driver those tools use, `WinRing0`, is
//! on Microsoft's vulnerable-driver blocklist and carries a known
//! privilege-escalation CVE, because a driver that grants arbitrary MSR and
//! physical-memory access to user mode grants it to *everything* in user mode.
//! HVCI blocks it outright on a correctly configured machine. A QC tool's whole
//! value is that a technician can run it on a customer's machine and hand the
//! machine back; leaving an exploitable driver behind on every box we touch
//! would be a poor trade for a wattage figure.
//!
//! So instead we read from a sensor daemon the operator installed deliberately.
//! **HWiNFO** publishes its whole sensor table into a named shared-memory
//! section, `Global\HWiNFO_SENS_SM2`, documented in its SDK. Reading it needs no
//! elevation, no COM, and no driver of ours — HWiNFO has already done the
//! privileged work, under a driver the operator chose to install and can remove.
//!
//! It also brings sensors nothing else on this machine can reach: **DIMM
//! temperatures** on boards that expose them over the SMBus, VRM temperatures,
//! and fan RPM — the RAM-side readings that are otherwise impossible.
//!
//! ## Setting it up
//!
//! Run HWiNFO in *Sensors-only* mode, then Settings → check **Shared Memory
//! Support**. Note that the **free version disables shared memory after about
//! 12 minutes** per session; a long soak needs the paid version, or HWiNFO
//! restarted. That limitation is HWiNFO's, and it is reported rather than worked
//! around — a sensor plane that quietly stops mid-soak must not look like a
//! machine that quietly cooled down.
//!
//! ## Reading a live table
//!
//! HWiNFO rewrites these values continuously while we read them. Each reading is
//! an aligned `f64` so a torn value is not a practical concern, but a snapshot
//! can mix samples from adjacent polls. That is fine for trend telemetry and is
//! why nothing here is treated as a synchronised instant.

use crate::json::Json;

/// `"SiWH"` little-endian — the shared-memory signature.
const SIGNATURE: u32 = 0x4857_6953;

/// Reading kinds, from the SDK's `SENSOR_READING_TYPE`.
const TYPE_TEMPERATURE: u32 = 1;
const TYPE_FAN: u32 = 3;
const TYPE_POWER: u32 = 5;

/// One sensor reading as HWiNFO publishes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub kind: u32,
    /// Which sensor group it belongs to (CPU, motherboard, a drive, …).
    pub sensor: String,
    pub label: String,
    pub unit: String,
    pub value: f64,
}

/// What this suite pulls out of that table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CpuSensors {
    /// CPU package power, watts — the number that cannot be had any other way.
    pub package_power_w: Option<f64>,
    /// Package / Tdie temperature, °C.
    pub package_temp_c: Option<f64>,
    /// Hottest individual core, °C. Often a couple of degrees above package.
    pub core_max_c: Option<f64>,
    /// DIMM temperatures, °C, where the board exposes them over SMBus.
    pub dimm_c: Vec<f64>,
    /// VRM / MOSFET temperature, °C — the part that actually fails first on a
    /// board pushed hard.
    pub vrm_c: Option<f64>,
    /// Whatever fans the daemon can see, RPM.
    pub fan_rpm: Vec<f64>,
}

impl CpuSensors {
    pub fn any(&self) -> bool {
        self.package_power_w.is_some()
            || self.package_temp_c.is_some()
            || self.core_max_c.is_some()
            || !self.dimm_c.is_empty()
    }

    pub fn line(&self) -> Option<String> {
        if !self.any() {
            return None;
        }
        let mut s = String::from("cpu:     ");
        match (self.package_power_w, self.package_temp_c) {
            (Some(w), Some(c)) => s.push_str(&format!("{w:.1} W package, {c:.0} °C")),
            (Some(w), None) => s.push_str(&format!("{w:.1} W package")),
            (None, Some(c)) => s.push_str(&format!("{c:.0} °C package")),
            (None, None) => s.push_str("(no package reading)"),
        }
        if let Some(c) = self.core_max_c {
            s.push_str(&format!(", hottest core {c:.0} °C"));
        }
        if !self.dimm_c.is_empty() {
            let hot = self.dimm_c.iter().cloned().fold(f64::MIN, f64::max);
            s.push_str(&format!(", DIMM peak {hot:.0} °C"));
        }
        if let Some(v) = self.vrm_c {
            s.push_str(&format!(", VRM {v:.0} °C"));
        }
        Some(s)
    }
}

/// Run-long summary of the CPU/board sensor plane.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CpuSummary {
    pub samples: u64,
    pub power_avg_w: f64,
    pub power_peak_w: f64,
    pub temp_peak_c: f64,
    pub core_peak_c: f64,
    pub dimm_peak_c: f64,
    pub vrm_peak_c: f64,
    pub saw_power: bool,
    /// The table stopped updating mid-run — HWiNFO's free-version shared-memory
    /// timeout, or the daemon being closed. Reported, because a sensor plane
    /// that silently stops looks exactly like hardware that silently cooled.
    pub went_stale: bool,
}

impl CpuSummary {
    pub fn accumulate(&mut self, s: &CpuSensors) {
        self.samples += 1;
        if let Some(w) = s.package_power_w {
            self.saw_power = true;
            self.power_peak_w = self.power_peak_w.max(w);
            self.power_avg_w += (w - self.power_avg_w) / self.samples as f64;
        }
        if let Some(c) = s.package_temp_c {
            self.temp_peak_c = self.temp_peak_c.max(c);
        }
        if let Some(c) = s.core_max_c {
            self.core_peak_c = self.core_peak_c.max(c);
        }
        for c in &s.dimm_c {
            self.dimm_peak_c = self.dimm_peak_c.max(*c);
        }
        if let Some(c) = s.vrm_c {
            self.vrm_peak_c = self.vrm_peak_c.max(c);
        }
    }

    pub fn line(&self) -> Option<String> {
        if self.samples == 0 {
            return None;
        }
        let mut s = String::from("cpu:     ");
        if self.saw_power {
            s.push_str(&format!(
                "package power avg {:.1} W, peak {:.1} W",
                self.power_avg_w, self.power_peak_w
            ));
        }
        if self.temp_peak_c > 0.0 {
            if self.saw_power {
                s.push_str(", ");
            }
            s.push_str(&format!("peak {:.0} °C", self.temp_peak_c));
        }
        if self.core_peak_c > 0.0 {
            s.push_str(&format!(" (hottest core {:.0} °C)", self.core_peak_c));
        }
        if self.dimm_peak_c > 0.0 {
            s.push_str(&format!(", DIMM peak {:.0} °C", self.dimm_peak_c));
        }
        if self.vrm_peak_c > 0.0 {
            s.push_str(&format!(", VRM peak {:.0} °C", self.vrm_peak_c));
        }
        if self.went_stale {
            s.push_str("  [SENSOR FEED STOPPED MID-RUN — HWiNFO free-version shared memory times out after ~12 min]");
        }
        if s.len() <= 9 {
            return None;
        }
        Some(s)
    }

    pub fn to_json(&self) -> Json {
        let f = |v: f64| if v > 0.0 { Json::F64((v * 10.0).round() / 10.0) } else { Json::Null };
        let mut o = Json::object();
        o.push("source", Json::str("HWiNFO shared memory"))
            .push("samples", Json::U64(self.samples))
            .push(
                "package_power_avg_w",
                if self.saw_power { f(self.power_avg_w) } else { Json::Null },
            )
            .push(
                "package_power_peak_w",
                if self.saw_power { f(self.power_peak_w) } else { Json::Null },
            )
            .push("package_temp_peak_c", f(self.temp_peak_c))
            .push("core_temp_peak_c", f(self.core_peak_c))
            .push("dimm_temp_peak_c", f(self.dimm_peak_c))
            .push("vrm_temp_peak_c", f(self.vrm_peak_c))
            .push("feed_stopped_mid_run", self.went_stale);
        o
    }
}

/// Pick the readings this suite cares about out of a full sensor table.
///
/// Label matching rather than fixed indices, because HWiNFO's table layout
/// depends on the board, the CPU and which sensors the user has hidden. The
/// patterns cover both vendors: Intel calls package power "CPU Package Power",
/// AMD reports "CPU Package Power" or "CPU PPT"; Intel's die sensor is
/// "CPU Package", AMD's is "CPU (Tctl/Tdie)".
pub fn select(readings: &[Reading]) -> CpuSensors {
    let mut out = CpuSensors::default();

    // Power, best match first. `Package Power` beats `PPT` beats anything else
    // containing both "CPU" and "Power", so a board that publishes several does
    // not hand us the wrong one.
    for want in ["cpu package power", "cpu ppt", "package power"] {
        if out.package_power_w.is_some() {
            break;
        }
        out.package_power_w = readings
            .iter()
            .filter(|r| r.kind == TYPE_POWER)
            .find(|r| r.label.to_ascii_lowercase().contains(want))
            .map(|r| r.value);
    }

    for want in ["cpu package", "cpu (tctl/tdie)", "cpu (tdie)", "core temperature"] {
        if out.package_temp_c.is_some() {
            break;
        }
        out.package_temp_c = readings
            .iter()
            .filter(|r| r.kind == TYPE_TEMPERATURE)
            .find(|r| r.label.to_ascii_lowercase().starts_with(want))
            .map(|r| r.value);
    }

    // Hottest individual core: "Core 0", "CPU Core 3", "P-core 2", "E-core 5".
    out.core_max_c = readings
        .iter()
        .filter(|r| r.kind == TYPE_TEMPERATURE)
        .filter(|r| {
            let l = r.label.to_ascii_lowercase();
            (l.contains("core") && !l.contains("package") && !l.contains("distance"))
                && l.chars().any(|c| c.is_ascii_digit())
        })
        .map(|r| r.value)
        .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));

    out.dimm_c = readings
        .iter()
        .filter(|r| r.kind == TYPE_TEMPERATURE)
        .filter(|r| {
            let l = r.label.to_ascii_lowercase();
            l.contains("dimm") || l.contains("memory module") || l.starts_with("spd")
        })
        .map(|r| r.value)
        .collect();

    out.vrm_c = readings
        .iter()
        .filter(|r| r.kind == TYPE_TEMPERATURE)
        .find(|r| {
            let l = r.label.to_ascii_lowercase();
            l.contains("vrm") || l.contains("vr mos") || l.contains("mosfet")
        })
        .map(|r| r.value);

    out.fan_rpm = readings
        .iter()
        .filter(|r| r.kind == TYPE_FAN && r.value > 0.0)
        .map(|r| r.value)
        .collect();

    out
}

/// A mapped view of HWiNFO's shared sensor table.
pub struct HwInfo {
    #[cfg(windows)]
    inner: win::Shared,
}

impl HwInfo {
    /// Map the table. `None` when HWiNFO is not running, or is running without
    /// Shared Memory Support enabled — both of which are normal and are reported
    /// to the operator rather than treated as failures.
    pub fn open() -> Option<HwInfo> {
        #[cfg(windows)]
        {
            win::Shared::open().map(|inner| HwInfo { inner })
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Every reading in the table, for diagnostics and for the `sensors` view.
    pub fn readings(&self) -> Vec<Reading> {
        #[cfg(windows)]
        {
            self.inner.readings()
        }
        #[cfg(not(windows))]
        {
            Vec::new()
        }
    }

    /// The subset this suite reports.
    pub fn sample(&self) -> CpuSensors {
        select(&self.readings())
    }

    /// HWiNFO's own poll timestamp. Used to notice the feed going stale — the
    /// free version stops updating shared memory after about twelve minutes, and
    /// a frozen value is indistinguishable from a steady one without this.
    pub fn poll_time(&self) -> u64 {
        #[cfg(windows)]
        {
            self.inner.poll_time()
        }
        #[cfg(not(windows))]
        {
            0
        }
    }
}

/// What to tell an operator who wants CPU power and has no daemon running.
pub const SETUP_HINT: &str = "CPU package power and die temperature need a sensor daemon: they live \
in model-specific registers that only a kernel driver can read, and this tool deliberately ships \
none. Install HWiNFO (free), run it in Sensors-only mode, and enable Settings -> Shared Memory \
Support. Note the free version stops publishing shared memory after ~12 minutes per session.";

#[cfg(windows)]
mod win {
    use super::{Reading, SIGNATURE};
    use core::ffi::c_void;

    const FILE_MAP_READ: u32 = 4;
    const MAP_NAME: &[u8] = b"Global\\HWiNFO_SENS_SM2\0";
    /// Fallback for a session-local mapping when HWiNFO is not running elevated.
    const MAP_NAME_LOCAL: &[u8] = b"HWiNFO_SENS_SM2\0";

    /// Field offsets in `HWiNFO_SENSORS_SHARED_MEM2`. Written out rather than
    /// mirrored as a struct so the 8-byte `__time64_t` in the middle cannot be
    /// silently repadded by the Rust layout.
    const H_SIGNATURE: usize = 0;
    const H_VERSION: usize = 4;
    const H_POLL_TIME: usize = 12; // __time64_t
    const H_SENSOR_OFFSET: usize = 20;
    const H_SENSOR_SIZE: usize = 24;
    const H_SENSOR_COUNT: usize = 28;
    const H_READING_OFFSET: usize = 32;
    const H_READING_SIZE: usize = 36;
    const H_READING_COUNT: usize = 40;
    const HEADER_LEN: usize = 44;

    /// Offsets inside `HWiNFO_SENSORS_READING_ELEMENT`.
    const R_TYPE: usize = 0;
    const R_SENSOR_INDEX: usize = 4;
    const R_LABEL_USER: usize = 140; // after type, index, id, and szLabelOrig[128]
    const R_UNIT: usize = 268;
    const R_VALUE: usize = 284;

    /// Offsets inside `HWiNFO_SENSORS_SENSOR_ELEMENT`.
    const S_NAME_USER: usize = 136; // after id, instance, szSensorNameOrig[128]

    /// A table larger than this is not something we should be walking.
    const MAX_ELEMENTS: u32 = 8192;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenFileMappingA(access: u32, inherit: i32, name: *const u8) -> isize;
        fn MapViewOfFile(
            mapping: isize,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(h: isize) -> i32;
    }

    pub(super) struct Shared {
        mapping: isize,
        view: *mut c_void,
    }

    // SAFETY: the view is read-only and never mutated by us; HWiNFO writes it
    // concurrently by design, which the module docs address.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    impl Shared {
        pub(super) fn open() -> Option<Shared> {
            for name in [MAP_NAME, MAP_NAME_LOCAL] {
                // SAFETY: NUL-terminated name literal.
                let mapping = unsafe { OpenFileMappingA(FILE_MAP_READ, 0, name.as_ptr()) };
                if mapping == 0 {
                    continue;
                }
                // 0 bytes = map the whole section.
                // SAFETY: mapping handle just obtained.
                let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
                if view.is_null() {
                    // SAFETY: opened above, closed exactly once.
                    unsafe { CloseHandle(mapping) };
                    continue;
                }
                let s = Shared { mapping, view };
                // Reject anything that is not the table we expect rather than
                // walking a stranger's shared memory as if it were sensors.
                if s.u32_at(H_SIGNATURE) != SIGNATURE || s.u32_at(H_VERSION) == 0 {
                    continue;
                }
                return Some(s);
            }
            None
        }

        fn u32_at(&self, off: usize) -> u32 {
            // SAFETY: reading within the mapped view; every caller's offset is a
            // header field inside HEADER_LEN, or bounds-checked below.
            unsafe { std::ptr::read_unaligned((self.view as *const u8).add(off) as *const u32) }
        }

        fn f64_at(&self, off: usize) -> f64 {
            // SAFETY: as above.
            unsafe { std::ptr::read_unaligned((self.view as *const u8).add(off) as *const f64) }
        }

        /// A fixed-width NUL-terminated ASCII field.
        fn str_at(&self, off: usize, cap: usize) -> String {
            // SAFETY: the caller bounds `off + cap` to the element it belongs to.
            let bytes = unsafe { std::slice::from_raw_parts((self.view as *const u8).add(off), cap) };
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(cap);
            String::from_utf8_lossy(&bytes[..end]).trim().to_string()
        }

        pub(super) fn poll_time(&self) -> u64 {
            // SAFETY: header field inside the mapped view.
            unsafe {
                std::ptr::read_unaligned((self.view as *const u8).add(H_POLL_TIME) as *const u64)
            }
        }

        pub(super) fn readings(&self) -> Vec<Reading> {
            if self.u32_at(H_SIGNATURE) != SIGNATURE {
                return Vec::new(); // the daemon closed the section under us
            }
            let s_off = self.u32_at(H_SENSOR_OFFSET) as usize;
            let s_size = self.u32_at(H_SENSOR_SIZE) as usize;
            let s_count = self.u32_at(H_SENSOR_COUNT).min(MAX_ELEMENTS) as usize;
            let r_off = self.u32_at(H_READING_OFFSET) as usize;
            let r_size = self.u32_at(H_READING_SIZE) as usize;
            let r_count = self.u32_at(H_READING_COUNT).min(MAX_ELEMENTS) as usize;

            // A layout that cannot hold the fields we read is a layout we do not
            // understand; walking it anyway would produce plausible nonsense.
            if s_off < HEADER_LEN || s_size <= S_NAME_USER || r_size <= R_VALUE + 8 {
                return Vec::new();
            }

            let names: Vec<String> = (0..s_count)
                .map(|i| self.str_at(s_off + i * s_size + S_NAME_USER, 128))
                .collect();

            (0..r_count)
                .map(|i| {
                    let base = r_off + i * r_size;
                    let idx = self.u32_at(base + R_SENSOR_INDEX) as usize;
                    Reading {
                        kind: self.u32_at(base + R_TYPE),
                        sensor: names.get(idx).cloned().unwrap_or_default(),
                        label: self.str_at(base + R_LABEL_USER, 128),
                        unit: self.str_at(base + R_UNIT, 16),
                        value: self.f64_at(base + R_VALUE),
                    }
                })
                .collect()
        }
    }

    impl Drop for Shared {
        fn drop(&mut self) {
            // SAFETY: both handles came from this module and are released once.
            unsafe {
                UnmapViewOfFile(self.view);
                CloseHandle(self.mapping);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(kind: u32, label: &str, value: f64, unit: &str) -> Reading {
        Reading {
            kind,
            sensor: "CPU [#0]".into(),
            label: label.into(),
            unit: unit.into(),
            value,
        }
    }

    #[test]
    fn intel_labels_resolve_to_package_power_and_temperature() {
        let t = vec![
            r(TYPE_POWER, "CPU Package Power", 142.5, "W"),
            r(TYPE_POWER, "CPU IA Cores Power", 120.0, "W"),
            r(TYPE_TEMPERATURE, "CPU Package", 78.0, "°C"),
            r(TYPE_TEMPERATURE, "Core 0", 74.0, "°C"),
            r(TYPE_TEMPERATURE, "Core 7", 81.0, "°C"),
        ];
        let s = select(&t);
        assert_eq!(s.package_power_w, Some(142.5), "IA Cores must not win");
        assert_eq!(s.package_temp_c, Some(78.0));
        assert_eq!(s.core_max_c, Some(81.0), "hottest core, not the first");
    }

    #[test]
    fn amd_labels_resolve_too() {
        let t = vec![
            r(TYPE_POWER, "CPU PPT", 88.0, "W"),
            r(TYPE_TEMPERATURE, "CPU (Tctl/Tdie)", 69.0, "°C"),
            r(TYPE_TEMPERATURE, "CPU Core 5", 71.0, "°C"),
        ];
        let s = select(&t);
        assert_eq!(s.package_power_w, Some(88.0));
        assert_eq!(s.package_temp_c, Some(69.0));
        assert_eq!(s.core_max_c, Some(71.0));
    }

    #[test]
    fn dimm_and_vrm_are_picked_up_because_nothing_else_can_reach_them() {
        // These are the readings that justify the bridge existing at all: no
        // driverless path reaches a DIMM sensor.
        let t = vec![
            r(TYPE_TEMPERATURE, "DIMM 1 Temperature", 44.0, "°C"),
            r(TYPE_TEMPERATURE, "DIMM 2 Temperature", 47.5, "°C"),
            r(TYPE_TEMPERATURE, "VRM MOS Temperature", 62.0, "°C"),
            r(TYPE_FAN, "CPU Fan", 1450.0, "RPM"),
        ];
        let s = select(&t);
        assert_eq!(s.dimm_c, vec![44.0, 47.5]);
        assert_eq!(s.vrm_c, Some(62.0));
        assert_eq!(s.fan_rpm, vec![1450.0]);
    }

    #[test]
    fn a_temperature_is_never_mistaken_for_a_power_reading() {
        // Kinds are checked, not just labels: a board publishing "CPU Package"
        // as a temperature must not become a wattage.
        let t = vec![r(TYPE_TEMPERATURE, "CPU Package Power", 78.0, "°C")];
        assert_eq!(select(&t).package_power_w, None);
    }

    #[test]
    fn an_empty_table_reports_nothing_rather_than_zeros() {
        let s = select(&[]);
        assert!(!s.any());
        assert_eq!(s.line(), None);
        assert_eq!(CpuSummary::default().line(), None);
    }

    #[test]
    fn a_stale_feed_is_called_out_in_the_summary() {
        // The free version stops publishing after ~12 minutes. A frozen value is
        // indistinguishable from a steady one, so the run has to say so.
        let mut sum = CpuSummary::default();
        sum.accumulate(&CpuSensors {
            package_power_w: Some(95.0),
            package_temp_c: Some(72.0),
            ..Default::default()
        });
        sum.went_stale = true;
        let l = sum.line().unwrap();
        assert!(l.contains("SENSOR FEED STOPPED"), "{l}");
        assert!(sum.to_json().to_compact().contains("\"feed_stopped_mid_run\":true"));
    }

    #[test]
    fn absent_power_is_null_in_json_not_zero() {
        let mut sum = CpuSummary::default();
        sum.accumulate(&CpuSensors {
            package_temp_c: Some(70.0),
            ..Default::default()
        });
        let j = sum.to_json().to_compact();
        assert!(j.contains("\"package_power_peak_w\":null"), "{j}");
        assert!(j.contains("\"package_temp_peak_c\":70"), "{j}");
    }

    #[test]
    fn the_live_machine_is_read_or_absent_but_never_implausible() {
        let Some(h) = HwInfo::open() else { return };
        let s = h.sample();
        if let Some(w) = s.package_power_w {
            assert!((0.1..1000.0).contains(&w), "implausible CPU package power {w} W");
        }
        if let Some(c) = s.package_temp_c {
            assert!((0.0..150.0).contains(&c), "implausible CPU temperature {c} °C");
        }
        for c in &s.dimm_c {
            assert!((0.0..150.0).contains(c), "implausible DIMM temperature {c} °C");
        }
    }
}
