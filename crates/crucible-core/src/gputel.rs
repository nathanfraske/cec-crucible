// SPDX-License-Identifier: MIT
//! GPU sensor plane — board power, temperatures, fan, clocks and throttle
//! reasons, via NVML.
//!
//! `nvml.dll` ships **with the NVIDIA driver**, not with the CUDA toolkit, so
//! this is `LoadLibrary` + `GetProcAddress` at runtime: no SDK, no admin, no
//! kernel driver, and a clean degrade to "unavailable" on AMD/Intel or on a box
//! with no NVIDIA GPU. Same posture as the CUDA transfer path.
//!
//! ## Why this earns its place in a QC tool
//!
//! Two of these readings answer questions nothing else in the suite can:
//!
//! * **Throttle reasons.** NVIDIA documents `HwPowerBrakeSlowdown` as an
//!   *external power-brake assertion — e.g. by the system power supply*. That
//!   means a marginal PSU asserting its protection is observable **in software**,
//!   without the bench rig, even on a machine that never shuts down. The only
//!   other symptom is a brief clock halving that no integrity test would notice.
//! * **Sustained board power and memory-junction temperature** are what turn "it
//!   passed" into "it passed with 12 °C of headroom" — the difference between a
//!   result that survives the customer's warm, dusty desk and one that does not.
//!
//! Everything here is measurement, never a gate on its own: we report, and the
//! operator (or a later margin policy) decides.

use crate::json::Json;

/// One sample of the GPU's sensors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuSample {
    /// Board power draw in watts.
    pub power_w: f64,
    /// Core temperature, °C.
    pub temp_c: u32,
    /// Memory-junction temperature, °C. 0 when the part does not report it
    /// (consumer cards frequently do not).
    pub mem_temp_c: u32,
    /// Fan duty, percent. 0 on passively cooled or non-reporting parts.
    pub fan_pct: u32,
    /// SM (core) clock, MHz.
    pub sm_mhz: u32,
    /// Memory clock, MHz.
    pub mem_mhz: u32,
    /// Raw throttle-reason bitmask (see [`throttle_names`]).
    pub throttle: u64,
}

/// Bits from NVML's clocks-event/throttle-reason mask that matter to QC.
pub mod throttle {
    pub const GPU_IDLE: u64 = 0x0000_0001;
    pub const APP_CLOCKS_SETTING: u64 = 0x0000_0002;
    pub const SW_POWER_CAP: u64 = 0x0000_0004;
    pub const HW_SLOWDOWN: u64 = 0x0000_0008;
    pub const SYNC_BOOST: u64 = 0x0000_0010;
    pub const SW_THERMAL_SLOWDOWN: u64 = 0x0000_0020;
    pub const HW_THERMAL_SLOWDOWN: u64 = 0x0000_0040;
    /// **The important one.** An external power-brake assertion — NVIDIA's docs
    /// name the system power supply as an example source.
    pub const HW_POWER_BRAKE: u64 = 0x0000_0080;
    pub const DISPLAY_CLOCK_SETTING: u64 = 0x0000_0100;
}

/// Human-readable throttle reasons present in `mask`, most serious first.
/// Deliberately omits `GPU_IDLE` and the benign clock-setting bits: reporting
/// "throttled: idle" during a burst test's off-phase would be noise.
pub fn throttle_names(mask: u64) -> Vec<&'static str> {
    let mut v = Vec::new();
    if mask & throttle::HW_POWER_BRAKE != 0 {
        v.push("HW power brake (external assertion — e.g. the PSU's protection)");
    }
    if mask & throttle::HW_THERMAL_SLOWDOWN != 0 {
        v.push("HW thermal slowdown");
    }
    if mask & throttle::SW_THERMAL_SLOWDOWN != 0 {
        v.push("SW thermal slowdown");
    }
    if mask & throttle::HW_SLOWDOWN != 0 {
        v.push("HW slowdown (thermal, power brake, or excessive draw)");
    }
    if mask & throttle::SW_POWER_CAP != 0 {
        v.push("SW power cap (at the board's power limit)");
    }
    v
}

/// Running summary across a run — what the report and the console line show.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuSummary {
    pub samples: u64,
    pub name: String,
    pub power_avg_w: f64,
    pub power_peak_w: f64,
    pub temp_peak_c: u32,
    pub mem_temp_peak_c: u32,
    pub fan_peak_pct: u32,
    pub sm_mhz_avg: u32,
    /// Union of every throttle mask seen during the run.
    pub throttle_seen: u64,
    /// Power limit the board is enforcing, watts (0 = unknown).
    pub power_limit_w: f64,
}

impl GpuSummary {
    pub fn accumulate(&mut self, s: &GpuSample) {
        self.samples += 1;
        // Running mean, so a long soak cannot overflow a sum.
        self.power_avg_w += (s.power_w - self.power_avg_w) / self.samples as f64;
        self.sm_mhz_avg = ((self.sm_mhz_avg as u64 * (self.samples - 1) + s.sm_mhz as u64)
            / self.samples) as u32;
        self.power_peak_w = self.power_peak_w.max(s.power_w);
        self.temp_peak_c = self.temp_peak_c.max(s.temp_c);
        self.mem_temp_peak_c = self.mem_temp_peak_c.max(s.mem_temp_c);
        self.fan_peak_pct = self.fan_peak_pct.max(s.fan_pct);
        self.throttle_seen |= s.throttle;
    }

    /// One-line console form.
    pub fn line(&self) -> String {
        let mut s = format!(
            "gpu: {} — power avg {:.0} W, peak {:.0} W",
            if self.name.is_empty() { "?" } else { &self.name },
            self.power_avg_w,
            self.power_peak_w
        );
        if self.power_limit_w > 0.0 {
            s.push_str(&format!(" (limit {:.0} W)", self.power_limit_w));
        }
        s.push_str(&format!(", peak {} °C", self.temp_peak_c));
        if self.mem_temp_peak_c > 0 {
            s.push_str(&format!(" (mem {} °C)", self.mem_temp_peak_c));
        }
        if self.fan_peak_pct > 0 {
            s.push_str(&format!(", fan {}%", self.fan_peak_pct));
        }
        let names = throttle_names(self.throttle_seen);
        if names.is_empty() {
            s.push_str(", no throttling");
        } else {
            s.push_str(&format!("  THROTTLED: {}", names.join("; ")));
        }
        s
    }

    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("name", Json::str(&self.name))
            .push("samples", Json::U64(self.samples))
            .push("power_avg_w", Json::F64((self.power_avg_w * 10.0).round() / 10.0))
            .push("power_peak_w", Json::F64((self.power_peak_w * 10.0).round() / 10.0))
            .push("power_limit_w", Json::F64(self.power_limit_w))
            .push("temp_peak_c", Json::U64(self.temp_peak_c as u64))
            // Null, not 0, when the board has no memory-junction sensor: a 0
            // here reads as a 0 °C measurement rather than as "not measured".
            .push(
                "mem_temp_peak_c",
                match self.mem_temp_peak_c {
                    0 => Json::Null,
                    v => Json::U64(v as u64),
                },
            )
            .push("fan_peak_pct", Json::U64(self.fan_peak_pct as u64))
            .push("sm_mhz_avg", Json::U64(self.sm_mhz_avg as u64));
        let names: Vec<Json> = throttle_names(self.throttle_seen)
            .into_iter()
            .map(Json::str)
            .collect();
        o.push("throttle_reasons", Json::Array(names));
        o
    }
}

/// An open NVML handle for one GPU.
pub struct GpuTelemetry {
    #[cfg(windows)]
    inner: win::Nvml,
    #[cfg(not(windows))]
    _priv: (),
}

impl GpuTelemetry {
    /// Open NVML for GPU `index`. `None` when NVML is absent (no NVIDIA driver,
    /// non-Windows, or a machine with no NVIDIA GPU) — callers degrade to "no
    /// GPU sensor data" rather than failing the run.
    pub fn open(index: u32) -> Option<GpuTelemetry> {
        #[cfg(windows)]
        {
            win::Nvml::open(index).map(|inner| GpuTelemetry { inner })
        }
        #[cfg(not(windows))]
        {
            let _ = index;
            None
        }
    }

    pub fn name(&self) -> String {
        #[cfg(windows)]
        {
            self.inner.name.clone()
        }
        #[cfg(not(windows))]
        {
            String::new()
        }
    }

    /// Enforced board power limit in watts (0 if unknown).
    pub fn power_limit_w(&self) -> f64 {
        #[cfg(windows)]
        {
            self.inner.power_limit_w
        }
        #[cfg(not(windows))]
        {
            0.0
        }
    }

    /// Take one sample. Fields that the part does not report come back 0.
    pub fn sample(&self) -> GpuSample {
        #[cfg(windows)]
        {
            self.inner.sample()
        }
        #[cfg(not(windows))]
        {
            GpuSample::default()
        }
    }
}

#[cfg(windows)]
mod win {
    use super::GpuSample;
    use core::ffi::c_void;

    const NVML_SUCCESS: i32 = 0;
    const NVML_TEMPERATURE_GPU: u32 = 0;
    const NVML_CLOCK_SM: u32 = 1;
    const NVML_CLOCK_MEM: u32 = 2;
    /// `NVML_FI_DEV_MEMORY_TEMP` — memory junction temperature.
    const NVML_FI_DEV_MEMORY_TEMP: u32 = 82;

    type Handle = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FieldValue {
        field_id: u32,
        scope_id: u32,
        timestamp: i64,
        latency_usec: i64,
        value_type: u32,
        nvml_return: u32,
        // The union, widest member is 8 bytes.
        value: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    macro_rules! sym {
        ($lib:expr, $name:literal, $t:ty) => {{
            // SAFETY: the name is a NUL-terminated literal; the transmute target
            // matches the documented NVML signature.
            let p = unsafe { GetProcAddress($lib, concat!($name, "\0").as_ptr()) };
            if p.is_null() {
                None
            } else {
                Some(unsafe { std::mem::transmute::<*mut c_void, $t>(p) })
            }
        }};
    }

    pub(super) struct Nvml {
        dev: Handle,
        pub(super) name: String,
        pub(super) power_limit_w: f64,
        f_power: Option<unsafe extern "C" fn(Handle, *mut u32) -> i32>,
        f_temp: Option<unsafe extern "C" fn(Handle, u32, *mut u32) -> i32>,
        f_fan: Option<unsafe extern "C" fn(Handle, *mut u32) -> i32>,
        f_clock: Option<unsafe extern "C" fn(Handle, u32, *mut u32) -> i32>,
        f_throttle: Option<unsafe extern "C" fn(Handle, *mut u64) -> i32>,
        f_fields: Option<unsafe extern "C" fn(Handle, i32, *mut FieldValue) -> i32>,
    }

    // SAFETY: NVML device handles are process-wide and the library is documented
    // as thread-safe; we only ever read.
    unsafe impl Send for Nvml {}
    unsafe impl Sync for Nvml {}

    impl Nvml {
        pub(super) fn open(index: u32) -> Option<Nvml> {
            // SAFETY: loading a system DLL by name; all pointers checked below.
            let lib = unsafe { LoadLibraryA(b"nvml.dll\0".as_ptr()) };
            if lib.is_null() {
                return None;
            }

            let init = sym!(lib, "nvmlInit_v2", unsafe extern "C" fn() -> i32)?;
            // SAFETY: documented no-arg initializer.
            if unsafe { init() } != NVML_SUCCESS {
                return None;
            }

            let get_handle = sym!(
                lib,
                "nvmlDeviceGetHandleByIndex_v2",
                unsafe extern "C" fn(u32, *mut Handle) -> i32
            )?;
            let mut dev: Handle = std::ptr::null_mut();
            // SAFETY: `dev` is a valid out-param.
            if unsafe { get_handle(index, &mut dev) } != NVML_SUCCESS || dev.is_null() {
                return None;
            }

            // Name: nice-to-have, so a failure here is not fatal.
            let mut name = String::new();
            if let Some(f) = sym!(
                lib,
                "nvmlDeviceGetName",
                unsafe extern "C" fn(Handle, *mut u8, u32) -> i32
            ) {
                let mut buf = [0u8; 96];
                // SAFETY: buffer and length match.
                if unsafe { f(dev, buf.as_mut_ptr(), buf.len() as u32) } == NVML_SUCCESS {
                    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                    name = String::from_utf8_lossy(&buf[..end]).into_owned();
                }
            }

            let mut power_limit_w = 0.0;
            if let Some(f) = sym!(
                lib,
                "nvmlDeviceGetEnforcedPowerLimit",
                unsafe extern "C" fn(Handle, *mut u32) -> i32
            ) {
                let mut mw = 0u32;
                // SAFETY: valid out-param.
                if unsafe { f(dev, &mut mw) } == NVML_SUCCESS {
                    power_limit_w = mw as f64 / 1000.0;
                }
            }

            Some(Nvml {
                dev,
                name,
                power_limit_w,
                f_power: sym!(
                    lib,
                    "nvmlDeviceGetPowerUsage",
                    unsafe extern "C" fn(Handle, *mut u32) -> i32
                ),
                f_temp: sym!(
                    lib,
                    "nvmlDeviceGetTemperature",
                    unsafe extern "C" fn(Handle, u32, *mut u32) -> i32
                ),
                f_fan: sym!(
                    lib,
                    "nvmlDeviceGetFanSpeed",
                    unsafe extern "C" fn(Handle, *mut u32) -> i32
                ),
                f_clock: sym!(
                    lib,
                    "nvmlDeviceGetClockInfo",
                    unsafe extern "C" fn(Handle, u32, *mut u32) -> i32
                ),
                // The _v2 name is the current one; both spellings exist in the wild.
                f_throttle: sym!(
                    lib,
                    "nvmlDeviceGetCurrentClocksThrottleReasons",
                    unsafe extern "C" fn(Handle, *mut u64) -> i32
                )
                .or_else(|| {
                    sym!(
                        lib,
                        "nvmlDeviceGetCurrentClocksEventReasons",
                        unsafe extern "C" fn(Handle, *mut u64) -> i32
                    )
                }),
                f_fields: sym!(
                    lib,
                    "nvmlDeviceGetFieldValues",
                    unsafe extern "C" fn(Handle, i32, *mut FieldValue) -> i32
                ),
            })
        }

        pub(super) fn sample(&self) -> GpuSample {
            let mut s = GpuSample::default();
            // SAFETY: every call below passes a valid device handle and a valid
            // out-param, and each return code is checked before the value is used.
            unsafe {
                if let Some(f) = self.f_power {
                    let mut mw = 0u32;
                    if f(self.dev, &mut mw) == NVML_SUCCESS {
                        s.power_w = mw as f64 / 1000.0;
                    }
                }
                if let Some(f) = self.f_temp {
                    let mut c = 0u32;
                    if f(self.dev, NVML_TEMPERATURE_GPU, &mut c) == NVML_SUCCESS {
                        s.temp_c = c;
                    }
                }
                if let Some(f) = self.f_fan {
                    let mut p = 0u32;
                    if f(self.dev, &mut p) == NVML_SUCCESS {
                        s.fan_pct = p;
                    }
                }
                if let Some(f) = self.f_clock {
                    let mut m = 0u32;
                    if f(self.dev, NVML_CLOCK_SM, &mut m) == NVML_SUCCESS {
                        s.sm_mhz = m;
                    }
                    let mut m2 = 0u32;
                    if f(self.dev, NVML_CLOCK_MEM, &mut m2) == NVML_SUCCESS {
                        s.mem_mhz = m2;
                    }
                }
                if let Some(f) = self.f_throttle {
                    let mut mask = 0u64;
                    if f(self.dev, &mut mask) == NVML_SUCCESS {
                        s.throttle = mask;
                    }
                }
                // Memory-junction temperature is a "field value", not a plain
                // getter, and many consumer parts simply do not report it.
                if let Some(f) = self.f_fields {
                    let mut fv = FieldValue {
                        field_id: NVML_FI_DEV_MEMORY_TEMP,
                        scope_id: 0,
                        timestamp: 0,
                        latency_usec: 0,
                        value_type: 0,
                        nvml_return: 0,
                        value: 0,
                    };
                    if f(self.dev, 1, &mut fv) == NVML_SUCCESS && fv.nvml_return == 0 {
                        // Reported as an unsigned int in the union's low bits.
                        let t = (fv.value & 0xFFFF_FFFF) as u32;
                        if t > 0 && t < 200 {
                            s.mem_temp_c = t;
                        }
                    }
                }
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_brake_is_named_first_and_idle_is_not_noise() {
        let names = throttle_names(throttle::HW_POWER_BRAKE | throttle::SW_POWER_CAP);
        assert!(names[0].contains("power brake"), "{names:?}");
        assert!(names[0].contains("PSU"), "the operator must be told what that means");
        // Idle and app-clock bits are normal and must not be reported as faults.
        assert!(throttle_names(throttle::GPU_IDLE).is_empty());
        assert!(throttle_names(throttle::APP_CLOCKS_SETTING).is_empty());
        assert!(throttle_names(0).is_empty());
    }

    #[test]
    fn summary_tracks_peaks_and_a_running_mean() {
        let mut g = GpuSummary {
            name: "Test GPU".into(),
            power_limit_w: 240.0,
            ..Default::default()
        };
        for (p, t) in [(100.0, 60u32), (200.0, 70), (150.0, 65)] {
            g.accumulate(&GpuSample {
                power_w: p,
                temp_c: t,
                sm_mhz: 1800,
                ..Default::default()
            });
        }
        assert_eq!(g.samples, 3);
        assert!((g.power_avg_w - 150.0).abs() < 1e-9, "avg {}", g.power_avg_w);
        assert!((g.power_peak_w - 200.0).abs() < 1e-9);
        assert_eq!(g.temp_peak_c, 70);
        let line = g.line();
        assert!(line.contains("peak 200 W"), "{line}");
        assert!(line.contains("limit 240 W"), "{line}");
        assert!(line.contains("no throttling"), "{line}");
    }

    #[test]
    fn a_power_brake_seen_once_is_reported_for_the_whole_run() {
        let mut g = GpuSummary::default();
        g.accumulate(&GpuSample::default());
        g.accumulate(&GpuSample {
            throttle: throttle::HW_POWER_BRAKE,
            ..Default::default()
        });
        g.accumulate(&GpuSample::default());
        // The union across the run — a brake that asserted for one sample is the
        // whole finding, and must not be averaged away.
        assert!(g.line().contains("THROTTLED"), "{}", g.line());
        assert!(g.to_json().to_compact().contains("power brake"));
    }

    #[test]
    fn open_degrades_rather_than_panicking() {
        // On a box with an NVIDIA driver this really opens NVML; elsewhere it is
        // None. Either is correct — the contract is only that it never panics.
        if let Some(t) = GpuTelemetry::open(0) {
            let s = t.sample();
            assert!(s.power_w >= 0.0 && s.power_w < 2000.0, "power {}", s.power_w);
            assert!(s.temp_c < 200, "temp {}", s.temp_c);
        }
        // A silly index must be None, not a crash.
        assert!(GpuTelemetry::open(9999).is_none());
    }
}
