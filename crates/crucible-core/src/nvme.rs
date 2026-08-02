// SPDX-License-Identifier: MIT
//! NVMe drive health: temperature, wear and endurance, straight from the
//! controller's SMART / Health Information log page.
//!
//! **This is the one component whose own sensors we can read without a driver.**
//! CPU die temperature and DIMM temperature both need ring 0 or the SMBus (see
//! [`crate::platform`]); an NVMe controller, by contrast, publishes its health
//! log through the standard storage stack, and Windows will hand it over via
//! `IOCTL_STORAGE_QUERY_PROPERTY` with the NVMe protocol-specific property.
//!
//! What that log carries, and what it is worth:
//!
//! * **Composite temperature**, plus up to eight discrete sensors. An NVMe drive
//!   throttles itself when it gets hot, so a storage test that came in slow next
//!   to an 80 °C composite temperature has explained itself. On this bench a
//!   980 PRO reads 63 °C at idle with a lifetime peak of 82 °C.
//! * **Percentage used** — the controller's own estimate of consumed endurance.
//!   A drive reporting 90% used in a machine being QC'd before delivery is a
//!   finding all by itself.
//! * **Unsafe shutdowns and media errors**, which are exactly the counters that
//!   distinguish "this machine crashed" from "this machine crashed *and the
//!   drive noticed*".
//!
//! **What NVMe does not report is power.** The spec exposes power *states* and
//! their advertised maximum draw, not live consumption; there is no watt figure
//! to read, and computing one from the state table would be inventing data.
//!
//! Temperatures are Kelvin in the log and converted here once, on the way out.

use crate::json::Json;

/// The health of one NVMe namespace/drive.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NvmeHealth {
    /// `\\.\PhysicalDriveN` index.
    pub index: u32,
    /// Composite temperature in °C — the controller's own headline number.
    pub temp_c: i32,
    /// Discrete sensors, °C. Empty when the drive populates none, which is
    /// common; the composite is mandatory, these are optional.
    pub sensors_c: Vec<i32>,
    /// Endurance consumed, percent. Can exceed 100 on a worn drive, and the
    /// spec says so — it is not clamped here.
    pub percentage_used: u8,
    /// Spare capacity remaining, percent.
    pub available_spare: u8,
    /// Bit field; any bit set is a controller-raised warning.
    pub critical_warning: u8,
    pub power_on_hours: u64,
    pub unsafe_shutdowns: u64,
    pub media_errors: u64,
}

impl NvmeHealth {
    /// Decoded critical-warning bits, per the NVMe base spec.
    pub fn warnings(&self) -> Vec<&'static str> {
        const BITS: [(u8, &str); 6] = [
            (0x01, "spare capacity below threshold"),
            (0x02, "temperature past a critical threshold"),
            (0x04, "internal reliability degraded"),
            (0x08, "media placed in read-only mode"),
            (0x10, "volatile memory backup failed"),
            (0x20, "persistent memory region unreliable"),
        ];
        BITS.iter()
            .filter(|(b, _)| self.critical_warning & b != 0)
            .map(|(_, s)| *s)
            .collect()
    }

    pub fn line(&self) -> String {
        let mut s = format!(
            "disk{}: {} °C, {}% endurance used, {}% spare",
            self.index, self.temp_c, self.percentage_used, self.available_spare
        );
        if self.power_on_hours > 0 {
            s.push_str(&format!(", {} h powered", self.power_on_hours));
        }
        if self.unsafe_shutdowns > 0 {
            s.push_str(&format!(", {} unsafe shutdown(s)", self.unsafe_shutdowns));
        }
        if self.media_errors > 0 {
            s.push_str(&format!(", {} MEDIA ERROR(S)", self.media_errors));
        }
        let w = self.warnings();
        if !w.is_empty() {
            s.push_str(&format!("  CRITICAL: {}", w.join("; ")));
        }
        s
    }

    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("index", Json::U64(self.index as u64))
            .push("temp_c", Json::I64(self.temp_c as i64))
            .push(
                "sensors_c",
                Json::Array(self.sensors_c.iter().map(|c| Json::I64(*c as i64)).collect()),
            )
            .push("percentage_used", Json::U64(self.percentage_used as u64))
            .push("available_spare", Json::U64(self.available_spare as u64))
            .push("power_on_hours", Json::U64(self.power_on_hours))
            .push("unsafe_shutdowns", Json::U64(self.unsafe_shutdowns))
            .push("media_errors", Json::U64(self.media_errors))
            .push(
                "critical_warnings",
                Json::Array(self.warnings().into_iter().map(Json::str).collect()),
            );
        o
    }
}

/// Read the health log for every NVMe drive the machine will let us open.
///
/// Drives that refuse are skipped rather than reported as healthy — a QC tool
/// that silently omits the drive it could not read is worse than one that says
/// it read nothing.
pub fn scan(max_drives: u32) -> Vec<NvmeHealth> {
    #[cfg(windows)]
    {
        (0..max_drives).filter_map(win::health).collect()
    }
    #[cfg(not(windows))]
    {
        let _ = max_drives;
        Vec::new()
    }
}

/// Parse a 512-byte NVMe SMART / Health Information log page.
///
/// Split out from the I/O so the field offsets — the part that is easy to get
/// wrong and impossible to notice — can be tested against a synthetic page.
pub fn parse_health_log(index: u32, log: &[u8]) -> Option<NvmeHealth> {
    if log.len() < 512 {
        return None;
    }
    let u16le = |o: usize| u16::from_le_bytes([log[o], log[o + 1]]);
    // The log stores 128-bit counters; the low 64 bits are far beyond any real
    // value and avoid needing a u128 in the report.
    let u64le = |o: usize| {
        let mut v = [0u8; 8];
        v.copy_from_slice(&log[o..o + 8]);
        u64::from_le_bytes(v)
    };
    // Kelvin -> °C. A zero here means "not reported", not −273 °C.
    let to_c = |k: u16| -> Option<i32> {
        if k == 0 {
            None
        } else {
            Some(k as i32 - 273)
        }
    };

    let temp_c = to_c(u16le(1))?;
    let mut sensors_c = Vec::new();
    for i in 0..8 {
        if let Some(c) = to_c(u16le(200 + i * 2)) {
            sensors_c.push(c);
        }
    }

    Some(NvmeHealth {
        index,
        temp_c,
        sensors_c,
        critical_warning: log[0],
        available_spare: log[3],
        percentage_used: log[5],
        power_on_hours: u64le(128),
        unsafe_shutdowns: u64le(144),
        media_errors: u64le(160),
    })
}

#[cfg(windows)]
mod win {
    use super::{parse_health_log, NvmeHealth};
    use core::ffi::c_void;

    const INVALID_HANDLE: isize = -1;
    const OPEN_EXISTING: u32 = 3;
    const FILE_SHARE_READ: u32 = 1;
    const FILE_SHARE_WRITE: u32 = 2;

    /// `CTL_CODE(IOCTL_STORAGE_BASE 0x2d, 0x0500, METHOD_BUFFERED, FILE_ANY_ACCESS)`.
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;
    /// `StorageDeviceProtocolSpecificProperty`
    const PROPERTY_ID: u32 = 50;
    /// `PropertyStandardQuery`
    const QUERY_TYPE: u32 = 0;
    /// `ProtocolTypeNvme`
    const PROTOCOL_NVME: u32 = 3;
    /// `NVMeDataTypeLogPage`
    const DATA_TYPE_LOG: u32 = 2;
    /// SMART / Health Information log page.
    const LOG_HEALTH: u32 = 0x02;

    const SPECIFIC_DATA_LEN: usize = 40;
    const LOG_LEN: usize = 512;
    /// `STORAGE_PROPERTY_QUERY` header is two DWORDs before the protocol block.
    const QUERY_HEADER: usize = 8;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: isize,
        ) -> isize;
        fn DeviceIoControl(
            device: isize,
            code: u32,
            in_buf: *mut c_void,
            in_size: u32,
            out_buf: *mut c_void,
            out_size: u32,
            returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        fn CloseHandle(h: isize) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Open a physical drive for a query-only IOCTL.
    ///
    /// `IOCTL_STORAGE_QUERY_PROPERTY` is `FILE_ANY_ACCESS`, so **zero** desired
    /// access is enough — and zero access does not require Administrator, which
    /// is why drive health can be read on a technician's normal session. Falls
    /// back to `GENERIC_READ` for drivers that insist on it.
    fn open_drive(index: u32) -> Option<isize> {
        const GENERIC_READ: u32 = 0x8000_0000;
        for access in [0u32, GENERIC_READ] {
            let path = wide(&format!(r"\\.\PhysicalDrive{index}"));
            // SAFETY: NUL-terminated path; all other arguments are scalars.
            let h = unsafe {
                CreateFileW(
                    path.as_ptr(),
                    access,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    std::ptr::null_mut(),
                    OPEN_EXISTING,
                    0,
                    0,
                )
            };
            if h != INVALID_HANDLE {
                return Some(h);
            }
        }
        None
    }

    pub(super) fn health(index: u32) -> Option<NvmeHealth> {
        let h = open_drive(index)?;

        // One buffer serves as both input and output: the query header, the
        // protocol-specific block, then room for the log page itself.
        let mut buf = vec![0u8; QUERY_HEADER + SPECIFIC_DATA_LEN + LOG_LEN];
        let put = |b: &mut [u8], off: usize, v: u32| {
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut buf, 0, PROPERTY_ID);
        put(&mut buf, 4, QUERY_TYPE);
        // STORAGE_PROTOCOL_SPECIFIC_DATA, at the end of the query header.
        let p = QUERY_HEADER;
        put(&mut buf, p, PROTOCOL_NVME);
        put(&mut buf, p + 4, DATA_TYPE_LOG);
        put(&mut buf, p + 8, LOG_HEALTH);
        put(&mut buf, p + 12, 0); // request sub-value
        // Offsets are relative to the start of the protocol block, so the log
        // lands immediately after it.
        put(&mut buf, p + 16, SPECIFIC_DATA_LEN as u32);
        put(&mut buf, p + 20, LOG_LEN as u32);

        let mut returned: u32 = 0;
        // SAFETY: one buffer used for both directions, sizes match its length,
        // and `returned` is a valid out-param.
        let ok = unsafe {
            DeviceIoControl(
                h,
                IOCTL_STORAGE_QUERY_PROPERTY,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as u32,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as u32,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        // SAFETY: opened above, closed exactly once.
        unsafe { CloseHandle(h) };
        if ok == 0 {
            return None; // not NVMe, or the driver declined
        }

        // On return the buffer holds STORAGE_PROTOCOL_DATA_DESCRIPTOR: two
        // DWORDs, then the protocol block echoed back with the offset at which
        // it actually placed the data.
        let off = u32::from_le_bytes([buf[p + 16], buf[p + 17], buf[p + 18], buf[p + 19]]) as usize;
        let len = u32::from_le_bytes([buf[p + 20], buf[p + 21], buf[p + 22], buf[p + 23]]) as usize;
        let start = QUERY_HEADER + off;
        if len < LOG_LEN || start + LOG_LEN > buf.len() {
            return None;
        }
        parse_health_log(index, &buf[start..start + LOG_LEN])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic health page with known values at the spec's offsets.
    fn page() -> Vec<u8> {
        let mut p = vec![0u8; 512];
        p[0] = 0x02; // critical warning: temperature threshold
        // Composite temperature: 336 K = 63 °C, the reading this bench's
        // 980 PRO actually returns.
        p[1..3].copy_from_slice(&336u16.to_le_bytes());
        p[3] = 99; // available spare
        p[5] = 4; // percentage used
        p[128..136].copy_from_slice(&12_345u64.to_le_bytes()); // power-on hours
        p[144..152].copy_from_slice(&7u64.to_le_bytes()); // unsafe shutdowns
        p[160..168].copy_from_slice(&2u64.to_le_bytes()); // media errors
        p[200..202].copy_from_slice(&340u16.to_le_bytes()); // sensor 1 = 67 °C
        p[202..204].copy_from_slice(&0u16.to_le_bytes()); // sensor 2 unpopulated
        p
    }

    #[test]
    fn every_field_lands_at_the_spec_offset() {
        let h = parse_health_log(0, &page()).expect("a full page parses");
        assert_eq!(h.temp_c, 63, "composite temperature is Kelvin at byte 1");
        assert_eq!(h.available_spare, 99);
        assert_eq!(h.percentage_used, 4);
        assert_eq!(h.power_on_hours, 12_345);
        assert_eq!(h.unsafe_shutdowns, 7);
        assert_eq!(h.media_errors, 2);
        assert_eq!(h.critical_warning, 0x02);
    }

    #[test]
    fn unpopulated_sensors_are_omitted_not_reported_as_minus_273() {
        // A zero Kelvin sensor means "this drive has no sensor here". Converting
        // it would put −273 °C on a chart.
        let h = parse_health_log(0, &page()).unwrap();
        assert_eq!(h.sensors_c, vec![67], "only the populated sensor survives");
    }

    #[test]
    fn critical_warning_bits_are_decoded_to_words() {
        let h = parse_health_log(0, &page()).unwrap();
        let w = h.warnings();
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("temperature"), "{w:?}");
        // And a clean drive says nothing.
        let mut clean = page();
        clean[0] = 0;
        assert!(parse_health_log(0, &clean).unwrap().warnings().is_empty());
    }

    #[test]
    fn a_short_or_empty_page_is_declined_rather_than_guessed() {
        assert!(parse_health_log(0, &[]).is_none());
        assert!(parse_health_log(0, &vec![0u8; 100]).is_none());
        // A full page whose composite temperature is unset is not a 0 °C drive.
        assert!(parse_health_log(0, &vec![0u8; 512]).is_none());
    }

    #[test]
    fn the_line_leads_with_what_a_technician_looks_for() {
        let h = parse_health_log(1, &page()).unwrap();
        let l = h.line();
        assert!(l.starts_with("disk1: 63 °C"), "{l}");
        assert!(l.contains("4% endurance used"), "{l}");
        assert!(l.contains("MEDIA ERROR"), "media errors must shout: {l}");
        assert!(l.contains("CRITICAL"), "{l}");
    }

    #[test]
    fn this_bench_reports_a_plausible_drive_or_none() {
        // Whatever is fitted, a temperature must be a temperature. This is the
        // check that would catch a Kelvin conversion left undone.
        for d in scan(4) {
            assert!(
                (0..120).contains(&d.temp_c),
                "implausible NVMe temperature {} °C on disk{}",
                d.temp_c,
                d.index
            );
            for s in &d.sensors_c {
                assert!((0..120).contains(s), "implausible sensor {s} °C");
            }
        }
    }
}
