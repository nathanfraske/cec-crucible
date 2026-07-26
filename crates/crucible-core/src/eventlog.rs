// SPDX-License-Identifier: MIT
//! Windows event-log detector plane — WHEA, TDR, bugchecks and disk resets.
//!
//! **This is the only plane that can see faults the hardware already handled.**
//! A corrected machine-check is, by definition, invisible to every checksum we
//! compute: the data came back right *because* the hardware fixed it. On AMD the
//! Infinity Fabric carries ECC and logs a corrected error as WHEA event 19 — so a
//! box running a marginal FCLK / SoC voltage can pass every integrity test in
//! this suite while quietly correcting errors the whole time. That machine is not
//! stable, and this is how we find out.
//!
//! Scanning is **on by default**: a detector that ships switched off catches
//! nothing. It is bracketed to the run window, so pre-existing events in the log
//! are never attributed to us.
//!
//! Non-admin: the System channel is readable by authenticated users (unlike
//! Security). No driver, no elevation, no external tool — hand-rolled `wevtapi`
//! FFI in the same style as the SMBIOS, PDH and QPC code elsewhere in this crate.
//! Every failure path degrades to "scan unavailable", which is reported as such
//! rather than being silently treated as "nothing found".

use crate::json::Json;

/// One event of interest found inside the run window.
#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    /// ISO-8601 UTC timestamp as Windows recorded it.
    pub time: String,
    /// Publisher, e.g. `Microsoft-Windows-WHEA-Logger`.
    pub provider: String,
    pub event_id: u32,
    /// Windows level: 1=Critical 2=Error 3=Warning 4=Information.
    pub level: u32,
    /// Flattened `<Data>` payload — for WHEA this carries the error source and
    /// component; truncated so a report stays readable.
    pub data: String,
    /// What this event means for the verdict.
    pub severity: Severity,
    /// Plain-English reading for the technician.
    pub meaning: &'static str,
}

/// How an event should weigh on the run's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Hardware error — the run cannot be called a pass.
    Fail,
    /// Real, but not necessarily this run's fault (e.g. a bugcheck logged at the
    /// last boot, a disk retry). Surfaced prominently, does not fail the run.
    Warn,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Fail => "fail",
            Severity::Warn => "warn",
        }
    }
}

/// Classify an event we pulled from the System channel. Returns `None` for the
/// overwhelming majority of log traffic, which we do not care about.
///
/// The WHEA identifiers are the load-bearing ones:
/// * **17** — corrected hardware error (the machine fixed it; you would never
///   otherwise know).
/// * **18** — uncorrectable / fatal hardware error.
/// * **19** — corrected *bus/interconnect* error. On Ryzen this is the Infinity
///   Fabric ECC correction that marginal FCLK / SoC voltage produces.
/// * **47** — corrected memory error with a PFN threshold exceeded.
fn classify(provider: &str, id: u32) -> Option<(Severity, &'static str)> {
    let p = provider.to_ascii_lowercase();
    if p.contains("whea") {
        return Some(match id {
            17 => (Severity::Fail, "corrected hardware error — the hardware fixed it, so no checksum could ever have caught this"),
            18 => (Severity::Fail, "UNCORRECTABLE hardware error"),
            19 => (Severity::Fail, "corrected bus/interconnect error — on AMD typically Infinity Fabric ECC (suspect FCLK / SoC voltage / memory training); on Intel the ring/uncore or a PCIe link"),
            47 => (Severity::Fail, "corrected memory error, page-retirement threshold exceeded — suspect a failing DIMM"),
            _  => (Severity::Fail, "WHEA hardware error"),
        });
    }
    // Display driver timeout & recovery. The provider is the display miniport
    // (nvlddmkm / amdkmdap / igfxn) or the generic "Display" channel.
    if id == 4101
        && (p.contains("display")
            || p.contains("nvlddmkm")
            || p.contains("amdkmdap")
            || p.contains("igfx"))
    {
        return Some((
            Severity::Fail,
            "TDR — the display driver stopped responding and was reset (this is what a GPU 'crash to desktop' looks like)",
        ));
    }
    if p.contains("kernel-power") && id == 41 {
        return Some((
            Severity::Warn,
            "system rebooted without a clean shutdown — power loss, hard hang, or PSU protection trip",
        ));
    }
    if p.contains("bugcheck") && id == 1001 {
        return Some((Severity::Warn, "bugcheck (BSOD) was recorded"));
    }
    // Storage stack: 153 = I/O retried, 129 = controller reset. Both are how a
    // marginal NVMe or a thermal-throttling controller shows up.
    if (p.contains("disk") || p.contains("storahci") || p.contains("stornvme"))
        && (id == 153 || id == 129)
    {
        return Some((
            Severity::Warn,
            "storage I/O was retried or the controller was reset — suspect a marginal or overheating drive",
        ));
    }
    None
}

/// Result of a scan. `available == false` means we could not read the log at all,
/// which is explicitly NOT the same as "no events found" — a run whose detector
/// plane was unavailable must not be reported as a clean pass on that basis.
#[derive(Debug, Clone, Default)]
pub struct EventScan {
    pub available: bool,
    pub unavailable_reason: String,
    pub events: Vec<EventRecord>,
}

impl EventScan {
    /// Events that should stop the run being called a pass.
    pub fn failing(&self) -> impl Iterator<Item = &EventRecord> {
        self.events.iter().filter(|e| e.severity == Severity::Fail)
    }

    pub fn fail_count(&self) -> usize {
        self.failing().count()
    }

    pub fn warn_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.severity == Severity::Warn)
            .count()
    }

    /// Report JSON.
    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("available", Json::Bool(self.available));
        if !self.available {
            o.push("unavailable_reason", Json::str(&self.unavailable_reason));
        }
        o.push("fail_count", Json::U64(self.fail_count() as u64));
        o.push("warn_count", Json::U64(self.warn_count() as u64));
        let items: Vec<Json> = self
            .events
            .iter()
            .map(|e| {
                let mut j = Json::object();
                j.push("time", Json::str(&e.time));
                j.push("provider", Json::str(&e.provider));
                j.push("event_id", Json::U64(e.event_id as u64));
                j.push("level", Json::U64(e.level as u64));
                j.push("severity", Json::str(e.severity.as_str()));
                j.push("meaning", Json::str(e.meaning));
                j.push("data", Json::str(&e.data));
                j
            })
            .collect();
        o.push("events", Json::Array(items));
        o
    }
}

/// Scan the System channel for events of interest in the last `window_ms`.
/// Never panics; any failure yields `available: false` with a reason.
pub fn scan_system_log(window_ms: u64) -> EventScan {
    #[cfg(windows)]
    {
        win::scan(window_ms)
    }
    #[cfg(not(windows))]
    {
        let _ = window_ms;
        EventScan {
            available: false,
            unavailable_reason: "event-log scanning is Windows-only".to_string(),
            events: Vec::new(),
        }
    }
}

#[cfg(windows)]
mod win {
    use super::{classify, EventRecord, EventScan};
    use core::ffi::c_void;

    const ERROR_NO_MORE_ITEMS: u32 = 259;
    /// `EvtQueryChannelPath | EvtQueryReverseDirection` — newest first, so a busy
    /// log cannot push our window off the end of a capped result set.
    const EVT_QUERY_CHANNEL_PATH: u32 = 0x1;
    const EVT_QUERY_REVERSE_DIRECTION: u32 = 0x200;
    /// `EvtRenderEventXml`
    const EVT_RENDER_EVENT_XML: u32 = 1;

    /// Hard cap so a pathological log cannot make a run hang or balloon a report.
    const MAX_EVENTS_SCANNED: usize = 4096;
    const MAX_EVENTS_KEPT: usize = 64;

    #[link(name = "wevtapi")]
    extern "system" {
        fn EvtQuery(
            session: isize,
            path: *const u16,
            query: *const u16,
            flags: u32,
        ) -> isize;
        fn EvtNext(
            result_set: isize,
            events_size: u32,
            events: *mut isize,
            timeout: u32,
            flags: u32,
            returned: *mut u32,
        ) -> i32;
        fn EvtRender(
            context: isize,
            fragment: isize,
            flags: u32,
            buffer_size: u32,
            buffer: *mut c_void,
            buffer_used: *mut u32,
            property_count: *mut u32,
        ) -> i32;
        fn EvtClose(object: isize) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetLastError() -> u32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Extract the text between `open` and `close`, starting the search at `from`.
    fn between<'a>(hay: &'a str, open: &str, close: &str) -> Option<&'a str> {
        let s = hay.find(open)? + open.len();
        let e = hay[s..].find(close)? + s;
        Some(&hay[s..e])
    }

    /// Value of an XML attribute, e.g. `Name='Microsoft-...'`.
    fn attr<'a>(hay: &'a str, name: &str) -> Option<&'a str> {
        let key = format!("{name}='");
        let s = hay.find(&key)? + key.len();
        let e = hay[s..].find('\'')? + s;
        Some(&hay[s..e])
    }

    pub(super) fn scan(window_ms: u64) -> EventScan {
        let mut out = EventScan {
            available: false,
            unavailable_reason: String::new(),
            events: Vec::new(),
        };

        // `timediff` is evaluated by the query engine against "now", which is
        // exactly the run window we want and avoids any timezone handling.
        let query = format!("*[System[TimeCreated[timediff(@SystemTime) <= {window_ms}]]]");
        let path = wide("System");
        let q = wide(&query);

        // SAFETY: null session = local machine; both strings are NUL-terminated.
        let results = unsafe {
            EvtQuery(
                0,
                path.as_ptr(),
                q.as_ptr(),
                EVT_QUERY_CHANNEL_PATH | EVT_QUERY_REVERSE_DIRECTION,
            )
        };
        if results == 0 {
            // SAFETY: trivially safe.
            let e = unsafe { GetLastError() };
            out.unavailable_reason = format!("EvtQuery failed (error {e})");
            return out;
        }

        out.available = true;
        let mut scanned = 0usize;
        let mut buf: Vec<u16> = vec![0; 16 * 1024];

        'outer: while scanned < MAX_EVENTS_SCANNED {
            let mut handles = [0isize; 32];
            let mut returned: u32 = 0;
            // SAFETY: `handles` is a valid out-array of the stated length.
            let ok = unsafe {
                EvtNext(
                    results,
                    handles.len() as u32,
                    handles.as_mut_ptr(),
                    2000,
                    0,
                    &mut returned,
                )
            };
            if ok == 0 {
                // SAFETY: trivially safe.
                let e = unsafe { GetLastError() };
                if e != ERROR_NO_MORE_ITEMS {
                    // Partial results are still worth reporting; note the cause.
                    out.unavailable_reason = format!("EvtNext stopped early (error {e})");
                }
                break;
            }

            for &h in handles.iter().take(returned as usize) {
                scanned += 1;
                let mut used: u32 = 0;
                let mut props: u32 = 0;
                // SAFETY: rendering into a byte buffer we own; sizes are in bytes.
                let rendered = unsafe {
                    EvtRender(
                        0,
                        h,
                        EVT_RENDER_EVENT_XML,
                        (buf.len() * 2) as u32,
                        buf.as_mut_ptr() as *mut c_void,
                        &mut used,
                        &mut props,
                    )
                };
                // SAFETY: handle came from EvtNext and is closed exactly once.
                unsafe { EvtClose(h) };
                if rendered == 0 {
                    continue;
                }
                let chars = (used as usize / 2).saturating_sub(1).min(buf.len());
                let xml = String::from_utf16_lossy(&buf[..chars]);

                let provider = attr(&xml, "Name").unwrap_or("").to_string();
                let id: u32 = between(&xml, "<EventID", "</EventID>")
                    .and_then(|s| s.rsplit('>').next())
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                let Some((severity, meaning)) = classify(&provider, id) else {
                    continue;
                };

                let time = attr(&xml, "SystemTime").unwrap_or("").to_string();
                let level: u32 = between(&xml, "<Level>", "</Level>")
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                // Flatten the <Data> payload; for WHEA this names the error source.
                let mut data = String::new();
                let mut rest = xml.as_str();
                while let Some(v) = between(rest, "<Data", "</Data>") {
                    if let Some(inner) = v.split_once('>') {
                        let t = inner.1.trim();
                        if !t.is_empty() && data.len() < 240 {
                            if !data.is_empty() {
                                data.push_str("; ");
                            }
                            data.push_str(t);
                        }
                    }
                    let adv = rest.find("</Data>").map(|i| i + 7).unwrap_or(rest.len());
                    rest = &rest[adv..];
                }

                out.events.push(EventRecord {
                    time,
                    provider,
                    event_id: id,
                    level,
                    data,
                    severity,
                    meaning,
                });
                if out.events.len() >= MAX_EVENTS_KEPT {
                    break 'outer;
                }
            }
            if returned == 0 {
                break;
            }
        }

        // SAFETY: result set came from EvtQuery and is closed exactly once.
        unsafe { EvtClose(results) };
        // Newest-first from the query; present oldest-first for reading.
        out.events.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whea_ids_are_failures_with_the_right_reading() {
        for id in [17u32, 18, 19, 47] {
            let (sev, why) = classify("Microsoft-Windows-WHEA-Logger", id).expect("classified");
            assert_eq!(sev, Severity::Fail, "WHEA {id} must fail the run");
            assert!(!why.is_empty());
        }
        // 19 is the one that names Infinity Fabric — the AMD FCLK signature.
        let (_, why) = classify("Microsoft-Windows-WHEA-Logger", 19).unwrap();
        assert!(why.contains("Infinity Fabric"), "{why}");
    }

    #[test]
    fn tdr_is_a_failure_and_ordinary_noise_is_ignored() {
        assert_eq!(classify("Display", 4101).unwrap().0, Severity::Fail);
        assert_eq!(classify("nvlddmkm", 4101).unwrap().0, Severity::Fail);
        // A 4101 from an unrelated provider is not a TDR.
        assert!(classify("Microsoft-Windows-Winlogon", 4101).is_none());
        // Routine log traffic must not be picked up.
        assert!(classify("Service Control Manager", 7040).is_none());
        assert!(classify("Microsoft-Windows-Kernel-General", 12).is_none());
    }

    #[test]
    fn power_and_storage_events_warn_but_do_not_fail() {
        assert_eq!(
            classify("Microsoft-Windows-Kernel-Power", 41).unwrap().0,
            Severity::Warn
        );
        assert_eq!(classify("disk", 153).unwrap().0, Severity::Warn);
        assert_eq!(classify("stornvme", 129).unwrap().0, Severity::Warn);
    }

    #[test]
    fn unavailable_is_distinguishable_from_clean() {
        let clean = EventScan {
            available: true,
            ..Default::default()
        };
        let broken = EventScan {
            available: false,
            unavailable_reason: "EvtQuery failed (error 5)".into(),
            ..Default::default()
        };
        assert_eq!(clean.fail_count(), 0);
        assert_eq!(broken.fail_count(), 0);
        // Same fail count — so callers MUST branch on `available`, which is the
        // whole point of keeping the flag.
        assert!(clean.available && !broken.available);
    }

    #[test]
    fn scan_does_not_panic_on_this_platform() {
        // Windows: exercises the real FFI. Elsewhere: the stub. Either way the
        // contract is "never panics, and says so when it cannot read".
        let s = scan_system_log(60_000);
        if !s.available {
            assert!(!s.unavailable_reason.is_empty());
        }
    }
}
