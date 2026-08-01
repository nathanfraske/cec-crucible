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
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
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
        /// Enumerate every channel registered on the machine.
        fn EvtOpenChannelEnum(session: isize, flags: u32) -> isize;
        fn EvtNextChannelPath(
            channel_enum: isize,
            path_buffer_size: u32,
            path_buffer: *mut u16,
            path_buffer_used: *mut u32,
        ) -> i32;
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
    // -----------------------------------------------------------------------
    // Full capture
    // -----------------------------------------------------------------------

    /// A machine registers on the order of a thousand channels; this only bounds
    /// a pathological enumeration.
    const MAX_CHANNELS: usize = 4096;
    /// Per-channel event cap, so one chatty provider cannot fill the archive and
    /// crowd out the channel that mattered.
    const MAX_PER_CHANNEL: usize = 512;

    /// Every channel path the machine will name.
    fn channel_paths() -> Vec<String> {
        let mut out = Vec::new();
        // SAFETY: null session = local machine.
        let e = unsafe { EvtOpenChannelEnum(0, 0) };
        if e == 0 {
            return out;
        }
        let mut buf: Vec<u16> = vec![0; 512];
        while out.len() < MAX_CHANNELS {
            let mut used: u32 = 0;
            // SAFETY: buffer and length agree; `used` is a valid out-param.
            let ok =
                unsafe { EvtNextChannelPath(e, buf.len() as u32, buf.as_mut_ptr(), &mut used) };
            if ok == 0 {
                // SAFETY: trivially safe.
                let err = unsafe { GetLastError() };
                if err == ERROR_INSUFFICIENT_BUFFER && used as usize > buf.len() {
                    buf.resize(used as usize, 0);
                    continue; // retry the same channel with room for its name
                }
                break; // ERROR_NO_MORE_ITEMS, or something unrecoverable
            }
            let chars = (used as usize).saturating_sub(1).min(buf.len());
            out.push(String::from_utf16_lossy(&buf[..chars]));
        }
        // SAFETY: handle came from EvtOpenChannelEnum and is closed once.
        unsafe { EvtClose(e) };
        out
    }

    pub(super) fn capture_all(window_ms: u64, max_records: usize) -> super::FullLog {
        use super::{FullLog, RawEvent};

        let mut out = FullLog::default();
        let channels = channel_paths();
        out.channels_total = channels.len();

        let query = format!("*[System[TimeCreated[timediff(@SystemTime) <= {window_ms}]]]");
        let q = wide(&query);
        let mut buf: Vec<u16> = vec![0; 16 * 1024];

        for channel in channels {
            if out.records.len() >= max_records {
                out.truncated = true;
                break;
            }
            let path = wide(&channel);
            // SAFETY: null session = local machine; both strings NUL-terminated.
            let results = unsafe {
                EvtQuery(
                    0,
                    path.as_ptr(),
                    q.as_ptr(),
                    EVT_QUERY_CHANNEL_PATH | EVT_QUERY_REVERSE_DIRECTION,
                )
            };
            if results == 0 {
                // Disabled, empty, or access-denied. Counted, never fatal: most
                // of a Windows machine's channels are simply not enabled, and
                // Security needs elevation.
                out.channels_denied += 1;
                continue;
            }
            out.channels_read += 1;

            let mut from_channel = 0usize;
            'chan: while from_channel < MAX_PER_CHANNEL {
                let mut handles = [0isize; 32];
                let mut returned: u32 = 0;
                // SAFETY: valid out-array of the stated length.
                let ok = unsafe {
                    EvtNext(
                        results,
                        handles.len() as u32,
                        handles.as_mut_ptr(),
                        1000,
                        0,
                        &mut returned,
                    )
                };
                if ok == 0 || returned == 0 {
                    break;
                }
                for &h in handles.iter().take(returned as usize) {
                    let mut used: u32 = 0;
                    let mut props: u32 = 0;
                    // SAFETY: rendering into a buffer we own; sizes in bytes.
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
                    // SAFETY: handle came from EvtNext, closed exactly once.
                    unsafe { EvtClose(h) };
                    if rendered == 0 {
                        continue;
                    }
                    let chars = (used as usize / 2).saturating_sub(1).min(buf.len());
                    let xml = String::from_utf16_lossy(&buf[..chars]);

                    let provider = attr(&xml, "Name").unwrap_or("").to_string();
                    let event_id: u32 = between(&xml, "<EventID", "</EventID>")
                        .and_then(|s| s.rsplit('>').next())
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                    let time = attr(&xml, "SystemTime").unwrap_or("").to_string();
                    let level: u32 = between(&xml, "<Level>", "</Level>")
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);

                    let mut data = String::new();
                    let mut rest = xml.as_str();
                    while let Some(v) = between(rest, "<Data", "</Data>") {
                        if let Some(inner) = v.split_once('>') {
                            let t = inner.1.trim();
                            if !t.is_empty() && data.len() < 400 {
                                if !data.is_empty() {
                                    data.push_str("; ");
                                }
                                data.push_str(t);
                            }
                        }
                        let adv = rest.find("</Data>").map(|i| i + 7).unwrap_or(rest.len());
                        rest = &rest[adv..];
                    }

                    out.records.push(RawEvent {
                        channel: channel.clone(),
                        time,
                        provider,
                        event_id,
                        level,
                        data,
                    });
                    from_channel += 1;
                    if out.records.len() >= max_records {
                        out.truncated = true;
                        break 'chan;
                    }
                }
            }
            // SAFETY: result set came from EvtQuery, closed exactly once.
            unsafe { EvtClose(results) };
        }

        // Newest-first per channel from the query; present oldest-first overall
        // so the archive reads as a timeline.
        out.records.sort_by(|a, b| a.time.cmp(&b.time));
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

// ---------------------------------------------------------------------------
// Full capture — every channel, not a curated list
// ---------------------------------------------------------------------------

/// One event, kept verbatim rather than classified.
///
/// [`EventRecord`] is the *detector* plane: it keeps only what it recognises, so
/// it can decide a verdict. This is the *archive* plane, and it keeps everything
/// — including the provider nobody has written a classifier for yet, which is
/// exactly the one that turns out to matter when a machine fails in a new way.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEvent {
    pub channel: String,
    pub time: String,
    pub provider: String,
    pub event_id: u32,
    pub level: u32,
    pub data: String,
}

impl RawEvent {
    /// One JSON object per line — the archive format. JSONL rather than one
    /// large document so a capture that is interrupted is still readable up to
    /// the point it stopped, and so `findstr`/`grep` works on it directly.
    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("channel", Json::str(&self.channel))
            .push("time", Json::str(&self.time))
            .push("provider", Json::str(&self.provider))
            .push("event_id", Json::U64(self.event_id as u64))
            .push("level", Json::U64(self.level as u64))
            .push("data", Json::str(&self.data));
        o
    }
}

/// The result of a full capture, including what it could *not* read.
#[derive(Debug, Clone, Default)]
pub struct FullLog {
    pub channels_total: usize,
    /// Channels that answered a query — most machines have hundreds registered
    /// and only a fraction enabled.
    pub channels_read: usize,
    /// Channels that refused. Overwhelmingly "disabled" or "needs elevation"
    /// (Security is the notable one), not errors worth failing over — but the
    /// count is reported, because "we read everything" and "we read what we were
    /// allowed to" are different claims.
    pub channels_denied: usize,
    pub records: Vec<RawEvent>,
    /// The cap was hit and events were dropped. A capture that silently
    /// truncates reads as a quiet log.
    pub truncated: bool,
}

impl FullLog {
    pub fn summary(&self) -> String {
        format!(
            "{} event(s) from {}/{} channel(s){}{}",
            self.records.len(),
            self.channels_read,
            self.channels_total,
            if self.channels_denied > 0 {
                format!(", {} not readable", self.channels_denied)
            } else {
                String::new()
            },
            if self.truncated { " (TRUNCATED)" } else { "" }
        )
    }

    /// Render the archive as JSONL.
    pub fn to_jsonl(&self) -> String {
        let mut s = String::new();
        for r in &self.records {
            s.push_str(&r.to_json().to_compact());
            s.push('\n');
        }
        s
    }
}

/// Capture **every** event the machine logged in the last `window_ms`, across
/// every channel it will let us read.
///
/// The curated [`scan_system_log`] decides the verdict; this decides whether the
/// evidence still exists tomorrow. A QC tool that reports "WHEA: clean" and
/// keeps nothing else has thrown away the context that explains the failure
/// nobody predicted.
///
/// `max_records` bounds the archive; hitting it sets `truncated`, because a
/// silently-capped capture reads exactly like a quiet machine.
pub fn capture_all_channels(window_ms: u64, max_records: usize) -> FullLog {
    #[cfg(windows)]
    {
        win::capture_all(window_ms, max_records)
    }
    #[cfg(not(windows))]
    {
        let _ = (window_ms, max_records);
        FullLog::default()
    }
}
