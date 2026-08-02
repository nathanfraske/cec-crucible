// SPDX-License-Identifier: MIT
//! The device-identified run report.
//!
//! A stress pass that does not crash but produced a wrong result is still a
//! FAIL — so the verdict rolls up per-kernel error counts, not just liveness.
//! The report also carries the Windows event-log detector plane (see
//! [`crate::eventlog`]): WHEA machine-checks, display TDRs, bugchecks and disk
//! resets inside the run window. A logged hardware error FAILS the run even when
//! every checksum matched — a *corrected* machine-check means the hardware fixed
//! something silently, which is exactly the fault class our own verification can
//! never observe.

use std::io;
use std::path::Path;

use crate::clock::{Clock, Timestamp};
use crate::device::DeviceId;
use crate::eventlog::EventScan;
use crate::json::Json;
use crate::kernel::{Kind, LoadResult};

/// Overall pass/fail state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// Ran clean but was cut short (operator stop / Ctrl-C) before completing.
    Partial,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Partial => "PARTIAL",
        }
    }
}

/// Result of a single kernel stage within a run.
#[derive(Debug, Clone)]
pub struct StageReport {
    pub kernel: String,
    pub kind: Kind,
    pub mode: String,
    pub seconds: f64,
    pub result: LoadResult,
}

impl StageReport {
    pub fn new(
        kernel: impl Into<String>,
        kind: Kind,
        mode: impl Into<String>,
        seconds: f64,
        result: LoadResult,
    ) -> StageReport {
        StageReport {
            kernel: kernel.into(),
            kind,
            mode: mode.into(),
            seconds,
            result,
        }
    }

    pub fn verdict(&self) -> Verdict {
        if self.result.passed() {
            Verdict::Pass
        } else {
            Verdict::Fail
        }
    }

    fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("kernel", self.kernel.as_str())
            .push("kind", self.kind.as_str())
            .push("mode", self.mode.as_str())
            .push("seconds", round2(self.seconds))
            .push("verdict", self.verdict().as_str())
            .push("ok", self.result.ok)
            .push("iterations", self.result.iterations)
            .push("checksum", self.result.checksum)
            .push("error_count", self.result.error_count)
            .push("detail", self.result.detail.as_str());
        o
    }
}

/// Present-pipeline summary from an ETW capture (PresentMon), when one ran.
/// Carried in the report so the *displayed* frame behaviour — which the app
/// cannot self-measure — lands in the JSON and the results CSV alongside
/// everything else, instead of only in a side file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PresentSummary {
    pub frames: u64,
    pub avg_fps: f64,
    pub low1pct_fps: f64,
    pub stutter_ms: f64,
    pub gpu_busy_ms: f64,
    pub cpu_busy_ms: f64,
    pub cpu_wait_ms: f64,
    pub display_latency_ms: f64,
    pub dropped: u64,
}

/// An ETW trace record, kept as pre-rendered JSON plus a console line so the
/// core crate does not have to know anything about Windows Performance Recorder
/// — the CLI owns that, and the report only carries the result.
#[derive(Debug, Clone)]
pub struct EtwRecord {
    pub available: bool,
    pub path: String,
    pub line: String,
    pub json: Json,
}

/// A complete, device-identified run report.
#[derive(Debug, Clone)]
pub struct Report {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    pub device: DeviceId,
    pub os: String,
    pub arch: String,
    pub qpc_frequency: u64,
    pub qpc_backed: bool,
    pub started: Option<Timestamp>,
    pub ended: Option<Timestamp>,
    pub stages: Vec<StageReport>,
    pub aborted: bool,
    pub markers_file: Option<String>,
    pub notes: Vec<String>,
    /// Windows event-log detector plane (WHEA / TDR / bugcheck / disk resets)
    /// bracketed to this run's window. `None` when the scan was not attempted.
    /// This is the only plane that sees faults the hardware already corrected —
    /// no checksum can, by definition.
    pub events: Option<EventScan>,
    /// ETW present-pipeline summary (`--presentmon`), when a capture succeeded.
    pub present: Option<PresentSummary>,
    /// GPU sensor summary (NVML): peak power, peak temps, throttle reasons.
    pub gpu: Option<crate::gputel::GpuSummary>,
    /// CPU package power / die temperature / DIMM temperatures, via the HWiNFO
    /// bridge. `None` when no sensor daemon is running — which is the common
    /// case and is reported as absent, never as zero.
    pub cpu_sensors: Option<crate::hwinfo::CpuSummary>,
    /// ACPI board zones and, where the firmware has one, a system power meter.
    pub platform: Option<crate::platform::PlatformSummary>,
    /// NVMe drive health: temperature, endurance, media errors.
    pub drives: Vec<crate::nvme::NvmeHealth>,
    /// OS-level ETW trace (`--etw`, via Windows Performance Recorder). Carried
    /// as an opaque record — the `.etl` is the artifact; this says whether one
    /// exists, where, and why not when it does not.
    pub etw: Option<EtwRecord>,
}

impl Report {
    pub fn new(tool_version: impl Into<String>, device: DeviceId, clock: &Clock) -> Report {
        Report {
            schema_version: 1,
            tool: "cec-crucible".to_string(),
            tool_version: tool_version.into(),
            device,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            qpc_frequency: clock.frequency(),
            qpc_backed: clock.is_qpc(),
            started: None,
            ended: None,
            stages: Vec::new(),
            aborted: false,
            markers_file: None,
            notes: Vec::new(),
            events: None,
            present: None,
            gpu: None,
            cpu_sensors: None,
            platform: None,
            drives: Vec::new(),
            etw: None,
        }
    }

    pub fn add_stage(&mut self, stage: StageReport) {
        self.stages.push(stage);
    }

    pub fn note(&mut self, msg: impl Into<String>) {
        self.notes.push(msg.into());
    }

    /// Roll up the overall verdict: any failed stage ⇒ FAIL; else an aborted
    /// run ⇒ PARTIAL; else PASS.
    pub fn verdict(&self) -> Verdict {
        // A hardware error the OS logged is a failure even when every checksum
        // matched — a CORRECTED machine-check means the hardware silently fixed
        // something, which is precisely the fault class our own verification can
        // never see. Ignoring it would be the worst kind of false pass.
        let logged_hw_fault = self
            .events
            .as_ref()
            .map(|e| e.fail_count() > 0)
            .unwrap_or(false);
        if self.stages.iter().any(|s| !s.result.passed()) || logged_hw_fault {
            Verdict::Fail
        } else if self.aborted {
            Verdict::Partial
        } else {
            Verdict::Pass
        }
    }

    /// Total detected errors across all stages.
    pub fn error_count(&self) -> u64 {
        self.stages.iter().map(|s| s.result.error_count).sum()
    }

    pub fn to_json(&self) -> Json {
        let mut root = Json::object();
        root.push("schema_version", self.schema_version)
            .push("tool", self.tool.as_str())
            .push("tool_version", self.tool_version.as_str())
            .push("verdict", self.verdict().as_str())
            .push("error_count", self.error_count())
            .push("aborted", self.aborted)
            .push("device", self.device.to_json())
            .push("os", self.os.as_str())
            .push("arch", self.arch.as_str());

        let mut qpc = Json::object();
        qpc.push("frequency", self.qpc_frequency)
            .push("backed_by_qpc", self.qpc_backed);
        root.push("qpc", qpc);

        root.push("started", opt_ts(&self.started))
            .push("ended", opt_ts(&self.ended))
            .push("duration_seconds", round2(self.duration_seconds()));

        let stages: Vec<Json> = self.stages.iter().map(StageReport::to_json).collect();
        root.push("stages", Json::Array(stages));

        root.push(
            "markers_file",
            match &self.markers_file {
                Some(f) => Json::str(f.as_str()),
                None => Json::Null,
            },
        );

        let notes: Vec<Json> = self.notes.iter().map(|n| Json::str(n.as_str())).collect();
        root.push("notes", Json::Array(notes));

        // The event-log detector plane. Always emitted when a scan was run, so a
        // report states plainly whether the plane was live — "no events found"
        // and "we could not read the log" must never look the same.
        if let Some(ev) = &self.events {
            root.push("event_log", ev.to_json());
        }
        if let Some(g) = &self.gpu {
            root.push("gpu", g.to_json());
        }
        if let Some(c) = &self.cpu_sensors {
            root.push("cpu_sensors", c.to_json());
        }
        if let Some(pl) = &self.platform {
            root.push("platform", pl.to_json());
        }
        if !self.drives.is_empty() {
            root.push(
                "drives",
                Json::Array(self.drives.iter().map(|d| d.to_json()).collect()),
            );
        }
        if let Some(t) = &self.etw {
            root.push("etw", t.json.clone());
        }
        if let Some(pm) = &self.present {
            let mut j = Json::object();
            j.push("frames", pm.frames)
                .push("avg_fps", round2(pm.avg_fps))
                .push("low1pct_fps", round2(pm.low1pct_fps))
                .push("stutter_ms", round2(pm.stutter_ms))
                .push("gpu_busy_ms", round2(pm.gpu_busy_ms))
                .push("cpu_busy_ms", round2(pm.cpu_busy_ms))
                .push("cpu_wait_ms", round2(pm.cpu_wait_ms))
                .push("display_latency_ms", round2(pm.display_latency_ms))
                .push("dropped", pm.dropped);
            root.push("present", j);
        }
        root
    }

    /// Wall-clock run duration in seconds, from the QPC window if available.
    pub fn duration_seconds(&self) -> f64 {
        match (self.started, self.ended) {
            (Some(a), Some(b)) => b.seconds_since(a),
            _ => 0.0,
        }
    }

    pub fn to_pretty_json(&self) -> String {
        self.to_json().to_pretty()
    }

    pub fn write_json(&self, path: &Path) -> io::Result<()> {
        std::fs::write(path, self.to_pretty_json())
    }

    /// Render the report as CSV — one row per stage, with the run-level context
    /// (device, OS, timestamp, overall verdict) repeated on every row so each row
    /// is self-contained and the file opens cleanly in Excel / Sheets / pandas.
    /// A header row is always emitted, even for a run with no stages.
    pub fn to_csv(&self) -> String {
        const HEADER: &str = "tool_version,device_short_id,host,system,board,os,arch,\
started_unix_nanos,run_verdict,kernel,kind,mode,seconds,stage_verdict,ok,iterations,\
checksum_hex,error_count,detail,\
whea_faults,event_warnings,pm_frames,pm_avg_fps,pm_1pct_low_fps,pm_stutter_ms,\
pm_gpu_busy_ms,pm_cpu_busy_ms,pm_cpu_wait_ms,pm_display_latency_ms,pm_dropped,\
gpu_name,gpu_power_avg_w,gpu_power_peak_w,gpu_power_limit_w,gpu_temp_peak_c,\
gpu_mem_temp_peak_c,gpu_fan_peak_pct,gpu_sm_mhz_avg,gpu_throttle_reasons\n";
        let mut s = String::from(HEADER);
        let started = self
            .started
            .map(|t| t.unix_nanos.to_string())
            .unwrap_or_default();
        let run_verdict = self.verdict().as_str();
        for st in &self.stages {
            let row = [
                csv_field(&self.tool_version),
                csv_field(&self.device.short_id),
                csv_field(&self.device.host),
                csv_field(&self.device.system),
                csv_field(&self.device.board),
                csv_field(&self.os),
                csv_field(&self.arch),
                csv_field(&started),
                run_verdict.to_string(),
                csv_field(&st.kernel),
                csv_field(st.kind.as_str()),
                csv_field(&st.mode),
                format!("{:.2}", st.seconds),
                st.verdict().as_str().to_string(),
                st.result.ok.to_string(),
                st.result.iterations.to_string(),
                format!("{:#018x}", st.result.checksum),
                st.result.error_count.to_string(),
                csv_field(&st.result.detail),
                // Run-level context repeated per row, same as device/verdict
                // above, so each row stays self-contained.
                self.events.as_ref().map(|e| e.fail_count().to_string()).unwrap_or_default(),
                self.events.as_ref().map(|e| e.warn_count().to_string()).unwrap_or_default(),
                pm_col(&self.present, |p| p.frames.to_string()),
                pm_col(&self.present, |p| format!("{:.1}", p.avg_fps)),
                pm_col(&self.present, |p| format!("{:.1}", p.low1pct_fps)),
                pm_col(&self.present, |p| format!("{:.2}", p.stutter_ms)),
                pm_col(&self.present, |p| format!("{:.2}", p.gpu_busy_ms)),
                pm_col(&self.present, |p| format!("{:.2}", p.cpu_busy_ms)),
                pm_col(&self.present, |p| format!("{:.2}", p.cpu_wait_ms)),
                pm_col(&self.present, |p| format!("{:.2}", p.display_latency_ms)),
                pm_col(&self.present, |p| p.dropped.to_string()),
                gpu_col(&self.gpu, |g| csv_field(&g.name)),
                gpu_col(&self.gpu, |g| format!("{:.1}", g.power_avg_w)),
                gpu_col(&self.gpu, |g| format!("{:.1}", g.power_peak_w)),
                gpu_col(&self.gpu, |g| format!("{:.0}", g.power_limit_w)),
                gpu_col(&self.gpu, |g| g.temp_peak_c.to_string()),
                // Blank when the board has no memory-junction sensor — see
                // `markers::gpu_cols` for why a 0 here would be a lie.
                gpu_col(&self.gpu, |g| match g.mem_temp_peak_c {
                    0 => String::new(),
                    v => v.to_string(),
                }),
                gpu_col(&self.gpu, |g| g.fan_peak_pct.to_string()),
                gpu_col(&self.gpu, |g| g.sm_mhz_avg.to_string()),
                gpu_col(&self.gpu, |g| {
                    csv_field(&crate::gputel::throttle_names(g.throttle_seen).join("; "))
                }),
            ]
            .join(",");
            s.push_str(&row);
            s.push('\n');
        }
        s
    }

    pub fn write_csv(&self, path: &Path) -> io::Result<()> {
        std::fs::write(path, self.to_csv())
    }
}

/// Quote a CSV field per RFC 4180 iff it contains a comma, quote, or newline,
/// doubling any embedded quotes. Detail strings can contain commas, so this
/// keeps the column count stable.
/// A PresentMon column: the formatted value, or empty when no capture ran.
/// Empty is deliberate — a blank cell reads as "not measured", whereas a 0
/// would read as "measured, and it was zero".
fn pm_col(p: &Option<PresentSummary>, f: impl Fn(&PresentSummary) -> String) -> String {
    p.as_ref().map(f).unwrap_or_default()
}

/// A GPU-sensor column, blank when NVML was unavailable — same reasoning as
/// [`pm_col`]: a zero-watt reading would look like a power failure on a graph.
fn gpu_col(
    g: &Option<crate::gputel::GpuSummary>,
    f: impl Fn(&crate::gputel::GpuSummary) -> String,
) -> String {
    g.as_ref().map(f).unwrap_or_default()
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn opt_ts(ts: &Option<Timestamp>) -> Json {
    match ts {
        Some(t) => ts_json(t),
        None => Json::Null,
    }
}

fn ts_json(ts: &Timestamp) -> Json {
    let mut o = Json::object();
    o.push("qpc_ticks", ts.qpc_ticks)
        .push("qpc_frequency", ts.qpc_frequency)
        // u128 nanos exceed JSON's safe-integer range, so emit as a string.
        .push("unix_nanos", ts.unix_nanos.to_string());
    o
}

/// Round to 2 decimals for tidy report numbers (throughput, seconds).
fn round2(x: f64) -> f64 {
    if x.is_finite() {
        (x * 100.0).round() / 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eventlog::{EventRecord, EventScan, Severity};

    fn scan_with(sev: Severity) -> EventScan {
        EventScan {
            available: true,
            unavailable_reason: String::new(),
            events: vec![EventRecord {
                time: "2026-07-24T20:15:38Z".into(),
                provider: "Microsoft-Windows-WHEA-Logger".into(),
                event_id: 19,
                level: 3,
                data: String::new(),
                severity: sev,
                meaning: "test",
            }],
        }
    }

    #[test]
    fn csv_header_and_row_have_the_same_field_count() {
        // This caught a real bug: the PresentMon/event columns were appended to
        // every data row while the header const still listed the old 19, so the
        // file did not parse. Any future column MUST be added in both places.
        let mut r = Report::new("t", DeviceId::detect(), &Clock::new());
        r.add_stage(StageReport::new(
            "cpu",
            Kind::Cpu,
            "steady",
            1.0,
            LoadResult::clean(1, 2, "ok"),
        ));
        let csv = r.to_csv();
        let mut lines = csv.lines();
        let header = lines.next().expect("header");
        let row = lines.next().expect("one data row");
        assert_eq!(
            header.split(',').count(),
            row.split(',').count(),
            "header/row column mismatch
header: {header}
row:    {row}"
        );
        assert!(header.contains("pm_avg_fps") && header.contains("whea_faults"));
        assert!(header.contains("gpu_power_peak_w") && header.contains("gpu_temp_peak_c"));
    }

    #[test]
    fn a_logged_hardware_fault_fails_a_run_with_no_stage_errors() {
        let mut r = Report::new("t", DeviceId::detect(), &Clock::new());
        assert_eq!(r.verdict(), Verdict::Pass, "clean baseline");
        r.events = Some(scan_with(Severity::Fail));
        assert_eq!(
            r.verdict(),
            Verdict::Fail,
            "a WHEA machine-check must fail the run even though every checksum matched"
        );
        assert_eq!(r.error_count(), 0, "it is not a STAGE error; the event plane is separate");
    }

    #[test]
    fn a_warning_event_does_not_fail_the_run() {
        let mut r = Report::new("t", DeviceId::detect(), &Clock::new());
        r.events = Some(scan_with(Severity::Warn));
        assert_eq!(r.verdict(), Verdict::Pass);
    }

    #[test]
    fn an_unavailable_scan_does_not_silently_pass_as_clean() {
        let mut r = Report::new("t", DeviceId::detect(), &Clock::new());
        r.events = Some(EventScan {
            available: false,
            unavailable_reason: "EvtQuery failed".into(),
            events: Vec::new(),
        });
        // It cannot fail the run (we saw nothing), but the report must record
        // that the plane was down so a reader can tell it apart from clean.
        assert_eq!(r.verdict(), Verdict::Pass);
        assert!(!r.events.as_ref().unwrap().available);
        assert!(r.to_pretty_json().contains("unavailable_reason"));
    }

    use super::*;

    fn dev() -> DeviceId {
        DeviceId::from_override("TEST-UUID")
    }

    #[test]
    fn clean_run_passes() {
        let clock = Clock::new();
        let mut r = Report::new("0.1.0", dev(), &clock);
        r.add_stage(StageReport::new(
            "cpu",
            Kind::Cpu,
            "steady",
            1.0,
            LoadResult::clean(10, 0xABCD, "ok"),
        ));
        assert_eq!(r.verdict(), Verdict::Pass);
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn any_error_fails_the_run() {
        let clock = Clock::new();
        let mut r = Report::new("0.1.0", dev(), &clock);
        r.add_stage(StageReport::new(
            "cpu",
            Kind::Cpu,
            "steady",
            1.0,
            LoadResult::clean(10, 0, "ok"),
        ));
        r.add_stage(StageReport::new(
            "mem",
            Kind::Mem,
            "steady",
            1.0,
            LoadResult::new(true, 5, 0, 3, "3 miscompares".into()),
        ));
        assert_eq!(r.verdict(), Verdict::Fail);
        assert_eq!(r.error_count(), 3);
    }

    #[test]
    fn aborted_clean_run_is_partial() {
        let clock = Clock::new();
        let mut r = Report::new("0.1.0", dev(), &clock);
        r.add_stage(StageReport::new(
            "cpu",
            Kind::Cpu,
            "steady",
            0.5,
            LoadResult::clean(10, 0, "ok"),
        ));
        r.aborted = true;
        assert_eq!(r.verdict(), Verdict::Partial);
    }

    #[test]
    fn setup_failure_fails() {
        let clock = Clock::new();
        let mut r = Report::new("0.1.0", dev(), &clock);
        r.add_stage(StageReport::new(
            "storage",
            Kind::Storage,
            "steady",
            0.0,
            LoadResult::setup_failure("no writable path"),
        ));
        assert_eq!(r.verdict(), Verdict::Fail);
    }

    #[test]
    fn csv_has_header_and_escapes_detail() {
        let clock = Clock::new();
        let mut r = Report::new("0.1.0", dev(), &clock);
        r.add_stage(StageReport::new(
            "mem",
            Kind::Mem,
            "steady",
            1.5,
            LoadResult::new(true, 5, 0xABCD, 2, "2 miscompares, first @ 0x40".into()),
        ));
        let csv = r.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines[0].starts_with("tool_version,"), "header row");
        assert_eq!(lines.len(), 2, "header + one stage row");
        // The detail contains a comma → must be quoted so the column count is stable.
        assert!(csv.contains("\"2 miscompares, first @ 0x40\""), "detail quoted");
        assert!(csv.contains(",mem,"), "kernel column");
        assert!(csv.contains("0x000000000000abcd"), "checksum as hex");
    }

    #[test]
    fn csv_empty_run_is_header_only() {
        let clock = Clock::new();
        let r = Report::new("0.1.0", dev(), &clock);
        assert_eq!(r.to_csv().lines().count(), 1);
    }

    #[test]
    fn json_has_required_top_level_keys() {
        let clock = Clock::new();
        let r = Report::new("0.1.0", dev(), &clock);
        let s = r.to_pretty_json();
        for key in [
            "schema_version",
            "tool",
            "verdict",
            "device",
            "qpc",
            "stages",
        ] {
            assert!(s.contains(&format!("\"{key}\"")), "missing key {key}");
        }
    }
}
