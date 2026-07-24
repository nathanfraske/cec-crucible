// SPDX-License-Identifier: MIT
//! The device-identified run report.
//!
//! A stress pass that does not crash but produced a wrong result is still a
//! FAIL — so the verdict rolls up per-kernel error counts, not just liveness.
//! (WHEA corrected-error gating is owned by the PowerShell harness around the
//! whole window; kernels here report their own compute/verify errors.)

use std::io;
use std::path::Path;

use crate::clock::{Clock, Timestamp};
use crate::device::DeviceId;
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
        if self.stages.iter().any(|s| !s.result.passed()) {
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
