// SPDX-License-Identifier: MIT
//! Opt-in ETW capture (`--etw`), driven through the in-box **Windows
//! Performance Recorder** (`%SystemRoot%\System32\wpr.exe`).
//!
//! Everything else this tool records is something *we* measured. An ETW trace is
//! the opposite: it is the operating system's own account of the run — every
//! context switch, DPC and ISR, every disk and GPU packet, every power-state
//! transition — written by providers we could never instrument ourselves. When a
//! machine stutters, drops a frame, or dies without a WHEA entry, that trace is
//! usually the only artifact that still contains the answer.
//!
//! **Why shell out to `wpr.exe` instead of hosting the session ourselves.**
//! A real ETW controller means `StartTrace` / `EnableTraceEx2` / `ProcessTrace`
//! plus a TDH decoder for every provider — and, more to the point, the *profile*
//! definitions: which of the several hundred providers to enable, at what level,
//! with which keyword masks, to make a trace WPA can actually open. Microsoft
//! already ships those as WPR profiles, and `wpr.exe` is present on every
//! Windows 10/11 install. Re-deriving them by hand would be a large FFI surface
//! whose only achievement is a worse version of a file the OS already knows how
//! to produce. The `.etl` opens directly in Windows Performance Analyzer.
//!
//! **This needs an elevated shell**, because arming system-wide ETW is a
//! privileged operation. Non-elevated, `wpr` fails with `0xc5585011`; we detect
//! that and say so plainly. A capture that could not run is reported as
//! *unavailable*, never as an empty-but-fine result — the same rule the event
//! log scan follows.
//!
//! **Traces are large.** `GeneralProfile` in file mode writes on the order of
//! 100 MB per minute under load. The default is therefore a single profile and
//! an explicit path, and the size is reported when the capture stops.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

use crucible_core::json::Json;

/// WPR profiles that make sense for a hardware QC run, in the order they are
/// suggested. These are in-box profile names (`wpr -profiles`), not our own.
#[allow(dead_code)] // referenced from --help prose and the Settings screen
pub const SUGGESTED: &[(&str, &str)] = &[
    ("GeneralProfile", "first-level triage: CPU sampling, context switches, DPC/ISR"),
    ("CPU", "CPU usage and scheduling detail"),
    ("GPU", "GPU work packets and scheduling"),
    ("Power", "power-state transitions and idle residency"),
    ("Thermal", "thermal-zone status"),
    ("DiskIO", "disk I/O activity"),
    ("Video", "video/present glitches"),
];

/// A capture in progress.
pub struct Capture {
    wpr: PathBuf,
    out: PathBuf,
    profiles: Vec<String>,
}

/// What the report records about the capture.
#[derive(Debug, Clone, Default)]
pub struct EtwTrace {
    /// A trace was actually written.
    pub available: bool,
    /// Why not, when `available` is false. Never empty in that case.
    pub unavailable_reason: String,
    pub profiles: Vec<String>,
    pub path: String,
    pub bytes: u64,
    /// How long `wpr -stop` took. Reported because the flush is the slowest part
    /// of an ETW capture by a wide margin, and it scales with trace size — which
    /// is the lever an operator actually has (fewer profiles, shorter run).
    pub flush_seconds: f64,
}

impl EtwTrace {
    fn unavailable(reason: impl Into<String>, profiles: &[String]) -> EtwTrace {
        EtwTrace {
            available: false,
            unavailable_reason: reason.into(),
            profiles: profiles.to_vec(),
            path: String::new(),
            bytes: 0,
            flush_seconds: 0.0,
        }
    }

    /// One-line console form.
    pub fn line(&self) -> String {
        if self.available {
            format!(
                "etw:     {}  ({}, {:.1} MB, flushed in {:.1}s)",
                self.path,
                self.profiles.join("+"),
                self.bytes as f64 / (1024.0 * 1024.0),
                self.flush_seconds
            )
        } else {
            format!("etw: UNAVAILABLE ({})", self.unavailable_reason)
        }
    }

    /// The report's carrier form: everything the JSON needs, plus the console
    /// line, so `crucible-core` never has to know what WPR is.
    pub fn record(&self) -> crucible_core::report::EtwRecord {
        crucible_core::report::EtwRecord {
            available: self.available,
            path: self.path.clone(),
            line: self.line(),
            json: self.to_json(),
        }
    }

    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("available", self.available)
            .push("unavailable_reason", Json::str(&self.unavailable_reason))
            .push(
                "profiles",
                Json::Array(self.profiles.iter().map(|p| Json::str(p)).collect()),
            )
            .push("path", Json::str(&self.path))
            .push("bytes", Json::U64(self.bytes))
            .push("flush_seconds", Json::F64((self.flush_seconds * 10.0).round() / 10.0));
        o
    }
}

/// `%SystemRoot%\System32\wpr.exe`, if it exists.
///
/// Deliberately not resolved through `PATH`: this must be the OS component, not
/// whatever a `wpr.exe` earlier on the path happens to be.
pub fn locate() -> Option<PathBuf> {
    let root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
    let p = Path::new(&root).join("System32").join("wpr.exe");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// True when a WPR session is already recording — ours or somebody else's.
/// Only one system-wide session can exist, so this decides between "we can
/// start" and "something else owns it".
pub fn already_recording(wpr: &Path) -> bool {
    match Command::new(wpr).arg("-status").output() {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            // `wpr -status` prints "WPR is not recording" when idle.
            !s.contains("not recording")
        }
        Err(_) => false,
    }
}

impl Capture {
    /// Arm a capture. `profiles` are in-box WPR profile names.
    ///
    /// Returns `Err(EtwTrace)` describing precisely why nothing is being
    /// recorded, so the caller can put that in the report rather than leaving a
    /// silent gap where a trace should be.
    pub fn start(profiles: &[String], out: PathBuf) -> Result<Capture, EtwTrace> {
        let wpr = match locate() {
            Some(w) => w,
            None => {
                return Err(EtwTrace::unavailable(
                    "wpr.exe not found in System32 (Windows Performance Recorder is in-box on Windows 10/11)",
                    profiles,
                ))
            }
        };
        if profiles.is_empty() {
            return Err(EtwTrace::unavailable("no profiles requested", profiles));
        }
        if already_recording(&wpr) {
            return Err(EtwTrace::unavailable(
                "another WPR session is already recording — only one system-wide session may exist \
                 (stop it with `wpr -cancel`)",
                profiles,
            ));
        }

        let mut cmd = Command::new(&wpr);
        for p in profiles {
            cmd.arg("-start").arg(p);
        }
        // File mode streams to disk instead of a bounded memory ring: a stress
        // run is minutes long, and the default memory buffers would silently
        // discard everything but the tail.
        cmd.arg("-filemode");

        match cmd.output() {
            Ok(o) if o.status.success() => Ok(Capture {
                wpr,
                out,
                profiles: profiles.to_vec(),
            }),
            Ok(o) => Err(EtwTrace::unavailable(explain(&o), profiles)),
            Err(e) => Err(EtwTrace::unavailable(
                format!("could not launch wpr.exe: {e}"),
                profiles,
            )),
        }
    }

    /// Stop the capture and flush the `.etl`.
    ///
    /// `wpr -stop` does not merely close a file: it *merges* the trace, and by
    /// default that includes generating NGEN and embedded PDBs so a profiler can
    /// resolve managed symbols later. On a machine with .NET assemblies loaded
    /// that symbol generation is routinely the majority of the stop time — tens
    /// of seconds, sometimes minutes, on a trace that took seconds to record.
    ///
    /// We are capturing hardware behaviour, not profiling managed code, so
    /// `-skipPdbGen` removes work whose output nothing here would ever read. The
    /// remaining merge — image-ID and machine-info injection, which is what makes
    /// the `.etl` openable in WPA on another machine — is kept.
    ///
    /// The flush is timed and reported: an operator watching a progress-less wait
    /// should at least be told afterwards what it cost.
    pub fn finish(self, description: &str) -> EtwTrace {
        let began = std::time::Instant::now();
        let r = Command::new(&self.wpr)
            .arg("-stop")
            .arg(&self.out)
            .arg(description)
            .arg("-skipPdbGen")
            .output();
        match r {
            Ok(o) if o.status.success() => {
                let bytes = std::fs::metadata(&self.out).map(|m| m.len()).unwrap_or(0);
                if bytes == 0 {
                    // A zero-byte .etl means the stop succeeded but nothing
                    // landed. Reporting that as a trace would send somebody
                    // looking for evidence that does not exist.
                    return EtwTrace::unavailable(
                        "wpr stopped but wrote an empty trace",
                        &self.profiles,
                    );
                }
                EtwTrace {
                    available: true,
                    unavailable_reason: String::new(),
                    profiles: self.profiles,
                    path: self.out.display().to_string(),
                    bytes,
                    flush_seconds: began.elapsed().as_secs_f64(),
                }
            }
            Ok(o) => EtwTrace::unavailable(explain(&o), &self.profiles),
            Err(e) => EtwTrace::unavailable(format!("wpr -stop failed to launch: {e}"), &self.profiles),
        }
    }

    /// Abandon the capture without writing a trace (used when a run is torn
    /// down early). Leaving the session armed would block the next run.
    pub fn cancel(self) {
        let _ = Command::new(&self.wpr).arg("-cancel").output();
    }
}

/// Stop an ETW session left armed by a run that died before it could stop its
/// own — the trace covering the crash is still in the kernel's buffers, and it
/// is the single most valuable artifact we could recover.
///
/// Returns the salvaged trace, or `None` when there was nothing to salvage.
pub fn salvage(out: PathBuf, description: &str) -> Option<EtwTrace> {
    let wpr = locate()?;
    if !already_recording(&wpr) {
        return None;
    }
    let cap = Capture {
        wpr,
        out,
        profiles: vec!["(recovered)".to_string()],
    };
    let t = cap.finish(description);
    Some(t)
}

/// Turn a failed `wpr` invocation into something an operator can act on.
///
/// `0xc5585011` is the one that matters: it is what non-elevated looks like, and
/// left as a hex code it reads like a broken tool rather than a missing right.
fn explain(o: &std::process::Output) -> String {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    if text.contains("0xc5585011") || text.contains("Failed to enable the policy") {
        return "ETW capture requires an elevated shell — re-run cec-crucible as administrator \
                (wpr: 0xc5585011, cannot enable the system performance profiling policy)"
            .to_string();
    }
    if text.contains("0xc5583000") || text.to_lowercase().contains("already") {
        return "a WPR session is already running (stop it with `wpr -cancel`)".to_string();
    }
    let first = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("Microsoft") && !l.starts_with("Copyright"))
        .unwrap_or("wpr failed with no output");
    format!("wpr: {first}")
}

/// Parse `--etw <a,b,c>` into profile names, defaulting to first-level triage.
///
/// Unknown names are passed through rather than rejected: `wpr` accepts custom
/// `.wprp` profiles too, and refusing one we do not recognise would block a
/// legitimate use for no benefit. A bad name fails loudly at `-start`.
pub fn parse_profiles(arg: Option<&str>) -> Vec<String> {
    match arg {
        None | Some("") => vec!["GeneralProfile".to_string()],
        Some(s) => s
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_string)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_default_to_triage_and_split_on_commas() {
        assert_eq!(parse_profiles(None), vec!["GeneralProfile"]);
        assert_eq!(parse_profiles(Some("")), vec!["GeneralProfile"]);
        assert_eq!(
            parse_profiles(Some("CPU, GPU ,Power")),
            vec!["CPU", "GPU", "Power"]
        );
        // A trailing comma must not produce an empty profile name, which wpr
        // would reject with a confusing error.
        assert_eq!(parse_profiles(Some("CPU,")), vec!["CPU"]);
    }

    #[test]
    fn wpr_is_in_box_on_this_windows() {
        // Not an environment assumption we can skip: if wpr.exe is missing the
        // feature cannot work at all, and the report must say so rather than
        // pretend. Locating it is the whole precondition.
        let found = locate();
        assert!(
            found.is_some(),
            "wpr.exe should exist in System32 on Windows 10/11"
        );
    }

    #[test]
    fn non_elevated_failure_is_explained_not_hex() {
        use std::os::windows::process::ExitStatusExt;
        let o = std::process::Output {
            status: std::process::ExitStatus::from_raw(1),
            stdout: b"\tFailed to enable the policy to profile system performance.\n\
                      \tError code: 0xc5585011\n"
                .to_vec(),
            stderr: Vec::new(),
        };
        let msg = explain(&o);
        assert!(
            msg.contains("elevated"),
            "the operator must be told to elevate, got: {msg}"
        );
    }

    #[test]
    fn an_unavailable_trace_never_reads_as_a_clean_one() {
        let t = EtwTrace::unavailable("no wpr", &["CPU".to_string()]);
        assert!(!t.available);
        assert!(t.line().contains("UNAVAILABLE"));
        assert!(!t.unavailable_reason.is_empty());
    }
}
