// SPDX-License-Identifier: MIT
//! Opt-in PresentMon integration — the ETW-depth half of the hybrid frame
//! telemetry (the self-measured half lives in the render kernel's own pacing).
//!
//! We do **not** bundle PresentMon (Intel's open-source, MIT frame-timing tool):
//! shipping a third-party profiler in the repo is a licensing + AV-false-positive
//! liability. Instead `--presentmon` locates an already-installed copy and drives
//! it alongside a presenting run, so its per-frame ETW capture — the *actually
//! displayed* frames, the ones the compositor dropped or duplicated, and display
//! latency, none of which the app can self-measure — lands in a CSV beside our
//! own pacing telemetry.
//!
//! PresentMon is targeted at **our own process id**, so it records exactly the
//! windows this run presents. It usually needs elevation for the ETW real-time
//! session; when it can't start (missing, unelevated, wrong version) that is a
//! `note:` and the run continues normally — the ETW capture is a bonus, never a
//! gate. Built for PresentMon 2.x CLI flags; older 1.x builds use single-dash
//! flags and are not supported.
//!
//! Windows + `gpu` only (ETW is Windows; the driver is meaningless without a
//! presenting GPU run).

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Locate an installed `PresentMon.exe`, in priority order: an explicit
/// `--presentmon-path`, the `CRUCIBLE_PRESENTMON` env var, then every directory
/// on `PATH` (exact name first, then a `PresentMon-*.exe` versioned build).
/// Returns `None` if none is found.
pub fn locate(explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(p) = std::env::var("CRUCIBLE_PRESENTMON") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["PresentMon.exe", "presentmon.exe"] {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        // Versioned release names, e.g. `PresentMon-2.3.0-x64.exe`.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let lower = e.file_name().to_string_lossy().to_ascii_lowercase();
                if lower.starts_with("presentmon") && lower.ends_with(".exe") {
                    return Some(e.path());
                }
            }
        }
    }
    None
}

/// A running PresentMon capture bound to this process. Self-terminates after its
/// timed window; killed on drop as a backstop if the run ends early.
pub struct Capture {
    child: Child,
    csv: PathBuf,
}

impl Capture {
    /// Spawn PresentMon capturing **this** process's presents to `csv` for up to
    /// `seconds` (a small margin is added so it outlives the run's own timing).
    /// Targets our pid so only the windows we present are recorded.
    pub fn start(exe: &Path, csv: PathBuf, seconds: u64) -> std::io::Result<Capture> {
        let pid = std::process::id();
        let child = Command::new(exe)
            .args([
                "--process_id".to_string(),
                pid.to_string(),
                "--output_file".to_string(),
                csv.to_string_lossy().into_owned(),
                "--timed".to_string(),
                (seconds + 2).to_string(),
                "--terminate_after_timed".to_string(),
                "--stop_existing_session".to_string(),
                "--no_console_stats".to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(Capture { child, csv })
    }

    /// Wait for PresentMon to finish flushing (it exits itself after `--timed`),
    /// then return the CSV path if it was actually written. A `None` means the
    /// tool started but produced nothing — almost always missing elevation.
    pub fn finish(mut self) -> Option<PathBuf> {
        let _ = self.child.wait();
        if self.csv.is_file() {
            Some(std::mem::take(&mut self.csv))
        } else {
            None
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // Backstop: if the run ended before PresentMon's timed window, stop it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse a one-line summary from a PresentMon CSV: the frame count and the
/// average *displayed* fps (from the per-frame `msBetweenDisplayChange` column
/// when present, else `msBetweenPresents`). Best-effort — returns `None` if the
/// header/columns aren't recognized. Header names track PresentMon 2.x.
pub fn summarize(csv: &Path) -> Option<String> {
    let text = std::fs::read_to_string(csv).ok()?;
    let mut lines = text.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let find = |want: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(want));
    // Prefer the display-change interval (true displayed pacing); fall back to
    // the present interval.
    let disp_idx = find("msBetweenDisplayChange").or_else(|| find("msBetweenPresents"))?;
    let dropped_idx = find("Dropped");

    let mut frames: u64 = 0;
    let mut sum_ms: f64 = 0.0;
    let mut dropped: u64 = 0;
    for row in lines {
        let f: Vec<&str> = row.split(',').collect();
        if let Some(ms) = f.get(disp_idx).and_then(|v| v.trim().parse::<f64>().ok()) {
            if ms > 0.0 {
                frames += 1;
                sum_ms += ms;
            }
        }
        if let Some(di) = dropped_idx {
            if f.get(di).map(|v| v.trim() == "1").unwrap_or(false) {
                dropped += 1;
            }
        }
    }
    if frames == 0 {
        return None;
    }
    let fps = 1000.0 * frames as f64 / sum_ms;
    Some(format!(
        "displayed {frames} frame(s) ~{fps:.0} fps, {dropped} dropped"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn locate_none_when_absent() {
        // A path that does not exist yields None from the explicit branch (and we
        // don't assume anything about the host PATH).
        assert!(locate(Some("Z:/definitely/not/here/PresentMon.exe")).is_none());
    }

    #[test]
    fn summarize_reads_display_pacing() {
        let dir = std::env::temp_dir();
        let path = dir.join("crucible-pmtest.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "Application,ProcessID,msBetweenDisplayChange,Dropped").unwrap();
            writeln!(f, "cec-crucible.exe,123,8.0,0").unwrap(); // 125 fps frame
            writeln!(f, "cec-crucible.exe,123,8.0,1").unwrap(); // dropped
            writeln!(f, "cec-crucible.exe,123,8.0,0").unwrap();
        }
        let s = summarize(&path).expect("summary");
        assert!(s.contains("displayed 3 frame(s)"), "{s}");
        assert!(s.contains("125 fps"), "{s}");
        assert!(s.contains("1 dropped"), "{s}");
        let _ = std::fs::remove_file(&path);
    }
}
