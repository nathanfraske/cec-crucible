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
use std::time::{Duration, Instant};

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
    // Next to our own executable. This is what makes the shipped bundle
    // portable: unzip the release (or install it) and PresentMon is simply
    // there, with no PATH entry, no install step and no configuration.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in ["PresentMon.exe", "presentmon.exe"] {
                let cand = dir.join(name);
                if cand.is_file() {
                    return Some(cand);
                }
            }
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
    /// When the capture was spawned, and the `--timed` window we gave it — so
    /// `finish` never declares the CSV complete before PresentMon has stopped
    /// writing it.
    started: Instant,
    window: Duration,
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
                // Pin the schema: v2 metrics carry the present-pipeline breakdown
                // (GPUBusy / CPUBusy / CPUWait / DisplayLatency) this suite reports.
                "--v2_metrics".to_string(),
                // An ETW real-time session needs elevation. When we are not already
                // elevated PresentMon relaunches ITSELF elevated (one UAC prompt) —
                // which is why `finish` waits on the CSV, not on our child handle:
                // the relaunched process is not ours.
                "--restart_as_admin".to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(Capture {
            child,
            csv,
            started: Instant::now(),
            window: Duration::from_secs(seconds + 2),
        })
    }

    /// Wait for the capture to land. PresentMon stops itself after its `--timed`
    /// window, but with `--restart_as_admin` the writing process is a *different*
    /// (elevated) one, so we wait on the artifact: poll until the CSV appears and
    /// its size stops growing. Returns `None` on timeout — practically always a
    /// declined UAC prompt (no elevation, no ETW session, no data).
    pub fn finish(mut self) -> Option<PathBuf> {
        // Our own child is only meaningful when it wasn't relaunched; don't block
        // on it for long either way.
        let _ = self.child.try_wait();

        // PresentMon writes in buffered bursts, so "size didn't change since last
        // poll" is NOT proof it finished — early on, the file is briefly stable
        // holding only its header. Wait out the `--timed` window first, then
        // require a stable size over several polls AND at least one data row.
        let remaining = self.window.saturating_sub(self.started.elapsed());
        std::thread::sleep(remaining);

        let deadline = Instant::now() + Duration::from_secs(20);
        let mut last_len: u64 = u64::MAX;
        let mut stable_for = 0u32;
        while Instant::now() < deadline {
            if let Ok(m) = std::fs::metadata(&self.csv) {
                let len = m.len();
                if len == last_len && len > 0 {
                    stable_for += 1;
                    // ~1.2s unchanged, and it holds more than just a header row.
                    if stable_for >= 4 && has_data_row(&self.csv) {
                        return Some(std::mem::take(&mut self.csv));
                    }
                } else {
                    stable_for = 0;
                    last_len = len;
                }
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        // Timed out still growing/empty: hand it back anyway if it has rows, so a
        // long capture is reported rather than silently dropped.
        if has_data_row(&self.csv) {
            return Some(std::mem::take(&mut self.csv));
        }
        None
    }
}

/// True once the CSV holds at least one row beyond its header — i.e. real frames
/// were captured, not just an opened file.
fn has_data_row(csv: &Path) -> bool {
    std::fs::read_to_string(csv)
        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count() > 1)
        .unwrap_or(false)
}

impl Drop for Capture {
    fn drop(&mut self) {
        // Backstop: if the run ended before PresentMon's timed window, stop it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The present-pipeline picture reduced from a PresentMon capture. This is the
/// half the app cannot self-measure: where each frame's time actually went, and
/// whether it reached the display at all.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PresentStats {
    /// Frames with a usable frame time.
    pub frames: u64,
    /// Average fps over the capture (from mean frame time).
    pub avg_fps: f64,
    /// 1%-low fps (99th-percentile frame time) — the felt stutter floor.
    pub low1pct_fps: f64,
    /// Worst single frame time (ms).
    pub stutter_ms: f64,
    /// Mean GPU work per frame (ms) — `GPUBusy`.
    pub gpu_busy_ms: f64,
    /// Mean CPU work per frame (ms) — `CPUBusy`.
    pub cpu_busy_ms: f64,
    /// Mean CPU stall waiting on the GPU/present (ms) — `CPUWait`.
    pub cpu_wait_ms: f64,
    /// Mean present-to-display latency (ms) — `DisplayLatency`.
    pub display_latency_ms: f64,
    /// Frames that never reached the display (no displayed time).
    pub dropped: u64,
}

impl PresentStats {
    /// One-line form for the console + the results detail column.
    pub fn line(&self) -> String {
        format!(
            "etw: {} frame(s) ~{:.0} fps (1%low {:.0}, stutter {:.1}ms), \
             gpu_busy {:.2}ms, cpu_busy {:.2}ms, cpu_wait {:.2}ms, \
             display_latency {:.2}ms, {} dropped",
            self.frames,
            self.avg_fps,
            self.low1pct_fps,
            self.stutter_ms,
            self.gpu_busy_ms,
            self.cpu_busy_ms,
            self.cpu_wait_ms,
            self.display_latency_ms,
            self.dropped
        )
    }
}

/// Reduce a PresentMon CSV to [`PresentStats`]. Written against PresentMon 2.x
/// **v2 metrics** (which `Capture::start` pins with `--v2_metrics`): per-frame
/// `FrameTime` / `GPUBusy` / `CPUBusy` / `CPUWait` / `DisplayLatency`, plus
/// `DisplayedTime` (blank when the frame never reached the screen). Falls back to
/// the 1.x `msBetweenPresents` / `Dropped` names so a v1 capture still summarizes.
/// Returns `None` if no frame rows parse.
pub fn summarize(csv: &Path) -> Option<PresentStats> {
    let text = std::fs::read_to_string(csv).ok()?;
    let mut lines = text.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let find = |want: &str| cols.iter().position(|c| c.eq_ignore_ascii_case(want));

    // v2 first, then the v1 equivalents.
    let ft_idx = find("FrameTime")
        .or_else(|| find("msBetweenPresents"))
        .or_else(|| find("MsBetweenPresents"))?;
    let gpu_idx = find("GPUBusy").or_else(|| find("msGPUActive"));
    let cpu_idx = find("CPUBusy");
    let wait_idx = find("CPUWait");
    let lat_idx = find("DisplayLatency").or_else(|| find("msUntilDisplayed"));
    let disp_idx = find("DisplayedTime");
    let dropped_idx = find("Dropped");

    let mut ft: Vec<f64> = Vec::new();
    let (mut gpu, mut cpu, mut wait, mut lat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut n_gpu, mut n_cpu, mut n_wait, mut n_lat) = (0u64, 0u64, 0u64, 0u64);
    let mut dropped: u64 = 0;

    for row in lines {
        let f: Vec<&str> = row.split(',').collect();
        let num = |i: Option<usize>| -> Option<f64> {
            i.and_then(|i| f.get(i))
                .map(|v| v.trim())
                .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("NA"))
                .and_then(|v| v.parse::<f64>().ok())
        };
        match num(Some(ft_idx)) {
            Some(v) if v > 0.0 => ft.push(v),
            _ => continue, // not a frame row
        }
        if let Some(v) = num(gpu_idx) {
            gpu += v;
            n_gpu += 1;
        }
        if let Some(v) = num(cpu_idx) {
            cpu += v;
            n_cpu += 1;
        }
        if let Some(v) = num(wait_idx) {
            wait += v;
            n_wait += 1;
        }
        if let Some(v) = num(lat_idx) {
            lat += v;
            n_lat += 1;
        }
        // Dropped: v1 has an explicit flag; in v2 a frame that never displayed has
        // no displayed time.
        if let Some(di) = dropped_idx {
            if f.get(di).map(|v| v.trim() == "1").unwrap_or(false) {
                dropped += 1;
            }
        } else if disp_idx.is_some() && num(disp_idx).unwrap_or(0.0) <= 0.0 {
            dropped += 1;
        }
    }

    if ft.is_empty() {
        return None;
    }
    let frames = ft.len() as u64;
    let sum: f64 = ft.iter().sum();
    let mean = sum / frames as f64;
    let mut sorted = ft.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 99th-percentile frame time = the 1%-low fps pivot.
    let p99 = sorted[(((frames - 1) as f64) * 0.99).round() as usize];
    let worst = *sorted.last().unwrap_or(&0.0);
    let avg = |sum: f64, n: u64| if n > 0 { sum / n as f64 } else { 0.0 };

    Some(PresentStats {
        frames,
        avg_fps: if mean > 0.0 { 1000.0 / mean } else { 0.0 },
        low1pct_fps: if p99 > 0.0 { 1000.0 / p99 } else { 0.0 },
        stutter_ms: worst,
        gpu_busy_ms: avg(gpu, n_gpu),
        cpu_busy_ms: avg(cpu, n_cpu),
        cpu_wait_ms: avg(wait, n_wait),
        display_latency_ms: avg(lat, n_lat),
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn a_bogus_explicit_path_is_never_returned() {
        // Deliberately does NOT assert `None`: this machine may legitimately have
        // PresentMon on PATH, in which case falling back to it is correct. The
        // real invariant is that a path we were handed but could not find is
        // never passed on as if it existed. (An earlier version asserted `None`
        // and only passed because PresentMon happened not to be installed.)
        let bogus = "Z:/definitely/not/here/PresentMon.exe";
        if let Some(found) = locate(Some(bogus)) {
            assert_ne!(
                found.to_string_lossy(),
                bogus,
                "a non-existent explicit path must never be returned"
            );
            assert!(found.is_file(), "any returned path must actually exist");
        }
    }

    #[test]
    fn summarize_reads_v2_present_pipeline() {
        let path = std::env::temp_dir().join("crucible-pmtest-v2.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(
                f,
                "Application,ProcessID,FrameTime,CPUBusy,CPUWait,GPUBusy,DisplayLatency,DisplayedTime"
            )
            .unwrap();
            // 8ms frames = 125 fps; the third never displayed (blank DisplayedTime).
            writeln!(f, "cec-crucible.exe,123,8.0,3.0,1.0,6.0,12.0,8.0").unwrap();
            writeln!(f, "cec-crucible.exe,123,8.0,3.0,1.0,6.0,12.0,8.0").unwrap();
            writeln!(f, "cec-crucible.exe,123,8.0,3.0,1.0,6.0,12.0,").unwrap();
        }
        let s = summarize(&path).expect("summary");
        assert_eq!(s.frames, 3);
        assert!((s.avg_fps - 125.0).abs() < 0.5, "fps {}", s.avg_fps);
        assert!((s.gpu_busy_ms - 6.0).abs() < 1e-6, "gpu {}", s.gpu_busy_ms);
        assert!((s.cpu_busy_ms - 3.0).abs() < 1e-6, "cpu {}", s.cpu_busy_ms);
        assert!((s.cpu_wait_ms - 1.0).abs() < 1e-6, "wait {}", s.cpu_wait_ms);
        assert!((s.display_latency_ms - 12.0).abs() < 1e-6);
        assert_eq!(s.dropped, 1, "undisplayed frame counts as dropped");
        assert!(s.line().contains("gpu_busy 6.00ms"), "{}", s.line());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn has_data_row_needs_more_than_a_header() {
        let path = std::env::temp_dir().join("crucible-pm-hdr.csv");
        std::fs::write(&path, "Application,ProcessID,FrameTime\n").unwrap();
        assert!(!has_data_row(&path), "header alone is not a capture");
        std::fs::write(&path, "Application,ProcessID,FrameTime\napp,1,8.0\n").unwrap();
        assert!(has_data_row(&path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn summarize_falls_back_to_v1_columns() {
        let path = std::env::temp_dir().join("crucible-pmtest-v1.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "Application,ProcessID,msBetweenPresents,Dropped").unwrap();
            writeln!(f, "cec-crucible.exe,123,10.0,0").unwrap();
            writeln!(f, "cec-crucible.exe,123,10.0,1").unwrap();
        }
        let s = summarize(&path).expect("summary");
        assert_eq!(s.frames, 2);
        assert!((s.avg_fps - 100.0).abs() < 0.5, "fps {}", s.avg_fps);
        assert_eq!(s.dropped, 1);
        let _ = std::fs::remove_file(&path);
    }
}
