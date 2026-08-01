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
use std::io::{BufRead, BufReader};
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
    /// Where PresentMon's own stderr went, so a failure can be explained rather
    /// than merely reported.
    err_log: PathBuf,
    /// True when PresentMon is genuinely our child (we were already elevated),
    /// which is what makes stopping it immediate rather than a wait.
    owns_child: bool,
    /// PresentMon's own path, so the capture can be stopped by session name.
    exe: PathBuf,
    /// When the capture was spawned, and the `--timed` backstop we gave it — so
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
        // Elevation decides whether the capture is OURS.
        //
        // PresentMon's trace session needs an Administrator token. Unelevated, it
        // relaunches itself with `--restart_as_admin`, and the process that ends
        // up writing the CSV is no longer our child — we cannot stop it, so the
        // capture only ends when its own `--timed` timer fires. Elevated, it is a
        // normal child we can end the instant the run does.
        let elevated = crucible_core::sysinfo::is_elevated();

        // `--timed` is a BACKSTOP, not the plan.
        //
        // It used to be `seconds + 2`, taken from the *requested* duration, which
        // was wrong in both directions: a run that overran (a cross-load asked for
        // 20s routinely takes 35) had its capture cut off early, and a run stopped
        // early left `finish` sleeping out the whole remaining window for nothing.
        // Now it is set well past any plausible run so it never truncates, and the
        // capture is ended actively instead.
        let backstop = backstop_seconds(seconds);

        let mut args = vec![
            "--process_id".to_string(),
            pid.to_string(),
            "--output_file".to_string(),
            csv.to_string_lossy().into_owned(),
            "--timed".to_string(),
            backstop.to_string(),
            "--terminate_after_timed".to_string(),
            "--stop_existing_session".to_string(),
            // A FIXED name, deliberately not per-run.
            //
            // An ETW session outlives the process that created it: kill
            // PresentMon and its session keeps running in the kernel. With a
            // per-pid name every run left a `crucible-<pid>` session behind and
            // `--stop_existing_session` — which only matches its own name —
            // never reclaimed any of them. Two were found running on this bench,
            // against a system-wide limit of 64. One fixed name makes the next
            // run's `--stop_existing_session` clean up the last run's leak, so
            // the failure mode is self-healing rather than cumulative.
            "--session_name".to_string(),
            SESSION_NAME.to_string(),
            // Never outlive us: if this run crashes, the session goes with it
            // rather than blocking the next one.
            "--terminate_on_proc_exit".to_string(),
            "--no_console_stats".to_string(),
            // Pin the schema: v2 metrics carry the present-pipeline breakdown
            // (GPUBusy / CPUBusy / CPUWait / DisplayLatency) this suite reports.
            "--v2_metrics".to_string(),
        ];
        if !elevated {
            args.push("--restart_as_admin".to_string());
        }

        // Keep stderr instead of discarding it.
        //
        // PresentMon explains itself when it fails — "access denied", "N ETW
        // events were lost" — and all of that was going to the null device, so
        // every failure looked identical from the outside: a long wait and no
        // data. On this bench it turned out to be losing ~180,000 events per six
        // seconds and capturing nothing, which the tool had no way to say.
        let err_log = csv.with_extension("log");
        let stderr = std::fs::File::create(&err_log)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let child = Command::new(exe)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()?;
        Ok(Capture {
            child,
            csv,
            started: Instant::now(),
            window: Duration::from_secs(backstop),
            err_log,
            owns_child: elevated,
            exe: exe.to_path_buf(),
        })
    }

    /// Where the capture is writing. Needed by the caller before `finish`
    /// consumes the handle, so a failure can still be explained afterwards.
    pub fn csv_path(&self) -> PathBuf {
        self.csv.clone()
    }

    /// Wait for the capture to land. PresentMon stops itself after its `--timed`
    /// window, but with `--restart_as_admin` the writing process is a *different*
    /// (elevated) one, so we wait on the artifact: poll until the CSV appears and
    /// its size stops growing. Returns `None` on timeout — practically always a
    /// declined UAC prompt (no elevation, no ETW session, no data).
    pub fn finish(mut self) -> Option<PathBuf> {
        // End the capture NOW rather than waiting for its timer.
        //
        // The old code slept out the remaining `--timed` window unconditionally.
        // On a run that finished on schedule that was a couple of wasted seconds;
        // on a soak the operator stopped early it was minutes of a progress-less
        // wait, which is what made this feel interminable.
        // Stop the SESSION, not merely the process.
        //
        // Killing PresentMon does not stop its ETW session — the session lives in
        // the kernel and keeps running, holding buffers, until something stops it
        // by name. `--terminate_existing_session` is that something, and it is
        // done FIRST so the session is gone whether or not the process is ours.
        let _ = Command::new(&self.exe)
            .args([
                "--terminate_existing_session",
                "--session_name",
                SESSION_NAME,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(|mut c| {
                let _ = c.wait();
            });

        if self.owns_child {
            // Elevated: it is our child, so it can also be ended directly, as a
            // backstop for the case where the session stop did not reach it.
            // PresentMon writes the CSV incrementally, so rows already on disk
            // survive either way.
            let _ = self.child.kill();
            let _ = self.child.wait();
        }

        // Nothing to wait for if nothing was ever captured.
        //
        // Unelevated, PresentMon needs a UAC prompt to relaunch itself; declined
        // or dismissed, no session starts and no CSV is ever created. The old
        // code still sat out the entire `--timed` window for a file that would
        // never appear — a 3-second test spent two and a half minutes waiting on
        // a capture that did not exist.
        if !self.csv.exists() {
            return None;
        }

        // PresentMon writes in buffered bursts, so "size didn't change since last
        // poll" is NOT proof it finished — early on the file is briefly stable
        // holding only its header. So completion still needs a stable size over
        // several polls AND at least one data row. What changed is that we start
        // checking immediately instead of after a fixed sleep.
        let deadline = Instant::now() + self.wait_budget();
        let mut last_len: u64 = u64::MAX;
        let mut stable_for = 0u32;
        while Instant::now() < deadline {
            if let Ok(m) = std::fs::metadata(&self.csv) {
                let len = m.len();
                if len == last_len && len > 0 {
                    stable_for += 1;
                    // ~1.2s unchanged, and it holds more than just a header row.
                    if stable_for >= 4 {
                        if has_data_row(&self.csv) {
                            return Some(std::mem::take(&mut self.csv));
                        }
                        // Stable, non-empty, and still only a header: the capture
                        // has stopped and no frame ever arrived. Nothing is
                        // coming, so waiting out the rest of the budget buys
                        // nothing — and this is the COMMON case, because most of
                        // the suite never presents. A `worst-case` cross-load
                        // drives compute, memory, storage and PCIe; not one of
                        // those puts a frame on screen, so PresentMon writes its
                        // header and then sits there while we wait for frames
                        // that cannot exist.
                        return None;
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

    /// How long to keep polling for the CSV to settle, once the capture has been
    /// asked to stop.
    ///
    /// **Seconds either way, never the length of the run.** PresentMon writes the
    /// CSV incrementally, so we never needed it to *exit* — only to stop writing
    /// long enough that we are not reading a half-flushed line. Waiting out its
    /// `--timed` window bought nothing and cost the whole window.
    ///
    /// A little longer when the writer is not our child, because the stop request
    /// has to travel through a second process; still bounded, and whatever rows
    /// are on disk when the budget expires are returned rather than discarded.
    fn wait_budget(&self) -> Duration {
        if self.owns_child {
            Duration::from_secs(6)
        } else {
            Duration::from_secs(12)
        }
    }
}

/// The ETW session name. Fixed, so a leak is cleaned up by the next run rather
/// than accumulating against the system's 64-session limit.
pub const SESSION_NAME: &str = "cec-crucible";

/// Whatever PresentMon last said on stderr, trimmed to something printable.
///
/// This is the difference between "PresentMon captured nothing" and "PresentMon
/// captured nothing because 183,445 ETW events were lost" — the second is a
/// diagnosis, and it was being thrown away.
///
/// Often empty in practice, and that is expected rather than broken: PresentMon
/// prints its summary warnings as it exits, and the normal path now ends its
/// session before it gets there. What this reliably catches is the failure that
/// happens at *startup* — "access denied" when the session cannot be created —
/// which is the one an operator most needs to be told about.
pub fn last_error(csv: &Path) -> Option<String> {
    let log = csv.with_extension("log");
    let text = std::fs::read_to_string(log).ok()?;
    let msg: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(3)
        .collect();
    if msg.is_empty() {
        None
    } else {
        Some(msg.join("; "))
    }
}

/// The `--timed` backstop for a run of `seconds`.
///
/// Generous on purpose. It exists only so a crashed run cannot leave a trace
/// session recording forever; it must never be the thing that ends a capture,
/// because a run routinely takes longer than it was asked for — a cross-load
/// requested at 20s measures 35s once every stage's setup and teardown is
/// counted. The old value of `seconds + 2` truncated exactly those runs.
fn backstop_seconds(seconds: u64) -> u64 {
    seconds.saturating_mul(2).saturating_add(120)
}

/// True once the CSV holds at least one row beyond its header — i.e. real frames
/// were captured, not just an opened file.
fn has_data_row(csv: &Path) -> bool {
    // Read the first couple of lines, not the file.
    //
    // This is called from a poll loop, and a long capture's CSV runs to hundreds
    // of megabytes — `read_to_string` on that, repeatedly, is its own stall. The
    // question is only "is there anything past the header", which the first two
    // non-empty lines answer.
    let Ok(f) = std::fs::File::open(csv) else {
        return false;
    };
    let mut seen = 0;
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        if seen > 1 {
            return true;
        }
    }
    false
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
    fn the_backstop_never_ends_a_run_that_overran() {
        // Measured on the bench: `run worst-case --seconds 20` takes ~35s wall,
        // and `--seconds 15` GPU runs take ~17s. The old `seconds + 2` cut the
        // first of those off 13 seconds early.
        for (asked, actual) in [(20u64, 35u64), (15, 17), (60, 95), (300, 380)] {
            assert!(
                backstop_seconds(asked) > actual,
                "a {asked}s run that actually took {actual}s would be truncated at {}",
                backstop_seconds(asked)
            );
        }
        // And it stays finite on an absurd request rather than overflowing.
        assert!(backstop_seconds(u64::MAX) >= u64::MAX - 1);
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
