// SPDX-License-Identifier: MIT
//! Load markers — the correlation feed for the external 1kHz+ power rig.
//!
//! The software cannot (and need not) sample power at 1kHz; it need only
//! *timestamp the load edges* precisely. Every run-, stage-, and burst-edge
//! transition is stamped with a QPC timestamp into a shared, thread-safe log
//! and written out as JSONL. The analog capture then aligns on `qpc_ticks`.

use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::clock::{Clock, Timestamp};
use crate::cpustats::CoreStat;
use crate::gputel::GpuSample;
use crate::json::Json;

/// Live per-lane activity for an in-process UI (`--ui`) or a harness readout.
///
/// Separate from the marker log (which only records *edges*, and only for bursty
/// shapes): a lane carries a running work counter and a current phase, so the UI
/// can show steady kernels and per-core activity too. Updates are plain relaxed
/// atomics on an `Arc` each worker holds — no lock on the hot path — and the
/// whole mechanism is inert unless a UI enabled it (see [`MarkerLog::live_on`]).
#[derive(Debug)]
pub struct LiveLane {
    pub label: String,
    pub work: AtomicU64,
    pub errors: AtomicU64,
    /// 0 = idle, 1 = working, 2 = done.
    pub phase: AtomicU8,
    /// Latest verification checksum the kernel is reproducing (0 = none yet) — so
    /// the UI can show the live "memory hash" the self-consistency check folds.
    pub hash: AtomicU64,
    /// A short multi-line "field: value" status the kernel pushes (throttled) so
    /// the UI can show what it is *actually* doing — the memory pattern, the value
    /// written vs expected, progress, watts. Empty until the kernel sets it.
    pub detail: Mutex<String>,
}

pub const PHASE_IDLE: u8 = 0;
pub const PHASE_WORK: u8 = 1;
pub const PHASE_DONE: u8 = 2;

impl LiveLane {
    fn new(label: String) -> LiveLane {
        LiveLane {
            label,
            work: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            phase: AtomicU8::new(PHASE_IDLE),
            hash: AtomicU64::new(0),
            detail: Mutex::new(String::new()),
        }
    }
    #[inline]
    pub fn bump_work(&self) {
        self.work.fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn set_phase(&self, p: u8) {
        self.phase.store(p, Ordering::Relaxed);
    }
    #[inline]
    pub fn set_hash(&self, h: u64) {
        self.hash.store(h, Ordering::Relaxed);
    }
    /// Replace the live status detail (a brief lock; called at a throttled rate,
    /// not on the hot inner loop). No-op churn when the text is unchanged.
    pub fn set_detail(&self, s: &str) {
        let mut g = self.detail.lock().unwrap_or_else(|e| e.into_inner());
        if *g != *s {
            g.clear();
            g.push_str(s);
        }
    }
}

/// An immutable snapshot of one lane for the renderer.
#[derive(Debug, Clone)]
pub struct LaneSnap {
    pub label: String,
    pub work: u64,
    pub errors: u64,
    pub phase: u8,
    pub hash: u64,
    pub detail: String,
}

/// CSV header for the periodic run-telemetry log (one row per lane per sample).
/// The trailing `eff_mhz,util_pct` carry PDH per-core telemetry: populated on
/// `core N` rows (and standalone `cpu N` rows), blank on other lanes.
pub fn telemetry_csv_header() -> &'static str {
    "elapsed_s,lane,work,phase,errors,hash_hex,eff_mhz,util_pct,\
gpu_power_w,gpu_temp_c,gpu_mem_temp_c,gpu_fan_pct,gpu_sm_mhz,gpu_throttle\n"
}

/// GPU sensor columns for one sample, or empty cells when no GPU telemetry is
/// available. Blank rather than 0 deliberately: a zero-watt reading would look
/// like a power failure on a graph, whereas an empty cell reads as "not measured".
fn gpu_cols(g: Option<&GpuSample>) -> String {
    match g {
        Some(s) => format!(
            "{:.1},{},{},{},{},{:#x}",
            s.power_w,
            s.temp_c,
            // Most consumer boards do not expose a memory-junction sensor, and
            // NVML answers with an error we carry as 0. Writing that 0 would
            // assert a 0 °C junction — which reads as a real measurement, and on
            // a shared temperature axis drags the whole chart to the floor.
            // Blank means "no sensor", which is the truth.
            blank_if_zero(s.mem_temp_c),
            s.fan_pct, // a genuine 0 here: zero-RPM idle mode
            s.sm_mhz,
            s.throttle
        ),
        None => ",,,,,".to_string(),
    }
}

fn blank_if_zero(v: u32) -> String {
    if v == 0 {
        String::new()
    } else {
        v.to_string()
    }
}

/// Parse the core index out of a `"core N"` lane label.
fn core_index(label: &str) -> Option<u32> {
    label.strip_prefix("core ").and_then(|n| n.trim().parse().ok())
}

/// Format one telemetry sample — every lane's counters at `elapsed_s` — as CSV
/// rows for the run-telemetry log: a time series (rate is `d(work)/dt`, i.e. the
/// derivative of the `work` column) you can graph after a run. Lane labels are
/// controlled ("core N", "mem", …) but a stray comma is guarded so the columns
/// stay stable.
///
/// `cpu` is the current PDH per-core snapshot (empty when unavailable). A core's
/// effective clock + utilization are attached to its matching `core N` load
/// lane; cores with telemetry but no load lane this sample (e.g. during a
/// GPU-only test) get a standalone `cpu N` row so the whole chip is still logged.
pub fn telemetry_csv_rows(
    elapsed_s: f64,
    lanes: &[LaneSnap],
    cpu: &[CoreStat],
    gpu: Option<&GpuSample>,
) -> String {
    use std::collections::{BTreeMap, BTreeSet};
    let by_core: BTreeMap<u32, &CoreStat> = cpu.iter().map(|c| (c.core, c)).collect();
    let mut charted: BTreeSet<u32> = BTreeSet::new();

    let mut s = String::new();
    for l in lanes {
        let phase = match l.phase {
            PHASE_WORK => "work",
            PHASE_DONE => "done",
            _ => "idle",
        };
        let label = l.label.replace(',', "_");
        // Attach per-core telemetry to this core's load lane, if we have it.
        let ci = core_index(&l.label);
        if let Some(i) = ci {
            charted.insert(i);
        }
        let (mhz, util) = match ci.and_then(|i| by_core.get(&i)) {
            Some(cs) => (cs.effective_mhz.to_string(), format!("{:.1}", cs.util_pct)),
            None => (String::new(), String::new()),
        };
        s.push_str(&format!(
            "{elapsed_s:.3},{label},{},{phase},{},{:#018x},{mhz},{util},{g}\n",
            l.work,
            l.errors,
            l.hash,
            g = gpu_cols(gpu)
        ));
    }

    // Cores that have telemetry but no load lane this sample — still log them so
    // throttling/boost is captured even during a pure GPU/storage test.
    for (core, cs) in &by_core {
        if !charted.contains(core) {
            s.push_str(&format!(
                "{elapsed_s:.3},cpu {core},0,idle,0,{:#018x},{},{:.1},{g}\n",
                0u64,
                cs.effective_mhz,
                cs.util_pct,
                g = gpu_cols(gpu)
            ));
        }
    }
    s
}

/// The kind of transition a marker records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Whole run started / stopped.
    RunStart,
    RunStop,
    /// A single kernel stage started / stopped.
    StageStart,
    StageStop,
    /// A burst duty-cycle rising / falling edge (the transients the rig hunts).
    BurstOn,
    BurstOff,
    /// A generic point marker.
    Mark,
}

impl Event {
    pub fn as_str(&self) -> &'static str {
        match self {
            Event::RunStart => "run_start",
            Event::RunStop => "run_stop",
            Event::StageStart => "stage_start",
            Event::StageStop => "stage_stop",
            Event::BurstOn => "burst_on",
            Event::BurstOff => "burst_off",
            Event::Mark => "mark",
        }
    }
}

/// One timestamped transition.
#[derive(Debug, Clone)]
pub struct Marker {
    pub seq: u64,
    pub event: Event,
    pub kernel: String,
    pub mode: String,
    pub ts: Timestamp,
    pub detail: String,
}

impl Marker {
    fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("seq", self.seq)
            .push("event", self.event.as_str())
            .push("kernel", self.kernel.as_str())
            .push("mode", self.mode.as_str())
            .push("qpc_ticks", self.ts.qpc_ticks)
            .push("qpc_frequency", self.ts.qpc_frequency)
            .push("unix_nanos", self.ts.unix_nanos.to_string())
            .push("detail", self.detail.as_str());
        o
    }
}

/// A shared, thread-safe marker log. One instance per run; kernels stamp their
/// own transitions into it concurrently (this is how cross-load stays on one
/// timeline). Cloneable handle semantics are provided by sharing `&MarkerLog`.
pub struct MarkerLog {
    clock: Clock,
    seq: AtomicU64,
    markers: Mutex<Vec<Marker>>,
    /// Live lanes for the UI. Registered once per worker; empty and untouched
    /// unless a UI turned tracking on.
    live: Mutex<Vec<Arc<LiveLane>>>,
    live_on: AtomicBool,
}

impl MarkerLog {
    pub fn new(clock: Clock) -> MarkerLog {
        MarkerLog {
            clock,
            seq: AtomicU64::new(0),
            markers: Mutex::new(Vec::new()),
            live: Mutex::new(Vec::new()),
            live_on: AtomicBool::new(false),
        }
    }

    /// Turn live-lane tracking on (an in-process UI does this). While off,
    /// [`register_lane`](Self::register_lane) returns `None`, so kernels do no
    /// live bookkeeping at all — the default, zero-overhead path.
    pub fn enable_live(&self) {
        self.live_on.store(true, Ordering::Relaxed);
    }

    pub fn live_enabled(&self) -> bool {
        self.live_on.load(Ordering::Relaxed)
    }

    /// Register (or fetch the existing) live lane for `label`. Returns `None`
    /// when tracking is off so the caller skips all per-tick bookkeeping.
    pub fn register_lane(&self, label: &str) -> Option<Arc<LiveLane>> {
        if !self.live_enabled() {
            return None;
        }
        let mut g = self.live.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(l) = g.iter().find(|l| l.label == label) {
            return Some(Arc::clone(l));
        }
        let lane = Arc::new(LiveLane::new(label.to_string()));
        g.push(Arc::clone(&lane));
        Some(lane)
    }

    /// Snapshot every lane's current counters for the renderer.
    pub fn live_snapshot(&self) -> Vec<LaneSnap> {
        let g = self.live.lock().unwrap_or_else(|e| e.into_inner());
        g.iter()
            .map(|l| LaneSnap {
                label: l.label.clone(),
                work: l.work.load(Ordering::Relaxed),
                errors: l.errors.load(Ordering::Relaxed),
                phase: l.phase.load(Ordering::Relaxed),
                hash: l.hash.load(Ordering::Relaxed),
                detail: l.detail.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            })
            .collect()
    }

    /// The clock backing this log — kernels read `now()` from it when they need
    /// a timestamp without stamping a marker.
    pub fn clock(&self) -> Clock {
        self.clock
    }

    /// Stamp a transition and return the timestamp captured.
    ///
    /// The QPC read happens *before* taking the lock so the timestamp reflects
    /// the true edge time, not lock-acquisition time.
    pub fn stamp(&self, event: Event, kernel: &str, mode: &str, detail: &str) -> Timestamp {
        let ts = self.clock.now();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let marker = Marker {
            seq,
            event,
            kernel: kernel.to_string(),
            mode: mode.to_string(),
            ts,
            detail: detail.to_string(),
        };
        // Poisoning is unrecoverable telemetry loss but must never take down a
        // stress run mid-soak; recover the guard and carry on.
        let mut guard = self.markers.lock().unwrap_or_else(|e| e.into_inner());
        guard.push(marker);
        ts
    }

    pub fn len(&self) -> usize {
        self.markers.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Render all markers as JSONL (one compact object per line), ordered by the
    /// sequence in which they were stamped.
    pub fn to_jsonl(&self) -> String {
        let guard = self.markers.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = String::new();
        for m in guard.iter() {
            out.push_str(&m.to_json().to_compact());
            out.push('\n');
        }
        out
    }

    pub fn write_jsonl(&self, path: &Path) -> io::Result<()> {
        std::fs::write(path, self.to_jsonl())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_are_sequenced_and_serialized() {
        let log = MarkerLog::new(Clock::new());
        log.stamp(Event::RunStart, "run", "", "");
        log.stamp(Event::BurstOn, "cpu", "burst", "core=all");
        log.stamp(Event::RunStop, "run", "", "");
        assert_eq!(log.len(), 3);

        let jsonl = log.to_jsonl();
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains(r#""event":"run_start""#));
        assert!(lines[0].contains(r#""seq":0"#));
        assert!(lines[1].contains(r#""event":"burst_on""#));
        assert!(lines[1].contains(r#""detail":"core=all""#));
        assert!(lines[2].contains(r#""seq":2"#));
    }

    #[test]
    fn concurrent_stamps_are_all_recorded() {
        use std::sync::Arc;
        let log = Arc::new(MarkerLog::new(Clock::new()));
        let mut handles = Vec::new();
        for t in 0..8 {
            let log = Arc::clone(&log);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    log.stamp(Event::Mark, "k", "", &format!("t{t}"));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(log.len(), 800);
        // Sequence numbers must be unique 0..800.
        let jsonl = log.to_jsonl();
        assert_eq!(jsonl.lines().count(), 800);
    }

    fn snap(label: &str, work: u64, phase: u8) -> LaneSnap {
        LaneSnap {
            label: label.to_string(),
            work,
            errors: 0,
            phase,
            hash: 0,
            detail: String::new(),
        }
    }

    #[test]
    fn telemetry_csv_attaches_and_backfills_cpu() {
        let lanes = vec![snap("core 0", 100, PHASE_WORK), snap("mem", 42, PHASE_WORK)];
        let cpu = vec![
            CoreStat { core: 0, effective_mhz: 4800, util_pct: 99.5 },
            CoreStat { core: 1, effective_mhz: 1200, util_pct: 3.0 },
        ];
        let out = telemetry_csv_rows(1.5, &lanes, &cpu, None);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3, "core 0 + mem + backfilled cpu 1");

        // Assert on FIELD POSITIONS, not on line suffixes: GPU sensor columns
        // now follow the CPU ones, and a suffix assertion silently stops testing
        // what it claims the moment another column is appended.
        let field = |line: &str, name: &str| -> String {
            let hdr: Vec<&str> = telemetry_csv_header().trim().split(',').collect();
            let i = hdr.iter().position(|h| *h == name).expect("known column");
            line.split(',').nth(i).unwrap_or_default().to_string()
        };

        // The core lane carries its own effective clock + utilization.
        let core0 = lines.iter().find(|l| l.contains(",core 0,")).unwrap();
        assert_eq!(field(core0, "eff_mhz"), "4800");
        assert_eq!(field(core0, "util_pct"), "99.5");
        // A non-core lane leaves the cpu columns blank.
        let mem = lines.iter().find(|l| l.contains(",mem,")).unwrap();
        assert_eq!(field(mem, "eff_mhz"), "");
        assert_eq!(field(mem, "util_pct"), "");
        // A core with telemetry but no load lane is backfilled as `cpu N`.
        let cpu1 = lines.iter().find(|l| l.contains(",cpu 1,")).unwrap();
        assert_eq!(field(cpu1, "eff_mhz"), "1200");
        assert_eq!(field(cpu1, "util_pct"), "3.0");
        assert!(cpu1.contains(",idle,"));

        // No GPU sample -> the sensor columns are BLANK, never 0. A zero-watt
        // reading would look like a power failure on a graph.
        assert_eq!(field(core0, "gpu_power_w"), "");
        assert_eq!(field(core0, "gpu_temp_c"), "");

        // Every row must have exactly as many fields as the header.
        let want = telemetry_csv_header().trim().split(',').count();
        for l in &lines {
            assert_eq!(l.split(',').count(), want, "column count mismatch: {l}");
        }
    }

    #[test]
    fn telemetry_csv_carries_gpu_sensors_when_present() {
        let g = GpuSample {
            power_w: 213.4,
            temp_c: 71,
            fan_pct: 62,
            sm_mhz: 1905,
            throttle: 0x4,
            ..Default::default()
        };
        let out = telemetry_csv_rows(2.0, &[snap("gpu", 9, PHASE_WORK)], &[], Some(&g));
        let line = out.lines().next().unwrap();
        let field = |name: &str| -> String {
            let hdr: Vec<&str> = telemetry_csv_header().trim().split(',').collect();
            let i = hdr.iter().position(|h| *h == name).expect("known column");
            line.split(',').nth(i).unwrap_or_default().to_string()
        };
        assert_eq!(field("gpu_power_w"), "213.4");
        assert_eq!(field("gpu_temp_c"), "71");
        assert_eq!(field("gpu_fan_pct"), "62");
        assert_eq!(field("gpu_sm_mhz"), "1905");
        assert_eq!(field("gpu_throttle"), "0x4");
        // No memory-junction sensor on this board: NVML answers with an error,
        // which we carry as 0. Writing that 0 would assert a 0 °C junction —
        // a reading, not a gap — and on a shared axis it drags the real curve
        // to the floor. Blank is the truth.
        assert_eq!(field("gpu_mem_temp_c"), "");

        let want = telemetry_csv_header().trim().split(',').count();
        assert_eq!(line.split(',').count(), want);
    }

    #[test]
    fn a_reported_memory_junction_temperature_is_written() {
        // The flip side: when the board DOES have the sensor, the value must
        // survive — the blank-if-zero rule must not swallow real readings.
        let g = GpuSample { mem_temp_c: 94, ..Default::default() };
        let out = telemetry_csv_rows(1.0, &[snap("gpu", 1, PHASE_WORK)], &[], Some(&g));
        let line = out.lines().next().unwrap();
        let hdr: Vec<&str> = telemetry_csv_header().trim().split(',').collect();
        let i = hdr.iter().position(|h| *h == "gpu_mem_temp_c").unwrap();
        assert_eq!(line.split(',').nth(i).unwrap(), "94");
    }

    #[test]
    fn telemetry_csv_blank_cpu_columns_without_pdh() {
        let out = telemetry_csv_rows(0.0, &[snap("core 0", 1, PHASE_WORK)], &[], None);
        assert_eq!(out.lines().count(), 1, "no PDH -> no backfill rows");
        let want = telemetry_csv_header().trim().split(',').count();
        assert_eq!(out.lines().next().unwrap().split(',').count(), want);
    }
}
