// SPDX-License-Identifier: MIT
//! Load markers — the correlation feed for the external 1kHz+ power rig.
//!
//! The software cannot (and need not) sample power at 1kHz; it need only
//! *timestamp the load edges* precisely. Every run-, stage-, and burst-edge
//! transition is stamped with a QPC timestamp into a shared, thread-safe log
//! and written out as JSONL. The analog capture then aligns on `qpc_ticks`.

use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::clock::{Clock, Timestamp};
use crate::json::Json;

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
}

impl MarkerLog {
    pub fn new(clock: Clock) -> MarkerLog {
        MarkerLog {
            clock,
            seq: AtomicU64::new(0),
            markers: Mutex::new(Vec::new()),
        }
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
}
