// SPDX-License-Identifier: MIT
// Part of the frame-pacing API (the score + its per-percentile fields) is used
// only by the windows+preview benchmark path; the normal render run uses the
// rest. Both live here, so some items read as dead in a non-preview build.
#![allow(dead_code)]
//! Frame-pacing recorder for the `render` benchmark mode.
//!
//! The benchmark presents every rendered frame at full rate (vsync off) and
//! stamps the wall-clock instant of each present here. The present-to-present
//! deltas *are* the frame times; everything a 3DMark-style score needs — average
//! throughput plus the 1% / 0.1% lows that expose stutter — is a reduction over
//! them. Frame-pacing (how *evenly* frames arrive), not peak throughput, is what
//! a player feels, so the score weights the lows heavily (see [`FrameStats`]).
//!
//! ## Bounded memory
//!
//! Only the most recent [`CAP`] deltas are retained for the distribution stats.
//! A fixed-duration bench never approaches the cap (200k frames is ~23 min at
//! 144 fps); the bound simply stops a pathological multi-hour run from growing
//! without limit. The frame count, summed frame time and worst-frame (stutter)
//! are tracked incrementally, so they stay correct across the whole run
//! regardless of the retention window.
//!
//! ## Gaps are not frames
//!
//! An off-phase of a burst shape, or the stall of a verification read-back, is
//! *not* a rendered frame — charging that idle span as one giant frame time
//! would wreck the percentiles. Callers ring [`FrameTimer::interrupt`] around
//! such gaps so the next present starts a fresh interval instead.

use std::collections::VecDeque;
use std::time::Instant;

/// Cap on retained per-frame samples (~23 min at 144 fps). Reached only by a
/// runaway multi-hour run; a normal timed bench stays far under it.
const CAP: usize = 200_000;

/// Score scale. The three score weights (see [`FrameStats::score`]) sum to 1.0,
/// so a *perfectly smooth* run — one whose average, 1%-low and 0.1%-low fps are
/// all equal — scores exactly `fps * SCALE`. `SCALE` is fixed so that the
/// reference "should feel gold-smooth" run at 144 fps lands on 10000. A
/// vsync-locked 60 fps run then scores ~4167 and an uncapped 240 fps run
/// ~16667, keeping the number an intuitive "smooth frames per second, ×69.4".
pub const SCALE: f64 = 10_000.0 / 144.0;

/// Present-to-present frame-time recorder.
#[derive(Default)]
pub struct FrameTimer {
    /// Instant of the previous present. `None` before the first present and
    /// after an [`interrupt`](Self::interrupt), so the next present opens a new
    /// interval rather than recording the gap as a frame.
    last: Option<Instant>,
    /// Total present-to-present intervals recorded — the measured frame count.
    frames: u64,
    /// Sum of all recorded frame times (ms). Drives the average independently of
    /// the retention window, so it is exact over the whole run.
    sum_ms: f64,
    /// Worst single frame time seen (ms) — the stutter, tracked globally so it
    /// survives the ring-buffer window.
    max_ms: f64,
    /// Recent per-frame deltas (ms), oldest at the front, capped at [`CAP`]. f32
    /// halves the footprint and is ample precision for a millisecond frame time.
    samples: VecDeque<f32>,
}

impl FrameTimer {
    pub fn new() -> FrameTimer {
        FrameTimer::default()
    }

    /// Record a present at `now`. The first present only seeds the interval (no
    /// delta yet); each subsequent present adds one present-to-present frame
    /// time. Cheap and allocation-free once the ring has grown.
    pub fn present(&mut self, now: Instant) {
        if let Some(prev) = self.last {
            let ms = now.saturating_duration_since(prev).as_secs_f64() * 1000.0;
            self.frames += 1;
            self.sum_ms += ms;
            if ms > self.max_ms {
                self.max_ms = ms;
            }
            if self.samples.len() >= CAP {
                self.samples.pop_front();
            }
            self.samples.push_back(ms as f32);
        }
        self.last = Some(now);
    }

    /// Break the interval: the next [`present`](Self::present) will not record a
    /// delta. Used around genuine gaps (a burst off-phase, or the verification
    /// read-back stall) so they never pollute the frame-time distribution.
    pub fn interrupt(&mut self) {
        self.last = None;
    }

    /// Running average fps over the whole run (`1000 * frames / summed_ms`), for
    /// the throttled live-status line. Zero until the first frame is timed.
    pub fn fps(&self) -> f64 {
        if self.sum_ms > 0.0 {
            1000.0 * self.frames as f64 / self.sum_ms
        } else {
            0.0
        }
    }

    /// Reduce the recorded frames to the summary stats a score is built from.
    /// Percentiles are taken over the retained window (all frames under the cap);
    /// count, summed time and stutter are global.
    pub fn stats(&self) -> FrameStats {
        let frames = self.frames;
        let elapsed_s = self.sum_ms / 1000.0;
        let avg_fps = self.fps();
        let mean_ms = if frames > 0 {
            self.sum_ms / frames as f64
        } else {
            0.0
        };

        let mut sorted: Vec<f32> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median_ms = percentile(&sorted, 0.50);
        let p95_ms = percentile(&sorted, 0.95);
        let p99_ms = percentile(&sorted, 0.99);
        let p999_ms = percentile(&sorted, 0.999);

        // A frame time's reciprocal is its fps; the 1%-low is thus the fps of the
        // 99th-percentile (slowest 1%) frame, the 0.1%-low that of the 99.9th.
        let low1pct_fps = if p99_ms > 0.0 { 1000.0 / p99_ms } else { 0.0 };
        let low01pct_fps = if p999_ms > 0.0 { 1000.0 / p999_ms } else { 0.0 };

        // A "dropped" frame is one that took more than twice the median — the
        // usual hitch heuristic.
        let dropped = if median_ms > 0.0 {
            let thresh = 2.0 * median_ms;
            self.samples.iter().filter(|&&d| d as f64 > thresh).count() as u64
        } else {
            0
        };

        FrameStats {
            frames,
            elapsed_s,
            avg_fps,
            mean_ms,
            median_ms,
            p95_ms,
            p99_ms,
            stutter_ms: self.max_ms,
            low1pct_fps,
            low01pct_fps,
            dropped,
        }
    }
}

/// The reduced frame-pacing statistics for one benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    /// Present-to-present intervals measured.
    pub frames: u64,
    /// Summed measured frame time (s) — excludes any interrupted gaps.
    pub elapsed_s: f64,
    /// Average fps over the run (`frames / elapsed_s`).
    pub avg_fps: f64,
    /// Mean frame time (ms).
    pub mean_ms: f64,
    /// Median (50th-pct) frame time (ms).
    pub median_ms: f64,
    /// 95th-percentile frame time (ms).
    pub p95_ms: f64,
    /// 99th-percentile frame time (ms) — the 1%-low pivot.
    pub p99_ms: f64,
    /// Worst single frame time (ms) — the stutter.
    pub stutter_ms: f64,
    /// 1%-low fps = `1000 / p99_ms`.
    pub low1pct_fps: f64,
    /// 0.1%-low fps = `1000 / p99.9_ms`.
    pub low01pct_fps: f64,
    /// Frames slower than 2× the median (hitches).
    pub dropped: u64,
}

impl FrameStats {
    /// The single normalized score. Rewards throughput *and* consistency by
    /// blending the average with the 1% / 0.1% lows — frame-pacing is the point,
    /// so the lows carry half the weight between them:
    ///
    /// `score = round((0.5·avg_fps + 0.35·low1pct + 0.15·low01pct) · SCALE)`
    ///
    /// See [`SCALE`] for the calibration (smooth 144 fps → 10000). Never
    /// negative; only meaningful for a VALID run (see [`detail_valid`]).
    ///
    /// [`detail_valid`]: Self::detail_valid
    pub fn score(&self) -> u64 {
        let weighted =
            0.5 * self.avg_fps + 0.35 * self.low1pct_fps + 0.15 * self.low01pct_fps;
        (weighted * SCALE).round().max(0.0) as u64
    }

    /// The compact, parseable benchmark line for `LoadResult.detail` — the score
    /// and every stat as `key=value` pairs, so it flows into the JSON report and
    /// results CSV detail column with no schema change. Emitted only for a run
    /// that verified clean; an INVALID run is reported by the caller with a
    /// score of 0 and never a throughput number a bad-pixel render could inflate.
    pub fn detail_valid(&self) -> String {
        format!(
            "bench: score={} VALID fps={:.1} 1%low={:.1} 0.1%low={:.1} mean_ms={:.2} \
             med_ms={:.2} p95_ms={:.2} p99_ms={:.2} stutter_ms={:.2} dropped={} frames={} \
             elapsed_s={:.1}",
            self.score(),
            self.avg_fps,
            self.low1pct_fps,
            self.low01pct_fps,
            self.mean_ms,
            self.median_ms,
            self.p95_ms,
            self.p99_ms,
            self.stutter_ms,
            self.dropped,
            self.frames,
            self.elapsed_s,
        )
    }
}

/// Nearest-rank percentile of an already-sorted slice; `q` in `[0, 1]`. Returns
/// 0 for an empty slice. Rank is on the `[0, n-1]` scale so `q = 0` / `q = 1`
/// land on the min / max.
fn percentile(sorted: &[f32], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let rank = (q * (n as f64 - 1.0)).round() as usize;
    sorted[rank.min(n - 1)] as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Build a timer fed a first (seeding) present then one present per delta.
    fn timer_with(deltas_ms: &[f64]) -> FrameTimer {
        let mut t = FrameTimer::new();
        let mut clock = Instant::now();
        t.present(clock); // seeds the interval; no delta recorded
        for &d in deltas_ms {
            clock += Duration::from_secs_f64(d / 1000.0);
            t.present(clock);
        }
        t
    }

    #[test]
    fn smooth_144fps_scores_about_10000() {
        // A perfectly even 144 fps run is the 10000 reference point.
        let dt = 1000.0 / 144.0;
        let s = timer_with(&vec![dt; 2000]).stats();
        assert_eq!(s.frames, 2000);
        assert!((s.avg_fps - 144.0).abs() < 0.5, "avg_fps = {}", s.avg_fps);
        let score = s.score();
        assert!((9950..=10050).contains(&score), "score = {score}");
    }

    #[test]
    fn locked_60fps_scores_about_4167() {
        let s = timer_with(&vec![1000.0 / 60.0; 1000]).stats();
        let score = s.score();
        assert!((4130..=4200).contains(&score), "score = {score}");
    }

    #[test]
    fn percentiles_and_dropped_track_a_spike() {
        // 999 even 5 ms frames plus one 50 ms stutter.
        let mut deltas = vec![5.0; 999];
        deltas.push(50.0);
        let s = timer_with(&deltas).stats();
        assert!((s.median_ms - 5.0).abs() < 0.01, "median = {}", s.median_ms);
        assert!(s.stutter_ms >= 49.0, "stutter = {}", s.stutter_ms);
        assert_eq!(s.dropped, 1, "the 50 ms frame is > 2x median");
    }

    #[test]
    fn interrupt_drops_the_gap_interval() {
        // A 10 s gap between two real frames must not be timed as a frame.
        let mut t = FrameTimer::new();
        let c = Instant::now();
        t.present(c);
        t.present(c + Duration::from_millis(5)); // 5 ms frame
        t.interrupt();
        t.present(c + Duration::from_secs(10)); // gap end — seeds, no delta
        t.present(c + Duration::from_millis(10_015)); // 15 ms frame
        let s = t.stats();
        assert_eq!(s.frames, 2, "the gap must not count");
        assert!(s.stutter_ms < 100.0, "gap leaked in: {}", s.stutter_ms);
    }

    #[test]
    fn empty_timer_is_all_zero() {
        let s = FrameTimer::new().stats();
        assert_eq!(s.frames, 0);
        assert_eq!(s.avg_fps, 0.0);
        assert_eq!(s.score(), 0);
    }
}
