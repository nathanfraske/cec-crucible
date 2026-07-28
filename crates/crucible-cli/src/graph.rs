// SPDX-License-Identifier: MIT
//! Telemetry graph renderer — a run's time-series CSV as one self-contained
//! HTML page of hand-written inline SVG.
//!
//! `--telemetry-csv` logs `crucible-<stamp>.telemetry.csv`: one row per lane per
//! sample, ~4 samples/second (see `crucible_core::markers::telemetry_csv_rows`).
//! That file already answers every question a QC tech has — did the GPU throttle,
//! did a core drop clock, did the work rate collapse forty minutes into a soak —
//! but only if somebody opens it in Excel and builds a chart by hand. This module
//! turns it into a single HTML file that renders offline from a `file://` URL, so
//! a run can be eyeballed in a browser and attached to a QC report as-is.
//!
//! Design constraints, in priority order:
//!
//! 1. **Columns are matched by name, never by index.** The schema grows (GPU
//!    sensor columns are landing now). Every column is optional; a chart whose
//!    series is entirely absent is *omitted with a stated reason* rather than
//!    drawn as an empty box.
//! 2. **A blank or malformed cell is dropped, never coerced to zero.** A dropped
//!    NVML sample must not draw as a zero-watt reading — on a power graph that
//!    looks exactly like a PSU failure that never happened. See [`num`].
//! 3. **Zero dependencies, offline.** SVG is emitted as strings and the
//!    stylesheet is inlined: no plotting crate, no serde, no fonts, no scripts,
//!    no network.
//! 4. **Nothing panics.** Empty, header-only, single-row, ragged, non-numeric,
//!    NaN and infinite input all still produce a valid page.
//!
//! Colours are the CEC brand palette from `theme.rs`, so a graph reads as the
//! same surface as the live dashboard.

// Only some of the surface is wired into the CLI yet; a renderer legitimately
// exposes helpers (and palette entries) ahead of their call sites.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io;
use std::path::Path;

// ------------------------------------------------------------------ palette --
// The CEC brand palette (see theme.rs), as CSS hex. Red is reserved for errors
// and never used as a series colour, so a red mark on a chart always means one
// thing.

const C_BG: &str = "#070711";
const C_SURFACE: &str = "#161526";
const C_TEXT: &str = "#f5f5f8";
const C_DIM: &str = "#a9a9b7";
const C_PINK: &str = "#ed2398";
const C_VIOLET: &str = "#8c67f2";
const C_CYAN: &str = "#41d9f8";
const C_LAVENDER: &str = "#9f9cff";
const C_AMBER: &str = "#f2a618";
const C_RED: &str = "#ee0b2a";

/// Series colours for the multi-lane chart, cycled with [`LANE_DASHES`]: five
/// on-brand hues × three dash patterns = fifteen distinguishable lanes without
/// inventing colours that are off-palette.
const LANE_COLORS: [&str; 5] = [C_PINK, C_CYAN, C_VIOLET, C_LAVENDER, C_AMBER];
const LANE_DASHES: [&str; 3] = ["", "7 4", "2 4"];

// ----------------------------------------------------------------- geometry --
// One viewBox shared by every chart, so they stack into a column that reads as a
// single aligned timeline. The SVG is scaled by CSS (`width:100%`), so these are
// user units, not pixels.

const VB_W: f64 = 960.0;
const VB_H: f64 = 260.0;
/// Plot frame edges inside the viewBox: left, top, right, bottom.
const PL_L: f64 = 64.0;
const PL_T: f64 = 28.0;
const PL_R: f64 = 944.0;
const PL_B: f64 = 220.0;
const PL_W: f64 = PL_R - PL_L; // 880
const PL_H: f64 = PL_B - PL_T; // 192

// ------------------------------------------------------------- CSV columns ---
// Named once so the parser, the "why is this chart missing" text and the tests
// can never drift apart.

const COL_TIME: &str = "elapsed_s";
const COL_LANE: &str = "lane";
const COL_WORK: &str = "work";
const COL_ERRORS: &str = "errors";
const COL_MHZ: &str = "eff_mhz";
const COL_UTIL: &str = "util_pct";
const COL_GPU_POWER: &str = "gpu_power_w";
const COL_GPU_TEMP: &str = "gpu_temp_c";
const COL_GPU_MEM_TEMP: &str = "gpu_mem_temp_c";

// ================================================================== public ===

/// Render a telemetry CSV into a self-contained HTML file with inline SVG.
///
/// `title` is the run name shown in the page header and the browser tab.
/// Reading the CSV or writing the page can fail (the returned [`io::Error`]);
/// *parsing* cannot — malformed content degrades into omitted charts and a
/// stated reason, never an error and never a panic.
pub fn render_html(csv_path: &Path, out_path: &Path, title: &str) -> io::Result<()> {
    // Lossy on purpose: a telemetry log truncated mid-write by a hard hang is
    // exactly the run somebody most wants to look at, and a stray invalid byte
    // must not cost them the whole graph.
    let raw = std::fs::read(csv_path)?;
    let text = String::from_utf8_lossy(&raw);
    let source = csv_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| csv_path.to_string_lossy().into_owned());
    std::fs::write(out_path, html_from_csv(&text, title, &source))
}

// ================================================================== parsing ==

/// Parse one numeric cell.
///
/// Blank, whitespace, non-numeric (`n/a`, a hex hash, a phase word) and
/// non-finite (`NaN`, `inf` — both of which `f64::from_str` happily accepts)
/// cells all return `None` and are **dropped from the series**. They are never
/// coerced to `0.0`: a sensor that failed to read for one sample is a gap in the
/// line, not a reading of zero.
fn num(cell: Option<&str>) -> Option<f64> {
    let s = cell?.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// [`num`] applied to one field of a row, by optional column index. A row
/// shorter than the header simply has no such field — cells go missing, they
/// never shift, so a truncated final line costs one sample and nothing else.
fn cell_num(fields: &[&str], i: Option<usize>) -> Option<f64> {
    num(i.and_then(|i| fields.get(i)).copied())
}

/// The core index behind a `core N` (load lane) or `cpu N` (backfilled
/// telemetry-only lane) label. Anything else is a domain lane.
fn core_index(label: &str) -> Option<u32> {
    for p in ["core ", "cpu "] {
        if let Some(rest) = label.strip_prefix(p) {
            if let Ok(n) = rest.trim().parse::<u32>() {
                return Some(n);
            }
        }
    }
    None
}

/// Every lane's state at one instant — the CSV's per-lane rows folded back into
/// the sample they were written from.
#[derive(Debug, Default, Clone)]
struct Frame {
    t: f64,
    /// Effective clock / utilization keyed by core index, so a `core N` row and
    /// a backfilled `cpu N` row for the same core can never double-count.
    mhz: BTreeMap<u32, f64>,
    util: BTreeMap<u32, f64>,
    /// GPU sensors. Carried per-sample rather than per-lane: whichever row of
    /// this sample first carries a readable value wins, so it does not matter
    /// whether the writer puts them on the `gpu` row or repeats them on all.
    gpu_power: Option<f64>,
    gpu_temp: Option<f64>,
    gpu_mem_temp: Option<f64>,
    /// Cumulative work counter per domain lane (core lanes excluded).
    work: BTreeMap<String, f64>,
    /// Cumulative error counters summed over every lane in this sample.
    errors: f64,
}

/// A parsed telemetry log.
#[derive(Debug, Default)]
struct Telemetry {
    /// Header names, lower-cased — the only thing the parser keys off.
    cols: Vec<String>,
    /// Samples, ascending in time, one per distinct `elapsed_s`.
    frames: Vec<Frame>,
    /// Domain lane labels seen anywhere in the run, sorted for stable colours.
    lanes: Vec<String>,
    /// Peak error count per lane, for the header banner.
    lane_errors: BTreeMap<String, f64>,
    data_rows: usize,
    /// Rows dropped for want of a usable timestamp.
    skipped_rows: usize,
    /// Why there is nothing to plot at all, if so.
    note: Option<String>,
}

impl Telemetry {
    fn has_col(&self, name: &str) -> bool {
        self.cols.iter().any(|c| c == name)
    }

    /// Plain text (backticks become `<code>` at render time) explaining why a
    /// series produced no points: the column was never in the file, or it was
    /// there and every single cell was unusable. A tech needs to tell those two
    /// apart — one is "old build", the other is "the sensor never answered".
    fn why_missing(&self, name: &str) -> String {
        if self.has_col(name) {
            format!("the `{name}` column is present but no sample carried a usable number")
        } else {
            format!("this CSV has no `{name}` column")
        }
    }

    fn series<F: Fn(&Frame) -> Option<f64>>(&self, pick: F) -> Vec<(f64, f64)> {
        self.frames.iter().filter_map(|f| pick(f).map(|v| (f.t, v))).collect()
    }

    /// The run's time span. Frames are time-ordered, so this is first..last.
    fn t_span(&self) -> (f64, f64) {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) => (a.t, b.t),
            _ => (0.0, 1.0),
        }
    }

    /// Timestamps at which the run's total error count *rose*.
    ///
    /// The `errors` column is a cumulative counter, so "every sample with a
    /// non-zero count" would paint the entire remainder of the run red and hide
    /// *when* things actually broke. Marking the increments puts the rule
    /// exactly on the moment of failure — and a run that keeps failing keeps
    /// getting rules, which is the loud outcome we want.
    fn error_marks(&self) -> Vec<f64> {
        let mut out = Vec::new();
        let mut prev = 0.0f64;
        for f in &self.frames {
            if f.errors > prev {
                out.push(f.t);
            }
            prev = f.errors;
        }
        out
    }

    fn total_errors(&self) -> f64 {
        self.frames.iter().map(|f| f.errors).fold(0.0, f64::max)
    }

    /// Distinct core indices that reported an effective clock or utilization.
    fn core_count(&self) -> usize {
        let mut set: BTreeSet<u32> = BTreeSet::new();
        for f in &self.frames {
            set.extend(f.mhz.keys().copied());
            set.extend(f.util.keys().copied());
        }
        set.len()
    }
}

/// Fold the CSV into time-ordered [`Frame`]s.
///
/// Lane labels are comma-guarded by the writer, so a plain `split(',')` keeps
/// the columns aligned; a ragged row simply has missing (not shifted) cells.
fn parse_csv(text: &str) -> Telemetry {
    let mut tel = Telemetry::default();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let header = match lines.next() {
        Some(h) => h,
        None => {
            tel.note = Some("this file is empty — there is nothing to plot.".to_string());
            return tel;
        }
    };
    let cols: Vec<String> = header
        .trim_start_matches('\u{feff}') // a UTF-8 BOM, if some editor added one
        .split(',')
        .map(|c| c.trim().to_ascii_lowercase())
        .collect();
    let pos = |name: &str| cols.iter().position(|c| c == name);

    // Every index is resolved by NAME, once, here. Nothing downstream knows a
    // column ordinal, so appending (or reordering) columns cannot break parsing.
    let i_lane = pos(COL_LANE);
    let i_work = pos(COL_WORK);
    let i_err = pos(COL_ERRORS);
    let i_mhz = pos(COL_MHZ);
    let i_util = pos(COL_UTIL);
    let i_pow = pos(COL_GPU_POWER);
    let i_tmp = pos(COL_GPU_TEMP);
    let i_mtmp = pos(COL_GPU_MEM_TEMP);
    let i_t = match pos(COL_TIME) {
        Some(i) => i,
        None => {
            tel.note = Some(format!(
                "no `{COL_TIME}` column in the header, so there is no timeline to plot against."
            ));
            tel.cols = cols;
            return tel;
        }
    };

    let mut by_ms: BTreeMap<i64, Frame> = BTreeMap::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let t = match cell_num(&f, Some(i_t)) {
            Some(v) => v,
            None => {
                tel.skipped_rows += 1;
                continue;
            }
        };
        tel.data_rows += 1;
        // The writer formats one identical `{:.3}` timestamp for every lane in a
        // sample, so bucketing to the millisecond regroups them exactly — and
        // tolerates a hand-edited file whose timestamps wobble slightly.
        let key = (t * 1000.0).round() as i64;
        let fr = by_ms.entry(key).or_insert_with(|| Frame { t, ..Frame::default() });

        let lane = i_lane.and_then(|i| f.get(i)).map(|s| s.trim()).unwrap_or("");

        if let Some(e) = cell_num(&f, i_err) {
            fr.errors += e;
            if e > 0.0 && !lane.is_empty() {
                let slot = tel.lane_errors.entry(lane.to_string()).or_insert(0.0);
                if e > *slot {
                    *slot = e;
                }
            }
        }

        // First readable value for this sample wins; a later blank cannot erase it.
        if fr.gpu_power.is_none() {
            fr.gpu_power = cell_num(&f, i_pow);
        }
        if fr.gpu_temp.is_none() {
            fr.gpu_temp = cell_num(&f, i_tmp);
        }
        if fr.gpu_mem_temp.is_none() {
            fr.gpu_mem_temp = cell_num(&f, i_mtmp);
        }

        match core_index(lane) {
            Some(ci) => {
                if let Some(v) = cell_num(&f, i_mhz) {
                    fr.mhz.insert(ci, v);
                }
                if let Some(v) = cell_num(&f, i_util) {
                    fr.util.insert(ci, v);
                }
            }
            None => {
                if !lane.is_empty() {
                    if let Some(w) = cell_num(&f, i_work) {
                        fr.work.insert(lane.to_string(), w);
                    }
                }
            }
        }
    }

    tel.frames = by_ms.into_values().collect();
    let mut lanes: BTreeSet<String> = BTreeSet::new();
    for fr in &tel.frames {
        for k in fr.work.keys() {
            lanes.insert(k.clone());
        }
    }
    tel.lanes = lanes.into_iter().collect();
    if tel.frames.is_empty() && tel.note.is_none() {
        tel.note = Some(
            "the header parsed, but the file holds no data rows with a usable timestamp \
             (a run that was cancelled before its first sample looks like this)."
                .to_string(),
        );
    }
    tel.cols = cols;
    tel
}

// ================================================================== scaling ==

/// Value → viewBox mapping for one chart. The time axis is shared by every
/// chart on the page so a spike lines up vertically across all of them.
#[derive(Debug, Clone, Copy)]
struct Scale {
    t0: f64,
    t1: f64,
    lo: f64,
    hi: f64,
    step: f64,
}

impl Scale {
    /// `t0 -> PL_L`, `t1 -> PL_R`. A zero-width span (a single sample) plots at
    /// the centre rather than dividing by zero.
    fn x(&self, t: f64) -> f64 {
        let span = self.t1 - self.t0;
        if !span.is_finite() || span <= 0.0 || !t.is_finite() {
            return PL_L + PL_W / 2.0;
        }
        PL_L + clamp01((t - self.t0) / span) * PL_W
    }

    /// `lo -> PL_B` (bottom), `hi -> PL_T` (top) — SVG y grows downward.
    fn y(&self, v: f64) -> f64 {
        let span = self.hi - self.lo;
        if !span.is_finite() || span <= 0.0 || !v.is_finite() {
            return PL_T + PL_H / 2.0;
        }
        PL_B - clamp01((v - self.lo) / span) * PL_H
    }
}

fn clamp01(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.5;
    }
    v.clamp(0.0, 1.0)
}

/// Round a raw axis interval up to the next 1/2/5×10ⁿ, so tick labels are
/// numbers a human reads at a glance instead of `3271.4`.
fn nice_step(raw: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let exp = raw.log10().floor().clamp(-30.0, 30.0) as i32;
    let mag = 10f64.powi(exp);
    let norm = raw / mag;
    let mult = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    };
    mult * mag
}

/// Expand `[min, max]` out to whole ticks: returns `(lo, hi, step)`.
///
/// Targets five intervals. A flat series gets padded rather than collapsing to a
/// zero-height plot, and a series that never goes negative never gets a negative
/// axis (there is no such thing as −50 W).
fn nice_bounds(min: f64, max: f64) -> (f64, f64, f64) {
    if !min.is_finite() || !max.is_finite() || min > max {
        return (0.0, 1.0, 0.25);
    }
    let (mut lo, mut hi) = (min, max);
    if hi - lo <= 0.0 {
        let pad = if lo.abs() > 10.0 { lo.abs() * 0.1 } else { 1.0 };
        lo -= pad;
        hi += pad;
    }
    let step = nice_step((hi - lo) / 5.0);
    let mut lo_n = (lo / step).floor() * step;
    let hi_n = (hi / step).ceil() * step;
    if min >= 0.0 && lo_n < 0.0 {
        lo_n = 0.0;
    }
    (lo_n, hi_n, step)
}

// ============================================================== formatting ===

/// A coordinate, at 2 decimal places with the noise trimmed — `64`, not
/// `64.00000000000001`. Non-finite input degrades to `0` so a bad value can
/// never emit `NaN` into an SVG attribute and blank the whole chart.
fn fnum(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_string();
    }
    let mut s = format!("{:.2}", (v * 100.0).round() / 100.0);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

/// A tick label, with just enough decimals for the step it sits on.
fn fmt_tick(v: f64, step: f64) -> String {
    if !v.is_finite() {
        return "-".to_string();
    }
    let prec = if !step.is_finite() || step >= 1.0 {
        0
    } else if step >= 0.1 {
        1
    } else if step >= 0.01 {
        2
    } else {
        3
    };
    format!("{v:.prec$}", prec = prec)
}

/// A time-axis label: raw seconds for short runs, `m:ss` for long soaks.
fn fmt_time(t: f64, span: f64) -> String {
    if !t.is_finite() {
        return "-".to_string();
    }
    if span >= 120.0 {
        let s = t.max(0.0).round() as u64;
        format!("{}:{:02}", s / 60, s % 60)
    } else if span >= 10.0 {
        format!("{t:.0}")
    } else {
        format!("{t:.1}")
    }
}

/// A human-scale magnitude for legend peaks — work counters run to millions.
fn fmt_si(v: f64) -> String {
    if !v.is_finite() {
        return "-".to_string();
    }
    let a = v.abs();
    if a >= 1e9 {
        format!("{:.2}G", v / 1e9)
    } else if a >= 1e6 {
        format!("{:.2}M", v / 1e6)
    } else if a >= 1e3 {
        format!("{:.1}k", v / 1e3)
    } else if v.fract() == 0.0 || a >= 10.0 {
        // Counts are counts: 3 errors, not "3.00".
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// A run duration for the header.
fn fmt_dur(s: f64) -> String {
    if !s.is_finite() || s < 0.0 {
        return "-".to_string();
    }
    let total = s.round() as u64;
    let (h, m, sec) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m {sec:02}s")
    } else if m > 0 {
        format!("{m}m {sec:02}s")
    } else {
        format!("{s:.1}s")
    }
}

/// Escape text for HTML/SVG text nodes and attribute values.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            _ => o.push(c),
        }
    }
    o
}

/// Escape *first*, then promote `back-ticked` spans to `<code>`. Lets prose
/// name a column without any path where un-escaped input reaches the page.
fn esc_code(s: &str) -> String {
    let escaped = esc(s);
    let mut out = String::with_capacity(escaped.len());
    let mut open = false;
    for c in escaped.chars() {
        if c == '`' {
            out.push_str(if open { "</code>" } else { "<code>" });
            open = !open;
        } else {
            out.push(c);
        }
    }
    if open {
        out.push_str("</code>");
    }
    out
}

// =================================================================== charts ==

/// Above this many points a series is decimated before it is drawn. A 4 Hz
/// sampler makes 14 400 points an hour and an overnight soak is the run you most
/// need to look at — undecimated, that page is megabytes of SVG the browser
/// chokes on, at a density no screen can resolve anyway (the plot is 880 units
/// wide).
const DECIMATE_TARGET: usize = 1000;

/// Min/max decimation: split into equal buckets and keep each bucket's lowest
/// and highest point, in time order.
///
/// This is what a scope does, and it is chosen over averaging for one reason —
/// **extremes survive**. A single-sample clock drop or power spike is exactly
/// the artifact a QC tech is hunting; a mean would erase it. Output is at most
/// `2 * DECIMATE_TARGET` points and the value range is bit-for-bit unchanged.
fn decimate(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if pts.len() <= DECIMATE_TARGET * 2 {
        return pts.to_vec();
    }
    let chunk = pts.len().div_ceil(DECIMATE_TARGET);
    let mut out = Vec::with_capacity(DECIMATE_TARGET * 2);
    for c in pts.chunks(chunk) {
        let mut lo = 0usize;
        let mut hi = 0usize;
        for (i, p) in c.iter().enumerate() {
            if p.1 < c[lo].1 {
                lo = i;
            }
            if p.1 > c[hi].1 {
                hi = i;
            }
        }
        let (first, second) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        out.push(c[first]);
        if second != first {
            out.push(c[second]);
        }
    }
    out
}

/// Decimate a band by keeping each bucket's outer envelope. Widening rather
/// than narrowing keeps the shaded spread honest: it can never hide a core that
/// fell out of the pack.
fn decimate_band(pts: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    if pts.len() <= DECIMATE_TARGET * 2 {
        return pts.to_vec();
    }
    let chunk = pts.len().div_ceil(DECIMATE_TARGET * 2);
    pts.chunks(chunk)
        .map(|c| {
            let lo = c.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let hi = c.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);
            (c[0].0, lo, hi)
        })
        .collect()
}

/// One plotted line.
struct Series {
    label: String,
    color: &'static str,
    /// SVG `stroke-dasharray`; empty for a solid line.
    dash: &'static str,
    pts: Vec<(f64, f64)>,
}

/// A shaded `(t, lo, hi)` envelope drawn behind the series — the min–max spread
/// across cores, so 32 lines collapse into one readable band.
struct Band {
    label: String,
    color: &'static str,
    pts: Vec<(f64, f64, f64)>,
}

struct Chart {
    /// Stable HTML anchor (`#gpu-power`) — also what the tests key off.
    id: &'static str,
    title: &'static str,
    sub: String,
    y_label: &'static str,
    /// Pin the value axis to `(lo, hi, step)` instead of fitting the data.
    /// Percentages get 0–100 so a "busy" chart cannot flatter itself by
    /// auto-scaling 3%–4% to fill the frame.
    y_fixed: Option<(f64, f64, f64)>,
    band: Option<Band>,
    series: Vec<Series>,
}

impl Chart {
    /// The value range the chart must cover, across band and every series.
    fn value_range(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for s in &self.series {
            for &(_, v) in &s.pts {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if let Some(b) = &self.band {
            for &(_, l, h) in &b.pts {
                lo = lo.min(l);
                hi = hi.max(h);
            }
        }
        (lo, hi)
    }
}

/// Emit one chart's `<svg>`: frame, grid, ticks, axis labels, error rules,
/// band, then lines (last, so nothing is drawn over the data).
fn render_chart(c: &Chart, t0: f64, t1: f64, marks: &[f64]) -> String {
    let (dlo, dhi) = c.value_range();
    let (lo, hi, step) = c.y_fixed.unwrap_or_else(|| nice_bounds(dlo, dhi));
    let sc = Scale { t0, t1, lo, hi, step };
    let span = t1 - t0;

    let mut s = String::new();
    let _ = writeln!(
        s,
        "<svg class=\"chartsvg\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" \
         role=\"img\" aria-label=\"{title}\">",
        w = fnum(VB_W),
        h = fnum(VB_H),
        title = esc(c.title)
    );
    let _ = writeln!(
        s,
        "<rect class=\"frame\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>",
        fnum(PL_L),
        fnum(PL_T),
        fnum(PL_W),
        fnum(PL_H)
    );

    // Value axis: horizontal gridlines + right-aligned labels.
    let ticks = if step.is_finite() && step > 0.0 {
        (((hi - lo) / step).round() as i64).clamp(1, 40)
    } else {
        1
    };
    for k in 0..=ticks {
        let v = lo + (k as f64) * step;
        let y = fnum(sc.y(v));
        let _ = writeln!(
            s,
            "<line class=\"grid\" x1=\"{}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\"/>",
            fnum(PL_L),
            fnum(PL_R)
        );
        let _ = writeln!(
            s,
            "<text class=\"tick\" text-anchor=\"end\" x=\"{}\" y=\"{}\">{}</text>",
            fnum(PL_L - 8.0),
            fnum(sc.y(v) + 3.6),
            esc(&fmt_tick(v, step))
        );
    }

    // Time axis: six evenly spaced marks (one, centred, for a single sample).
    let xt: Vec<f64> = if span.is_finite() && span > 0.0 {
        (0..=5).map(|k| t0 + span * (k as f64) / 5.0).collect()
    } else {
        vec![t0]
    };
    for t in xt {
        let x = fnum(sc.x(t));
        let _ = writeln!(
            s,
            "<line class=\"grid\" x1=\"{x}\" y1=\"{}\" x2=\"{x}\" y2=\"{}\"/>",
            fnum(PL_T),
            fnum(PL_B)
        );
        let _ = writeln!(
            s,
            "<text class=\"tick\" text-anchor=\"middle\" x=\"{x}\" y=\"{}\">{}</text>",
            fnum(PL_B + 16.0),
            esc(&fmt_time(t, span))
        );
    }
    let x_unit = if span >= 120.0 { "elapsed (m:ss)" } else { "elapsed (s)" };
    let _ = writeln!(
        s,
        "<text class=\"axlab\" text-anchor=\"middle\" x=\"{}\" y=\"{}\">{}</text>",
        fnum(PL_L + PL_W / 2.0),
        fnum(VB_H - 8.0),
        esc(x_unit)
    );
    let _ = writeln!(
        s,
        "<text class=\"axlab\" text-anchor=\"middle\" transform=\"rotate(-90 {x} {y})\" \
         x=\"{x}\" y=\"{y}\">{lab}</text>",
        x = fnum(14.0),
        y = fnum(PL_T + PL_H / 2.0),
        lab = esc(c.y_label)
    );

    // Error rules, under the data so they never hide a line. Deduped by rendered
    // x so a burst of failures cannot balloon the file with identical elements.
    let mut drawn: BTreeSet<String> = BTreeSet::new();
    for &t in marks {
        let x = fnum(sc.x(t));
        if !drawn.insert(x.clone()) {
            continue;
        }
        let _ = writeln!(
            s,
            "<line class=\"err\" x1=\"{x}\" y1=\"{}\" x2=\"{x}\" y2=\"{}\"/>",
            fnum(PL_T),
            fnum(PL_B)
        );
    }

    if let Some(b) = &c.band {
        let bpts = decimate_band(&b.pts);
        if !bpts.is_empty() {
            let mut d: Vec<String> = Vec::with_capacity(bpts.len() * 2 + 1);
            for (i, &(t, _, h)) in bpts.iter().enumerate() {
                let head = if i == 0 { "M" } else { "L" };
                d.push(format!("{head}{},{}", fnum(sc.x(t)), fnum(sc.y(h))));
            }
            for &(t, l, _) in bpts.iter().rev() {
                d.push(format!("L{},{}", fnum(sc.x(t)), fnum(sc.y(l))));
            }
            d.push("Z".to_string());
            let _ = writeln!(
                s,
                "<path class=\"band\" d=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
                d.join(" "),
                b.color,
                b.color
            );
        }
    }

    for ser in &c.series {
        let pts: Vec<String> = decimate(&ser.pts)
            .iter()
            .map(|&(t, v)| format!("{},{}", fnum(sc.x(t)), fnum(sc.y(v))))
            .collect();
        let dash = if ser.dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{}\"", ser.dash)
        };
        let _ = writeln!(
            s,
            "<polyline class=\"ln\" points=\"{}\" stroke=\"{}\"{dash}/>",
            pts.join(" "),
            ser.color
        );
        // A lone sample has no line to draw; a dot is the honest rendering of
        // "we have exactly one reading" (and beats a blank chart).
        if ser.pts.len() == 1 {
            let (t, v) = ser.pts[0];
            let _ = writeln!(
                s,
                "<circle cx=\"{}\" cy=\"{}\" r=\"3.5\" fill=\"{}\"/>",
                fnum(sc.x(t)),
                fnum(sc.y(v)),
                ser.color
            );
        }
    }

    s.push_str("</svg>\n");
    s
}

/// A legend swatch drawn as a miniature of the mark it stands for, so a dashed
/// line reads as dashed and a band reads as a translucent fill.
fn legend_item(color: &str, dash: &str, band: bool, label: &str) -> String {
    let mark = if band {
        format!("<rect x=\"1\" y=\"1\" width=\"22\" height=\"10\" fill=\"{color}\" fill-opacity=\".22\" stroke=\"{color}\" stroke-opacity=\".5\"/>")
    } else {
        let d = if dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        format!("<line x1=\"1\" y1=\"6\" x2=\"23\" y2=\"6\" stroke=\"{color}\" stroke-width=\"3\"{d}/>")
    };
    format!(
        "<li><svg class=\"sw\" viewBox=\"0 0 24 12\" width=\"24\" height=\"12\" aria-hidden=\"true\">\
         {mark}</svg><span class=\"k\">{}</span></li>",
        esc(label)
    )
}

/// Wrap one chart in its section: heading, sub-caption, SVG, legend.
fn render_section(c: &Chart, t0: f64, t1: f64, marks: &[f64], has_errors: bool) -> String {
    let mut legend = String::new();
    if let Some(b) = &c.band {
        legend.push_str(&legend_item(b.color, "", true, &b.label));
    }
    for ser in &c.series {
        legend.push_str(&legend_item(ser.color, ser.dash, false, &ser.label));
    }
    if has_errors {
        legend.push_str(&legend_item(C_RED, "", false, "error reported"));
    }
    format!(
        "<section class=\"chart\" id=\"{id}\">\n<h2>{title}</h2>\n<p class=\"sub\">{sub}</p>\n\
         {svg}<ul class=\"legend\">{legend}</ul>\n</section>\n",
        id = c.id,
        title = esc(c.title),
        sub = esc_code(&c.sub),
        svg = render_chart(c, t0, t1, marks),
    )
}

// ===================================================== chart construction ====

/// Discard a temperature series that reads a constant 0 °C, returning
/// `(points, was_dropped)`.
///
/// This is a *sensor absence*, not a cryogenic GPU: NVML reports 0 for a part
/// with no memory-junction probe, which consumer cards frequently lack, and the
/// writer forwards that verbatim. Drawing it would assert a reading that was
/// never taken and, worse, drag the shared axis down to 0 and crush the real
/// core-temperature curve into the top of the frame. The blank-is-not-zero rule
/// this module is built on has to hold for a sentinel zero too.
fn drop_zero_sensor(pts: Vec<(f64, f64)>) -> (Vec<(f64, f64)>, bool) {
    if !pts.is_empty() && pts.iter().all(|&(_, v)| v == 0.0) {
        (Vec::new(), true)
    } else {
        (pts, false)
    }
}

fn zero_sensor_reason(tel: &Telemetry, col: &str, zeroed: bool) -> String {
    if zeroed {
        format!(
            "`{col}` read a constant 0 C for the whole run, which is how a part with no such \
             sensor reports - treated as absent rather than plotted as a reading"
        )
    } else {
        tel.why_missing(col)
    }
}

/// Build every chart the data supports, and a reason for each one it does not.
fn build_charts(tel: &Telemetry) -> (Vec<Chart>, Vec<(&'static str, String)>) {
    let mut charts: Vec<Chart> = Vec::new();
    let mut omitted: Vec<(&'static str, String)> = Vec::new();

    // 1 — GPU board power.
    let power = tel.series(|f| f.gpu_power);
    if power.is_empty() {
        omitted.push(("GPU power", tel.why_missing(COL_GPU_POWER)));
    } else {
        charts.push(Chart {
            id: "gpu-power",
            title: "GPU power",
            sub: format!("`{COL_GPU_POWER}` — board power draw, one point per sample."),
            y_label: "watts (W)",
            y_fixed: None,
            band: None,
            series: vec![Series {
                label: "board power".to_string(),
                color: C_PINK,
                dash: "",
                pts: power,
            }],
        });
    }

    // 2 — GPU temperatures: core and memory junction share one axis, because
    // the gap between them is the thing worth seeing.
    let (core_t, core_zero) = drop_zero_sensor(tel.series(|f| f.gpu_temp));
    let (mem_t, mem_zero) = drop_zero_sensor(tel.series(|f| f.gpu_mem_temp));
    let core_why = zero_sensor_reason(tel, COL_GPU_TEMP, core_zero);
    let mem_why = zero_sensor_reason(tel, COL_GPU_MEM_TEMP, mem_zero);
    if core_t.is_empty() && mem_t.is_empty() {
        omitted.push(("GPU temperature", format!("{core_why}; {mem_why}")));
    } else {
        let mut series = Vec::new();
        if !core_t.is_empty() {
            series.push(Series {
                label: "GPU core".to_string(),
                color: C_AMBER,
                dash: "",
                pts: core_t,
            });
        }
        if !mem_t.is_empty() {
            series.push(Series {
                label: "memory junction".to_string(),
                color: C_VIOLET,
                dash: "",
                pts: mem_t,
            });
        }
        let mut sub = format!("`{COL_GPU_TEMP}` / `{COL_GPU_MEM_TEMP}`, on a shared axis.");
        if core_zero || mem_zero {
            let dropped = if core_zero { &core_why } else { &mem_why };
            let _ = write!(sub, " Not drawn: {dropped}.");
        }
        charts.push(Chart {
            id: "gpu-temp",
            title: "GPU temperature",
            sub,
            y_label: "degrees C",
            y_fixed: None,
            band: None,
            series,
        });
    }

    // 3 — CPU effective clock: mean line over a min–max band. One band beats 32
    // overlapping lines, and still shows a single core dropping out.
    let mut mean_mhz: Vec<(f64, f64)> = Vec::new();
    let mut band_mhz: Vec<(f64, f64, f64)> = Vec::new();
    for f in &tel.frames {
        if f.mhz.is_empty() {
            continue;
        }
        let n = f.mhz.len() as f64;
        let sum: f64 = f.mhz.values().sum();
        let lo = f.mhz.values().copied().fold(f64::INFINITY, f64::min);
        let hi = f.mhz.values().copied().fold(f64::NEG_INFINITY, f64::max);
        mean_mhz.push((f.t, sum / n));
        band_mhz.push((f.t, lo, hi));
    }
    if mean_mhz.is_empty() {
        omitted.push(("CPU effective clock", tel.why_missing(COL_MHZ)));
    } else {
        let cores = tel.core_count();
        charts.push(Chart {
            id: "cpu-clock",
            title: "CPU effective clock",
            sub: format!(
                "`{COL_MHZ}` aggregated over {cores} core lane(s): the mean, with the \
                 per-sample min-max spread shaded behind it."
            ),
            y_label: "MHz",
            y_fixed: None,
            band: Some(Band {
                label: "min-max across cores".to_string(),
                color: C_LAVENDER,
                pts: band_mhz,
            }),
            series: vec![Series {
                label: "mean effective clock".to_string(),
                color: C_LAVENDER,
                dash: "",
                pts: mean_mhz,
            }],
        });
    }

    // 4 — CPU utilization, pinned to 0-100 so the eye reads the real level.
    let mut mean_util: Vec<(f64, f64)> = Vec::new();
    for f in &tel.frames {
        if f.util.is_empty() {
            continue;
        }
        let n = f.util.len() as f64;
        let sum: f64 = f.util.values().sum();
        mean_util.push((f.t, sum / n));
    }
    if mean_util.is_empty() {
        omitted.push(("CPU utilization", tel.why_missing(COL_UTIL)));
    } else {
        charts.push(Chart {
            id: "cpu-util",
            title: "CPU utilization",
            sub: format!("`{COL_UTIL}`, meaned across every reporting core lane."),
            y_label: "percent busy",
            y_fixed: Some((0.0, 100.0, 25.0)),
            band: None,
            series: vec![Series {
                label: "mean utilization".to_string(),
                color: C_CYAN,
                dash: "",
                pts: mean_util,
            }],
        });
    }

    // 5 — per-lane work rate. `work` is a cumulative counter, so the interesting
    // quantity is its derivative. Lanes differ by orders of magnitude (memory
    // passes vs. storage writes), so each is normalized to its own peak and the
    // absolute peak moves into the legend — shape stays comparable, magnitude
    // stays honest.
    let mut lane_series: Vec<Series> = Vec::new();
    let mut idle_lanes: Vec<String> = Vec::new();
    for (i, lane) in tel.lanes.iter().enumerate() {
        let mut raw: Vec<(f64, f64)> = Vec::new();
        let mut prev: Option<(f64, f64)> = None;
        for f in &tel.frames {
            if let Some(&w) = f.work.get(lane) {
                if let Some((pt, pw)) = prev {
                    let dt = f.t - pt;
                    let dw = w - pw;
                    // A negative delta means the counter restarted, not that the
                    // machine ran backwards; drop the point rather than draw a
                    // spike that never happened.
                    if dt > 0.0 && dw >= 0.0 {
                        raw.push((f.t, dw / dt));
                    }
                }
                prev = Some((f.t, w));
            }
        }
        let peak = raw.iter().map(|&(_, r)| r).fold(0.0f64, f64::max);
        if raw.is_empty() || peak <= 0.0 {
            idle_lanes.push(lane.clone());
            continue;
        }
        lane_series.push(Series {
            label: format!("{lane} (peak {}/s)", fmt_si(peak)),
            color: LANE_COLORS[i % LANE_COLORS.len()],
            dash: LANE_DASHES[(i / LANE_COLORS.len()) % LANE_DASHES.len()],
            pts: raw.into_iter().map(|(t, r)| (t, r / peak * 100.0)).collect(),
        });
    }
    if lane_series.is_empty() {
        let why = if !tel.has_col(COL_WORK) {
            tel.why_missing(COL_WORK)
        } else if tel.frames.len() < 2 {
            "a rate needs at least two samples and this run has fewer".to_string()
        } else {
            "no domain lane advanced its `work` counter during this run".to_string()
        };
        omitted.push(("Per-lane work rate", why));
    } else {
        let mut sub = format!(
            "d(`{COL_WORK}`)/dt per domain lane. Each lane is normalized to its own peak - the \
             absolute peak is in the legend - so lanes orders of magnitude apart stay readable."
        );
        if !idle_lanes.is_empty() {
            let _ = write!(sub, " Never advanced, so not drawn: {}.", idle_lanes.join(", "));
        }
        charts.push(Chart {
            id: "work-rate",
            title: "Per-lane work rate",
            sub,
            y_label: "% of lane peak",
            y_fixed: Some((0.0, 100.0, 25.0)),
            band: None,
            series: lane_series,
        });
    }

    (charts, omitted)
}

// ===================================================================== page ==

const STYLE: &str = concat!(
    ":root{color-scheme:dark}",
    "*{box-sizing:border-box}",
    "body{margin:0;padding:24px 20px 40px;background:", "#070711", ";color:", "#f5f5f8",
    ";font:15px/1.55 'Segoe UI',system-ui,-apple-system,Helvetica,Arial,sans-serif}",
    ".wrap{max-width:1120px;margin:0 auto}",
    "header{border-bottom:1px solid rgba(140,103,242,.35);padding-bottom:14px;margin-bottom:6px}",
    "h1{font-size:21px;margin:0 0 6px;font-weight:600;letter-spacing:.01em}",
    "h1 .brand{color:#ed2398;font-weight:700;letter-spacing:.14em;font-size:12px;display:block;",
    "margin-bottom:4px}",
    ".meta{color:#a9a9b7;font-size:13px;margin:2px 0}",
    ".meta b{color:#f5f5f8;font-weight:600}",
    ".alert{margin:12px 0 0;padding:10px 12px;border-radius:8px;font-size:13px;",
    "background:rgba(238,11,42,.13);border:1px solid #ee0b2a;color:#f5f5f8}",
    ".alert b{color:#ee0b2a;letter-spacing:.08em}",
    ".ok{margin:12px 0 0;padding:8px 12px;border-radius:8px;font-size:13px;",
    "background:rgba(65,217,248,.08);border:1px solid rgba(65,217,248,.4);color:#a9a9b7}",
    "section{background:#161526;border:1px solid rgba(140,103,242,.3);border-radius:10px;",
    "padding:14px 16px 10px;margin:18px 0}",
    "section h2{font-size:12px;margin:0;color:#9f9cff;font-weight:700;letter-spacing:.12em;",
    "text-transform:uppercase}",
    ".sub{color:#a9a9b7;font-size:12.5px;margin:5px 0 10px}",
    "code{font-family:'Cascadia Code',Consolas,ui-monospace,monospace;font-size:12px;",
    "color:#41d9f8;background:rgba(65,217,248,.09);padding:1px 5px;border-radius:4px}",
    "svg.chartsvg{display:block;width:100%;max-width:100%;height:auto}",
    "svg .frame{fill:#070711;stroke:#8c67f2;stroke-opacity:.32;stroke-width:1}",
    "svg .grid{stroke:#8c67f2;stroke-opacity:.15;stroke-width:1}",
    "svg .err{stroke:#ee0b2a;stroke-width:2;stroke-opacity:.85}",
    "svg .band{fill-opacity:.16;stroke-opacity:.42;stroke-width:1}",
    "svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}",
    "svg text{font-family:'Cascadia Code',Consolas,ui-monospace,monospace}",
    "svg .tick{fill:#a9a9b7;font-size:11px}",
    "svg .axlab{fill:#9f9cff;font-size:11px;letter-spacing:.05em}",
    ".legend{list-style:none;display:flex;flex-wrap:wrap;gap:5px 18px;margin:8px 0 4px;",
    "padding:0;font-size:12px;color:#a9a9b7}",
    ".legend li{display:flex;align-items:center;gap:7px}",
    ".legend .sw{flex:0 0 auto}",
    ".legend .k{color:#f5f5f8}",
    "ul.why{margin:6px 0 4px;padding-left:20px;color:#a9a9b7;font-size:13px}",
    "ul.why li{margin:3px 0}",
    "ul.why b{color:#f5f5f8}",
    "footer{color:#a9a9b7;font-size:12px;margin-top:22px;text-align:center;opacity:.8}",
    "@media (max-width:640px){body{padding:14px 10px 28px}.sub{font-size:12px}}",
);

/// Render a whole page from CSV text. Split out from [`render_html`] so the
/// renderer is testable without touching the filesystem.
fn html_from_csv(csv: &str, title: &str, source: &str) -> String {
    let tel = parse_csv(csv);
    let (t0, t1) = tel.t_span();
    let marks = tel.error_marks();
    let (charts, omitted) = build_charts(&tel);
    let total_errors = tel.total_errors();
    let has_errors = total_errors > 0.0;

    let mut h = String::with_capacity(16 * 1024);
    h.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    h.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n");
    let _ = writeln!(h, "<title>{} - cec-crucible telemetry</title>", esc(title));
    let _ = writeln!(h, "<style>{STYLE}</style>");
    h.push_str("</head>\n<body>\n<div class=\"wrap\">\n<header>\n");
    let _ = writeln!(
        h,
        "<h1><span class=\"brand\">CEC CRUCIBLE // TELEMETRY</span>{}</h1>",
        esc(title)
    );

    let duration = if tel.frames.is_empty() { 0.0 } else { t1 - t0 };
    let _ = writeln!(
        h,
        "<p class=\"meta\">duration <b>{}</b> &middot; <b>{}</b> samples &middot; <b>{}</b> \
         core lanes &middot; <b>{}</b> domain lanes &middot; <b>{}</b> data rows</p>",
        esc(&fmt_dur(duration)),
        tel.frames.len(),
        tel.core_count(),
        tel.lanes.len(),
        tel.data_rows
    );
    let _ = writeln!(h, "<p class=\"meta\">source <b>{}</b></p>", esc(source));
    if tel.skipped_rows > 0 {
        let _ = writeln!(
            h,
            "<p class=\"meta\">{} row(s) had no usable timestamp and were skipped.</p>",
            tel.skipped_rows
        );
    }

    if has_errors {
        let first = marks.first().copied().unwrap_or(f64::NAN);
        let per_lane: Vec<String> = tel
            .lane_errors
            .iter()
            .map(|(k, v)| format!("{k} {}", fmt_si(*v)))
            .collect();
        let _ = writeln!(
            h,
            "<p class=\"alert\"><b>ERRORS: {}</b> &mdash; first at {} s ({}). Every chart carries \
             a red vertical rule at each moment the count rose.</p>",
            fmt_si(total_errors),
            esc(&fmt_tick(first, 0.1)),
            esc(&per_lane.join(", "))
        );
    } else if !tel.frames.is_empty() {
        h.push_str("<p class=\"ok\">No lane reported an error during this run.</p>\n");
    }
    h.push_str("</header>\n");

    if let Some(note) = &tel.note {
        let _ = writeln!(
            h,
            "<section id=\"no-data\"><h2>No data</h2><p class=\"sub\">{}</p></section>",
            esc_code(note)
        );
    }

    for c in &charts {
        h.push_str(&render_section(c, t0, t1, &marks, has_errors));
    }

    if !omitted.is_empty() {
        h.push_str("<section id=\"not-charted\">\n<h2>Not charted</h2>\n");
        h.push_str("<p class=\"sub\">These charts were skipped because their data is not in \
                    this CSV. Columns are matched by name, so they appear automatically once \
                    the run logs them.</p>\n<ul class=\"why\">\n");
        for (name, why) in &omitted {
            let _ = writeln!(h, "<li><b>{}</b> &mdash; {}</li>", esc(name), esc_code(why));
        }
        h.push_str("</ul>\n</section>\n");
    }

    let _ = writeln!(
        h,
        "<footer>Generated by cec-crucible from the run telemetry CSV. Self-contained: no \
         scripts, no fonts, no network. Series longer than {} points are drawn by min/max \
         decimation, so peaks and dips survive but not every sample is a vertex.</footer>",
        DECIMATE_TARGET * 2
    );
    h.push_str("</div>\n</body>\n</html>\n");
    h
}

// ==================================================================== tests ==

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// The header as it ships today.
    const HDR_OLD: &str = "elapsed_s,lane,work,phase,errors,hash_hex,eff_mhz,util_pct";
    /// The header with the GPU sensor columns appended.
    const HDR_NEW: &str = "elapsed_s,lane,work,phase,errors,hash_hex,eff_mhz,util_pct,\
                           gpu_power_w,gpu_temp_c,gpu_mem_temp_c,gpu_fan_pct,gpu_sm_mhz,\
                           gpu_throttle";

    /// Slice out one `<section id="...">` so a test can assert on just it.
    fn section<'a>(html: &'a str, id: &str) -> Option<&'a str> {
        let start = html.find(&format!("id=\"{id}\""))?;
        let end = html[start..].find("</section>")? + start;
        Some(&html[start..end])
    }

    /// Every `points="..."` in order — one per series, band paths use `d=`.
    fn points(sec: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = sec;
        while let Some(i) = rest.find("points=\"") {
            let after = &rest[i + 8..];
            match after.find('"') {
                Some(j) => {
                    out.push(&after[..j]);
                    rest = &after[j..];
                }
                None => break,
            }
        }
        out
    }

    fn tmp_path(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("crucible-graph-{}-{tag}-{n}", std::process::id()))
    }

    /// Three samples, GPU sensors present: power 100/200/300 W, core temp
    /// 60/65/70 C, memory junction 70/75/80 C.
    fn csv_gpu() -> String {
        let mut s = String::from(HDR_NEW);
        s.push('\n');
        for (i, (p, t, m)) in [(100, 60, 70), (200, 65, 75), (300, 70, 80)].iter().enumerate() {
            let _ = writeln!(
                s,
                "{:.3},gpu,{},work,0,0x0000000000000000,,,{p},{t},{m},55,2400,",
                i as f64,
                i * 10
            );
        }
        s
    }

    #[test]
    fn gpu_power_points_match_hand_computed_scaling() {
        let html = html_from_csv(&csv_gpu(), "scale check", "t.csv");
        let sec = section(&html, "gpu-power").expect("gpu-power section");
        // t spans 0..2 -> x = 64, 504, 944 (plot is 64..944, width 880).
        // Values 100..300 -> nice bounds 100..300 step 50 -> y = 220, 124, 28
        // (plot is 28..220, height 192).
        assert_eq!(points(sec), vec!["64,220 504,124 944,28"], "{sec}");
    }

    #[test]
    fn gpu_temp_series_share_one_axis() {
        let html = html_from_csv(&csv_gpu(), "temps", "t.csv");
        let sec = section(&html, "gpu-temp").expect("gpu-temp section");
        // Combined range 60..80 -> nice bounds 60..80 step 5.
        // core 60/65/70 -> y 220/172/124 ; memory 70/75/80 -> y 124/76/28.
        assert_eq!(
            points(sec),
            vec!["64,220 504,172 944,124", "64,124 504,76 944,28"],
            "{sec}"
        );
    }

    #[test]
    fn cpu_clock_plots_the_mean_over_a_min_max_band() {
        let mut s = String::from(HDR_OLD);
        s.push('\n');
        // Two cores; means 4500, 4500, 3000 over a 3000..5000 spread.
        for (i, (a, b)) in [(4000, 5000), (4200, 4800), (3000, 3000)].iter().enumerate() {
            let _ = writeln!(s, "{:.3},core 0,0,work,0,0x0,{a},100.0", i as f64);
            let _ = writeln!(s, "{:.3},core 1,0,work,0,0x0,{b},50.0", i as f64);
        }
        let html = html_from_csv(&s, "cpu", "t.csv");

        let sec = section(&html, "cpu-clock").expect("cpu-clock section");
        // 3000..5000 -> nice bounds 3000..5000 step 500.
        // mean 4500 -> 220 - (1500/2000)*192 = 76 ; mean 3000 -> 220.
        assert_eq!(points(sec), vec!["64,76 504,76 944,220"], "{sec}");
        assert!(sec.contains("class=\"band\""), "min-max band missing: {sec}");
        // Exactly one line: 32 cores must never become 32 polylines.
        assert_eq!(points(sec).len(), 1);

        let sec = section(&html, "cpu-util").expect("cpu-util section");
        // Mean of 100 and 50 is 75; axis pinned 0..100 -> 220 - .75*192 = 76.
        assert_eq!(points(sec), vec!["64,76 504,76 944,76"], "{sec}");
    }

    #[test]
    fn work_rate_is_the_derivative_normalized_per_lane() {
        let mut s = String::from(HDR_OLD);
        s.push('\n');
        // mem: 0 -> 1000 -> 3000, i.e. rates of 1000/s then 2000/s (peak 2000).
        for (i, w) in [0u64, 1000, 3000].iter().enumerate() {
            let _ = writeln!(s, "{:.3},mem,{w},work,0,0x0,,", i as f64);
        }
        let html = html_from_csv(&s, "rate", "t.csv");
        let sec = section(&html, "work-rate").expect("work-rate section");
        // First sample has no delta, so the line starts at t=1 (x=504).
        // 1000/2000 = 50% -> y 124 ; 2000/2000 = 100% -> y 28.
        assert_eq!(points(sec), vec!["504,124 944,28"], "{sec}");
        assert!(sec.contains("peak 2.0k/s"), "absolute peak missing: {sec}");
    }

    #[test]
    fn new_gpu_columns_render_and_their_absence_omits_those_charts() {
        // With the new columns: both GPU charts exist.
        let with = html_from_csv(&csv_gpu(), "with", "t.csv");
        assert!(section(&with, "gpu-power").is_some());
        assert!(section(&with, "gpu-temp").is_some());

        // Same rows under today's 8-column header: omitted, with a reason.
        let mut s = String::from(HDR_OLD);
        s.push('\n');
        for i in 0..3 {
            let _ = writeln!(s, "{:.3},gpu,{},work,0,0x0,,", i as f64, i * 10);
        }
        let without = html_from_csv(&s, "without", "t.csv");
        assert!(section(&without, "gpu-power").is_none());
        assert!(section(&without, "gpu-temp").is_none());
        let why = section(&without, "not-charted").expect("not-charted section");
        assert!(why.contains("GPU power"), "{why}");
        assert!(why.contains("GPU temperature"), "{why}");
        // The stated reason must be about the column, not a nonexistent sensor.
        assert!(why.contains("no <code>gpu_power_w</code> column"), "{why}");
        assert!(why.contains("no <code>gpu_mem_temp_c</code> column"), "{why}");
        // The charts that DO have data are still drawn.
        assert!(section(&without, "work-rate").is_some());

        // A column that exists but never carried a number reads differently
        // from one that was never logged at all.
        let mut blank = String::from(HDR_NEW);
        blank.push('\n');
        for i in 0..3 {
            let _ = writeln!(blank, "{:.3},gpu,{},work,0,0x0,,,,,,,,", i as f64, i * 10);
        }
        let blank = html_from_csv(&blank, "no nvml", "t.csv");
        assert!(section(&blank, "gpu-power").is_none());
        let why = section(&blank, "not-charted").expect("not-charted section");
        assert!(why.contains("is present but no sample carried a usable number"), "{why}");
    }

    #[test]
    fn columns_are_found_by_name_not_position() {
        // Deliberately hostile ordering: sensors first, extra unknown columns,
        // padded whitespace and mixed case in the header.
        let mut s = String::from(
            "GPU_Power_W , gpu_temp_c ,note,lane,ELAPSED_S,work,errors,gpu_mem_temp_c,util_pct,eff_mhz\n",
        );
        for (i, (p, t, m)) in [(100, 60, 70), (200, 65, 75), (300, 70, 80)].iter().enumerate() {
            let _ = writeln!(s, "{p},{t},x,gpu,{:.3},0,0,{m},,", i as f64);
        }
        let html = html_from_csv(&s, "reordered", "t.csv");
        let sec = section(&html, "gpu-power").expect("gpu-power section");
        assert_eq!(points(sec), vec!["64,220 504,124 944,28"], "{sec}");
        let sec = section(&html, "gpu-temp").expect("gpu-temp section");
        assert_eq!(
            points(sec),
            vec!["64,220 504,172 944,124", "64,124 504,76 944,28"],
            "{sec}"
        );
    }

    #[test]
    fn blank_and_malformed_cells_are_dropped_not_read_as_zero() {
        assert_eq!(num(Some("")), None);
        assert_eq!(num(Some("   ")), None);
        assert_eq!(num(None), None);
        assert_eq!(num(Some("n/a")), None);
        assert_eq!(num(Some("--")), None);
        assert_eq!(num(Some("0x0000000000000000")), None);
        assert_eq!(num(Some("NaN")), None, "NaN parses as f64 and must be rejected");
        assert_eq!(num(Some("nan")), None);
        assert_eq!(num(Some("inf")), None);
        assert_eq!(num(Some("-inf")), None);
        assert_eq!(num(Some("infinity")), None);
        // A real zero is still a real reading.
        assert_eq!(num(Some("0")), Some(0.0));
        assert_eq!(num(Some(" 12.5 ")), Some(12.5));

        // End to end: two dropped samples between two good ones. If a blank read
        // as 0 W the axis would start at 0 and there would be four points.
        let mut s = String::from(HDR_NEW);
        s.push('\n');
        for (i, p) in ["100", "", "n/a", "300"].iter().enumerate() {
            let _ = writeln!(s, "{:.3},gpu,0,work,0,0x0,,,{p},,,,,", i as f64);
        }
        let html = html_from_csv(&s, "gaps", "t.csv");
        let sec = section(&html, "gpu-power").expect("gpu-power section");
        // t spans 0..3 -> x 64 and 944; values 100..300 -> y 220 and 28.
        assert_eq!(points(sec), vec!["64,220 944,28"], "{sec}");
        assert!(!sec.contains(">0<"), "a zero tick means a blank became 0 W: {sec}");
    }

    #[test]
    fn empty_header_only_and_single_row_never_panic() {
        for (tag, csv) in [
            ("empty", String::new()),
            ("blank-lines", "\n\n   \n".to_string()),
            ("header-only", format!("{HDR_NEW}\n")),
            ("old-header-only", format!("{HDR_OLD}\n")),
            (
                "single-row",
                format!("{HDR_NEW}\n0.000,gpu,7,work,0,0x0,,,250,61,72,55,2400,\n"),
            ),
            ("no-header-match", "a,b,c\n1,2,3\n".to_string()),
            ("ragged", format!("{HDR_NEW}\n0.000,gpu\n1.000\n2.000,gpu,1,work,0\n")),
            ("junk", "\u{feff}not,a,telemetry,file\r\nx,y,z\r\n".to_string()),
        ] {
            let html = html_from_csv(&csv, "edge", "t.csv");
            assert!(html.starts_with("<!doctype html>"), "{tag}");
            assert!(html.ends_with("</html>\n"), "{tag}");
            assert!(html.contains("<title>"), "{tag}");
            assert!(!html.contains("NaN"), "NaN leaked into the page: {tag}");
            assert!(!html.contains("inf\""), "inf leaked into an attribute: {tag}");
        }

        // A single sample still draws: a lone point gets a dot, not a blank box.
        let one = format!("{HDR_NEW}\n0.000,gpu,7,work,0,0x0,,,250,61,72,55,2400,\n");
        let html = html_from_csv(&one, "one", "t.csv");
        let sec = section(&html, "gpu-power").expect("gpu-power section");
        assert!(sec.contains("<circle"), "single sample should draw a dot: {sec}");
        // Centred, because a zero-width time span has no left or right.
        assert!(sec.contains("cx=\"504\""), "{sec}");

        // Nothing at all still explains itself.
        let none = html_from_csv("", "empty", "t.csv");
        assert!(section(&none, "no-data").is_some(), "{none}");
        assert!(none.contains("this file is empty"), "{none}");
    }

    #[test]
    fn a_constant_zero_memory_junction_is_a_missing_sensor_not_a_reading() {
        // What a consumer card actually logs: a real core temperature and a
        // memory junction pinned at 0 because the part has no such probe.
        let mut s = String::from(HDR_NEW);
        s.push('\n');
        for (i, t) in [60, 65, 70].iter().enumerate() {
            let _ = writeln!(s, "{:.3},gpu,0,work,0,0x0,,,300,{t},0,55,2400,0x0", i as f64);
        }
        let html = html_from_csv(&s, "consumer card", "t.csv");
        let sec = section(&html, "gpu-temp").expect("gpu-temp section");
        // One series only, and the axis fits 60-70 rather than being dragged to
        // 0 (which would flatten the real curve against the top of the frame).
        assert_eq!(points(sec), vec!["64,220 504,124 944,28"], "{sec}");
        assert!(sec.contains("constant 0 C"), "the omission must be explained: {sec}");

        // A card that does report it still gets both lines.
        let both = html_from_csv(&csv_gpu(), "pro card", "t.csv");
        assert_eq!(points(section(&both, "gpu-temp").unwrap()).len(), 2);

        // Both sensors dead -> no chart at all, with the reason stated.
        let mut s = String::from(HDR_NEW);
        s.push('\n');
        for i in 0..3 {
            let _ = writeln!(s, "{:.3},gpu,0,work,0,0x0,,,300,0,0,55,2400,0x0", i as f64);
        }
        let dead = html_from_csv(&s, "no temps", "t.csv");
        assert!(section(&dead, "gpu-temp").is_none());
        assert!(section(&dead, "not-charted").unwrap().contains("constant 0 C"));
        // Power is unaffected: a real 0 W reading must still be visible.
        assert!(section(&dead, "gpu-power").is_some());
    }

    #[test]
    fn crlf_and_bom_parse_the_same_as_plain_lf() {
        let plain = csv_gpu();
        let windows = format!("\u{feff}{}", plain.replace('\n', "\r\n"));
        let a = html_from_csv(&plain, "t", "t.csv");
        let b = html_from_csv(&windows, "t", "t.csv");
        let sa = section(&a, "gpu-power").expect("a");
        let sb = section(&b, "gpu-power").expect("b");
        assert_eq!(points(sa), points(sb));
        assert_eq!(points(sb), vec!["64,220 504,124 944,28"]);
    }

    #[test]
    fn errors_are_marked_on_every_chart_and_called_out_in_the_header() {
        let mut s = String::from(HDR_NEW);
        s.push('\n');
        // Cumulative error counter on the mem lane: clean, clean, then 3.
        for (i, e) in [0u32, 0, 3].iter().enumerate() {
            let _ = writeln!(s, "{:.3},mem,{},work,{e},0x0,,,{},60,70,,,", i as f64, i * 100, 200 + i);
            let _ = writeln!(s, "{:.3},core 0,0,work,0,0x0,4000,90.0", i as f64);
        }
        let html = html_from_csv(&s, "bad run", "t.csv");
        assert!(html.contains("ERRORS: 3"), "{html}");
        assert!(html.contains("mem 3"), "which lane failed must be named");

        // A rule at t=2 (x=944) on every chart, not just one.
        for id in ["gpu-power", "gpu-temp", "cpu-clock", "cpu-util", "work-rate"] {
            let sec = section(&html, id).unwrap_or_else(|| panic!("{id} missing"));
            assert!(
                sec.contains("<line class=\"err\" x1=\"944\""),
                "no error rule on {id}: {sec}"
            );
        }

        // And a clean run says so, with no rules anywhere.
        let clean = html_from_csv(&csv_gpu(), "good run", "t.csv");
        assert!(clean.contains("No lane reported an error"));
        assert!(!clean.contains("class=\"err\""));
    }

    #[test]
    fn cumulative_errors_mark_the_moment_of_failure_not_the_whole_tail() {
        let mut tel = Telemetry::default();
        for (t, e) in [(0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 5.0), (4.0, 5.0)] {
            tel.frames.push(Frame { t, errors: e, ..Frame::default() });
        }
        assert_eq!(tel.error_marks(), vec![1.0, 3.0]);
        assert_eq!(tel.total_errors(), 5.0);
    }

    #[test]
    fn decimation_bounds_the_page_without_losing_extremes() {
        // Short series are passed through untouched, so the exact-coordinate
        // assertions above describe every realistic short run.
        let short: Vec<(f64, f64)> = (0..1500).map(|i| (i as f64, i as f64)).collect();
        assert_eq!(decimate(&short), short);

        // A long soak: 12 hours at 4 Hz. One catastrophic single-sample dip and
        // one spike, buried in 172 800 points, must both survive.
        let mut long: Vec<(f64, f64)> = (0..172_800).map(|i| (i as f64 * 0.25, 4700.0)).collect();
        long[99_999].1 = 800.0;
        long[140_001].1 = 5600.0;
        let d = decimate(&long);
        assert!(d.len() <= DECIMATE_TARGET * 2, "{} points", d.len());
        assert!(d.iter().any(|p| p.1 == 800.0), "the dip was decimated away");
        assert!(d.iter().any(|p| p.1 == 5600.0), "the spike was decimated away");
        // Time order is preserved, or the polyline would zig-zag backwards.
        assert!(d.windows(2).all(|w| w[0].0 <= w[1].0));
        // The value range is unchanged, so the axis still tells the truth.
        let lo = d.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let hi = d.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!((lo, hi), (800.0, 5600.0));

        // A band's envelope may widen, never narrow.
        let band: Vec<(f64, f64, f64)> =
            (0..172_800).map(|i| (i as f64 * 0.25, 4000.0, 4800.0)).collect();
        let db = decimate_band(&band);
        assert!(db.len() <= DECIMATE_TARGET * 2 + 1, "{} points", db.len());
        assert!(db.iter().all(|p| p.1 <= 4000.0 && p.2 >= 4800.0));
    }

    #[test]
    fn nice_bounds_round_out_to_readable_ticks() {
        assert_eq!(nice_bounds(100.0, 300.0), (100.0, 300.0, 50.0));
        assert_eq!(nice_bounds(60.0, 80.0), (60.0, 80.0, 5.0));
        assert_eq!(nice_bounds(3000.0, 5000.0), (3000.0, 5000.0, 500.0));
        // A run that never went negative never gets a negative axis.
        let (lo, _, _) = nice_bounds(3.0, 97.0);
        assert!(lo >= 0.0, "lo = {lo}");
        // A flat series gets air instead of a zero-height plot.
        let (lo, hi, step) = nice_bounds(60.0, 60.0);
        assert!(lo < 60.0 && hi > 60.0 && step > 0.0, "{lo} {hi} {step}");
        // Garbage in, usable axis out.
        assert_eq!(nice_bounds(f64::NAN, 1.0), (0.0, 1.0, 0.25));
        assert_eq!(nice_bounds(f64::INFINITY, f64::NEG_INFINITY), (0.0, 1.0, 0.25));
        assert_eq!(nice_bounds(5.0, 1.0), (0.0, 1.0, 0.25));
    }

    #[test]
    fn scale_survives_degenerate_ranges() {
        let s = Scale { t0: 0.0, t1: 0.0, lo: 5.0, hi: 5.0, step: 1.0 };
        assert_eq!(s.x(0.0), PL_L + PL_W / 2.0);
        assert_eq!(s.y(5.0), PL_T + PL_H / 2.0);
        let s = Scale { t0: 0.0, t1: 10.0, lo: 0.0, hi: 100.0, step: 25.0 };
        assert_eq!(s.x(f64::NAN), PL_L + PL_W / 2.0);
        assert_eq!(s.y(f64::NAN), PL_T + PL_H / 2.0);
        // Out-of-range values pin to the frame instead of drawing outside it.
        assert_eq!(s.y(-50.0), PL_B);
        assert_eq!(s.y(500.0), PL_T);
        assert_eq!(fnum(f64::NAN), "0");
        assert_eq!(fnum(f64::INFINITY), "0");
        assert_eq!(fnum(-0.0), "0");
        assert_eq!(fnum(63.999_999_9), "64");
    }

    #[test]
    fn core_and_cpu_lanes_aggregate_without_double_counting() {
        // The writer backfills a `cpu N` row for a core with no load lane. If a
        // file ever carried both for one core, the core must still count once.
        let mut s = String::from(HDR_OLD);
        s.push('\n');
        let _ = writeln!(s, "0.000,core 0,0,work,0,0x0,4000,100.0");
        let _ = writeln!(s, "0.000,cpu 0,0,idle,0,0x0,2000,10.0");
        let _ = writeln!(s, "0.000,cpu 1,0,idle,0,0x0,2000,10.0");
        let tel = parse_csv(&s);
        assert_eq!(tel.frames.len(), 1);
        assert_eq!(tel.frames[0].mhz.len(), 2, "core 0 must not be counted twice");
        assert_eq!(tel.core_count(), 2);
        // Core lanes never appear in the per-lane work chart.
        assert!(tel.lanes.is_empty(), "{:?}", tel.lanes);
    }

    #[test]
    fn lane_labels_are_escaped_into_the_page() {
        let mut s = String::from(HDR_OLD);
        s.push('\n');
        for (i, w) in [0u32, 100].iter().enumerate() {
            let _ = writeln!(s, "{:.3},<script>x</script>,{w},work,0,0x0,,", i as f64);
        }
        let html = html_from_csv(&s, "<b>title</b> & more", "t.csv");
        assert!(!html.contains("<script>"), "lane label injected markup");
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("&lt;b&gt;title&lt;/b&gt; &amp; more"));
    }

    #[test]
    fn render_html_writes_a_self_contained_file() {
        let csv = tmp_path("in.csv");
        let out = tmp_path("out.html");
        std::fs::write(&csv, csv_gpu()).expect("write csv");
        render_html(&csv, &out, "round trip").expect("render");

        let html = std::fs::read_to_string(&out).expect("read html");
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("round trip"));
        assert!(html.contains("in.csv"), "source file should be named");
        assert!(section(&html, "gpu-power").is_some());
        // Self-contained: nothing may be fetched at view time.
        for bad in ["<script", "http://", "https://", "src=", "@import", "url("] {
            assert!(!html.contains(bad), "external reference {bad:?} in output");
        }
        let _ = std::fs::remove_file(&csv);
        let _ = std::fs::remove_file(&out);

        // A missing input is an io::Error, never a panic.
        assert!(render_html(Path::new("no-such-telemetry.csv"), &out, "x").is_err());
        let _ = std::fs::remove_file(&out);
    }
}
