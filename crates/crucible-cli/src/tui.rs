// SPDX-License-Identifier: MIT
//! Live monitoring dashboard (`--ui`), built on ratatui.
//!
//! A render thread reads [`MarkerLog::live_snapshot`] ~16×/s and paints a
//! pinned, in-place dashboard — never a scrolling log:
//!
//! * a **per-thread bar grid** (one bar per CPU core, rising/falling with its
//!   commanded activity), and
//! * a **panel per domain** (GPU / VRAM / RAM / storage / PCIe / render / …) with
//!   a stable-formatted rate, a gauge, phase, verify + error counts, the live
//!   verification hash, and whatever status the kernel is publishing (the memory
//!   pattern, the value written vs expected, watts, …).
//!
//! It only *reads* the shared lanes the [`ShapeDriver`](crucible_core::kernel)
//! feeds; it never touches the kernels. `q` / Ctrl-C flips the run's
//! [`StopFlag`], so quitting ends the stress run cleanly, not just the display.
//! The whole module is behind the `tui` feature so the default build stays
//! zero-dependency.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, Paragraph};
use ratatui::{Frame, Terminal};

use crucible_core::cpustats::{aggregate, CoreStat, CpuStats};
use crucible_core::kernel::StopFlag;
use crucible_core::markers::{LaneSnap, MarkerLog, PHASE_DONE, PHASE_WORK};

use crate::fx::Fx;

use crate::theme;

/// EMA-smoothed work rate for one lane, so the bars/values move smoothly and the
/// displayed number does not jitter frame-to-frame.
struct RateEma {
    last_work: u64,
    last_t: Instant,
    rate: f64,
}

/// Run the dashboard until `ui_stop` is set (by `Runner::finish`). `q` / Ctrl-C
/// flips `run_stop` so the whole run ends.
pub fn render_loop(
    markers: Arc<MarkerLog>,
    ui_stop: Arc<AtomicBool>,
    run_stop: StopFlag,
    title: String,
) {
    let mut stdout = std::io::stdout();
    if enable_raw_mode().is_err() {
        return;
    }
    let _ = execute!(stdout, EnterAlternateScreen);
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(_) => {
            let _ = disable_raw_mode();
            return;
        }
    };
    let _ = terminal.clear();

    let start = Instant::now();
    let mut rates: HashMap<String, RateEma> = HashMap::new();

    // Per-core clock/utilization via PDH (None on non-Windows or if PDH fails —
    // the grid then falls back to lane-rate heat). Sampled ~4×/s, not every
    // redraw: the counters are rate-based and need spacing between collects.
    let mut cpu = CpuStats::new();
    let mut cpu_stats: Vec<CoreStat> = Vec::new();
    let mut last_cpu = Instant::now() - Duration::from_secs(1);

    // Reactive border FX, advanced once per redraw.
    let mut fx = Fx::new();

    while !ui_stop.load(Ordering::Relaxed) {
        // Drain input — quit keys end the *run* (raw mode swallows console Ctrl-C).
        while event::poll(Duration::from_millis(0)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                if k.kind == KeyEventKind::Press {
                    let ctrl_c = k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL);
                    if ctrl_c || k.code == KeyCode::Char('q') {
                        run_stop.stop();
                    }
                }
            }
        }

        let lanes = markers.live_snapshot();
        update_rates(&mut rates, &lanes);
        if let Some(c) = cpu.as_mut() {
            let now = Instant::now();
            if now.duration_since(last_cpu) >= Duration::from_millis(250) {
                last_cpu = now;
                let s = c.sample();
                if !s.is_empty() {
                    cpu_stats = s;
                }
            }
        }
        // Feed the border: how busy we are, whether anything failed, and whether
        // a fresh self-consistency checksum landed since the last frame.
        let working = lanes.iter().filter(|l| l.phase == PHASE_WORK).count();
        let activity = if lanes.is_empty() {
            0.0
        } else {
            working as f32 / lanes.len() as f32
        };
        let errors: u64 = lanes.iter().map(|l| l.errors).sum();
        let verify_mix = lanes
            .iter()
            .fold(0u64, |acc, l| acc.rotate_left(7) ^ l.hash);
        fx.update(activity, errors, verify_mix);

        let n_markers = markers.len();
        let _ = terminal
            .draw(|f| dashboard(f, &title, start, &lanes, &rates, &cpu_stats, n_markers, &fx));
        std::thread::sleep(Duration::from_millis(60));
    }

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
}

/// Refresh each lane's EMA rate roughly 5×/s (independent of the 16 Hz redraw).
fn update_rates(rates: &mut HashMap<String, RateEma>, lanes: &[LaneSnap]) {
    let now = Instant::now();
    for l in lanes {
        let e = rates.entry(l.label.clone()).or_insert(RateEma {
            last_work: l.work,
            last_t: now,
            rate: 0.0,
        });
        let dt = now.duration_since(e.last_t).as_secs_f64();
        if dt >= 0.20 {
            let inst = l.work.saturating_sub(e.last_work) as f64 / dt;
            e.rate = e.rate * 0.6 + inst * 0.4;
            e.last_work = l.work;
            e.last_t = now;
        }
    }
}

fn rate_of(rates: &HashMap<String, RateEma>, label: &str) -> f64 {
    rates.get(label).map(|e| e.rate).unwrap_or(0.0)
}

/// Split cores from domain lanes and lay out header / core-grid / panels.
// The draw fn legitimately needs the whole frame's state (lanes, rates, cpu
// telemetry, marker count, border FX); bundling it into a struct would only move
// the arguments, not reduce them.
#[allow(clippy::too_many_arguments)]
fn dashboard(
    f: &mut Frame,
    title: &str,
    start: Instant,
    lanes: &[LaneSnap],
    rates: &HashMap<String, RateEma>,
    cpu: &[CoreStat],
    n_markers: usize,
    fx: &Fx,
) {
    // On-brand full-screen backdrop (CEC --bg); later widgets keep this bg since
    // they only set fg, so the whole dashboard reads as one dark surface.
    f.render_widget(Block::default().style(Style::default().bg(theme::BG)), f.area());

    let mut cores: Vec<(u32, &LaneSnap)> = lanes
        .iter()
        .filter_map(|l| l.label.strip_prefix("core ").map(|n| (n.parse().unwrap_or(0), l)))
        .collect();
    cores.sort_by_key(|(n, _)| *n);
    let domains: Vec<&LaneSnap> = lanes
        .iter()
        .filter(|l| !l.label.starts_with("core "))
        .collect();

    let has_cores = !cores.is_empty();
    // Height the core panel to the heatmap it will draw (see draw_core_grid):
    // chunky = 3 terminal rows per grid-row, dense (>48 cores) = 1.
    let core_h: u16 = if has_cores {
        let w = (f.area().width.saturating_sub(2)).max(1) as usize;
        // Keep in sync with draw_core_grid: chunky = 4-wide cells over 3 rows,
        // dense (>48 cores) = 1-wide cells on a single row.
        let (cell, per_gridrow) = if cores.len() > 48 { (1usize, 1u16) } else { (4usize, 3u16) };
        let rows = cores.len().div_ceil((w / cell).max(1)).max(1) as u16;
        (2 + rows * per_gridrow).clamp(4, f.area().height / 2)
    } else {
        0
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),        // header
            Constraint::Length(core_h),   // core grid
            Constraint::Min(6),           // domain panels
        ])
        .split(f.area());

    draw_header(f, chunks[0], title, start, lanes, n_markers);
    if has_cores {
        draw_core_grid(f, chunks[1], &cores, rates, cpu);
    }
    draw_panels(f, chunks[2], &domains, rates);

    // Last: the reactive border rides on top of the panel borders it shares
    // cells with, so sparks and lightning are never painted over.
    fx.render(f, f.area());
}

fn draw_header(
    f: &mut Frame,
    area: Rect,
    title: &str,
    start: Instant,
    lanes: &[LaneSnap],
    n_markers: usize,
) {
    let el = start.elapsed().as_secs_f64();
    let errors: u64 = lanes.iter().map(|l| l.errors).sum();
    let any_work = lanes.iter().any(|l| l.phase == PHASE_WORK);
    let all_done = !lanes.is_empty() && lanes.iter().all(|l| l.phase == PHASE_DONE);

    let (status, status_color) = if errors > 0 {
        ("● FAIL", theme::BAD)
    } else if all_done {
        ("● DONE", theme::DIM)
    } else if any_work {
        ("● RUNNING", theme::GOOD)
    } else {
        ("● IDLE", theme::DIM)
    };

    let line = Line::from(vec![
        Span::styled(" ⚠ CRUCIBLE ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{title}  "), Style::default().fg(theme::TEXT)),
        Span::styled(status, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("   {:>6.1}s   markers {n_markers}   err {errors}", el),
            Style::default().fg(theme::DIM),
        ),
        Span::styled("   (q stops the run)", Style::default().fg(theme::FAINT)),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER));
    f.render_widget(Paragraph::new(line).block(block), area);
}

/// Per-core activity as a **heatmap grid** — one fixed cell per core, coloured by
/// its load (dim = idle → cyan→pink = busy). Reads at a glance without the jitter
/// of variable-height bars, and it *scales*: chunky labelled cells for a normal
/// desktop, but a single dense column per core beyond 48 threads so a
/// Threadripper / Xeon (128+) still fits the panel by wrapping.
fn draw_core_grid(
    f: &mut Frame,
    area: Rect,
    cores: &[(u32, &LaneSnap)],
    rates: &HashMap<String, RateEma>,
    cpu: &[CoreStat],
) {
    let max_rate = cores
        .iter()
        .map(|(_, l)| rate_of(rates, &l.label))
        .fold(1.0_f64, f64::max);

    // Real per-core telemetry (PDH: effective clock + utilization), keyed by
    // core index. Empty on non-Windows / PDH failure — the grid then falls back
    // to lane work-rate for heat and the core index for the label.
    let by_core: HashMap<u32, &CoreStat> = cpu.iter().map(|c| (c.core, c)).collect();

    // The title carries the whole-chip aggregate so even the dense 128-thread
    // view has hard numbers (avg / min–max effective clock, avg util), not just
    // a colour field.
    let title = if let Some((avg, min, max, util)) = aggregate(cpu) {
        format!(
            " CPU · {} threads · {:.2} GHz avg  {:.1}–{:.1}  · {:.0}% util ",
            cores.len(),
            avg as f64 / 1000.0,
            min as f64 / 1000.0,
            max as f64 / 1000.0,
            util
        )
    } else {
        format!(" CPU · {} threads ", cores.len())
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            title,
            Style::default().fg(theme::LABEL).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    // Heat prefers real utilization; without PDH it uses the lane's work-rate.
    // A verify error always wins (red), whatever the clock is doing.
    let heat_of = |core: u32, l: &LaneSnap| -> Color {
        if l.errors > 0 {
            return theme::BAD;
        }
        let frac = match by_core.get(&core) {
            // Real utilization is authoritative when we have it.
            Some(cs) => cs.util_pct as f64 / 100.0,
            // Fallback: lane work-rate, but only while the lane is actually
            // working — a finished core shouldn't glow on a stale EMA.
            None if l.phase == PHASE_WORK => rate_of(rates, &l.label) / max_rate,
            None => 0.0,
        };
        if frac > 0.02 {
            core_heat(frac.clamp(0.0, 1.0))
        } else {
            theme::IDLE_BAR
        }
    };
    // Per-core label: the effective clock in GHz when PDH is live (the number
    // the user asked to see), or the plain core index when it isn't.
    let label_of = |core: u32| -> (String, Color) {
        match by_core.get(&core) {
            Some(cs) => (
                format!("{:<4}", format!("{:.1}", cs.effective_mhz as f64 / 1000.0)),
                theme::VALUE,
            ),
            None => (format!("{core:<4}"), theme::FAINT),
        }
    };

    let mut lines: Vec<Line> = Vec::new();
    if cores.len() > 48 {
        // Dense: one column per core, no labels — a 128-thread part still fits.
        // Colour is utilization, and the aggregate clock lives in the title.
        let per_row = (inner.width as usize).max(1);
        let mut spans: Vec<Span> = Vec::new();
        for (i, e) in cores.iter().enumerate() {
            spans.push(Span::styled("█", Style::default().fg(heat_of(e.0, e.1))));
            if (i + 1) % per_row == 0 {
                lines.push(Line::from(std::mem::take(&mut spans)));
            }
        }
        if !spans.is_empty() {
            lines.push(Line::from(spans));
        }
    } else {
        // Chunky: a 4-wide cell — a heat block (utilization) over the core's
        // effective clock in GHz, so each core reads as "how hot + how fast".
        let per_row = (inner.width as usize / 4).max(1);
        let (mut blocks, mut labels): (Vec<Span>, Vec<Span>) = (Vec::new(), Vec::new());
        for (i, e) in cores.iter().enumerate() {
            blocks.push(Span::styled("██", Style::default().fg(heat_of(e.0, e.1))));
            blocks.push(Span::raw("  "));
            let (txt, col) = label_of(e.0);
            labels.push(Span::styled(txt, Style::default().fg(col)));
            if (i + 1) % per_row == 0 {
                lines.push(Line::from(std::mem::take(&mut blocks)));
                lines.push(Line::from(std::mem::take(&mut labels)));
                lines.push(Line::default());
            }
        }
        if !blocks.is_empty() {
            lines.push(Line::from(blocks));
            lines.push(Line::from(labels));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// Heatmap ramp for a core by its normalised load: a neutral bluish slate at idle
/// → amber → CEC pink when hot. The amber midpoint bridges the cool idle and the
/// warm brand pink, so it reads as a smooth thermal-ish gradient rather than a
/// gray→pink clash — on-brand but still legible as "how loaded".
fn core_heat(frac: f64) -> Color {
    let f = frac.clamp(0.0, 1.0);
    // neutral (78,84,104) → amber (242,166,24) → pink (237,35,152)
    let (a, b, t) = if f < 0.5 {
        ([78.0, 84.0, 104.0], [242.0, 166.0, 24.0], f * 2.0)
    } else {
        ([242.0, 166.0, 24.0], [237.0, 35.0, 152.0], (f - 0.5) * 2.0)
    };
    let lerp = |i: usize| (a[i] + (b[i] - a[i]) * t) as u8;
    Color::Rgb(lerp(0), lerp(1), lerp(2))
}

/// Domain lanes as a wrapping grid of rounded panels (≤3 per row).
fn draw_panels(f: &mut Frame, area: Rect, domains: &[&LaneSnap], rates: &HashMap<String, RateEma>) {
    if domains.is_empty() {
        return;
    }
    let per_row = 3usize;
    let rows = domains.len().div_ceil(per_row);
    let row_areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Ratio(1, rows as u32); rows])
        .split(area);

    for (ri, chunk) in domains.chunks(per_row).enumerate() {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Ratio(1, per_row as u32); per_row])
            .split(row_areas[ri]);
        for (ci, lane) in chunk.iter().enumerate() {
            draw_panel(f, cols[ci], lane, rate_of(rates, &lane.label));
        }
    }
}

fn draw_panel(f: &mut Frame, area: Rect, lane: &LaneSnap, rate: f64) {
    let working = lane.phase == PHASE_WORK;
    let (phase_txt, phase_color) = if lane.errors > 0 {
        ("FAIL", theme::BAD)
    } else if lane.phase == PHASE_WORK {
        ("WORK", theme::GOOD)
    } else if lane.phase == PHASE_DONE {
        ("done", theme::DIM)
    } else {
        ("idle", theme::DIM)
    };

    let title = Span::styled(
        format!(" {} ", lane.label.to_uppercase()),
        Style::default().fg(theme::LABEL).add_modifier(Modifier::BOLD),
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if lane.errors > 0 { theme::BAD } else { theme::BORDER }))
        .title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split the panel: a top text block, a gauge line at the bottom.
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(format!("{:<9}", fmt_rate(rate)), Style::default().fg(theme::VALUE).add_modifier(Modifier::BOLD)),
        Span::styled(phase_txt, Style::default().fg(phase_color).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  ✓{}  err {}", short_count(lane.work), lane.errors),
            Style::default().fg(theme::DIM),
        ),
    ]));
    if lane.hash != 0 {
        lines.push(Line::from(Span::styled(
            format!("hash {:#018x}", lane.hash),
            Style::default().fg(theme::HASH),
        )));
    }
    // Kernel-published status: each "\n" line shown, label dimmed, value bright.
    for raw in lane.detail.lines().take(6) {
        lines.push(status_line(raw));
    }
    f.render_widget(Paragraph::new(lines), parts[0]);

    // Activity gauge: log-scaled so KB/s and GB/s lanes both read sensibly.
    let ratio = gauge_ratio(rate);
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(if working { theme::GOOD } else { theme::IDLE_BAR }))
        .ratio(if working { ratio } else { 0.0 })
        .label("");
    f.render_widget(gauge, parts[1]);
}

/// Render a kernel status line: split on the first colon so the field label is
/// dimmed and the value pops.
fn status_line(raw: &str) -> Line<'static> {
    if let Some((k, v)) = raw.split_once(':') {
        Line::from(vec![
            Span::styled(format!("{}:", k), Style::default().fg(theme::FAINT)),
            Span::styled(v.to_string(), Style::default().fg(theme::TEXT)),
        ])
    } else {
        Line::from(Span::styled(raw.to_string(), Style::default().fg(theme::TEXT)))
    }
}

/// Stable rate string: fixed one-decimal, unit-scaled, so digits never jump.
fn fmt_rate(r: f64) -> String {
    if r >= 1.0e9 {
        format!("{:.1} G", r / 1.0e9)
    } else if r >= 1.0e6 {
        format!("{:.1} M", r / 1.0e6)
    } else if r >= 1.0e3 {
        format!("{:.1} k", r / 1.0e3)
    } else {
        format!("{r:.0}")
    }
}

/// Compact count for the ✓ verify tally (K/M), stable width-ish.
fn short_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1.0e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1.0e3)
    } else {
        n.to_string()
    }
}

/// Log-ish gauge fill so a KB/s lane and a GB/s lane both animate.
fn gauge_ratio(rate: f64) -> f64 {
    if rate <= 1.0 {
        0.0
    } else {
        (rate.log10() / 10.0).clamp(0.05, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
//
// ratatui's `TestBackend` renders a frame into an in-memory `Buffer` we can
// assert on cell-by-cell — no real terminal, no escape-code grepping. The same
// buffer is walked into a self-contained coloured-HTML snapshot (written to
// `target/tui-dashboard.html`) so the exact rendered dashboard can be eyeballed
// and shared. Feeds the real `dashboard()` draw fn synthetic-but-realistic lane
// data (cores + memory/GPU/VRAM/pathtrace panels with pattern, values, hash).

#[cfg(test)]
mod tests {
    use super::*;
    use crucible_core::markers::PHASE_IDLE;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Position;

    fn lane(label: &str, work: u64, phase: u8, hash: u64, detail: &str) -> LaneSnap {
        LaneSnap {
            label: label.to_string(),
            work,
            errors: 0,
            phase,
            hash,
            detail: detail.to_string(),
        }
    }

    fn synth_lanes() -> Vec<LaneSnap> {
        let mut v = Vec::new();
        for n in 0..20u32 {
            // A deterministic but varied per-core activity so the grid looks alive.
            let work = (((n.wrapping_mul(37).wrapping_add(9)) % 100) as u64) * 800 + 200;
            let phase = if n % 7 == 3 { PHASE_IDLE } else { PHASE_WORK };
            v.push(lane(&format!("core {n}"), work, phase, 0, ""));
        }
        v.push(lane(
            "mem",
            42_000,
            PHASE_WORK,
            0x9e37_79b9_7f4a_7c15,
            "pattern: moving-inv 0xaaaaaaaaaaaaaaaa\nread:   0xaaaaaaaaaaaaaaaa\nexpect: 0xaaaaaaaaaaaaaaaa  OK\nword:   1310719\nverified: 4.21 GiB",
        ));
        v.push(lane(
            "gpu",
            5_200_000,
            PHASE_WORK,
            0x0000_0000_12ab_34cd,
            "watts:  218 W (92%)\nkernel: fma-thrash x4096\nbackend: vulkan",
        ));
        v.push(lane(
            "vram",
            900_000,
            PHASE_WORK,
            0xdead_beef_cafe_f00d,
            "pattern: checkerboard\nregion: 0x4000..0x0400_0000\nverified: 447 GiB",
        ));
        v.push(lane(
            "pathtrace",
            41,
            PHASE_WORK,
            0x0000_0000_5a3c_11e9,
            "material: glass\nsamples: 64 x8\nrays: 12.3 Gray/s",
        ));
        v.push(lane(
            "storage",
            188_000,
            PHASE_WORK,
            0x7b1d_44c2_9f30_a8e1,
            "mode: unbuffered (FILE_FLAG_NO_BUFFERING)\nphase: VERIFY\nrate: 181 MB/s\npass: 3\nverified: 22.5 GiB\ndir: D:\\crucible-scratch",
        ));
        v.push(lane(
            "pcie",
            6_400,
            PHASE_WORK,
            0x0000_0000_c0ff_ee11,
            "dir: H2D+D2H (full-duplex)\nH2D: 12.84 GB/s\nD2H: 12.61 GB/s\nmoved: 74.30 GiB\nverified: 3200\nerrors: 0",
        ));
        v
    }

    fn synth_rates(lanes: &[LaneSnap]) -> HashMap<String, RateEma> {
        let now = Instant::now();
        lanes
            .iter()
            .map(|l| {
                let rate = match l.label.as_str() {
                    "mem" => 6.5e9,
                    "gpu" => 5.2e9,
                    "vram" => 2.2e10,
                    "pathtrace" => 1.2e10,
                    "storage" => 1.81e8,
                    "pcie" => 2.545e10,
                    _ => (l.work as f64) + 500.0,
                };
                (
                    l.label.clone(),
                    RateEma {
                        last_work: l.work,
                        last_t: now,
                        rate,
                    },
                )
            })
            .collect()
    }

    /// Synthetic PDH telemetry aligned to `synth_lanes`' 20 cores: busy cores
    /// boost high at ~90% util, the parked one (n % 7 == 3) idles low.
    fn synth_cpu() -> Vec<CoreStat> {
        (0..20u32)
            .map(|n| {
                let idle = n % 7 == 3;
                let (effective_mhz, util_pct) = if idle {
                    (1200 + (n % 5) * 90, 3.0 + (n % 4) as f32 * 1.5)
                } else {
                    (4600 + (n * 37) % 520, (88.0 + (n % 11) as f32).min(100.0))
                };
                CoreStat {
                    core: n,
                    effective_mhz,
                    util_pct,
                }
            })
            .collect()
    }

    fn render(w: u16, h: u16) -> Buffer {
        let lanes = synth_lanes();
        let rates = synth_rates(&lanes);
        let cpu = synth_cpu();
        // Warm the border FX so the snapshot shows a live edge: a busy run, with
        // a verification having just landed. Deterministic (stateless hashing).
        let mut fx = Fx::new();
        for i in 0..40 {
            fx.update(0.85, 0, 0x9e37_79b9_7f4a_7c15 ^ (i / 12));
        }
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        let start = Instant::now();
        terminal
            .draw(|f| {
                dashboard(f, "RTX 3070 · worst-case", start, &lanes, &rates, &cpu, 20590, &fx)
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn buffer_text(buf: &Buffer) -> String {
        let a = buf.area;
        let mut s = String::new();
        for y in 0..a.height {
            for x in 0..a.width {
                s.push_str(buf[Position::new(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn dashboard_renders_key_content() {
        // A range of sizes: the layout must fit (no panic) and keep the essentials.
        for (w, h) in [(100u16, 30u16), (140, 44), (200, 60)] {
            let buf = render(w, h);
            let text = buffer_text(&buf);
            assert!(text.contains("CRUCIBLE"), "brand missing @ {w}x{h}");
            assert!(text.contains("CPU"), "core grid title missing @ {w}x{h}");
            assert!(text.contains("MEM"), "mem panel missing @ {w}x{h}");
            assert!(text.contains("moving-inv"), "mem pattern missing @ {w}x{h}");
            assert!(text.contains("GPU"), "gpu panel missing @ {w}x{h}");
            assert!(text.contains("RUNNING"), "status missing @ {w}x{h}");
            // Per-core telemetry: the aggregate clock in the title and at least
            // one boosting per-core effective clock (4.6xx GHz) from synth_cpu.
            assert!(text.contains("GHz avg"), "cpu clock aggregate missing @ {w}x{h}");
            assert!(text.contains("% util"), "cpu util aggregate missing @ {w}x{h}");
            assert!(text.contains("4.6"), "per-core effective clock missing @ {w}x{h}");
        }
    }

    // A visual dump, not an assertion — run on demand:
    //   cargo test -p crucible-cli --features tui -- --ignored emit_html_snapshot
    #[test]
    #[ignore = "writes target/tui-dashboard.html; run explicitly for a visual"]
    fn emit_html_snapshot() {
        let buf = render(150, 46);
        let html = buffer_to_html(&buf);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tui-dashboard.html");
        std::fs::write(path, html).expect("write html snapshot");
        eprintln!("wrote dashboard snapshot -> {path}");
    }

    /// Walk the buffer into a self-contained coloured-HTML `<pre>`, coalescing
    /// runs of same colour/weight so the file stays small.
    fn buffer_to_html(buf: &Buffer) -> String {
        let a = buf.area;
        let mut out = String::from(
            "<!doctype html><html><head><meta charset=\"utf-8\">\
             <title>cec-crucible · live dashboard</title></head>\
             <body style=\"background:#040409;margin:0;padding:24px;\
             display:flex;justify-content:center\">\
             <pre style=\"margin:0;padding:18px;background:#070711;\
             color:#c8d0dc;font:13px/1.22 'Cascadia Code',Consolas,monospace;\
             border-radius:10px;overflow:auto;display:inline-block;\
             box-shadow:0 8px 40px rgba(0,0,0,.6)\">",
        );
        for y in 0..a.height {
            let mut run = String::new();
            let mut run_key: Option<(String, bool)> = None;
            let flush = |out: &mut String, run: &mut String, key: &Option<(String, bool)>| {
                if run.is_empty() {
                    return;
                }
                if let Some((hex, bold)) = key {
                    let weight = if *bold { "font-weight:700" } else { "" };
                    out.push_str(&format!("<span style=\"color:{hex};{weight}\">{run}</span>"));
                }
                run.clear();
            };
            for x in 0..a.width {
                let cell = &buf[Position::new(x, y)];
                let hex = color_hex(cell.fg);
                let bold = cell.modifier.contains(Modifier::BOLD);
                let key = (hex, bold);
                if run_key.as_ref() != Some(&key) {
                    flush(&mut out, &mut run, &run_key);
                    run_key = Some(key);
                }
                match cell.symbol() {
                    "<" => run.push_str("&lt;"),
                    ">" => run.push_str("&gt;"),
                    "&" => run.push_str("&amp;"),
                    s => run.push_str(s),
                }
            }
            flush(&mut out, &mut run, &run_key);
            out.push('\n');
        }
        out.push_str("</pre></body></html>");
        out
    }

    fn color_hex(c: Color) -> String {
        match c {
            Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
            Color::Green => "#78e696".into(),
            Color::Red => "#f56868".into(),
            Color::Cyan => "#60cdff".into(),
            Color::White => "#dce4ec".into(),
            Color::DarkGray => "#54617a".into(),
            Color::Gray => "#8894a6".into(),
            _ => "#c8d0dc".into(),
        }
    }
}
