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
use ratatui::widgets::{Bar, BarChart, BarGroup, Block, BorderType, Borders, Gauge, Paragraph};
use ratatui::{Frame, Terminal};

use crucible_core::kernel::StopFlag;
use crucible_core::markers::{LaneSnap, MarkerLog, PHASE_DONE, PHASE_WORK};

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
        let n_markers = markers.len();
        let _ = terminal.draw(|f| dashboard(f, &title, start, &lanes, &rates, n_markers));
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
fn dashboard(
    f: &mut Frame,
    title: &str,
    start: Instant,
    lanes: &[LaneSnap],
    rates: &HashMap<String, RateEma>,
    n_markers: usize,
) {
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
    let core_h: u16 = if has_cores { 11 } else { 0 };
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
        draw_core_grid(f, chunks[1], &cores, rates);
    }
    draw_panels(f, chunks[2], &domains, rates);
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
        Span::styled(" ◆ CRUCIBLE ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
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

/// One bar per CPU core, height = its EMA work rate, coloured by phase — the
/// "big grid of bars raising and lowering per core".
fn draw_core_grid(
    f: &mut Frame,
    area: Rect,
    cores: &[(u32, &LaneSnap)],
    rates: &HashMap<String, RateEma>,
) {
    let max_rate = cores
        .iter()
        .map(|(_, l)| rate_of(rates, &l.label))
        .fold(1.0_f64, f64::max);

    let bars: Vec<Bar> = cores
        .iter()
        .map(|(n, l)| {
            let r = rate_of(rates, &l.label);
            let working = l.phase == PHASE_WORK;
            let frac = (r / max_rate).clamp(0.0, 1.0);
            // Value scaled 0..100 so the tallest core fills the panel.
            let v = (frac * 100.0).round() as u64;
            let color = if l.errors > 0 {
                theme::BAD
            } else if working {
                core_heat(frac)
            } else {
                theme::IDLE_BAR
            };
            Bar::default()
                .value(v)
                .text_value(String::new()) // no number label; keep the grid clean
                .label(Line::from(format!("{n}")))
                .style(Style::default().fg(color))
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            format!(" CPU · {} threads ", cores.len()),
            Style::default().fg(theme::LABEL).add_modifier(Modifier::BOLD),
        ));

    let chart = BarChart::default()
        .block(block)
        .data(BarGroup::default().bars(&bars))
        .bar_width(3)
        .bar_gap(1)
        .max(100);
    f.render_widget(chart, area);
}

/// Green→amber heat ramp for an active core by its normalised load.
fn core_heat(frac: f64) -> Color {
    let f = frac.clamp(0.0, 1.0);
    let r = (80.0 + f * 175.0) as u8;
    let g = (200.0 - f * 40.0) as u8;
    Color::Rgb(r, g, 90)
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
