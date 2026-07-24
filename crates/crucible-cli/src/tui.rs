// SPDX-License-Identifier: MIT
//! Opt-in live terminal UI (`--ui`), built on crossterm.
//!
//! A render thread reads [`MarkerLog::live_snapshot`] ~15×/s and draws each
//! kernel's activity — a grid of CPU cores that flicker as they spike/idle, and
//! a row per GPU/RAM/storage/… lane with its phase and work rate. It never
//! touches the kernels; it only reads the shared lanes the [`ShapeDriver`] feeds.
//! Kept deliberately small (crossterm only, no TUI framework) so the default
//! build stays zero-dependency — this whole module is behind the `tui` feature.
//!
//! Quitting (`q` or Ctrl-C) flips the run's [`StopFlag`], so it ends the stress
//! run cleanly, not just the display.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal::{
    self, disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use crossterm::{cursor, queue};

use crucible_core::kernel::StopFlag;
use crucible_core::markers::{LaneSnap, MarkerLog, PHASE_DONE, PHASE_WORK};

/// Run the live UI until `ui_stop` is set (by `Runner::finish`). `q` / Ctrl-C
/// flips `run_stop` so the whole run ends.
pub fn render_loop(
    markers: Arc<MarkerLog>,
    ui_stop: Arc<AtomicBool>,
    run_stop: StopFlag,
    title: String,
) {
    let mut out = std::io::stdout();
    let _ = enable_raw_mode();
    let _ = queue!(out, EnterAlternateScreen, cursor::Hide);
    let _ = out.flush();

    let start = Instant::now();
    let mut rate: HashMap<String, (u64, Instant, f64)> = HashMap::new();

    while !ui_stop.load(Ordering::Relaxed) {
        // Quit keys end the *run* (raw mode swallows the console Ctrl-C).
        while event::poll(Duration::from_millis(0)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                let ctrl_c =
                    k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
                if ctrl_c || k.code == KeyCode::Char('q') {
                    run_stop.stop();
                }
            }
        }
        draw(&mut out, &markers, &title, start, &mut rate);
        std::thread::sleep(Duration::from_millis(66));
    }

    let _ = queue!(out, ResetColor, cursor::Show, LeaveAlternateScreen);
    let _ = out.flush();
    let _ = disable_raw_mode();
}

fn draw(
    out: &mut std::io::Stdout,
    markers: &MarkerLog,
    title: &str,
    start: Instant,
    rate: &mut HashMap<String, (u64, Instant, f64)>,
) {
    let (cols, _rows) = terminal::size().unwrap_or((100, 40));
    let lanes = markers.live_snapshot();
    let now = Instant::now();

    // Work rate per lane (EMA-smoothed), updated from the previous snapshot.
    for l in &lanes {
        let e = rate.entry(l.label.clone()).or_insert((l.work, now, 0.0));
        let dt = now.duration_since(e.1).as_secs_f64();
        if dt >= 0.20 {
            let inst = (l.work.saturating_sub(e.0)) as f64 / dt;
            e.2 = e.2 * 0.6 + inst * 0.4;
            e.0 = l.work;
            e.1 = now;
        }
    }

    // Split CPU cores from the rest.
    let mut cores: Vec<(u32, &LaneSnap)> = lanes
        .iter()
        .filter_map(|l| {
            l.label
                .strip_prefix("core ")
                .map(|n| (n.parse().unwrap_or(0), l))
        })
        .collect();
    cores.sort_by_key(|(n, _)| *n);
    let others: Vec<&LaneSnap> = lanes
        .iter()
        .filter(|l| !l.label.starts_with("core ") && (l.work > 0 || l.phase == PHASE_WORK))
        .collect();

    let _ = queue!(out, cursor::MoveTo(0, 0), Clear(ClearType::All));
    let el = start.elapsed().as_secs_f64();

    // Header.
    let _ = queue!(
        out,
        SetForegroundColor(Color::Cyan),
        Print(" cec-crucible  "),
        SetForegroundColor(Color::White),
        Print(title),
        SetForegroundColor(Color::DarkGrey),
        Print(format!(
            "    {el:6.1}s    markers {}    (q / Ctrl-C to stop)",
            markers.len()
        )),
        ResetColor,
    );
    let mut row: u16 = 2;

    // CPU core grid — each core a block that lights green while it spikes.
    if !cores.is_empty() {
        let _ = queue!(
            out,
            cursor::MoveTo(0, row),
            SetForegroundColor(Color::White),
            Print(format!(" CPU  {} core(s):", cores.len())),
            ResetColor,
        );
        row += 1;
        let per_row = ((cols as usize).saturating_sub(2) / 4).clamp(8, 32);
        let mut placed = 0;
        let _ = queue!(out, cursor::MoveTo(1, row));
        for (n, l) in &cores {
            let color = lane_color(l);
            let _ = queue!(
                out,
                SetForegroundColor(color),
                Print(format!("{n:>3}")),
                SetForegroundColor(Color::DarkGrey),
                Print("|"),
            );
            placed += 1;
            if placed % per_row == 0 {
                row += 1;
                let _ = queue!(out, cursor::MoveTo(1, row));
            }
        }
        let _ = queue!(out, ResetColor);
        row += 2;
    }

    // Other lanes (GPU / RAM / storage / VRAM / PCIe / render).
    for l in &others {
        let r = rate.get(&l.label).map(|e| e.2).unwrap_or(0.0);
        let color = lane_color(l);
        let phase = match l.phase {
            PHASE_WORK => "WORK",
            PHASE_DONE => "done",
            _ => "idle",
        };
        let bar = bar_for(r);
        let _ = queue!(
            out,
            cursor::MoveTo(1, row),
            SetForegroundColor(Color::White),
            Print(format!("{:<8}", l.label)),
            SetForegroundColor(color),
            Print(format!("[{phase:>4}] ")),
            SetForegroundColor(Color::Green),
            Print(format!("{bar:<16}")),
            SetForegroundColor(Color::DarkGrey),
            Print(format!(" {}/s", fmt_rate(r))),
            ResetColor,
        );
        if l.errors > 0 {
            let _ = queue!(
                out,
                SetForegroundColor(Color::Red),
                Print(format!("  ERR {}", l.errors)),
                ResetColor,
            );
        }
        row += 1;
    }

    let _ = out.flush();
}

fn lane_color(l: &LaneSnap) -> Color {
    if l.errors > 0 {
        Color::Red
    } else if l.phase == PHASE_WORK {
        Color::Green
    } else {
        Color::DarkGrey
    }
}

fn bar_for(rate: f64) -> String {
    // Log-ish bar so both KB/s and GB/s lanes read sensibly.
    let filled = if rate <= 0.0 {
        0
    } else {
        ((rate.log10().max(0.0)) * 2.0).round() as usize
    }
    .min(16);
    let mut s = String::new();
    for _ in 0..filled {
        s.push('#');
    }
    s
}

fn fmt_rate(r: f64) -> String {
    if r >= 1.0e9 {
        format!("{:.1}G", r / 1.0e9)
    } else if r >= 1.0e6 {
        format!("{:.1}M", r / 1.0e6)
    } else if r >= 1.0e3 {
        format!("{:.1}k", r / 1.0e3)
    } else {
        format!("{r:.0}")
    }
}
