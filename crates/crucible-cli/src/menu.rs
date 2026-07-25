// SPDX-License-Identifier: MIT
//! Interactive main-menu launcher (`cec-crucible menu`), built on crossterm.
//!
//! One place to browse and launch every test, profile, and diagnostic the binary
//! was compiled with. Selecting a row builds the exact argv you would have typed
//! and hands it to the normal command dispatch ([`crate::run`]) — so a menu launch
//! is identical to a CLI run, including the live `--ui` monitor and reports. The
//! menu only owns the terminal while you are choosing; during a run it steps out
//! of the way and lets the command drive the screen.
//!
//! Menu items are feature-gated, so the menu offers exactly what is built in
//! (e.g. the OptiX row only appears in an `--features optix` build). Kept to
//! crossterm only (no TUI framework); the whole module is behind the `tui`
//! feature, so the default build stays zero-dependency.

#![cfg(feature = "tui")]

use std::io::{Stdout, Write};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{cursor, queue};

use crate::VERSION;

/// Duration presets cycled by the `d` key (seconds).
const DURATIONS: [u64; 6] = [10, 30, 60, 120, 300, 600];

/// What a selectable row launches when you press Enter.
enum Action {
    /// A load test — argv becomes `[cmd, --seconds N, (--shape burst)?, (--preview)?, --ui]`.
    Load {
        cmd: &'static str,
        shape: bool,   // honours the Shape setting
        preview: bool, // honours the Preview setting
    },
    /// A profile / campaign — argv becomes `[run, <profile>, --seconds N, --ui]`.
    Profile(&'static str),
    /// A diagnostic that just prints (no duration / shape / live UI).
    Info(&'static str),
}

/// One catalog entry: either a non-selectable section header or a launchable row.
enum Entry {
    Section(&'static str),
    Item {
        label: &'static str,
        hint: &'static str,
        action: Action,
    },
}

/// Adjustable launch settings, shown at the top and applied to every launch.
struct Settings {
    dur_idx: usize, // into DURATIONS
    burst: bool,    // false = steady, true = burst
    preview: bool,  // pop a live render window for GPU tests that support it
}

impl Settings {
    fn seconds(&self) -> u64 {
        DURATIONS[self.dur_idx]
    }
}

/// Build the catalog. Rows are feature-gated so the menu reflects the build.
#[allow(clippy::vec_init_then_push)] // the pushes are #[cfg]-conditional
fn catalog() -> Vec<Entry> {
    let mut v = Vec::new();

    v.push(Entry::Section("DIAGNOSTICS"));
    v.push(Entry::Item {
        label: "System info",
        hint: "CPU / RAM / board identity",
        action: Action::Info("info"),
    });
    v.push(Entry::Item {
        label: "Drives",
        hint: "enumerate physical disks",
        action: Action::Info("drives"),
    });
    #[cfg(feature = "gpu")]
    v.push(Entry::Item {
        label: "GPU info",
        hint: "adapters + limits",
        action: Action::Info("gpu-info"),
    });

    v.push(Entry::Section("COMPUTE · MEMORY · STORAGE"));
    v.push(Entry::Item {
        label: "CPU burn",
        hint: "AVX2+FMA, recompute-verified",
        action: Action::Load { cmd: "cpu", shape: true, preview: false },
    });
    v.push(Entry::Item {
        label: "Memory test",
        hint: "moving-inversion battery",
        action: Action::Load { cmd: "mem", shape: true, preview: false },
    });
    v.push(Entry::Item {
        label: "Storage",
        hint: "uncached write / verify",
        action: Action::Load { cmd: "storage", shape: true, preview: false },
    });

    #[cfg(feature = "gpu")]
    {
        v.push(Entry::Section("GPU"));
        v.push(Entry::Item {
            label: "GPU thrash",
            hint: "ALU + VRAM power virus",
            action: Action::Load { cmd: "gpu", shape: true, preview: false },
        });
        v.push(Entry::Item {
            label: "VRAM integrity",
            hint: "moving-inversion on VRAM",
            action: Action::Load { cmd: "vram", shape: true, preview: false },
        });
        v.push(Entry::Item {
            label: "PCIe link",
            hint: "transfer + verify bandwidth",
            action: Action::Load { cmd: "link", shape: true, preview: false },
        });
        v.push(Entry::Item {
            label: "Render",
            hint: "rasteriser / TMU / ROP",
            action: Action::Load { cmd: "render", shape: true, preview: true },
        });
        #[cfg(feature = "tensor")]
        v.push(Entry::Item {
            label: "Tensor cores",
            hint: "f16 cmma GEMM",
            action: Action::Load { cmd: "tensor", shape: true, preview: false },
        });
        #[cfg(feature = "rt")]
        v.push(Entry::Item {
            label: "Ray tracing (rt)",
            hint: "ray-query, RT cores",
            action: Action::Load { cmd: "rt", shape: true, preview: true },
        });
        #[cfg(feature = "rt")]
        v.push(Entry::Item {
            label: "Path tracer",
            hint: "multi-bounce GI, materials",
            action: Action::Load { cmd: "pathtrace", shape: true, preview: true },
        });
        #[cfg(feature = "optix")]
        v.push(Entry::Item {
            label: "OptiX path tracer",
            hint: "NVIDIA-native RT + SM",
            action: Action::Load { cmd: "optix", shape: false, preview: false },
        });
    }

    v.push(Entry::Section("PROFILES · CAMPAIGNS"));
    for (p, h) in [
        ("quick", "CPU + RAM + storage, ~15s"),
        ("soak", "long steady CPU + RAM"),
        ("cross", "concurrent cross-load"),
        ("worst-case", "every domain at once"),
        ("chaos", "independent jitter storm"),
        ("game-load", "frame-paced CPU → GPU"),
        ("core-cycle", "rotate boost per core"),
        ("c-states", "idle / pulse per core"),
    ] {
        v.push(Entry::Item {
            label: p,
            hint: h,
            action: Action::Profile(p),
        });
    }

    v
}

/// Turn the selected row + current settings into an argv for [`crate::run`].
fn build_argv(action: &Action, s: &Settings) -> Vec<String> {
    match action {
        Action::Info(cmd) => vec![(*cmd).to_string()],
        Action::Profile(p) => vec![
            "run".to_string(),
            (*p).to_string(),
            "--seconds".to_string(),
            s.seconds().to_string(),
            "--ui".to_string(),
        ],
        Action::Load { cmd, shape, preview } => {
            let mut a = vec![
                (*cmd).to_string(),
                "--seconds".to_string(),
                s.seconds().to_string(),
            ];
            if *shape && s.burst {
                a.push("--shape".to_string());
                a.push("burst".to_string());
            }
            if *preview && s.preview {
                a.push("--preview".to_string());
            }
            a.push("--ui".to_string());
            a
        }
    }
}

/// Entry point for the `menu` command (and bare `cec-crucible`).
pub fn run_menu() -> Result<u8, String> {
    let entries = catalog();
    // Indices of the selectable rows (skip section headers).
    let selectable: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, Entry::Item { .. }))
        .map(|(i, _)| i)
        .collect();
    if selectable.is_empty() {
        return Err("no menu items compiled in".to_string());
    }

    let mut cur = 0usize; // index into `selectable`
    let mut settings = Settings {
        dur_idx: 1, // 30s
        burst: false,
        preview: false,
    };

    let mut out = std::io::stdout();
    enter_ui(&mut out);

    let code = loop {
        draw(&mut out, &entries, &selectable, cur, &settings);

        let ev = match event::read() {
            Ok(e) => e,
            Err(_) => break 0,
        };
        let Event::Key(k) = ev else { continue };
        // Ignore key-release events (crossterm can emit them on Windows).
        if k.kind != crossterm::event::KeyEventKind::Press {
            continue;
        }
        let ctrl_c =
            k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Up | KeyCode::Char('k') => cur = cur.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                cur = (cur + 1).min(selectable.len() - 1)
            }
            KeyCode::Home => cur = 0,
            KeyCode::End => cur = selectable.len() - 1,
            KeyCode::Char('d') => {
                settings.dur_idx = (settings.dur_idx + 1) % DURATIONS.len()
            }
            KeyCode::Char('s') => settings.burst = !settings.burst,
            KeyCode::Char('p') => settings.preview = !settings.preview,
            KeyCode::Char('q') | KeyCode::Esc => break 0,
            _ if ctrl_c => break 0,
            KeyCode::Enter => {
                if let Entry::Item { action, .. } = &entries[selectable[cur]] {
                    let argv = build_argv(action, &settings);
                    exit_ui(&mut out);
                    launch(&argv);
                    enter_ui(&mut out);
                }
            }
            _ => {}
        }
    };

    exit_ui(&mut out);
    Ok(code)
}

/// Run one selection to completion on the normal terminal, then pause so the
/// verdict stays on screen until the user acknowledges.
fn launch(argv: &[String]) {
    println!("\n\x1b[36m▶ cec-crucible {}\x1b[0m\n", argv.join(" "));
    match crate::run(argv) {
        Ok(code) => println!("\n[exit {code}]"),
        Err(e) => println!("\nerror: {e}"),
    }
    print!("\n── done · press any key to return to the menu ──");
    let _ = std::io::stdout().flush();
    let _ = enable_raw_mode();
    loop {
        if let Ok(Event::Key(k)) = event::read() {
            if k.kind == crossterm::event::KeyEventKind::Press {
                break;
            }
        }
    }
    let _ = disable_raw_mode();
}

fn enter_ui(out: &mut Stdout) {
    let _ = enable_raw_mode();
    let _ = queue!(out, EnterAlternateScreen, cursor::Hide);
    let _ = out.flush();
}

fn exit_ui(out: &mut Stdout) {
    let _ = queue!(out, ResetColor, cursor::Show, LeaveAlternateScreen);
    let _ = out.flush();
    let _ = disable_raw_mode();
}

fn draw(
    out: &mut Stdout,
    entries: &[Entry],
    selectable: &[usize],
    cur: usize,
    settings: &Settings,
) {
    let sel_entry = selectable[cur];
    let _ = queue!(out, cursor::MoveTo(0, 0), Clear(ClearType::All));

    // Title.
    let _ = queue!(
        out,
        SetForegroundColor(Color::Cyan),
        Print(" cec-crucible "),
        SetForegroundColor(Color::DarkGrey),
        Print(VERSION),
        SetForegroundColor(Color::White),
        Print("   —   MAIN MENU"),
        ResetColor,
    );

    // Settings line.
    let shape = if settings.burst { "burst" } else { "steady" };
    let preview = if settings.preview { "on" } else { "off" };
    let _ = queue!(
        out,
        cursor::MoveTo(0, 1),
        SetForegroundColor(Color::DarkGrey),
        Print(" duration "),
        SetForegroundColor(Color::Yellow),
        Print(format!("{}s", settings.seconds())),
        SetForegroundColor(Color::DarkGrey),
        Print(" [d]    shape "),
        SetForegroundColor(Color::Yellow),
        Print(shape),
        SetForegroundColor(Color::DarkGrey),
        Print(" [s]    preview "),
        SetForegroundColor(Color::Yellow),
        Print(preview),
        SetForegroundColor(Color::DarkGrey),
        Print(" [p]"),
        ResetColor,
    );

    // Catalog.
    let mut row: u16 = 3;
    for (i, e) in entries.iter().enumerate() {
        match e {
            Entry::Section(name) => {
                row += 1;
                let _ = queue!(
                    out,
                    cursor::MoveTo(1, row),
                    SetForegroundColor(Color::Blue),
                    SetAttribute(Attribute::Bold),
                    Print(*name),
                    SetAttribute(Attribute::Reset),
                    ResetColor,
                );
                row += 1;
            }
            Entry::Item { label, hint, .. } => {
                let selected = i == sel_entry;
                let _ = queue!(out, cursor::MoveTo(2, row));
                if selected {
                    let _ = queue!(
                        out,
                        SetAttribute(Attribute::Reverse),
                        SetForegroundColor(Color::Green),
                        Print(format!(" ▶ {label:<20} ")),
                        SetForegroundColor(Color::White),
                        Print(format!("{hint} ")),
                        SetAttribute(Attribute::Reset),
                        ResetColor,
                    );
                } else {
                    let _ = queue!(
                        out,
                        SetForegroundColor(Color::White),
                        Print(format!("   {label:<20} ")),
                        SetForegroundColor(Color::DarkGrey),
                        Print(*hint),
                        ResetColor,
                    );
                }
                row += 1;
            }
        }
    }

    // Footer.
    row += 1;
    let _ = queue!(
        out,
        cursor::MoveTo(1, row),
        SetForegroundColor(Color::DarkGrey),
        Print("↑/↓ move   Enter launch   d duration · s shape · p preview   q quit"),
        ResetColor,
    );
    let _ = out.flush();
}
