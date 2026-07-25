// SPDX-License-Identifier: MIT
//! Interactive main-menu launcher (`cec-crucible menu`), built on ratatui.
//!
//! A branded, four-screen flow that browses and launches every test, profile
//! and diagnostic the binary was compiled with:
//!
//! 1. **Main menu** — a centred card of category-keyed rounded panels (it
//!    re-centres on resize and scrolls when a category overflows); `s` opens
//!    Settings.
//! 2. **Test setup** — the selected test's name, description, a list of
//!    adjustable parameters (each a `‹ value ›` ring cycled with ←/→) and a big
//!    red ▶ FIRE button.
//! 3. **Settings** — global CSV-logging + output-directory options (each a
//!    `‹ value ›` ring) injected into every load and profile launch, so one
//!    place flips `--csv` / `--telemetry-csv` / `--out` for the whole session.
//! 4. **Launch** — FIRE leaves the menu UI, prints a one-line header and hands
//!    the built argv to the normal command dispatch ([`crate::run`]) so a menu
//!    launch is byte-identical to the equivalent CLI run — including the live
//!    `--ui` dashboard and the reports — then re-enters the menu.
//!
//! Menu items are feature-gated, so the menu offers exactly what is built in
//! (e.g. the OptiX row only appears in an `--features optix` build) and every
//! parameter maps to a real CLI flag its command actually accepts. Shares the
//! [`crate::theme`] palette with the live dashboard so the whole TUI reads as
//! one surface. The whole module is behind the `tui` feature, so the default
//! build stays zero-dependency.

#![cfg(feature = "tui")]

use std::io::{Stdout, Write};

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::cursor::{Hide, Show};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::{Frame, Terminal};

use crate::theme;
use crate::VERSION;

/// The card widths — content is centred inside the terminal, so these are the
/// intended reading widths (clamped to the terminal when it is narrower).
const MENU_W: u16 = 78;
const SETUP_W: u16 = 74;

/// Tagline shown under the wordmark.
const TAGLINE: &str = "PC-BUILD STRESS & VALIDATION";

/// Duration presets (seconds) offered by the Duration field.
const DURATIONS: [u64; 6] = [10, 30, 60, 120, 300, 600];
/// Path-trace / OptiX samples-per-pixel presets (CLI default 16).
#[cfg(any(feature = "rt", feature = "optix"))]
const SAMPLES: [u64; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
/// Path-trace / OptiX bounce presets (CLI default 8).
#[cfg(any(feature = "rt", feature = "optix"))]
const BOUNCES: [u64; 7] = [1, 2, 4, 8, 16, 32, 64];
/// Ray-tracing re-trace presets (CLI default 192).
#[cfg(feature = "rt")]
const RT_ITERS: [u64; 7] = [32, 64, 128, 192, 256, 512, 1024];
/// Memory-buffer size presets in MiB (`auto` = the CLI default, 50% of free RAM).
const MEM_MB: [u64; 5] = [512, 1024, 2048, 4096, 8192];
/// Storage scratch-file size presets in MiB (`default` = the CLI default, 1024).
const STORAGE_MB: [u64; 5] = [512, 1024, 4096, 16384, 65536];

// ---------------------------------------------------------------------------
// Catalog model
// ---------------------------------------------------------------------------

/// A menu category, its display title and its accent colour.
#[derive(Clone, Copy)]
enum Category {
    Diagnostics,
    Compute,
    #[cfg(feature = "gpu")]
    Gpu,
    Profiles,
}

impl Category {
    fn title(self) -> &'static str {
        match self {
            Category::Diagnostics => "DIAGNOSTICS",
            Category::Compute => "COMPUTE · MEMORY · STORAGE",
            #[cfg(feature = "gpu")]
            Category::Gpu => "GPU",
            Category::Profiles => "PROFILES · CAMPAIGNS",
        }
    }

    fn color(self) -> ratatui::style::Color {
        match self {
            Category::Diagnostics => theme::CAT_DIAG,
            Category::Compute => theme::CAT_CPU,
            #[cfg(feature = "gpu")]
            Category::Gpu => theme::CAT_GPU,
            Category::Profiles => theme::CAT_PROFILE,
        }
    }
}

/// How a row launches. The built argv is handed verbatim to [`crate::run`].
enum Launch {
    /// A diagnostic that just prints — argv is `[cmd]`, no duration / live UI.
    Info(&'static str),
    /// A load test — argv is `[cmd, <field args>…, --ui]`.
    Load(&'static str),
    /// A profile / campaign — argv is `[run, <profile>, <field args>…, --ui]`.
    Profile(&'static str),
    /// A composite benchmark — argv is `[cmd, <field args>…, <--out …>]`. No
    /// `--ui`: it prints its own per-engine + composite scores, not the live
    /// lane dashboard, and it takes only the output-dir setting (it writes its
    /// own benchmark CSV, so `--csv` / `--telemetry-csv` do not apply).
    Bench(&'static str),
}

/// One selectable value for a [`Field`]: what to show, and the argv fragment it
/// contributes to the launch (empty = "the command's own default", so we never
/// emit a redundant flag).
struct Opt {
    show: String,
    args: Vec<String>,
}

/// A single adjustable launch parameter: a labelled ring of options that ←/→
/// cycles. Numeric knobs are a ring of preset values, categorical knobs (shape,
/// material, direction) a ring of names, a toggle a two-entry ring.
struct Field {
    label: &'static str,
    opts: Vec<Opt>,
    idx: usize,
}

impl Field {
    fn left(&mut self) {
        let n = self.opts.len();
        self.idx = (self.idx + n - 1) % n;
    }

    fn right(&mut self) {
        self.idx = (self.idx + 1) % self.opts.len();
    }

    fn show(&self) -> &str {
        &self.opts[self.idx].show
    }

    fn args(&self) -> &[String] {
        &self.opts[self.idx].args
    }
}

/// One launchable row: its label, one-line description, how it launches, and the
/// parameters it exposes (each mapped to a flag the command actually accepts).
struct Test {
    label: &'static str,
    desc: &'static str,
    launch: Launch,
    fields: Vec<Field>,
}

impl Test {
    /// Turn the row + its current field values into an argv for [`crate::run`].
    /// A menu launch is byte-identical to typing the same command by hand.
    /// `Info` diagnostics stay a bare command; loads and profiles also carry the
    /// global [`Settings`] (CSV / output) flags.
    fn build_argv(&self, settings: &Settings) -> Vec<String> {
        match self.launch {
            Launch::Info(cmd) => vec![cmd.to_string()],
            Launch::Load(cmd) => self.load_argv(vec![cmd.to_string()], settings),
            Launch::Profile(p) => self.load_argv(vec!["run".to_string(), p.to_string()], settings),
            Launch::Bench(cmd) => {
                let mut argv = vec![cmd.to_string()];
                for f in &self.fields {
                    argv.extend(f.args().iter().cloned());
                }
                // Benchmark writes its own CSV and has no live UI: only the
                // output-dir setting applies (no --csv / --telemetry-csv / --ui).
                argv.extend(settings.out_args());
                argv
            }
        }
    }

    /// Append every field's args, then the global settings flags, then `--ui` so
    /// the load lands in the live dashboard (`tui::render_loop`) exactly as `--ui`
    /// from the CLI would. Settings ride in beside `--ui` (never on `Info`), so a
    /// configured launch is still byte-identical to the hand-typed command.
    fn load_argv(&self, mut argv: Vec<String>, settings: &Settings) -> Vec<String> {
        for f in &self.fields {
            argv.extend(f.args().iter().cloned());
        }
        settings.append_flags(&mut argv);
        argv.push("--ui".to_string());
        argv
    }
}

/// A category and the rows under it.
struct Group {
    cat: Category,
    tests: Vec<Test>,
}

// --- Field constructors -----------------------------------------------------

/// A numeric ring: each preset emits `--flag <n>`, displayed as `<n><unit>`.
fn num_field(label: &'static str, flag: &'static str, vals: &[u64], default: usize, unit: &str) -> Field {
    let opts = vals
        .iter()
        .map(|&v| Opt {
            show: format!("{v}{unit}"),
            args: vec![flag.to_string(), v.to_string()],
        })
        .collect();
    Field { label, opts, idx: default }
}

/// Duration in seconds — common to every load test and profile.
fn duration_field() -> Field {
    num_field("Duration", "--seconds", &DURATIONS, 1, "s") // default 30s
}

/// Load shape — only offered where the command accepts `--shape` (steady is the
/// default, so it emits nothing; burst emits `--shape burst`).
fn shape_field() -> Field {
    Field {
        label: "Shape",
        opts: vec![
            Opt { show: "steady".into(), args: vec![] },
            Opt { show: "burst".into(), args: vec!["--shape".into(), "burst".into()] },
        ],
        idx: 0,
    }
}

/// Live-preview toggle (render / rt / pathtrace) — `--preview` when on.
#[cfg(feature = "gpu")]
fn preview_field() -> Field {
    Field {
        label: "Preview",
        opts: vec![
            Opt { show: "off".into(), args: vec![] },
            Opt { show: "on".into(), args: vec!["--preview".into()] },
        ],
        idx: 0,
    }
}

/// Path-trace BSDF material — `--material <name>`.
#[cfg(feature = "rt")]
fn material_field() -> Field {
    let mats = ["metal", "matte", "plastic", "mirror", "glass", "velvet", "marble", "fur"];
    Field {
        label: "Material",
        opts: mats
            .iter()
            .map(|&m| Opt { show: m.into(), args: vec!["--material".into(), m.into()] })
            .collect(),
        idx: 0,
    }
}

/// PCIe transfer direction — `--link-dir up|down|bidir` (default bidir).
#[cfg(feature = "gpu")]
fn direction_field() -> Field {
    let dirs = ["bidir", "down", "up"];
    Field {
        label: "Direction",
        opts: dirs
            .iter()
            .map(|&d| Opt { show: d.into(), args: vec!["--link-dir".into(), d.into()] })
            .collect(),
        idx: 0,
    }
}

/// Memory buffer size — `auto` (the CLI default) plus fixed `--mb` presets.
fn mem_size_field() -> Field {
    let mut opts = vec![Opt { show: "auto".into(), args: vec![] }];
    for &v in &MEM_MB {
        opts.push(Opt { show: format!("{v} MB"), args: vec!["--mb".into(), v.to_string()] });
    }
    Field { label: "Size", opts, idx: 0 }
}

/// Storage scratch-file size — `default` (the CLI's 1024 MiB) plus `--size-mb`
/// presets. `default` emits nothing, so an untouched launch stays byte-identical.
fn storage_size_field() -> Field {
    let mut opts = vec![Opt { show: "default".into(), args: vec![] }];
    for &v in &STORAGE_MB {
        opts.push(Opt { show: format!("{v} MB"), args: vec!["--size-mb".into(), v.to_string()] });
    }
    Field { label: "Size", opts, idx: 0 }
}

/// Storage target — the current dir (default) or every fixed SSD at once
/// (`--all-drives`: solo baseline vs concurrent per drive).
fn all_drives_field() -> Field {
    Field {
        label: "Target",
        opts: vec![
            Opt { show: "this dir".into(), args: vec![] },
            Opt { show: "all SSDs".into(), args: vec!["--all-drives".into()] },
        ],
        idx: 0,
    }
}

/// Build the catalog. Rows and whole categories are feature-gated exactly like
/// the CLI, so the menu reflects the build.
#[allow(clippy::vec_init_then_push)] // the GPU pushes are #[cfg]-conditional
fn catalog() -> Vec<Group> {
    let mut groups = Vec::new();

    // --- Diagnostics (print-and-exit, no load / no live UI) ---
    let mut diag = Vec::new();
    diag.push(Test {
        label: "System info",
        desc: "CPU / RAM / board identity — no load.",
        launch: Launch::Info("info"),
        fields: vec![],
    });
    diag.push(Test {
        label: "Drives",
        desc: "Enumerate fixed physical disks — no load.",
        launch: Launch::Info("drives"),
        fields: vec![],
    });
    #[cfg(feature = "gpu")]
    diag.push(Test {
        label: "GPU info",
        desc: "List usable adapters + limits — no load.",
        launch: Launch::Info("gpu-info"),
        fields: vec![],
    });
    groups.push(Group { cat: Category::Diagnostics, tests: diag });

    // --- Compute · Memory · Storage ---
    groups.push(Group {
        cat: Category::Compute,
        tests: vec![
            Test {
                label: "CPU burn",
                desc: "AVX2+FMA power virus, recompute-verified.",
                launch: Launch::Load("cpu"),
                fields: vec![duration_field(), shape_field()],
            },
            Test {
                label: "Memory test",
                desc: "Moving-inversion pattern battery over RAM.",
                launch: Launch::Load("mem"),
                fields: vec![duration_field(), mem_size_field()],
            },
            Test {
                label: "Storage",
                desc: "Uncached scratch write / read-back verify.",
                launch: Launch::Load("storage"),
                fields: vec![duration_field(), storage_size_field(), all_drives_field()],
            },
        ],
    });

    // --- GPU (feature-gated, mirroring the CLI's cfg layout) ---
    #[cfg(feature = "gpu")]
    {
        let mut gpu = Vec::new();
        gpu.push(Test {
            label: "GPU thrash",
            desc: "ALU + VRAM power virus (watts).",
            launch: Launch::Load("gpu"),
            fields: vec![duration_field(), shape_field()],
        });
        gpu.push(Test {
            label: "VRAM integrity",
            desc: "Moving-inversion pattern over VRAM.",
            launch: Launch::Load("vram"),
            fields: vec![duration_field()],
        });
        gpu.push(Test {
            label: "PCIe link",
            desc: "Host<->device transfer + verify bandwidth.",
            launch: Launch::Load("link"),
            fields: vec![duration_field(), direction_field()],
        });
        gpu.push(Test {
            label: "Render",
            desc: "Rasteriser / TMU / ROP + framebuffer verify.",
            launch: Launch::Load("render"),
            fields: vec![duration_field(), shape_field(), preview_field()],
        });
        #[cfg(feature = "tensor")]
        gpu.push(Test {
            label: "Tensor cores",
            desc: "f16 cooperative-matrix GEMM + verify.",
            launch: Launch::Load("tensor"),
            fields: vec![duration_field(), shape_field()],
        });
        #[cfg(feature = "rt")]
        gpu.push(Test {
            label: "Ray tracing",
            desc: "Hardware BVH traversal on the RT cores.",
            launch: Launch::Load("rt"),
            fields: vec![
                duration_field(),
                shape_field(),
                num_field("Iters", "--rt-iters", &RT_ITERS, 3, ""),
                preview_field(),
            ],
        });
        #[cfg(feature = "rt")]
        gpu.push(Test {
            label: "Path tracer",
            desc: "Multi-bounce GI — deep, divergent RT + SM.",
            launch: Launch::Load("pathtrace"),
            fields: vec![
                duration_field(),
                shape_field(),
                num_field("Samples", "--pt-samples", &SAMPLES, 4, ""),
                num_field("Bounces", "--pt-bounces", &BOUNCES, 3, ""),
                material_field(),
                preview_field(),
            ],
        });
        #[cfg(feature = "optix")]
        gpu.push(Test {
            label: "OptiX path tracer",
            desc: "NVIDIA-native OptiX RT + SM path tracer.",
            launch: Launch::Load("optix"),
            fields: vec![
                duration_field(),
                num_field("Samples", "--optix-samples", &SAMPLES, 4, ""),
                num_field("Bounces", "--optix-bounces", &BOUNCES, 3, ""),
            ],
        });
        // Composite "3DMark-class" score across the graphics engines that are
        // built in. No live UI — it prints per-engine + composite scores.
        #[cfg(any(feature = "rt", feature = "preview"))]
        gpu.push(Test {
            label: "Graphics benchmark",
            desc: "Composite score: render + rt + pathtrace, per-engine + total.",
            launch: Launch::Bench("benchmark"),
            fields: vec![duration_field()],
        });
        groups.push(Group { cat: Category::Gpu, tests: gpu });
    }

    // --- Profiles · Campaigns (each `run <profile> --seconds N --ui`) ---
    #[allow(unused_mut)] // the GPU cross-profiles are #[cfg]-conditional
    let mut profiles: Vec<(&str, &str)> = vec![
        ("quick", "CPU + RAM + storage, ~15s."),
        ("soak", "Long steady CPU + RAM."),
        ("cross", "Concurrent cross-load, all domains."),
        ("power", "CPU burst, dense markers for the power rig."),
        ("storage-cross", "Multi-SSD: solo baseline vs concurrent."),
        ("worst-case", "Every domain at once."),
        ("chaos", "Independent seeded jitter storm."),
        ("game-load", "Frame-paced CPU -> GPU handoff."),
        ("core-cycle", "Rotate boost over each core."),
        ("c-states", "Idle / pulse per core."),
    ];
    // The CPU+GPU electrical cross-profiles need a GPU build.
    #[cfg(feature = "gpu")]
    profiles.extend_from_slice(&[
        ("in-phase", "CPU+GPU burst together — peak PSU/OCP."),
        ("anti-phase", "CPU/GPU alternate — VRM/PSU chase load."),
        ("beat", "Offset periods — sweep every phase."),
    ]);
    groups.push(Group {
        cat: Category::Profiles,
        tests: profiles
            .iter()
            .map(|&(p, h)| Test {
                label: p,
                desc: h,
                launch: Launch::Profile(p),
                fields: vec![duration_field()],
            })
            .collect(),
    });

    groups
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// The output-directory presets offered by the Settings screen's Output ring.
/// `Default` injects nothing (each command writes to its own default dir);
/// `./crucible-reports` pins a relative dir; `Home` targets a `crucible-reports`
/// dir under the user profile — only offered when `%USERPROFILE%` resolves, so an
/// unknown home never yields a bad `--out` path. Each entry reuses [`Opt`]: its
/// `show` is the ring label, its `args` the `--out <DIR>` fragment it injects.
fn out_presets() -> Vec<Opt> {
    let mut presets = vec![
        Opt { show: "Default".into(), args: vec![] },
        Opt {
            show: "./crucible-reports".into(),
            args: vec!["--out".into(), "crucible-reports".into()],
        },
    ];
    if let Ok(home) = std::env::var("USERPROFILE") {
        let dir = std::path::Path::new(&home).join("crucible-reports");
        presets.push(Opt {
            show: "Home".into(),
            args: vec!["--out".into(), dir.to_string_lossy().into_owned()],
        });
    }
    presets
}

/// Rows on the Settings screen: results CSV, telemetry CSV, output directory.
const SETTINGS_ROWS: usize = 3;

/// Global CSV-logging + output settings, injected into every load and profile
/// launch (never `Info` diagnostics). All-off by default, so an untouched menu
/// launch stays byte-identical to the equivalent hand-typed command.
#[derive(Default)]
struct Settings {
    /// `--csv`: also write a per-stage results CSV.
    results_csv: bool,
    /// `--telemetry-csv`: also log the time-series telemetry CSV.
    telemetry_csv: bool,
    /// Index into [`out_presets`] — the `--out` directory preset (0 = Default).
    out: usize,
}

impl Settings {
    /// Label of the selected output preset, for the Output ring.
    fn out_label(&self) -> String {
        out_presets()
            .get(self.out)
            .map(|o| o.show.clone())
            .unwrap_or_else(|| "Default".into())
    }

    /// The `--out <DIR>` fragment for the selected preset (empty for Default).
    fn out_args(&self) -> Vec<String> {
        out_presets().get(self.out).map(|o| o.args.clone()).unwrap_or_default()
    }

    /// Cycle the focused row. Rows 0/1 are on/off toggles (←/→ both flip); row 2
    /// steps the output-preset ring by `delta` (−1 / +1), wrapping.
    fn cycle(&mut self, row: usize, delta: isize) {
        match row {
            0 => self.results_csv = !self.results_csv,
            1 => self.telemetry_csv = !self.telemetry_csv,
            _ => {
                let n = out_presets().len() as isize;
                self.out = (self.out as isize + delta).rem_euclid(n) as usize;
            }
        }
    }

    /// Append the enabled flags in a stable order (results, telemetry, out). With
    /// everything off (the default) this pushes nothing, so a default launch is
    /// unchanged and the byte-identical-to-CLI guarantee still holds.
    fn append_flags(&self, argv: &mut Vec<String>) {
        if self.results_csv {
            argv.push("--csv".to_string());
        }
        if self.telemetry_csv {
            argv.push("--telemetry-csv".to_string());
        }
        argv.extend(self.out_args());
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Menu,
    Setup,
    Settings,
}

/// The whole interactive app: the catalog, a flat index of every selectable row
/// (for main-menu navigation), and the current screen + cursor positions.
struct App {
    groups: Vec<Group>,
    /// `(group, test)` for every selectable row, in display order.
    flat: Vec<(usize, usize)>,
    screen: Screen,
    /// Index into `flat` — the highlighted row on the main menu.
    sel: usize,
    /// Setup screen: the focused field, or the FIRE button when `== fields.len()`.
    field_sel: usize,
    /// Global CSV / output settings, injected into every load + profile launch.
    settings: Settings,
    /// Settings screen: the focused row (`0..SETTINGS_ROWS`).
    set_sel: usize,
}

impl App {
    fn new() -> App {
        let groups = catalog();
        let flat = groups
            .iter()
            .enumerate()
            .flat_map(|(gi, g)| (0..g.tests.len()).map(move |ti| (gi, ti)))
            .collect();
        App {
            groups,
            flat,
            screen: Screen::Menu,
            sel: 0,
            field_sel: 0,
            settings: Settings::default(),
            set_sel: 0,
        }
    }

    fn cur(&self) -> (usize, usize) {
        self.flat[self.sel]
    }

    fn enter_setup(&mut self) {
        self.screen = Screen::Setup;
        self.field_sel = 0;
    }

    fn enter_settings(&mut self) {
        self.screen = Screen::Settings;
        self.set_sel = 0;
    }

    /// Height in rows of a category panel (rounded border + one row per test).
    fn group_h(&self, gi: usize) -> u16 {
        2 + self.groups[gi].tests.len() as u16
    }

    fn total_groups_h(&self) -> u16 {
        (0..self.groups.len()).map(|gi| self.group_h(gi)).sum()
    }

    /// The first category to draw so the selected row is fully visible: keep the
    /// selected group and as many groups above it as fit in `viewport_h`.
    fn menu_top_group(&self, viewport_h: u16) -> usize {
        let sg = self.cur().0;
        let mut top = sg;
        let mut used = self.group_h(sg) as usize;
        while top > 0 {
            let h = self.group_h(top - 1) as usize;
            if used + h <= viewport_h as usize {
                used += h;
                top -= 1;
            } else {
                break;
            }
        }
        top
    }

    // --- Event loop ---------------------------------------------------------

    fn event_loop(&mut self, terminal: &mut Term) -> Result<u8, String> {
        loop {
            terminal.draw(|f| self.draw(f)).map_err(|e| e.to_string())?;

            let ev = match event::read() {
                Ok(e) => e,
                Err(_) => return Ok(0),
            };
            let Event::Key(k) = ev else { continue };
            // Ignore key-release events (crossterm can emit them on Windows).
            if k.kind != KeyEventKind::Press {
                continue;
            }
            if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(0);
            }
            match self.screen {
                Screen::Menu => match k.code {
                    KeyCode::Up | KeyCode::Char('k') => self.sel = self.sel.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.sel = (self.sel + 1).min(self.flat.len() - 1)
                    }
                    KeyCode::Home => self.sel = 0,
                    KeyCode::End => self.sel = self.flat.len() - 1,
                    KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Right | KeyCode::Char('l') => {
                        self.enter_setup()
                    }
                    KeyCode::Char('s') => self.enter_settings(),
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(0),
                    _ => {}
                },
                Screen::Setup => {
                    let (g, t) = self.cur();
                    let nf = self.groups[g].tests[t].fields.len();
                    match k.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            self.field_sel = self.field_sel.saturating_sub(1)
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            self.field_sel = (self.field_sel + 1).min(nf) // nf == FIRE
                        }
                        KeyCode::Left | KeyCode::Char('h') => {
                            if self.field_sel < nf {
                                self.groups[g].tests[t].fields[self.field_sel].left();
                            }
                        }
                        KeyCode::Right | KeyCode::Char('l') => {
                            if self.field_sel < nf {
                                self.groups[g].tests[t].fields[self.field_sel].right();
                            }
                        }
                        KeyCode::Enter => {
                            let argv = self.groups[g].tests[t].build_argv(&self.settings);
                            self.fire(terminal, &argv);
                        }
                        KeyCode::Esc | KeyCode::Char('q') => self.screen = Screen::Menu,
                        _ => {}
                    }
                }
                Screen::Settings => match k.code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.set_sel = self.set_sel.saturating_sub(1)
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.set_sel = (self.set_sel + 1).min(SETTINGS_ROWS - 1)
                    }
                    KeyCode::Left | KeyCode::Char('h') => self.settings.cycle(self.set_sel, -1),
                    KeyCode::Right | KeyCode::Char('l') => self.settings.cycle(self.set_sel, 1),
                    KeyCode::Esc | KeyCode::Char('q') => self.screen = Screen::Menu,
                    _ => {}
                },
            }
        }
    }

    /// Run one selection: step out of the menu UI, drive the command on the real
    /// terminal (its own `--ui` dashboard takes over), pause on the verdict, then
    /// step back into the menu. The proven handoff from the old crossterm menu.
    fn fire(&mut self, terminal: &mut Term, argv: &[String]) {
        restore_terminal(terminal);
        launch(argv);
        reenter_terminal(terminal);
    }

    // --- Rendering ----------------------------------------------------------

    fn draw(&self, f: &mut Frame) {
        // On-brand full-screen backdrop (CEC --bg); the card + text render on top
        // and only set fg, so the whole menu reads as one dark surface.
        f.render_widget(Block::default().style(Style::default().bg(theme::BG)), f.area());
        match self.screen {
            Screen::Menu => self.draw_menu(f),
            Screen::Setup => self.draw_setup(f),
            Screen::Settings => self.draw_settings(f),
        }
    }

    fn draw_menu(&self, f: &mut Frame) {
        let area = f.area();
        // Content-sized card, clamped + centred → it re-centres on every resize.
        let inner_h = 2 + self.total_groups_h() + 1; // header + panels + footer
        let card = centered_rect(MENU_W, inner_h.saturating_add(2), area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER));
        let inner = block.inner(card);
        f.render_widget(block, card);

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // header
                Constraint::Min(3),    // category panels
                Constraint::Length(1), // footer
            ])
            .split(inner);

        draw_header(f, rows[0], "MAIN MENU");
        let (more_up, more_down) = self.draw_menu_body(f, rows[1]);
        draw_menu_footer(f, rows[2], more_up, more_down);
    }

    /// Draw the category panels top-down with whole-panel scrolling. Returns
    /// `(more_above, more_below)` so the footer can show scroll hints.
    fn draw_menu_body(&self, f: &mut Frame, area: Rect) -> (bool, bool) {
        let top = self.menu_top_group(area.height);
        let bottom = area.y + area.height;
        let mut y = area.y;
        let mut last = top;
        let mut clipped = false;

        for gi in top..self.groups.len() {
            if y >= bottom {
                break;
            }
            let gh = self.group_h(gi);
            let h = gh.min(bottom - y);
            if h < 3 {
                // No room for a border + a row — treat as overflow and stop.
                clipped = true;
                break;
            }
            self.draw_group(f, Rect { x: area.x, y, width: area.width, height: h }, gi);
            last = gi;
            if h < gh {
                clipped = true;
                break;
            }
            y += gh;
        }
        (top > 0, clipped || last + 1 < self.groups.len())
    }

    fn draw_group(&self, f: &mut Frame, rect: Rect, gi: usize) {
        let g = &self.groups[gi];
        let cat = g.cat;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER))
            .title(Span::styled(
                format!(" {} ", cat.title()),
                Style::default().fg(cat.color()).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(rect);
        f.render_widget(block, rect);

        let (sg, st) = self.cur();
        let lines: Vec<Line> = g
            .tests
            .iter()
            .enumerate()
            .map(|(ti, test)| {
                let selected = gi == sg && ti == st;
                let (marker, label_style) = if selected {
                    ("▸ ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD))
                } else {
                    ("  ", Style::default().fg(cat.color()))
                };
                let hint = Style::default().fg(if selected { theme::TEXT } else { theme::DIM });
                Line::from(vec![
                    Span::styled(marker, Style::default().fg(theme::ACCENT)),
                    Span::styled(format!("{:<16}", test.label), label_style),
                    Span::styled(format!("  {}", test.desc), hint),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(lines), inner);

        // A subtle full-width bar behind the selected row — painted after the
        // text so it only tints the background (leaves the symbols + accent fg).
        // Terminal polish only: the HTML snapshot keys off the ▸ marker + accent
        // instead, since it captures foreground colour, not background.
        if gi == sg && (st as u16) < inner.height {
            let bar = Rect { x: inner.x, y: inner.y + st as u16, width: inner.width, height: 1 };
            f.render_widget(Block::default().style(Style::default().bg(theme::SEL_BG)), bar);
        }
    }

    fn draw_setup(&self, f: &mut Frame) {
        let area = f.area();
        let (g, t) = self.cur();
        let test = &self.groups[g].tests[t];
        let cat = self.groups[g].cat;
        let nf = test.fields.len();
        let fields_h = nf.max(1) as u16;

        // brand + name + desc + spacer + fields + gap + preview + FIRE + footer.
        let inner_h = 4 + fields_h + 1 + 1 + 3 + 1;
        let card = centered_rect(SETUP_W, inner_h + 2, area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(cat.color()));
        let inner = block.inner(card);
        f.render_widget(block, card);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),        // brand
                Constraint::Length(1),        // test name
                Constraint::Length(1),        // description
                Constraint::Length(1),        // spacer
                Constraint::Length(fields_h), // parameter fields
                Constraint::Min(1),           // flexible gap
                Constraint::Length(1),        // argv preview
                Constraint::Length(3),        // FIRE button
                Constraint::Length(1),        // footer
            ])
            .split(inner);

        let w = inner.width as usize;
        draw_header(f, chunks[0], "TEST SETUP");
        // Test name, category-coloured, with a status-dot marker.
        f.render_widget(
            Paragraph::new(lr_line(
                vec![
                    Span::styled(" ● ", Style::default().fg(cat.color())),
                    Span::styled(test.label, Style::default().fg(cat.color()).add_modifier(Modifier::BOLD)),
                ],
                vec![Span::styled(format!("{} ", cat.title()), Style::default().fg(theme::FAINT))],
                w,
            )),
            chunks[1],
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("   {}", test.desc),
                Style::default().fg(theme::DIM),
            ))),
            chunks[2],
        );

        self.draw_fields(f, chunks[4], test, w);

        // The exact command FIRE will run — proof the menu launch is a CLI run.
        // Any Settings flags show here too, so what you see is what runs.
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" ▶ cec-crucible {}", test.build_argv(&self.settings).join(" ")),
                Style::default().fg(theme::FAINT),
            ))),
            chunks[6],
        );

        draw_fire(f, chunks[7], self.field_sel >= nf);

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " ←/→ adjust   ↑/↓ field   Enter ▶ FIRE   Esc back",
                Style::default().fg(theme::FAINT),
            ))),
            chunks[8],
        );
    }

    fn draw_fields(&self, f: &mut Frame, area: Rect, test: &Test, w: usize) {
        if test.fields.is_empty() {
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "   no parameters — press ▶ FIRE to run",
                    Style::default().fg(theme::DIM),
                ))),
                area,
            );
            return;
        }
        let lines: Vec<Line> = test
            .fields
            .iter()
            .enumerate()
            .map(|(i, fld)| {
                let focused = self.field_sel == i;
                let label_style = if focused {
                    Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::FAINT)
                };
                let left = vec![
                    Span::styled(if focused { "  ▸ " } else { "    " }, Style::default().fg(theme::ACCENT)),
                    Span::styled(fld.label, label_style),
                ];
                // The focused field shows the ‹ value › adjuster; others dim it.
                let right = if focused {
                    vec![
                        Span::styled("‹ ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                        Span::styled(fld.show().to_string(), Style::default().fg(theme::VALUE).add_modifier(Modifier::BOLD)),
                        Span::styled(" › ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                    ]
                } else {
                    vec![Span::styled(format!("{}  ", fld.show()), Style::default().fg(theme::DIM))]
                };
                lr_line(left, right, w)
            })
            .collect();
        f.render_widget(Paragraph::new(lines), area);
    }

    /// The Settings screen: a centred card of three `‹ value ›` rings (results
    /// CSV / telemetry CSV / output dir) whose values inject into every launch.
    /// Styled like the setup card — rounded border, brand header, footer hints.
    fn draw_settings(&self, f: &mut Frame) {
        let area = f.area();
        // header + subtitle + spacer + rings + gap + hint + footer.
        let inner_h = 2 + 1 + 1 + SETTINGS_ROWS as u16 + 1 + 1 + 1;
        let card = centered_rect(SETUP_W, inner_h + 2, area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::ACCENT));
        let inner = block.inner(card);
        f.render_widget(block, card);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),                    // brand header (+ SETTINGS tag)
                Constraint::Length(1),                    // subtitle
                Constraint::Length(1),                    // spacer
                Constraint::Length(SETTINGS_ROWS as u16), // the three rings
                Constraint::Min(1),                       // flexible gap
                Constraint::Length(1),                    // hint
                Constraint::Length(1),                    // footer
            ])
            .split(inner);

        let w = inner.width as usize;
        draw_header(f, chunks[0], "SETTINGS");
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "   CSV logging — added to every load & profile launch",
                Style::default().fg(theme::DIM),
            ))),
            chunks[1],
        );

        self.draw_settings_rows(f, chunks[3], w);

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "   written to the --out directory alongside the JSON report",
                Style::default().fg(theme::FAINT),
            ))),
            chunks[5],
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " ←/→ change   ↑/↓ move   Esc back",
                Style::default().fg(theme::FAINT),
            ))),
            chunks[6],
        );
    }

    /// The three settings rows as `‹ value ›` rings, mirroring `draw_fields`: the
    /// focused row shows the adjuster, the others dim their value.
    fn draw_settings_rows(&self, f: &mut Frame, area: Rect, w: usize) {
        let rows: [(&str, String); SETTINGS_ROWS] = [
            ("Results CSV", onoff(self.settings.results_csv)),
            ("Telemetry CSV", onoff(self.settings.telemetry_csv)),
            ("Output", self.settings.out_label()),
        ];
        let lines: Vec<Line> = rows
            .iter()
            .enumerate()
            .map(|(i, (label, value))| {
                let focused = self.set_sel == i;
                let label_style = if focused {
                    Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::FAINT)
                };
                let left = vec![
                    Span::styled(if focused { "  ▸ " } else { "    " }, Style::default().fg(theme::ACCENT)),
                    Span::styled(*label, label_style),
                ];
                // The focused row shows the ‹ value › adjuster; others dim it.
                let right = if focused {
                    vec![
                        Span::styled("‹ ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                        Span::styled(value.clone(), Style::default().fg(theme::VALUE).add_modifier(Modifier::BOLD)),
                        Span::styled(" › ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                    ]
                } else {
                    vec![Span::styled(format!("{value}  "), Style::default().fg(theme::DIM))]
                };
                lr_line(left, right, w)
            })
            .collect();
        f.render_widget(Paragraph::new(lines), area);
    }
}

// ---------------------------------------------------------------------------
// Shared rendering helpers
// ---------------------------------------------------------------------------

/// A `Rect` of the given size centred inside `area` (clamped when it is larger).
fn centered_rect(w: u16, h: u16, area: Rect) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w,
        height: h,
    }
}

/// A single line with `left` spans flush-left and `right` spans flush-right,
/// separated by an elastic run of spaces sized to `width`.
fn lr_line<'a>(left: Vec<Span<'a>>, right: Vec<Span<'a>>, width: usize) -> Line<'a> {
    let span_w = |spans: &[Span]| spans.iter().map(|s| s.content.chars().count()).sum::<usize>();
    let gap = width.saturating_sub(span_w(&left) + span_w(&right)).max(1);
    let mut spans = left;
    spans.push(Span::raw(" ".repeat(gap)));
    spans.extend(right);
    Line::from(spans)
}

/// `on` / `off` label for a boolean Settings ring.
fn onoff(b: bool) -> String {
    if b { "on".to_string() } else { "off".to_string() }
}

/// The two-line brand header shared by both screens: the wordmark + version on
/// top, the tagline + a right-aligned screen tag under it.
fn draw_header(f: &mut Frame, area: Rect, tag: &str) {
    let w = area.width as usize;
    let l0 = lr_line(
        vec![Span::styled(" ⚠ CRUCIBLE", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD))],
        vec![
            Span::styled(format!("v{VERSION} · "), Style::default().fg(theme::DIM)),
            Span::styled("CEC", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(" · cec.direct", Style::default().fg(theme::DIM)),
        ],
        w,
    );
    let l1 = lr_line(
        vec![Span::styled(format!(" {TAGLINE}"), Style::default().fg(theme::FAINT))],
        vec![Span::styled(format!("{tag} "), Style::default().fg(theme::LABEL).add_modifier(Modifier::BOLD))],
        w,
    );
    f.render_widget(Paragraph::new(vec![l0, l1]), area);
}

fn draw_menu_footer(f: &mut Frame, area: Rect, more_up: bool, more_down: bool) {
    let arrows = format!(
        " {}{} ",
        if more_up { "▲" } else { " " },
        if more_down { "▼" } else { " " },
    );
    let line = Line::from(vec![
        Span::styled(arrows, Style::default().fg(theme::ACCENT)),
        Span::styled(
            " ↑/↓ move   Enter configure   s settings   Home/End jump   q quit",
            Style::default().fg(theme::FAINT),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// The big red ▶ FIRE button — filled and bold when focused, outlined otherwise.
fn draw_fire(f: &mut Frame, area: Rect, focused: bool) {
    let border_style = if focused {
        Style::default().fg(theme::FIRE).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme::DIM)
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style);
    if focused {
        block = block.style(Style::default().bg(theme::FIRE_BG));
    }
    let inner = block.inner(area);
    f.render_widget(block, area);

    let label = if focused { "▶  F I R E  ◀" } else { "▶ FIRE" };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            label,
            Style::default().fg(theme::FIRE).add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center),
        inner,
    );
}

// ---------------------------------------------------------------------------
// Terminal lifecycle + launch handoff
// ---------------------------------------------------------------------------

type Term = Terminal<CrosstermBackend<Stdout>>;

/// Entry point for the `menu` command (and bare `cec-crucible`).
pub fn run_menu() -> Result<u8, String> {
    let mut app = App::new();
    if app.flat.is_empty() {
        return Err("no menu items compiled in".to_string());
    }
    let mut terminal = setup_terminal()?;
    let result = app.event_loop(&mut terminal);
    restore_terminal(&mut terminal);
    result
}

fn setup_terminal() -> Result<Term, String> {
    enable_raw_mode().map_err(|e| e.to_string())?;
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen, Hide).map_err(|e| e.to_string())?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out)).map_err(|e| e.to_string())?;
    terminal.clear().map_err(|e| e.to_string())?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Term) {
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen, Show);
    let _ = terminal.show_cursor();
}

/// Re-enter the menu UI after a run has driven the real terminal.
fn reenter_terminal(terminal: &mut Term) {
    let _ = enable_raw_mode();
    let _ = execute!(terminal.backend_mut(), EnterAlternateScreen, Hide);
    let _ = terminal.clear();
}

/// Run one selection to completion on the normal terminal, then pause so the
/// verdict stays on screen until the user acknowledges. Identical in spirit to
/// the old crossterm menu's launch: the menu owns the terminal only while you
/// choose; the command drives it while it runs.
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
            if k.kind == KeyEventKind::Press {
                break;
            }
        }
    }
    let _ = disable_raw_mode();
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
//
// Mirrors tui.rs's harness: ratatui's `TestBackend` renders a screen into an
// in-memory `Buffer` we assert on cell-by-cell — no real terminal. The same
// buffer is walked into a self-contained coloured-HTML snapshot (the MAIN MENU
// to `target/tui-menu.html`, a TEST-SETUP screen to `target/tui-config.html`,
// the SETTINGS screen to `target/tui-settings.html`) so the exact rendered
// screens can be eyeballed and shared.

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Position;
    use ratatui::style::{Color, Modifier};

    fn render(app: &App, w: u16, h: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        terminal.backend().buffer().clone()
    }

    /// An app parked on a given command's setup screen (with FIRE focused, so the
    /// button shows in its prominent state).
    fn setup_app_on(cmd: &str) -> App {
        let mut app = App::new();
        let idx = app
            .flat
            .iter()
            .position(|&(g, t)| match app.groups[g].tests[t].launch {
                Launch::Load(c) | Launch::Info(c) | Launch::Bench(c) => c == cmd,
                Launch::Profile(_) => false,
            })
            .expect("command present in catalog");
        app.sel = idx;
        app.enter_setup();
        let (g, t) = app.cur();
        app.field_sel = app.groups[g].tests[t].fields.len(); // focus FIRE
        app
    }

    /// An app parked on the Settings screen (top row focused).
    fn settings_app() -> App {
        let mut app = App::new();
        app.enter_settings();
        app
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
    fn menu_renders_key_content() {
        // A range of sizes: the layout must fit (no panic) and keep the essentials.
        for (w, h) in [(100u16, 30u16), (140, 44), (200, 60)] {
            let app = App::new();
            let text = buffer_text(&render(&app, w, h));
            assert!(text.contains("CRUCIBLE"), "brand missing @ {w}x{h}");
            assert!(text.contains("DIAGNOSTICS"), "category missing @ {w}x{h}");
            assert!(text.contains("CPU burn"), "test label missing @ {w}x{h}");
            assert!(text.contains(VERSION), "version missing @ {w}x{h}");
        }
    }

    #[test]
    fn setup_renders_key_content() {
        // `cpu` exists in every build (Duration + Shape fields), so this holds
        // whatever GPU features are (or are not) compiled in.
        for (w, h) in [(100u16, 30u16), (140, 44), (200, 60)] {
            let app = setup_app_on("cpu");
            let text = buffer_text(&render(&app, w, h));
            assert!(text.contains("CRUCIBLE"), "brand missing @ {w}x{h}");
            assert!(text.contains("CPU burn"), "test name missing @ {w}x{h}");
            assert!(text.contains("Duration"), "field missing @ {w}x{h}");
            assert!(text.contains("FIRE"), "fire button missing @ {w}x{h}");
            assert!(text.contains(VERSION), "version missing @ {w}x{h}");
        }
    }

    #[test]
    fn argv_is_identical_to_a_cli_run() {
        // The whole point: a menu launch must be byte-for-byte a CLI invocation.
        let strs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        let app = setup_app_on("cpu");
        let (g, t) = app.cur();
        // Defaults: Duration 30s, Shape steady (steady emits nothing), and the
        // default all-off Settings (so no CSV / --out flags leak in).
        assert_eq!(
            app.groups[g].tests[t].build_argv(&app.settings),
            strs(&["cpu", "--seconds", "30", "--ui"])
        );

        // Flip Shape → burst; Duration → 60s and re-check.
        let mut app = setup_app_on("cpu");
        let (g, t) = app.cur();
        app.groups[g].tests[t].fields[0].right(); // Duration 30 -> 60
        app.groups[g].tests[t].fields[1].right(); // Shape steady -> burst
        assert_eq!(
            app.groups[g].tests[t].build_argv(&app.settings),
            strs(&["cpu", "--seconds", "60", "--shape", "burst", "--ui"])
        );

        // A diagnostic is a bare command with no duration / live UI.
        let app = setup_app_on("info");
        let (g, t) = app.cur();
        assert_eq!(app.groups[g].tests[t].build_argv(&app.settings), strs(&["info"]));
    }

    #[test]
    fn profiles_launch_via_run() {
        let app = App::new();
        // Find the `quick` profile row and check its argv shape.
        let (gi, ti) = app
            .flat
            .iter()
            .copied()
            .find(|&(g, t)| matches!(app.groups[g].tests[t].launch, Launch::Profile("quick")))
            .expect("quick profile present");
        let argv = app.groups[gi].tests[ti].build_argv(&app.settings);
        assert_eq!(
            argv,
            ["run", "quick", "--seconds", "30", "--ui"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[cfg(any(feature = "rt", feature = "preview"))]
    #[test]
    fn benchmark_launch_has_no_ui() {
        let app = App::new();
        let (gi, ti) = app
            .flat
            .iter()
            .copied()
            .find(|&(g, t)| matches!(app.groups[g].tests[t].launch, Launch::Bench("benchmark")))
            .expect("benchmark entry present");
        let argv = app.groups[gi].tests[ti].build_argv(&app.settings);
        // Carries its duration field but never --ui (it has no live dashboard).
        let want: Vec<String> = ["benchmark", "--seconds", "30"].iter().map(|s| s.to_string()).collect();
        assert_eq!(argv, want);
        assert!(!argv.contains(&"--ui".to_string()), "benchmark must not carry --ui");
    }

    #[test]
    fn settings_renders_key_content() {
        // The card + its three rings must fit and keep the essentials at both sizes.
        for (w, h) in [(100u16, 30u16), (140, 44)] {
            let app = settings_app();
            let text = buffer_text(&render(&app, w, h));
            assert!(text.contains("SETTINGS"), "title missing @ {w}x{h}");
            assert!(text.contains("Results CSV"), "results row missing @ {w}x{h}");
            assert!(text.contains("Telemetry"), "telemetry row missing @ {w}x{h}");
            assert!(text.contains("Output"), "output row missing @ {w}x{h}");
        }
    }

    #[test]
    fn settings_inject_flags_into_loads() {
        let strs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        // All-off (the default): a load launch is byte-identical to a CLI run.
        let app = setup_app_on("cpu");
        let (g, t) = app.cur();
        assert_eq!(
            app.groups[g].tests[t].build_argv(&app.settings),
            strs(&["cpu", "--seconds", "30", "--ui"])
        );

        // Both CSV flags on → injected ahead of `--ui` on a load test.
        let mut app = setup_app_on("cpu");
        app.settings.results_csv = true;
        app.settings.telemetry_csv = true;
        let (g, t) = app.cur();
        let argv = app.groups[g].tests[t].build_argv(&app.settings);
        assert!(argv.contains(&"--csv".to_string()), "--csv missing: {argv:?}");
        assert!(
            argv.contains(&"--telemetry-csv".to_string()),
            "--telemetry-csv missing: {argv:?}"
        );
        assert!(argv.contains(&"--ui".to_string()), "--ui missing: {argv:?}");

        // The `./crucible-reports` output preset injects `--out crucible-reports`.
        let mut app = setup_app_on("cpu");
        app.settings.out = 1;
        let (g, t) = app.cur();
        let argv = app.groups[g].tests[t].build_argv(&app.settings);
        assert!(
            argv.windows(2).any(|pair| pair[0] == "--out" && pair[1] == "crucible-reports"),
            "--out preset missing: {argv:?}"
        );

        // Profiles carry the settings flags too (they route through `load_argv`).
        let mut app = App::new();
        app.settings.results_csv = true;
        let (gi, ti) = app
            .flat
            .iter()
            .copied()
            .find(|&(g, t)| matches!(app.groups[g].tests[t].launch, Launch::Profile("quick")))
            .expect("quick profile present");
        let argv = app.groups[gi].tests[ti].build_argv(&app.settings);
        assert_eq!(argv.first().map(String::as_str), Some("run"));
        assert!(argv.contains(&"--csv".to_string()), "profile --csv missing: {argv:?}");

        // Info diagnostics never carry settings flags, even with everything on.
        let mut app = setup_app_on("info");
        app.settings.results_csv = true;
        app.settings.telemetry_csv = true;
        app.settings.out = 1;
        let (g, t) = app.cur();
        assert_eq!(app.groups[g].tests[t].build_argv(&app.settings), strs(&["info"]));
    }

    // A visual dump, not an assertion — run on demand:
    //   cargo test -p crucible-cli --features tui -- --ignored emit_html_snapshot
    #[test]
    #[ignore = "writes target/tui-menu.html + target/tui-config.html; run explicitly for a visual"]
    fn emit_html_snapshot() {
        let menu = App::new();
        let menu_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tui-menu.html");
        std::fs::write(
            menu_path,
            buffer_to_html(&render(&menu, 150, 44), "cec-crucible · main menu"),
        )
        .expect("write menu snapshot");
        eprintln!("wrote menu snapshot -> {menu_path}");

        let cfg = setup_app_on("cpu");
        let cfg_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tui-config.html");
        std::fs::write(
            cfg_path,
            buffer_to_html(&render(&cfg, 150, 32), "cec-crucible · test setup"),
        )
        .expect("write config snapshot");
        eprintln!("wrote config snapshot -> {cfg_path}");
    }

    // A visual dump, not an assertion — run on demand:
    //   cargo test -p crucible-cli --features tui -- --ignored emit_settings
    #[test]
    #[ignore = "writes target/tui-settings.html; run explicitly for a visual"]
    fn emit_settings_html_snapshot() {
        // A representative state: both CSVs on, the relative-dir output preset,
        // Output focused so its ‹ value › adjuster shows.
        let mut app = settings_app();
        app.settings.results_csv = true;
        app.settings.telemetry_csv = true;
        app.settings.out = 1;
        app.set_sel = 2;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/tui-settings.html");
        std::fs::write(
            path,
            buffer_to_html(&render(&app, 150, 26), "cec-crucible · settings"),
        )
        .expect("write settings snapshot");
        eprintln!("wrote settings snapshot -> {path}");
    }

    /// Walk the buffer into a self-contained coloured-HTML `<pre>`, coalescing
    /// runs of same colour/weight so the file stays small. (Duplicated from
    /// tui.rs's harness — the two snapshots are deliberately independent.)
    fn buffer_to_html(buf: &Buffer, title: &str) -> String {
        let a = buf.area;
        let mut out = format!(
            "<!doctype html><html><head><meta charset=\"utf-8\">\
             <title>{title}</title></head>\
             <body style=\"background:#040409;margin:0;padding:24px;\
             display:flex;justify-content:center\">\
             <pre style=\"margin:0;padding:18px;background:#070711;\
             color:#c8d0dc;font:13px/1.22 'Cascadia Code',Consolas,monospace;\
             border-radius:10px;overflow:auto;display:inline-block;\
             box-shadow:0 8px 40px rgba(0,0,0,.6)\">"
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
