// SPDX-License-Identifier: MIT
//! # cec-crucible
//!
//! The orchestrator binary. It is both the per-test tool the PowerShell QC
//! harness invokes (`cec-crucible cpu --seconds 60 --device-id <uuid>`) and a
//! standalone runner (`cec-crucible run cross`).
//!
//! Responsibilities: parse args, resolve device identity, build the requested
//! kernel(s) + budget, run them under one [`StopFlag`] and one marker timeline,
//! then write a device-ID'd JSON report and the JSONL marker feed.

mod args;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, Shape, StopFlag};
use crucible_core::markers::{Event, MarkerLog};
use crucible_core::report::{Report, StageReport, Verdict};
use crucible_core::{sysinfo, Clock, DeviceId};

use crucible_cpu::{CoreSel, CpuKernel};
use crucible_mem::{MemKernel, MemSize};
use crucible_storage::{StorageConfig, StorageKernel, StorageStats};

use args::Parsed;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Options accepted by (nearly) every command.
const COMMON_BOOLS: &[&str] = &[
    "json",
    "no-report",
    "keep",
    "help",
    "buffered",
    "unbuffered",
    "all-drives",
];

const USAGE: &str = "\
cec-crucible — CEC in-house PC-build stress/validation suite

USAGE:
    cec-crucible <command> [options]

COMMANDS:
    info                 Print device id, CPU, memory and QPC info, then exit.
    drives               List fixed physical drives (NVMe/SATA), then exit.
    cpu                  Run the CPU FMA/AVX burn kernel.
    mem                  Run the RAM pattern kernel.
    storage              Run the storage scratch-file kernel.
    run <profile>        Profile: quick | soak | cross | power | storage-cross.
    version              Print version.
    help                 Print this help.

COMMON OPTIONS:
    --seconds <N>        Run duration in seconds.
    --device-id <ID>     Machine id from the harness (else auto-detected).
    --out <DIR>          Output directory for report + markers.
    --no-report          Do not write report/marker files.
    --json               Emit the report as JSON on stdout (pipe-friendly).

CPU OPTIONS:
    --core <all|N>       All cores (default) or a single logical core index.
    --shape <steady|burst>
    --burst-on <MS> --burst-off <MS>   Burst duty cycle (default 20/20 ms).

MEM OPTIONS:
    --mb <N>             Buffer size in MiB (default: 50% of free RAM).

STORAGE OPTIONS:
    --path <DIR>         Target directory for the scratch file (default: .).
    --size-mb <N>        Scratch file size in MiB (default 1024).
    --block-kb <N>       Block size in KiB (default 1024).
    --keep               Keep the scratch file after the run.
    --unbuffered         Force device-direct I/O (FILE_FLAG_NO_BUFFERING).
    --buffered           Force buffered I/O (default off Windows).
    --all-drives         Cross-load every fixed physical drive: solo baseline
                         then concurrent, reporting per-drive slowdown.
                         (Same as `run storage-cross`.)

EXIT CODES:
    0 PASS/PARTIAL   1 FAIL   2 usage error

Zero external dependencies. Emits QPC-timestamped markers for correlation with
an external power-monitoring rig, and a device-ID'd JSON report.
";

// Set by the console-control handler; polled by a bridge thread that trips the
// run's StopFlag so a Ctrl-C ends the current stage cleanly with a report.
static CTRLC: AtomicBool = AtomicBool::new(false);

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => ExitCode::from(code),
        Err(msg) => {
            eprintln!("error: {msg}");
            eprintln!("\nrun `cec-crucible help` for usage.");
            ExitCode::from(2)
        }
    }
}

fn run(argv: &[String]) -> Result<u8, String> {
    let command = match argv.first() {
        Some(c) => c.as_str(),
        None => {
            print!("{USAGE}");
            return Ok(2);
        }
    };
    let rest = &argv[1..];

    match command {
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(0)
        }
        "version" | "-V" | "--version" => {
            println!("cec-crucible {VERSION}");
            Ok(0)
        }
        "info" => cmd_info(rest),
        "drives" => cmd_drives(rest),
        "cpu" => cmd_cpu(rest),
        "mem" => cmd_mem(rest),
        "storage" => cmd_storage(rest),
        "run" => cmd_run(rest),
        other => Err(format!("unknown command '{other}'")),
    }
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&["json", "device-id", "help"])?;

    let clock = Clock::new();
    let device = resolve_device(&p);
    let mem = sysinfo::memory();
    let cpus = sysinfo::logical_cpus();

    if p.has("json") {
        let mut o = crucible_core::Json::object();
        o.push("tool", "cec-crucible")
            .push("version", VERSION)
            .push("device", device.to_json())
            .push("logical_cpus", cpus)
            .push("cpu_backend", CpuKernel::backend())
            .push("qpc_frequency", clock.frequency())
            .push("qpc_backed", clock.is_qpc());
        if let Some(m) = mem {
            let mut mj = crucible_core::Json::object();
            mj.push("total_bytes", m.total_bytes)
                .push("avail_bytes", m.avail_bytes);
            o.push("memory", mj);
        } else {
            o.push("memory", crucible_core::Json::Null);
        }
        println!("{}", o.to_pretty());
        return Ok(0);
    }

    println!("cec-crucible {VERSION}");
    println!("device:");
    println!("  short-id : {}", device.short_id);
    println!("  uuid     : {}", device.uuid);
    println!(
        "  board    : {} (serial {})",
        device.board, device.board_serial
    );
    println!("  system   : {}", device.system);
    println!("  host     : {}", device.host);
    println!("  id-source: {}", device.source);
    println!("cpu:");
    println!("  logical  : {cpus}");
    println!("  backend  : {}", CpuKernel::backend());
    if let Some(m) = mem {
        println!("memory:");
        println!("  total    : {:.1} GiB", gib(m.total_bytes));
        println!("  available: {:.1} GiB", gib(m.avail_bytes));
    } else {
        println!("memory: (unavailable on this platform)");
    }
    println!("timing:");
    println!(
        "  qpc      : {:.3} MHz{}",
        clock.frequency() as f64 / 1.0e6,
        if clock.is_qpc() {
            ""
        } else {
            " (fallback, not QPC)"
        }
    );
    Ok(0)
}

// ---------------------------------------------------------------------------
// drives
// ---------------------------------------------------------------------------

fn cmd_drives(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&["json", "help"])?;
    let drives = crucible_storage::drives::discover();

    if p.has("json") {
        let arr: Vec<crucible_core::Json> = drives
            .iter()
            .map(|d| {
                let mut o = crucible_core::Json::object();
                o.push("number", d.number)
                    .push("bus", d.bus.as_str())
                    .push("primary_root", d.primary_root.as_str())
                    .push("roots", d.roots.clone());
                o
            })
            .collect();
        println!("{}", crucible_core::Json::Array(arr).to_pretty());
        return Ok(0);
    }

    if drives.is_empty() {
        println!("no fixed physical drives detected (or unsupported platform)");
        return Ok(0);
    }
    println!("fixed physical drives ({}):", drives.len());
    for d in &drives {
        println!(
            "  disk {:<2} [{:<5}]  scratch target {}  (volumes: {})",
            d.number,
            d.bus.as_str(),
            d.primary_root,
            d.roots.join(", "),
        );
    }
    Ok(0)
}

// ---------------------------------------------------------------------------
// single-kernel commands
// ---------------------------------------------------------------------------

fn cmd_cpu(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "core",
        "shape",
        "burst-on",
        "burst-off",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let seconds = seconds_arg(&p, 60)?;
    let shape = shape_from(&p)?;
    let cores = core_from(&p)?;
    let kernel = CpuKernel::new(cores);
    let mode = format!("{} {}", shape.mode_str(), cores_label(cores));
    let budget = Budget {
        duration: Duration::from_secs(seconds),
        shape,
        target_watts: None,
    };

    let mut runner = Runner::new(&p)?;
    runner.single_stage(&kernel, &budget, &mode)
}

fn cmd_mem(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "mb",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let seconds = seconds_arg(&p, 60)?;
    let size = mem_size_from(&p, None)?;
    let kernel = MemKernel::new(size);
    let budget = Budget::steady(Duration::from_secs(seconds));

    let mut runner = Runner::new(&p)?;
    runner.single_stage(&kernel, &budget, "steady")
}

fn cmd_storage(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "path",
        "size-mb",
        "block-kb",
        "keep",
        "buffered",
        "unbuffered",
        "all-drives",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let seconds = seconds_arg(&p, 60)?;

    let mut runner = Runner::new(&p)?;
    if p.has("all-drives") {
        return runner.all_drives_storage(&p, seconds);
    }

    let cfg = storage_cfg_from(&p, 1024)?;
    let kernel = StorageKernel::new(cfg);
    let mode = if unbuffered_from(&p) {
        "unbuffered"
    } else {
        "buffered"
    };
    let budget = Budget::steady(Duration::from_secs(seconds));
    runner.single_stage(&kernel, &budget, mode)
}

// ---------------------------------------------------------------------------
// run <profile>
// ---------------------------------------------------------------------------

fn cmd_run(rest: &[String]) -> Result<u8, String> {
    let p = Parsed::parse(rest, COMMON_BOOLS)?;
    p.reject_unknown(&[
        "seconds",
        "core",
        "shape",
        "burst-on",
        "burst-off",
        "mb",
        "path",
        "size-mb",
        "block-kb",
        "keep",
        "buffered",
        "unbuffered",
        "all-drives",
        "device-id",
        "out",
        "no-report",
        "json",
        "help",
    ])?;
    let profile = p
        .positional
        .first()
        .cloned()
        .ok_or("run needs a profile: quick | soak | cross | power | storage-cross")?;

    let mut runner = Runner::new(&p)?;

    match profile.as_str() {
        "quick" => {
            let dur = Duration::from_secs(seconds_arg(&p, 15)?);
            let stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, Some(1024))?)),
                    "steady".into(),
                ),
                (
                    Box::new(StorageKernel::new(storage_cfg_from(&p, 256)?)),
                    "steady".into(),
                ),
            ];
            runner.sequential(stages, |_| Budget::steady(dur))
        }
        "soak" => {
            let dur = Duration::from_secs(seconds_arg(&p, 600)?);
            let stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, None)?)),
                    "steady".into(),
                ),
            ];
            runner.sequential(stages, |_| Budget::steady(dur))
        }
        "cross" => {
            // Concurrent CPU + RAM + storage under one stop and one timeline —
            // the worst-case transient mix that steady single-domain tests miss.
            // (GPU joins this profile in Phase 3.)
            let budget = Budget::steady(Duration::from_secs(seconds_arg(&p, 60)?));
            let stages: Vec<(Box<dyn LoadKernel>, String)> = vec![
                (
                    Box::new(CpuKernel::new(CoreSel::All)),
                    "steady core=all".into(),
                ),
                (
                    Box::new(MemKernel::new(mem_size_from(&p, Some(1024))?)),
                    "steady".into(),
                ),
                (
                    Box::new(StorageKernel::new(storage_cfg_from(&p, 512)?)),
                    "steady".into(),
                ),
            ];
            runner.concurrent(stages, &budget)
        }
        "power" => {
            // Dense-marker CPU burst for the 1kHz power rig to profile the rails.
            let shape = shape_from_burst(&p)?;
            let budget = Budget {
                duration: Duration::from_secs(seconds_arg(&p, 60)?),
                shape,
                target_watts: None,
            };
            runner.note(
                "power profile: CPU burst only in Phase 1; GPU wattage sweeps arrive in Phase 3",
            );
            let kernel = CpuKernel::new(core_from(&p)?);
            let mode = format!("burst {}", cores_label(core_from(&p)?));
            runner.single_stage_budget(&kernel, &budget, &mode)
        }
        "storage-cross" => {
            // Multi-SSD cross-load: solo baseline vs concurrent, to expose
            // shared-lane / chipset-uplink slowdowns across physical drives.
            let seconds = seconds_arg(&p, 60)?;
            runner.all_drives_storage(&p, seconds)
        }
        other => Err(format!(
            "unknown profile '{other}' (expected: quick | soak | cross | power | storage-cross)"
        )),
    }
}

// ---------------------------------------------------------------------------
// Runner: shared orchestration state + output
// ---------------------------------------------------------------------------

struct Runner {
    clock: Clock,
    device: DeviceId,
    markers: MarkerLog,
    stop: StopFlag,
    out_dir: Option<PathBuf>,
    json: bool,
    stamp: String,
    report: Report,
    bridge_done: Arc<AtomicBool>,
    bridge: Option<JoinHandle<()>>,
}

impl Runner {
    fn new(p: &Parsed) -> Result<Runner, String> {
        let clock = Clock::new();
        let device = resolve_device(p);
        let markers = MarkerLog::new(clock);
        let stop = StopFlag::new();

        let out_dir = if p.has("no-report") {
            None
        } else {
            Some(match p.get("out") {
                Some(dir) => PathBuf::from(dir),
                None => default_out_dir(),
            })
        };

        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stamp = format!("{}-{}-{}", device.short_id, unix_secs, std::process::id());

        let report = Report::new(VERSION, device.clone(), &clock);

        // Register the console-control handler and start the bridge that trips
        // the run's StopFlag when Ctrl-C is seen, so a soak ends with a report
        // rather than a hard kill.
        install_ctrlc();
        let bridge_done = Arc::new(AtomicBool::new(false));
        let bridge = spawn_ctrlc_bridge(stop.clone(), bridge_done.clone());

        Ok(Runner {
            clock,
            device,
            markers,
            stop,
            out_dir,
            json: p.has("json"),
            stamp,
            report,
            bridge_done,
            bridge: Some(bridge),
        })
    }

    fn note(&mut self, msg: &str) {
        self.report.note(msg);
        eprintln!("note: {msg}");
    }

    /// Run one kernel as a whole run (steady default budget already built).
    fn single_stage(
        &mut self,
        kernel: &dyn LoadKernel,
        budget: &Budget,
        mode: &str,
    ) -> Result<u8, String> {
        self.single_stage_budget(kernel, budget, mode)
    }

    fn single_stage_budget(
        &mut self,
        kernel: &dyn LoadKernel,
        budget: &Budget,
        mode: &str,
    ) -> Result<u8, String> {
        self.begin();
        self.run_one(kernel, budget, mode);
        self.finish()
    }

    /// Run kernels back to back; a Ctrl-C between stages ends the run.
    fn sequential(
        &mut self,
        stages: Vec<(Box<dyn LoadKernel>, String)>,
        budget_for: impl Fn(usize) -> Budget,
    ) -> Result<u8, String> {
        self.begin();
        for (i, (kernel, mode)) in stages.iter().enumerate() {
            if self.stop.stopped() {
                self.note("stop requested — skipping remaining stages");
                break;
            }
            let budget = budget_for(i);
            self.run_one(kernel.as_ref(), &budget, mode);
        }
        self.finish()
    }

    /// Run kernels concurrently under one stop flag and one timeline.
    fn concurrent(
        &mut self,
        stages: Vec<(Box<dyn LoadKernel>, String)>,
        budget: &Budget,
    ) -> Result<u8, String> {
        self.begin();
        self.markers
            .stamp(Event::StageStart, "cross", "concurrent", "");
        eprintln!("running {} kernels concurrently…", stages.len());
        let t0 = Instant::now();

        let stop = &self.stop;
        let markers = &self.markers;
        let results: Vec<LoadResult> = std::thread::scope(|scope| {
            let handles: Vec<_> = stages
                .iter()
                .map(|(kernel, _)| scope.spawn(move || kernel.run(budget, stop, markers)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| LoadResult::setup_failure("kernel thread panicked"))
                })
                .collect()
        });

        let secs = t0.elapsed().as_secs_f64();
        self.markers
            .stamp(Event::StageStop, "cross", "concurrent", "");

        for ((kernel, mode), result) in stages.iter().zip(results) {
            print_stage(kernel.name(), mode, secs, &result);
            self.report.add_stage(StageReport::new(
                kernel.name(),
                kernel.kind(),
                mode.clone(),
                secs,
                result,
            ));
        }
        self.finish()
    }

    /// Multi-SSD cross-load: measure each physical drive's storage throughput
    /// solo, then all drives concurrently, and report the per-drive slowdown —
    /// the signature of shared PCIe/DMI lanes or a saturated chipset uplink.
    fn all_drives_storage(&mut self, p: &Parsed, seconds: u64) -> Result<u8, String> {
        let drives = crucible_storage::drives::discover();
        if drives.is_empty() {
            // No enumeration (non-Windows, or nothing fixed) — degrade to a
            // single-drive run against --path so the command still does work.
            self.note("no fixed physical drives detected; running single-drive storage on --path");
            let cfg = storage_cfg_from(p, 1024)?;
            let kernel = StorageKernel::new(cfg);
            let budget = Budget::steady(Duration::from_secs(seconds));
            let mode = if unbuffered_from(p) {
                "unbuffered"
            } else {
                "buffered"
            };
            return self.single_stage(&kernel, &budget, mode);
        }

        // Split the budget between the solo baseline and the concurrent phase.
        let solo_secs = (seconds * 2 / 5).max(1); // ~40%
        let conc_secs = seconds.saturating_sub(solo_secs).max(1);

        self.begin();
        self.markers.stamp(
            Event::StageStart,
            "storage-cross",
            "multi-drive",
            &format!("{} drive(s)", drives.len()),
        );
        eprintln!(
            "multi-SSD cross-load: {} physical drive(s), {solo_secs}s solo baseline then {conc_secs}s concurrent",
            drives.len()
        );

        // ---- Phase A: solo baseline (sequential) ----
        let mut runs: Vec<DriveRun> = Vec::new();
        for d in &drives {
            if self.stop.stopped() {
                break;
            }
            let cfg = storage_cfg_for(p, PathBuf::from(&d.primary_root), 1024)?;
            let kernel = StorageKernel::new(cfg);
            let budget = Budget::steady(Duration::from_secs(solo_secs));
            let mode = format!("solo {} [{}]", d.primary_root, d.bus.as_str());
            self.markers.stamp(Event::StageStart, "storage", &mode, "");
            let (result, stats) = kernel.run_measured(&budget, &self.stop, &self.markers);
            self.markers
                .stamp(Event::StageStop, "storage", &mode, &result.detail);
            // A drive we can't write to (e.g. a locked system root without
            // elevation) is skipped, not failed — it must not poison the verdict.
            if !result.ok && result.iterations == 0 {
                eprintln!("  [storage] SKIP  {} — {}", d.primary_root, result.detail);
                self.note(&format!("skipped {} — {}", d.primary_root, result.detail));
                continue;
            }
            print_stage("storage", &mode, solo_secs as f64, &result);
            self.report.add_stage(StageReport::new(
                "storage",
                Kind::Storage,
                mode,
                solo_secs as f64,
                result,
            ));
            runs.push(DriveRun {
                drive: d.clone(),
                solo: stats,
                conc: None,
            });
        }

        // ---- Phase B: concurrent (all writable drives at once) ----
        if runs.len() < 2 {
            self.note("fewer than 2 writable drives — concurrent phase has no cross-load signal");
        }
        if !runs.is_empty() && !self.stop.stopped() {
            let mut kernels: Vec<(StorageKernel, String)> = Vec::new();
            for dr in &runs {
                let cfg = storage_cfg_for(p, PathBuf::from(&dr.drive.primary_root), 1024)?;
                let mode = format!(
                    "concurrent {} [{}]",
                    dr.drive.primary_root,
                    dr.drive.bus.as_str()
                );
                kernels.push((StorageKernel::new(cfg), mode));
            }
            let budget = Budget::steady(Duration::from_secs(conc_secs));
            let budget_ref = &budget;
            let stop = &self.stop;
            let markers = &self.markers;
            eprintln!("running {} drive(s) concurrently…", kernels.len());
            let results: Vec<(LoadResult, StorageStats)> = std::thread::scope(|scope| {
                let handles: Vec<_> = kernels
                    .iter()
                    .map(|(k, _)| scope.spawn(move || k.run_measured(budget_ref, stop, markers)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            (
                                LoadResult::setup_failure("kernel thread panicked"),
                                StorageStats::default(),
                            )
                        })
                    })
                    .collect()
            });
            for (i, ((_, mode), (result, stats))) in kernels.iter().zip(results).enumerate() {
                print_stage("storage", mode, conc_secs as f64, &result);
                runs[i].conc = Some(stats);
                self.report.add_stage(StageReport::new(
                    "storage",
                    Kind::Storage,
                    mode.clone(),
                    conc_secs as f64,
                    result,
                ));
            }
        }

        self.markers
            .stamp(Event::StageStop, "storage-cross", "multi-drive", "");

        // ---- Comparison: solo vs concurrent per drive ----
        eprintln!("\nmulti-SSD cross-load summary (solo -> concurrent):");
        for dr in &runs {
            let Some(c) = dr.conc else { continue };
            let dw = pct_delta(dr.solo.write_mibps, c.write_mibps);
            let dr_ = pct_delta(dr.solo.read_mibps, c.read_mibps);
            let line = format!(
                "disk {} [{}] {}: write {:.0}->{:.0} MiB/s ({dw:+.0}%), read {:.0}->{:.0} MiB/s ({dr_:+.0}%)",
                dr.drive.number,
                dr.drive.bus.as_str(),
                dr.drive.primary_root,
                dr.solo.write_mibps,
                c.write_mibps,
                dr.solo.read_mibps,
                c.read_mibps,
            );
            eprintln!("  {line}");
            self.report.note(line);
            // A large drop under concurrency is the shared-lane / uplink signal.
            if dw <= -15.0 || dr_ <= -15.0 {
                self.report.note(format!(
                    "drive {} shows >15% slowdown under concurrent load — possible shared-lane/uplink contention",
                    dr.drive.primary_root
                ));
            }
        }

        self.finish()
    }

    fn begin(&mut self) {
        eprintln!(
            "cec-crucible {VERSION}  device {} ({})  qpc {:.1} MHz{}",
            self.device.short_id,
            self.device.board,
            self.clock.frequency() as f64 / 1.0e6,
            if self.clock.is_qpc() {
                ""
            } else {
                " (fallback)"
            },
        );
        let ts = self.markers.stamp(Event::RunStart, "run", "", &self.stamp);
        self.report.started = Some(ts);
    }

    fn run_one(&mut self, kernel: &dyn LoadKernel, budget: &Budget, mode: &str) {
        self.markers
            .stamp(Event::StageStart, kernel.name(), mode, "");
        eprintln!(
            "running {} for {}s ({mode})…",
            kernel.name(),
            budget.duration.as_secs()
        );
        let t0 = Instant::now();
        let result = kernel.run(budget, &self.stop, &self.markers);
        let secs = t0.elapsed().as_secs_f64();
        self.markers
            .stamp(Event::StageStop, kernel.name(), mode, &result.detail);
        print_stage(kernel.name(), mode, secs, &result);
        self.report.add_stage(StageReport::new(
            kernel.name(),
            kernel.kind(),
            mode,
            secs,
            result,
        ));
    }

    fn finish(&mut self) -> Result<u8, String> {
        let ts = self.markers.stamp(Event::RunStop, "run", "", "");
        self.report.ended = Some(ts);
        self.report.aborted = self.stop.stopped();

        // Retire the Ctrl-C bridge thread now that the run is over.
        self.bridge_done.store(true, Ordering::SeqCst);
        if let Some(h) = self.bridge.take() {
            let _ = h.join();
        }

        // Write outputs.
        let mut report_path: Option<PathBuf> = None;
        let mut markers_path: Option<PathBuf> = None;
        if let Some(dir) = self.out_dir.clone() {
            match std::fs::create_dir_all(&dir) {
                Ok(()) => {
                    let rp = dir.join(format!("crucible-{}.report.json", self.stamp));
                    let mp = dir.join(format!("crucible-{}.markers.jsonl", self.stamp));
                    self.report.markers_file =
                        Some(mp.file_name().unwrap().to_string_lossy().into_owned());
                    if let Err(e) = self.report.write_json(&rp) {
                        eprintln!("warning: could not write report to {}: {e}", rp.display());
                    } else {
                        report_path = Some(rp);
                    }
                    if let Err(e) = self.markers.write_jsonl(&mp) {
                        eprintln!("warning: could not write markers to {}: {e}", mp.display());
                    } else {
                        markers_path = Some(mp);
                    }
                }
                Err(e) => eprintln!("warning: could not create out dir {}: {e}", dir.display()),
            }
        }

        let verdict = self.report.verdict();

        if self.json {
            // Machine-readable: report JSON is the sole stdout content.
            println!("{}", self.report.to_pretty_json());
        } else {
            println!(
                "verdict: {}   errors: {}   markers: {}",
                verdict.as_str(),
                self.report.error_count(),
                self.markers.len(),
            );
            if let Some(rp) = &report_path {
                println!("report:  {}", rp.display());
            }
            if let Some(mp) = &markers_path {
                println!("markers: {}", mp.display());
            }
        }

        Ok(match verdict {
            Verdict::Pass | Verdict::Partial => 0,
            Verdict::Fail => 1,
        })
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Upper bound on a run duration, so an absurd `--seconds` can't overflow the
/// `Instant + Duration` deadline math (7 days is far beyond any real soak).
const MAX_SECONDS: u64 = 7 * 24 * 3600;
/// Upper bound on a burst half-period in milliseconds.
const MAX_BURST_MS: u64 = 60_000;

/// Read `--seconds`, clamped to a sane range.
fn seconds_arg(p: &Parsed, default: u64) -> Result<u64, String> {
    Ok(p.get_u64("seconds")?
        .unwrap_or(default)
        .clamp(1, MAX_SECONDS))
}

fn resolve_device(p: &Parsed) -> DeviceId {
    match p.get("device-id") {
        Some(id) => DeviceId::from_override(id),
        None => DeviceId::detect(),
    }
}

fn shape_from(p: &Parsed) -> Result<Shape, String> {
    match p.get("shape") {
        None | Some("steady") => Ok(Shape::Steady),
        Some("burst") => {
            let on = p.get_u64("burst-on")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
            let off = p.get_u64("burst-off")?.unwrap_or(20).clamp(0, MAX_BURST_MS);
            Ok(Shape::Burst {
                on: Duration::from_millis(on),
                off: Duration::from_millis(off),
            })
        }
        Some(other) => Err(format!("--shape expects steady|burst, got '{other}'")),
    }
}

/// A burst shape from `--burst-on`/`--burst-off` (clamped), always bursty.
fn shape_from_burst(p: &Parsed) -> Result<Shape, String> {
    let on = p.get_u64("burst-on")?.unwrap_or(20).clamp(1, MAX_BURST_MS);
    let off = p.get_u64("burst-off")?.unwrap_or(20).clamp(0, MAX_BURST_MS);
    Ok(Shape::Burst {
        on: Duration::from_millis(on),
        off: Duration::from_millis(off),
    })
}

fn core_from(p: &Parsed) -> Result<CoreSel, String> {
    match p.get("core") {
        None | Some("all") => Ok(CoreSel::All),
        Some(n) => n
            .parse::<usize>()
            .map(CoreSel::One)
            .map_err(|_| format!("--core expects 'all' or an index, got '{n}'")),
    }
}

fn cores_label(sel: CoreSel) -> String {
    match sel {
        CoreSel::All => "core=all".to_string(),
        CoreSel::One(i) => format!("core={i}"),
    }
}

fn mem_size_from(p: &Parsed, default_mb: Option<u64>) -> Result<MemSize, String> {
    match p.get_u64("mb")? {
        // saturating_mul so an absurd --mb can't overflow/panic (the kernel
        // caps to available RAM via try_reserve anyway).
        Some(mb) => Ok(MemSize::Bytes(mb.saturating_mul(1024 * 1024))),
        None => match default_mb {
            Some(mb) => Ok(MemSize::Bytes(mb.saturating_mul(1024 * 1024))),
            None => Ok(MemSize::default()),
        },
    }
}

/// Resolve buffering: explicit `--buffered`/`--unbuffered` win, else default to
/// unbuffered (true device load) on Windows.
fn unbuffered_from(p: &Parsed) -> bool {
    // Explicit `--unbuffered` wins; else `--buffered` opts out; else default to
    // device-direct I/O on Windows. (Single expression so the `cfg!` literal
    // doesn't fold into a lintable bool-literal if/else.)
    p.has("unbuffered") || (!p.has("buffered") && cfg!(windows))
}

fn storage_cfg_from(p: &Parsed, default_size_mb: u64) -> Result<StorageConfig, String> {
    storage_cfg_for(
        p,
        PathBuf::from(p.get("path").unwrap_or(".")),
        default_size_mb,
    )
}

/// Storage config for a specific directory (used per-drive by `--all-drives`).
fn storage_cfg_for(
    p: &Parsed,
    dir: PathBuf,
    default_size_mb: u64,
) -> Result<StorageConfig, String> {
    let size_mb = p.get_u64("size-mb")?.unwrap_or(default_size_mb);
    let block_kb = p.get_u64("block-kb")?.unwrap_or(1024);
    Ok(StorageConfig {
        dir,
        file_bytes: size_mb.saturating_mul(1024 * 1024),
        block_bytes: block_kb.saturating_mul(1024).min(usize::MAX as u64) as usize,
        keep: p.has("keep"),
        unbuffered: unbuffered_from(p),
    })
}

/// One physical drive's solo baseline and concurrent throughput, for the
/// multi-SSD cross-load comparison.
struct DriveRun {
    drive: crucible_storage::drives::PhysicalDrive,
    solo: StorageStats,
    conc: Option<StorageStats>,
}

/// Percent change from `base` to `val` (negative = slowdown).
fn pct_delta(base: f64, val: f64) -> f64 {
    if base > 0.0 {
        (val - base) / base * 100.0
    } else {
        0.0
    }
}

fn print_stage(name: &str, mode: &str, secs: f64, result: &LoadResult) {
    let verdict = if result.passed() { "PASS" } else { "FAIL" };
    eprintln!(
        "  [{name:<7}] {verdict}  {secs:6.1}s  ({mode})  {}",
        result.detail
    );
}

/// Default output directory: the harness's collection dir if it already exists,
/// otherwise the current directory (never silently create ProgramData paths).
fn default_out_dir() -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(pd) = std::env::var("ProgramData") {
            let p = PathBuf::from(pd).join("firstboot").join("logs");
            if p.is_dir() {
                return p;
            }
        }
    }
    PathBuf::from(".")
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

// --- Ctrl-C handling ---

fn install_ctrlc() {
    #[cfg(windows)]
    {
        win_ctrlc::install();
    }
}

/// Poll the console-handler flag and trip the run's stop when Ctrl-C is seen.
/// Exits promptly once `done` is set at end of run.
fn spawn_ctrlc_bridge(stop: StopFlag, done: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || loop {
        if done.load(Ordering::SeqCst) {
            break;
        }
        if CTRLC.load(Ordering::SeqCst) {
            eprintln!("\nstop requested (Ctrl-C) — finishing current stage and writing report…");
            stop.stop();
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    })
}

#[cfg(windows)]
mod win_ctrlc {
    use super::CTRLC;
    use std::sync::atomic::Ordering;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }

    unsafe extern "system" fn handler(_ctrl_type: u32) -> i32 {
        CTRLC.store(true, Ordering::SeqCst);
        1 // TRUE: handled — suppress the default terminate so we can report.
    }

    pub fn install() {
        // SAFETY: registering a 'static handler fn; idempotent enough for our use.
        unsafe {
            SetConsoleCtrlHandler(Some(handler), 1);
        }
    }
}
