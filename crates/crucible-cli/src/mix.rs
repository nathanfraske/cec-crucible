// SPDX-License-Identifier: MIT
//! `mix` — compose an arbitrary run: N tests, each with its own parameters, all
//! at once. Design of record: `docs/composable-runs.md`.
//!
//! ```text
//! cec-crucible mix [RUN OPTIONS] -- <test> [TEST OPTIONS] [-- <test> [OPTS]]…
//! ```
//!
//! Group 0 holds run-level options; every later group is one participant. The
//! separator is a bare `--`, so the *shell* does the tokenizing and we never
//! invent a second escaping layer — a path with spaces works exactly as it does
//! for a solo command.
//!
//! The orchestration underneath is [`Runner::concurrent_phased`], unchanged:
//! one `StopFlag`, one `MarkerLog`, and one shared phase epoch for every
//! participant, which is what makes per-test phase offsets (`--at`) meaningful
//! rather than a race against per-kernel setup time.
//!
//! Per-test options beyond the command's own:
//! * `--seconds <N>` — this participant's own duration (default: the run's).
//! * `--at <DUR>` — phase offset from the shared epoch (`20ms`, `1500us`, `2s`).
//! * `--as <NAME>` — label for the report, so repeated kernels are tellable apart.

use std::time::Duration;

use crucible_core::kernel::{LoadKernel, Shape};

use crate::args::Parsed;
use crate::{budget_with, seconds_arg, PhasedStage, Runner, COMMON_BOOLS};

/// Per-test options `mix` adds on top of whatever the command itself accepts.
const PER_TEST_OPTS: &[&str] = &["seconds", "at", "as"];

/// Run-level options, valid only in group 0.
const RUN_OPTS: &[&str] = &[
    "seconds",
    "out",
    "device-id",
    "no-report",
    "json",
    "csv",
    "telemetry-csv",
    "ui",
    "dry-run",
    "help",
];

/// One resolved participant, before it becomes a [`PhasedStage`].
struct Participant {
    cmd: &'static str,
    kernel: Box<dyn LoadKernel>,
    mode: String,
    shape: Shape,
    /// Per-test duration override.
    seconds: Option<u64>,
    /// Phase offset from the shared epoch.
    at: Duration,
    /// `--as` label, if given.
    label: Option<String>,
    /// Directory this participant will write into (storage only), for the
    /// same-scratch-file collision guard.
    scratch_dir: Option<String>,
    /// Whether this participant opens a preview window.
    preview: bool,
}

/// Parse a duration: `20ms`, `1500us`, `2s`, or a bare integer meaning ms.
fn parse_dur(s: &str) -> Result<Duration, String> {
    let t = s.trim();
    let (num, mult_us) = if let Some(v) = t.strip_suffix("ms") {
        (v, 1_000u64)
    } else if let Some(v) = t.strip_suffix("us") {
        (v, 1)
    } else if let Some(v) = t.strip_suffix('s') {
        (v, 1_000_000)
    } else {
        (t, 1_000) // bare = milliseconds
    };
    let n: u64 = num
        .trim()
        .parse()
        .map_err(|_| format!("bad duration '{s}' — use e.g. 20ms, 1500us, 2s"))?;
    Ok(Duration::from_micros(n.saturating_mul(mult_us)))
}

/// Build one participant from its own argv (`["cpu", "--shape", "burst", …]`).
///
/// Every knob is parsed by the SAME builder the solo command uses, so `mix` and
/// the standalone command can never drift on what a flag means. Feature gating
/// is inherited too: an unavailable kernel returns the identical "rebuild with
/// --features X" message the solo command gives — and because every participant
/// is built before any thread is spawned, that failure costs zero load time.
fn participant_from(group: &[String]) -> Result<Participant, String> {
    let cmd = group
        .first()
        .ok_or_else(|| "mix: empty '--' group — each one needs a test name".to_string())?
        .clone();
    let args = &group[1..];
    let p = Parsed::parse(args, COMMON_BOOLS)?;

    let at = match p.get("at") {
        Some(v) => parse_dur(v)?,
        None => Duration::ZERO,
    };
    let label = p.get("as").map(|s| s.to_string());
    let seconds = p.get_u64("seconds")?;
    let preview = p.has("preview");

    // Allowlist: the command's own options plus the per-test ones. Run-level
    // flags are rejected inside a group so `-- cpu --out X` is an error, not a
    // silent no-op.
    let mut allowed: Vec<&str> = PER_TEST_OPTS.to_vec();
    allowed.extend_from_slice(crate::SHAPE_OPTS);
    allowed.extend_from_slice(crate::GPU_OPTS);
    allowed.extend_from_slice(&[
        "core", "shape", "burst-on", "burst-off", "mb", "size-mb", "path", "buffered",
        "unbuffered", "preview", "material", "scene",
    ]);
    p.reject_unknown(&allowed)?;

    // Dispatch. Each arm mirrors the solo command's construction.
    let (kernel, mode, shape, scratch_dir): (Box<dyn LoadKernel>, String, Shape, Option<String>) =
        match cmd.as_str() {
            "cpu" => {
                let shape = crate::shape_from(&p)?;
                let cores = crate::core_from(&p)?;
                let mut k = crucible_cpu::CpuKernel::new(cores);
                k.per_core_jitter = p.has("per-core");
                let mode = format!(
                    "{} {}",
                    crate::shape_label(&shape),
                    crate::cores_label(cores)
                );
                (Box::new(k), mode, shape, None)
            }
            "mem" => {
                let size = crate::mem_size_from(&p, None)?;
                (
                    Box::new(crucible_mem::MemKernel::new(size)),
                    "steady".to_string(),
                    Shape::Steady,
                    None,
                )
            }
            "storage" => {
                if p.has("all-drives") {
                    return Err("mix: --all-drives is not composable — run \
                                `run storage-cross` for the multi-drive comparison"
                        .to_string());
                }
                let cfg = crate::storage_cfg_from(&p, 1024)?;
                let dir = cfg.dir.display().to_string();
                (
                    Box::new(crucible_storage::StorageKernel::new(cfg)),
                    "steady".to_string(),
                    Shape::Steady,
                    Some(dir),
                )
            }
            #[cfg(feature = "gpu")]
            "gpu" => {
                let shape = crate::shape_from(&p)?;
                let k = crate::gpu_kernel_from(&p, shape)?;
                (
                    Box::new(k),
                    format!("{} gpu", crate::shape_label(&shape)),
                    shape,
                    None,
                )
            }
            #[cfg(feature = "gpu")]
            "vram" => (
                Box::new(crate::vram_kernel_from(&p)?),
                "integrity".to_string(),
                Shape::Steady,
                None,
            ),
            #[cfg(feature = "gpu")]
            "link" => (
                Box::new(crate::link_kernel_from(&p)?),
                "transfer+verify".to_string(),
                Shape::Steady,
                None,
            ),
            #[cfg(feature = "gpu")]
            "render" => {
                let shape = crate::shape_from(&p)?;
                let k = crate::render_kernel_from(&p)?;
                let mode = format!(
                    "{} render {}x{}",
                    crate::shape_label(&shape),
                    k.width,
                    k.height
                );
                (Box::new(k), mode, shape, None)
            }
            #[cfg(feature = "tensor")]
            "tensor" => {
                let shape = crate::shape_from(&p)?;
                let k = crate::tensor_kernel_from(&p)?;
                (
                    Box::new(k),
                    format!("{} tensor", crate::shape_label(&shape)),
                    shape,
                    None,
                )
            }
            #[cfg(feature = "rt")]
            "rt" => {
                let shape = crate::shape_from(&p)?;
                let k = crate::rt_kernel_from(&p)?;
                (
                    Box::new(k),
                    format!("{} rt", crate::shape_label(&shape)),
                    shape,
                    None,
                )
            }
            #[cfg(feature = "rt")]
            "pathtrace" => {
                let shape = crate::shape_from(&p)?;
                let k = crate::pathtrace_kernel_from(&p)?;
                (
                    Box::new(k),
                    format!("{} pathtrace", crate::shape_label(&shape)),
                    shape,
                    None,
                )
            }
            // Feature-gated kernels that are not in this build: same message the
            // solo command gives.
            #[cfg(not(feature = "gpu"))]
            "gpu" | "vram" | "link" | "render" => {
                return Err(format!("the {cmd} test needs a build with `--features gpu`"))
            }
            #[cfg(not(feature = "tensor"))]
            "tensor" => {
                return Err("the tensor test needs a build with `--features tensor`".to_string())
            }
            #[cfg(not(feature = "rt"))]
            "rt" | "pathtrace" => {
                return Err(format!("the {cmd} test needs a build with `--features rt`"))
            }
            other => {
                return Err(format!(
                    "mix: unknown test '{other}' — expected one of \
                     cpu|mem|storage|gpu|vram|link|render|tensor|rt|pathtrace"
                ))
            }
        };

    // A phase offset only means something for a shape that HAS a phase.
    if at > Duration::ZERO && matches!(shape, Shape::Steady) {
        return Err(format!(
            "mix: --at on a steady '{cmd}' does nothing — it shifts the burst phase, \
             not the start time. Give it a bursty --shape, or drop --at."
        ));
    }

    // Leak the command name so the participant can carry a 'static str.
    let cmd: &'static str = Box::leak(cmd.into_boxed_str());
    Ok(Participant {
        cmd,
        kernel,
        mode,
        shape,
        seconds,
        at,
        label,
        scratch_dir,
        preview,
    })
}

/// Reject compositions that would produce a WRONG result rather than a useful
/// one. These are hard errors, not warnings: a false FAIL in a QC gate is worse
/// than no test at all.
fn validate(ps: &[Participant]) -> Result<(), String> {
    // Two storage participants in one directory share a per-PROCESS scratch
    // filename, so they would overwrite each other's data and both report
    // miscompares that are not real.
    for i in 0..ps.len() {
        for j in (i + 1)..ps.len() {
            if let (Some(a), Some(b)) = (&ps[i].scratch_dir, &ps[j].scratch_dir) {
                if a.eq_ignore_ascii_case(b) {
                    return Err(format!(
                        "mix: two storage tests both target '{a}' — they would share one \
                         scratch file and report false miscompares. Give them different \
                         --path values."
                    ));
                }
            }
        }
    }
    // Two preview windows in one process: closing either stops the whole run.
    if ps.iter().filter(|p| p.preview).count() > 1 {
        return Err("mix: only one participant may use --preview".to_string());
    }
    Ok(())
}

/// Warn about compositions that are legitimate but will not measure what the
/// user probably expects. These print and continue — "three GPU engines at once"
/// is a thing you should be allowed to try.
fn warn(ps: &[Participant], runner: &mut Runner) {
    let n_cpu = ps.iter().filter(|p| p.cmd == "cpu").count();
    if n_cpu > 1 {
        runner.note(&format!(
            "{n_cpu} cpu participants each spawn a thread per core — oversubscription \
             will flatten the burst edges this test exists to produce"
        ));
    }
    let n_mem = ps.iter().filter(|p| p.cmd == "mem").count();
    if n_mem > 1 {
        runner.note(&format!(
            "{n_mem} mem participants — each sizes itself from free RAM by default; \
             give explicit --mb values or they may exhaust memory"
        ));
    }
    let n_gpu = ps
        .iter()
        .filter(|p| matches!(p.cmd, "gpu" | "vram" | "render" | "rt" | "pathtrace" | "tensor"))
        .count();
    if n_gpu > 2 {
        runner.note(&format!(
            "{n_gpu} GPU participants share one device — expect VRAM pressure and \
             contention; throughput numbers are not comparable to a solo run"
        ));
    }
}

/// `mix` entry point.
pub fn cmd_mix(rest: &[String]) -> Result<u8, String> {
    // Split on the bare `--` BEFORE any option parsing: the arg parser would
    // otherwise treat `--` as an empty-named option and swallow the next token.
    let mut groups = rest.split(|t| t == "--");
    let head: Vec<String> = groups.next().unwrap_or(&[]).to_vec();
    let tests: Vec<Vec<String>> = groups.map(|g| g.to_vec()).filter(|g| !g.is_empty()).collect();

    if tests.is_empty() {
        return Err("mix: no tests given — put each one after a '--', e.g. \
                    `mix --seconds 60 -- cpu --shape burst -- mem --mb 2048`"
            .to_string());
    }

    let p = Parsed::parse(&head, COMMON_BOOLS)?;
    if let Some(stray) = p.positional.first() {
        return Err(format!(
            "mix: unexpected '{stray}' before the first '--' — run options come first, \
             then each test after its own '--'"
        ));
    }
    p.reject_unknown(RUN_OPTS)?;
    let run_seconds = seconds_arg(&p, 60)?;

    // Build EVERY participant before spawning anything, so a typo or a missing
    // feature fails instantly instead of after a long run.
    let mut participants = Vec::new();
    for g in &tests {
        participants.push(participant_from(g)?);
    }
    validate(&participants)?;

    // --dry-run: show exactly what would run, then stop.
    if p.has("dry-run") {
        println!("mix: {} participant(s), {run_seconds}s run", participants.len());
        for (i, pt) in participants.iter().enumerate() {
            let name = pt.label.clone().unwrap_or_else(|| format!("#{}", i + 1));
            println!(
                "  {name:<10} {:<10} {:<28} {}s{}",
                pt.cmd,
                pt.mode,
                pt.seconds.unwrap_or(run_seconds),
                if pt.at.is_zero() {
                    String::new()
                } else {
                    format!("  @{:?}", pt.at)
                }
            );
        }
        return Ok(0);
    }

    let mut runner = Runner::new(&p)?;
    warn(&participants, &mut runner);
    runner.note(&format!("mix: {}", rest.join(" ")));

    let stages: Vec<PhasedStage> = participants
        .into_iter()
        .enumerate()
        .map(|(i, pt)| {
            // Slot-tag the mode so repeated kernels are tellable apart in the
            // report and the results CSV.
            let tag = pt
                .label
                .clone()
                .unwrap_or_else(|| format!("#{}", i + 1));
            PhasedStage {
                kernel: pt.kernel,
                mode: format!("{tag} {}", pt.mode),
                budget: budget_with(
                    Duration::from_secs(pt.seconds.unwrap_or(run_seconds)),
                    pt.shape,
                ),
                phase_offset: pt.at,
            }
        })
        .collect();

    runner.concurrent_phased(stages, "mix")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_grammar() {
        assert_eq!(parse_dur("20ms").unwrap(), Duration::from_millis(20));
        assert_eq!(parse_dur("1500us").unwrap(), Duration::from_micros(1500));
        assert_eq!(parse_dur("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_dur("50").unwrap(), Duration::from_millis(50));
        assert!(parse_dur("soon").is_err());
    }

    fn group(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    #[test]
    fn builds_a_cpu_participant_with_its_own_shape() {
        let pt = participant_from(&group("cpu --shape burst --burst-on 20 --burst-off 20"))
            .expect("cpu participant");
        assert_eq!(pt.cmd, "cpu");
        assert!(matches!(pt.shape, Shape::Burst { .. }), "shape {:?}", pt.shape);
    }

    #[test]
    fn rejects_unknown_test_and_run_flags_inside_a_group() {
        assert!(participant_from(&group("nonsense")).is_err());
        // --out is a RUN option; inside a test group it must be rejected rather
        // than silently ignored.
        let e = participant_from(&group("cpu --out C:/tmp")).map(|_| ()).unwrap_err();
        assert!(e.contains("out"), "{e}");
    }

    #[test]
    fn at_on_a_steady_test_is_an_error_not_a_silent_noop() {
        let e = participant_from(&group("mem --at 20ms")).map(|_| ()).unwrap_err();
        assert!(e.contains("--at"), "{e}");
        assert!(e.contains("steady"), "{e}");
    }

    #[test]
    fn same_directory_storage_pair_is_refused() {
        let a = participant_from(&group("storage --path .")).expect("a");
        let b = participant_from(&group("storage --path .")).expect("b");
        let e = validate(&[a, b]).unwrap_err();
        assert!(e.contains("scratch"), "{e}");
    }

    #[test]
    fn mix_needs_at_least_one_test() {
        let e = cmd_mix(&group("--seconds 60")).unwrap_err();
        assert!(e.contains("no tests"), "{e}");
    }

    #[test]
    fn stray_positional_before_the_first_separator_is_caught() {
        let e = cmd_mix(&group("cpu -- mem")).unwrap_err();
        assert!(e.contains("unexpected"), "{e}");
    }
}
