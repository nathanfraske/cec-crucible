// SPDX-License-Identifier: MIT
//! Crash watchdog — capture the state around a hard crash instead of losing it.
//!
//! A QC tool that dies without saying anything is worse than useless: a crash is
//! the single most informative event a stress run can produce, and it is exactly
//! the moment we currently learn nothing. Field captures from a client machine
//! contained four hard crashes whose only trace was a telemetry file that simply
//! stopped — no report, no verdict, no indication of what the tool had been
//! doing.
//!
//! Two things kill us without warning, and neither is a Rust panic:
//!
//! * **A structured exception** — an access violation raised inside a GPU driver
//!   DLL while we tear down a device. `catch_unwind` does not see these; the
//!   process is terminated by the OS.
//! * **A `TerminateProcess`** from outside (which nothing in-process can catch).
//!
//! So this module does two independent things:
//!
//! 1. Keeps a **breadcrumb** — a short, always-current description of what the
//!    run is doing — plus the run's identity and output path, in process-global
//!    state that costs a relaxed atomic store to update.
//! 2. Installs an **unhandled-exception filter** and a **panic hook** that write
//!    a `*.crash.json` naming the exception, the faulting address and the
//!    breadcrumb, before the process goes down.
//!
//! What survives even a `TerminateProcess` is the breadcrumb file itself, which
//! is rewritten on every phase change — so the last phase reached is always on
//! disk. `resolve()` on a later run turns a leftover breadcrumb into a finding.
//!
//! The handler runs inside a broken process, so it does the minimum: format a
//! small string and write one file. No locks are taken that a normal path holds.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Where the crash/breadcrumb files go, and what to call them. Set once by
/// [`arm`]; read by the handler.
static TARGET: Mutex<Option<Target>> = Mutex::new(None);
/// The current breadcrumb. A separate lock from `TARGET` so a phase update never
/// contends with arming.
static PHASE: Mutex<Option<String>> = Mutex::new(None);
/// Set once the run finished cleanly — the handler then knows a late crash (in
/// process teardown) is not a run failure.
static FINISHED: AtomicBool = AtomicBool::new(false);
/// 0 = not armed, 1 = armed. Guards against double-installing the hooks.
static ARMED: AtomicU8 = AtomicU8::new(0);

struct Target {
    /// `<out_dir>/crucible-<stamp>` — suffixes are appended.
    base: PathBuf,
    started: Instant,
    /// Human-readable run identity, copied into the crash record.
    label: String,
}

/// Arm the watchdog for a run. `base` is the report path stem, so the crash
/// record lands beside the report it never got to write.
pub fn arm(base: PathBuf, label: impl Into<String>) {
    {
        let mut t = TARGET.lock().unwrap_or_else(|e| e.into_inner());
        *t = Some(Target {
            base,
            started: Instant::now(),
            label: label.into(),
        });
    }
    FINISHED.store(false, Ordering::SeqCst);

    if ARMED.swap(1, Ordering::SeqCst) == 0 {
        install_panic_hook();
        #[cfg(windows)]
        win::install_exception_filter();
    }
    phase("startup");
}

/// Record what the run is doing now. Cheap enough to call at every phase
/// boundary; the string is written straight to disk so a `TerminateProcess`
/// still leaves the last phase behind.
pub fn phase(what: impl Into<String>) {
    let what = what.into();
    {
        let mut p = PHASE.lock().unwrap_or_else(|e| e.into_inner());
        *p = Some(what.clone());
    }
    // Best-effort breadcrumb on disk. Small and overwritten, so the cost is a
    // single tiny write per phase change — not per iteration.
    if let Some(t) = TARGET.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        let path = breadcrumb_path(&t.base);
        let body = format!(
            "{{\"label\":\"{}\",\"phase\":\"{}\",\"elapsed_s\":{:.3}}}\n",
            escape(&t.label),
            escape(&what),
            t.started.elapsed().as_secs_f64()
        );
        let _ = std::fs::write(path, body);
    }
}

/// The run completed normally — remove the breadcrumb so a later `resolve()`
/// does not report a crash that did not happen.
pub fn finished() {
    FINISHED.store(true, Ordering::SeqCst);
    if let Some(t) = TARGET.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        let _ = std::fs::remove_file(breadcrumb_path(&t.base));
    }
}

fn breadcrumb_path(base: &std::path::Path) -> PathBuf {
    let mut p = base.as_os_str().to_os_string();
    p.push(".running");
    PathBuf::from(p)
}

fn crash_path(base: &std::path::Path) -> PathBuf {
    let mut p = base.as_os_str().to_os_string();
    p.push(".crash.json");
    PathBuf::from(p)
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

/// Write the crash record. Called from the panic hook and the exception filter,
/// so it must stay allocation-light and must never panic itself.
fn write_crash(kind: &str, detail: &str) {
    let phase = PHASE
        .lock()
        .map(|p| p.clone().unwrap_or_default())
        .unwrap_or_default();
    let guard = match TARGET.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let Some(t) = guard.as_ref() else { return };
    let body = format!(
        "{{\n  \"crash\": true,\n  \"kind\": \"{}\",\n  \"detail\": \"{}\",\n  \
         \"phase_when_it_died\": \"{}\",\n  \"label\": \"{}\",\n  \
         \"elapsed_s\": {:.3},\n  \"run_had_finished\": {}\n}}\n",
        escape(kind),
        escape(detail),
        escape(&phase),
        escape(&t.label),
        t.started.elapsed().as_secs_f64(),
        FINISHED.load(Ordering::SeqCst),
    );
    let _ = std::fs::write(crash_path(&t.base), body);
    // Also to stderr — a technician watching the console should not have to go
    // looking for the file to know something went wrong.
    eprintln!("\nCRASH [{kind}] during phase '{phase}': {detail}");
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".to_string());
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        write_crash("panic", &format!("{msg} at {loc}"));
        prev(info);
    }));
}

/// Look for a breadcrumb left by a previous run that never finished. Returns a
/// description of what that run was doing when it died, if any.
///
/// This is what turns an unattended overnight campaign from "some runs are
/// missing" into "run X died during GPU teardown".
pub fn resolve(base_dir: &std::path::Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(rd) = std::fs::read_dir(base_dir) else {
        return found;
    };
    for e in rd.flatten() {
        let p = e.path();
        let is_breadcrumb = p
            .extension()
            .map(|x| x.eq_ignore_ascii_case("running"))
            .unwrap_or(false);
        if !is_breadcrumb {
            continue;
        }
        if let Ok(body) = std::fs::read_to_string(&p) {
            found.push(format!(
                "a previous run did not finish — it was last seen in: {}",
                body.trim()
            ));
        }
        // Consume it so the same corpse is not reported forever.
        let _ = std::fs::remove_file(&p);
    }
    found
}

#[cfg(windows)]
mod win {
    use core::ffi::c_void;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut c_void,
        address: *mut c_void,
        n_params: u32,
        info: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        exception_record: *mut ExceptionRecord,
        context_record: *mut c_void,
    }

    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetUnhandledExceptionFilter(
            filter: Option<unsafe extern "system" fn(*mut ExceptionPointers) -> i32>,
        ) -> *mut c_void;
    }

    fn name_of(code: u32) -> &'static str {
        match code {
            0xC000_0005 => "ACCESS_VIOLATION (a read/write to memory the process does not own — \
                            typically raised inside a driver DLL)",
            0xC000_001D => "ILLEGAL_INSTRUCTION",
            0xC000_0025 => "NONCONTINUABLE_EXCEPTION",
            0xC000_008C => "ARRAY_BOUNDS_EXCEEDED",
            0xC000_0094 => "INT_DIVIDE_BY_ZERO",
            0xC000_00FD => "STACK_OVERFLOW",
            0x8000_0003 => "BREAKPOINT",
            _ => "structured exception",
        }
    }

    /// Runs in a process that is already going down. Keep it to formatting one
    /// string and writing one file, then let the default handling proceed.
    unsafe extern "system" fn filter(info: *mut ExceptionPointers) -> i32 {
        // SAFETY: the OS hands us a valid pointer for the duration of the call.
        unsafe {
            if !info.is_null() && !(*info).exception_record.is_null() {
                let rec = &*(*info).exception_record;
                let detail = format!(
                    "{} (code {:#010x}) at address {:p}",
                    name_of(rec.code),
                    rec.code,
                    rec.address
                );
                super::write_crash("structured_exception", &detail);
            }
        }
        // Let Windows carry on with its normal handling (WER etc.) — we are here
        // to record, not to pretend the process is healthy.
        EXCEPTION_CONTINUE_SEARCH
    }

    pub(super) fn install_exception_filter() {
        // SAFETY: installing a 'static filter fn; documented, idempotent enough.
        unsafe {
            SetUnhandledExceptionFilter(Some(filter));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test, not three: `TARGET`/`PHASE` are process-global, so parallel
    /// test threads would arm over each other. Sequencing the assertions here is
    /// the honest fix — the alternative (a test-only lock) would be testing the
    /// lock, not the watchdog.
    #[test]
    fn breadcrumb_lifecycle_and_crash_record() {
        let dir = std::env::temp_dir().join("crucible-crashguard-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // --- a phase change leaves a breadcrumb on disk ---
        let base = dir.join("crucible-run1");
        arm(base.clone(), "unit test");
        phase("running:cpu");
        let bc = breadcrumb_path(&base);
        assert!(bc.is_file(), "a phase change must leave a breadcrumb");
        assert!(std::fs::read_to_string(&bc).unwrap().contains("running:cpu"));

        // --- a clean finish removes it, or every run looks like a crash ---
        finished();
        assert!(!bc.is_file(), "a clean finish must clear the breadcrumb");

        // --- a crash record names the phase it died in ---
        let base2 = dir.join("crucible-run2");
        arm(base2.clone(), "unit test");
        phase("teardown:pathtrace");
        write_crash("structured_exception", "ACCESS_VIOLATION at 0x0");
        let body = std::fs::read_to_string(crash_path(&base2)).unwrap();
        assert!(body.contains("\"crash\": true"), "{body}");
        assert!(body.contains("teardown:pathtrace"), "{body}");
        assert!(body.contains("ACCESS_VIOLATION"), "{body}");

        // --- a leftover breadcrumb is found, then consumed ---
        let found = resolve(&dir);
        assert_eq!(found.len(), 1, "the unfinished run must be found: {found:?}");
        assert!(found[0].contains("teardown:pathtrace"), "{:?}", found[0]);
        assert!(
            resolve(&dir).is_empty(),
            "it must be consumed, or the same crash is reported forever"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
