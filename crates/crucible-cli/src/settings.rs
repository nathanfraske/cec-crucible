// SPDX-License-Identifier: MIT
//! Settings that survive a restart.
//!
//! The Settings screen used to reset on every launch, so a technician who
//! wanted telemetry, PresentMon and an ETW trace had to re-arm all three before
//! every run — and the one they forgot is the one they needed. These now
//! persist to a small file next to the rest of the user's application data.
//!
//! **Format is deliberately `key = value` lines, not JSON.** This crate has a
//! JSON *writer* and no reader, and a settings file is exactly the thing an
//! operator ends up opening in Notepad on a bench machine when something looks
//! wrong. A malformed line is skipped rather than failing the load: a settings
//! file should never be able to stop the tool from starting.
//!
//! **Precedence: an explicit command-line flag always wins.** The stored value
//! is a *default*, so `--no-presentmon`-style intent on one run never gets
//! silently overwritten by what the menu happened to be set to last week.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// ETW profile sets offered on the Settings ring, cheapest first. Index 0 is
/// off, which is why an untouched install captures nothing: a trace needs
/// elevation and writes hundreds of megabytes, so it is never something an
/// operator gets by accident.
///
/// Lives here rather than in the menu so a stored index still resolves in a
/// build without the TUI, and on the CLI path where there is no menu at all.
pub const ETW_RINGS: &[(&str, &str)] = &[
    ("off", ""),
    ("triage", "GeneralProfile"),
    ("cpu+gpu", "CPU,GPU"),
    ("power+thermal", "Power,Thermal"),
    ("everything (big)", "GeneralProfile,CPU,GPU,Power,Thermal,DiskIO"),
];

/// The profiles a stored ring index means, empty when off.
pub fn etw_profiles(ring: usize) -> Vec<String> {
    match ETW_RINGS.get(ring) {
        Some((_, profiles)) if !profiles.is_empty() => {
            profiles.split(',').map(str::to_string).collect()
        }
        _ => Vec::new(),
    }
}

/// Persisted run settings. Field-for-field the Settings screen, plus anything
/// else worth remembering between sessions.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// `--csv`: per-stage results CSV.
    pub results_csv: bool,
    /// `--telemetry-csv`: the time-series log (and, with it, the chart page).
    pub telemetry_csv: bool,
    /// Index into the menu's output-directory presets.
    pub out: usize,
    /// `--presentmon`.
    pub presentmon: bool,
    /// 0 = above normal (default), 1 = high, 2 = normal/off.
    pub priority: usize,
    /// Index into the menu's ETW profile ring; 0 = off.
    pub etw: usize,
    /// Render the chart page alongside the telemetry CSV.
    pub graph: bool,
    /// Open the machine's package page in a browser when a run finishes.
    pub open_report: bool,
}

impl Default for Settings {
    fn default() -> Self {
        // All-off, matching a hand-typed command with no flags: an untouched
        // install must behave exactly like the documented defaults.
        Settings {
            results_csv: false,
            telemetry_csv: false,
            out: 0,
            presentmon: false,
            priority: 0,
            etw: 0,
            graph: true,
            open_report: false,
        }
    }
}

/// Where the settings file lives: `%APPDATA%\cec-crucible\settings.conf`.
///
/// Roaming rather than local so a domain profile carries a technician's
/// preferences between the machines they work on.
pub fn path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("cec-crucible").join("settings.conf"))
}

impl Settings {
    /// Load, falling back to defaults on anything unreadable.
    ///
    /// Never returns an error: a corrupt or half-written settings file must not
    /// be able to stop a QC run from starting.
    pub fn load() -> Settings {
        let Some(p) = path() else {
            return Settings::default();
        };
        let Ok(text) = std::fs::read_to_string(&p) else {
            return Settings::default();
        };
        Settings::parse(&text)
    }

    pub fn parse(text: &str) -> Settings {
        let mut kv: BTreeMap<&str, &str> = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                kv.insert(k.trim(), v.trim());
            }
        }
        let d = Settings::default();
        let b = |key: &str, dflt: bool| -> bool {
            match kv.get(key).map(|v| v.to_ascii_lowercase()) {
                Some(v) => matches!(v.as_str(), "1" | "true" | "yes" | "on"),
                None => dflt,
            }
        };
        let n = |key: &str, dflt: usize| -> usize {
            kv.get(key).and_then(|v| v.parse().ok()).unwrap_or(dflt)
        };
        Settings {
            results_csv: b("results_csv", d.results_csv),
            telemetry_csv: b("telemetry_csv", d.telemetry_csv),
            out: n("out", d.out),
            presentmon: b("presentmon", d.presentmon),
            priority: n("priority", d.priority),
            etw: n("etw", d.etw),
            graph: b("graph", d.graph),
            open_report: b("open_report", d.open_report),
        }
    }

    pub fn to_text(&self) -> String {
        format!(
            "# cec-crucible settings — written by the Settings screen.\n\
             # Safe to edit by hand; unknown or malformed lines are ignored.\n\
             results_csv = {}\n\
             telemetry_csv = {}\n\
             out = {}\n\
             presentmon = {}\n\
             priority = {}\n\
             etw = {}\n\
             graph = {}\n\
             open_report = {}\n",
            self.results_csv,
            self.telemetry_csv,
            self.out,
            self.presentmon,
            self.priority,
            self.etw,
            self.graph,
            self.open_report
        )
    }

    /// Write the settings back. Best-effort: a read-only profile or a full disk
    /// must not fail a run that is otherwise fine.
    pub fn save(&self) -> std::io::Result<()> {
        let p = path().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no APPDATA or HOME to store settings in",
            )
        })?;
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&p, self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_trip_preserves_every_field() {
        let s = Settings {
            results_csv: true,
            telemetry_csv: true,
            out: 2,
            presentmon: true,
            priority: 1,
            etw: 3,
            graph: false,
            open_report: true,
        };
        assert_eq!(Settings::parse(&s.to_text()), s);
    }

    #[test]
    fn an_unreadable_file_yields_defaults_rather_than_failing() {
        // A settings file must never be able to stop the tool from starting, so
        // garbage in means defaults out — not an error, and not a panic.
        let garbage = "\u{0}\u{1}not=even=close\n[section]\n= =\nresults_csv\n";
        assert_eq!(Settings::parse(garbage), Settings::default());
    }

    #[test]
    fn unknown_keys_are_ignored_and_known_ones_still_apply() {
        // Forward compatibility: an older build reading a newer file must keep
        // working rather than reset everything the operator configured.
        let s = Settings::parse("presentmon = on\nfuture_option = 7\npriority = 2\n");
        assert!(s.presentmon);
        assert_eq!(s.priority, 2);
        assert_eq!(s.out, Settings::default().out);
    }

    #[test]
    fn comments_and_whitespace_are_tolerated() {
        let s = Settings::parse("  # a comment\n\n  telemetry_csv   =   TRUE  \n");
        assert!(s.telemetry_csv);
    }

    #[test]
    fn the_etw_ring_is_off_at_zero_and_names_profiles_beyond_it() {
        // Index 0 must be genuinely empty: a stored 0 that produced a profile
        // would arm a several-hundred-megabyte trace on every run of an
        // install nobody configured.
        assert!(etw_profiles(0).is_empty());
        assert_eq!(etw_profiles(1), vec!["GeneralProfile"]);
        assert_eq!(etw_profiles(2), vec!["CPU", "GPU"]);
        // An index past the end (an older build reading a newer file) must be
        // off rather than a panic or an arbitrary profile.
        assert!(etw_profiles(999).is_empty());
    }

    #[test]
    fn graph_defaults_on_and_can_be_turned_off() {
        // graph is the one option whose default is true, so it is the one a
        // naive "absent means false" parser would silently break.
        assert!(Settings::default().graph);
        assert!(Settings::parse("").graph);
        assert!(!Settings::parse("graph = false\n").graph);
    }
}
