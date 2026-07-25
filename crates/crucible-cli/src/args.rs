// SPDX-License-Identifier: MIT
//! A minimal, dependency-free argument parser.
//!
//! Supports `--key value`, `--key=value`, and boolean `--flag` forms. Boolean
//! flag names are declared up front so `--flag` is never mistaken for a
//! value-taking option.

use std::collections::{BTreeMap, BTreeSet};

/// Parsed command-line arguments (excluding the leading subcommand).
#[derive(Debug, Default)]
pub struct Parsed {
    pub positional: Vec<String>,
    pub values: BTreeMap<String, String>,
    pub bools: BTreeSet<String>,
}

impl Parsed {
    pub fn parse(args: &[String], bool_flags: &[&str]) -> Result<Parsed, String> {
        let mut out = Parsed::default();
        let mut i = 0;
        while i < args.len() {
            let tok = &args[i];
            if let Some(name) = tok.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    out.values.insert(k.to_string(), v.to_string());
                } else if bool_flags.contains(&name) {
                    out.bools.insert(name.to_string());
                } else {
                    // Consume the next token as this option's value.
                    let val = args
                        .get(i + 1)
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    if val.starts_with("--") {
                        return Err(format!("--{name} needs a value"));
                    }
                    out.values.insert(name.to_string(), val.clone());
                    i += 1;
                }
            } else {
                out.positional.push(tok.clone());
            }
            i += 1;
        }
        Ok(out)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    pub fn has(&self, key: &str) -> bool {
        self.bools.contains(key)
    }

    pub fn get_u64(&self, key: &str) -> Result<Option<u64>, String> {
        match self.values.get(key) {
            None => Ok(None),
            Some(s) => s
                .parse::<u64>()
                .map(Some)
                .map_err(|_| format!("--{key} expects an integer, got '{s}'")),
        }
    }

    /// Reject any option key not in `allowed` (typo protection).
    pub fn reject_unknown(&self, allowed: &[&str]) -> Result<(), String> {
        for k in self.values.keys().chain(self.bools.iter()) {
            // A few flags are accepted by every command: `--ui` (live terminal UI)
            // and the CSV logging opt-ins.
            if matches!(k.as_str(), "ui" | "csv" | "telemetry-csv") {
                continue;
            }
            if !allowed.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_values_bools_and_positionals() {
        let args = s(&["run", "soak", "--seconds", "60", "--json", "--mb=512"]);
        let p = Parsed::parse(&args, &["json"]).unwrap();
        assert_eq!(p.positional, vec!["run", "soak"]);
        assert_eq!(p.get("seconds"), Some("60"));
        assert_eq!(p.get("mb"), Some("512"));
        assert!(p.has("json"));
    }

    #[test]
    fn missing_value_errors() {
        let args = s(&["--seconds"]);
        assert!(Parsed::parse(&args, &[]).is_err());
    }

    #[test]
    fn value_flag_followed_by_flag_errors() {
        let args = s(&["--seconds", "--json"]);
        assert!(Parsed::parse(&args, &["json"]).is_err());
    }

    #[test]
    fn u64_parsing_reports_bad_values() {
        let args = s(&["--seconds", "abc"]);
        let p = Parsed::parse(&args, &[]).unwrap();
        assert!(p.get_u64("seconds").is_err());
    }

    #[test]
    fn unknown_option_rejected() {
        let args = s(&["--bogus", "1"]);
        let p = Parsed::parse(&args, &[]).unwrap();
        assert!(p.reject_unknown(&["seconds"]).is_err());
    }
}
