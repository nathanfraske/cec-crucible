// SPDX-License-Identifier: MIT
//! CPU package power and die temperature via **LibreHardwareMonitor**.
//!
//! The second backend for the sensor vocabulary defined in [`crate::hwinfo`],
//! and the one that can actually be made to work from a downloaded package.
//!
//! ## Why this one and not the HWiNFO bridge
//!
//! Both read a daemon rather than shipping a kernel driver of our own (see
//! [`crate::hwinfo`] for why that line exists). The difference is whether the
//! daemon can be *set up without a human*:
//!
//! * HWiNFO's Shared Memory Support is a GUI checkbox. It is not in the INI, not
//!   in the registry, and does not survive being written there — measured, not
//!   assumed. A technician has to tick it on every machine.
//! * LibreHardwareMonitor keeps its settings in a plain XML config next to the
//!   executable, and exposes a local HTTP endpoint. Two keys and a launch, no
//!   interaction, which is what "works with the downloaded package" requires.
//!
//! LHM 0.9.5 also replaced **WinRing0** with **PawnIO**. That matters: WinRing0
//! is on Microsoft's vulnerable-driver blocklist and Defender now flags it as
//! `VulnerableDriver:WinNT/Winring0`, which broke a swathe of monitoring tools.
//! PawnIO is signed, open source, and runs verified bytecode in a sandbox rather
//! than handing user mode arbitrary ring-0 access — the objection that ruled out
//! bundling LHM a year ago no longer applies.
//!
//! PawnIO installs separately (`winget install -e --id namazso.PawnIO`); without
//! it LHM runs but reports no CPU power, which this module reports as absent
//! rather than as zero.
//!
//! ## The endpoint
//!
//! `/metrics`, Prometheus text, one sensor per line:
//!
//! ```text
//! lhm_cpu_power_watts {"sensorName"="CPU Package", …} 166.08
//! lhm_cpu_temperature_celsius {"sensorName"="Core Max", …} 91.25
//! ```
//!
//! Deliberately *not* `/data.json`: that is a nested tree with values as
//! unit-suffixed strings, and parsing it would mean carrying a JSON reader this
//! crate has never needed. A line format needs a `split`.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::hwinfo::{Reading, TYPE_FAN, TYPE_POWER, TYPE_TEMPERATURE};

/// Port LHM's web server listens on. Its own default, kept so a machine an
/// operator already configured by hand keeps working.
pub const DEFAULT_PORT: u16 = 8085;

/// Bound on a response. The full table is ~72 KB on this bench; a megabyte is
/// far past anything real and stops a wedged server from growing without limit.
const MAX_RESPONSE: usize = 1024 * 1024;

/// Short by design. This runs inside a 4 Hz sampling loop, so a daemon that has
/// stopped answering must fail fast rather than stall the telemetry thread.
const TIMEOUT: Duration = Duration::from_millis(1500);

/// Fetch and parse LHM's sensor table.
pub fn read(port: u16) -> Option<Vec<Reading>> {
    let body = http_get(port, "/metrics")?;
    let r = parse_metrics(&body);
    if r.is_empty() {
        None
    } else {
        Some(r)
    }
}

/// Is a LHM web server answering on this port?
pub fn available(port: u16) -> bool {
    read(port).is_some()
}

/// A minimal HTTP/1.1 GET against loopback.
///
/// Hand-rolled for the same reason as every other bit of I/O here: one request
/// to a known local server does not justify a dependency. Loopback only — this
/// must never be pointed at a remote host, and binding LHM to 127.0.0.1 is part
/// of the configuration we write.
fn http_get(port: u16, path: &str) -> Option<String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut s = TcpStream::connect_timeout(&addr, TIMEOUT).ok()?;
    s.set_read_timeout(Some(TIMEOUT)).ok()?;
    s.set_write_timeout(Some(TIMEOUT)).ok()?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         User-Agent: cec-crucible\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).ok()?;
    let _ = s.flush();

    let mut buf = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > MAX_RESPONSE {
                    break;
                }
            }
            Err(_) => break, // timeout or reset: use whatever arrived
        }
    }
    let _ = s.shutdown(Shutdown::Both);

    let text = String::from_utf8_lossy(&buf).into_owned();
    // Split headers from body. A response without the separator is not one.
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string())?;
    if !text.starts_with("HTTP/1.1 200") && !text.starts_with("HTTP/1.0 200") {
        return None;
    }
    Some(body)
}

/// Parse Prometheus exposition lines into readings.
///
/// The metric *name* carries the kind (`..._power_watts`,
/// `..._temperature_celsius`, `..._fan_rpm`), the label blob carries
/// `sensorName` and `hardwareName`, and the value is the last field. Anything
/// that does not fit that shape is skipped rather than guessed at.
pub fn parse_metrics(body: &str) -> Vec<Reading> {
    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((head, value)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = value.trim().parse::<f64>() else {
            continue;
        };
        let metric = head.split_whitespace().next().unwrap_or("");
        let kind = if metric.ends_with("_watts") {
            TYPE_POWER
        } else if metric.ends_with("_celsius") {
            TYPE_TEMPERATURE
        } else if metric.ends_with("_rpm") {
            TYPE_FAN
        } else {
            continue; // clocks, loads, voltages: not what this plane reports
        };
        out.push(Reading {
            kind,
            sensor: label(head, "hardwareName"),
            label: label(head, "sensorName"),
            unit: match kind {
                TYPE_POWER => "W".into(),
                TYPE_TEMPERATURE => "°C".into(),
                _ => "RPM".into(),
            },
            value,
        });
    }
    out
}

/// Pull `"key"="value"` out of the label blob.
fn label(head: &str, key: &str) -> String {
    let needle = format!("\"{key}\"=\"");
    let Some(start) = head.find(&needle) else {
        return String::new();
    };
    let rest = &head[start + needle.len()..];
    match rest.find('"') {
        Some(end) => rest[..end].to_string(),
        None => String::new(),
    }
}

/// Where LibreHardwareMonitor might be, newest-looking first.
///
/// Next to our own executable comes first so a bundled copy always wins over
/// whatever else the machine happens to have.
pub fn locate() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("LibreHardwareMonitor"));
            roots.push(dir.to_path_buf());
        }
    }
    for var in ["LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"] {
        if let Some(v) = std::env::var_os(var) {
            let p = PathBuf::from(v);
            roots.push(p.join("Programs").join("LibreHardwareMonitor"));
            roots.push(p.join("LibreHardwareMonitor"));
            // winget keeps packages under a versioned directory.
            roots.push(p.join("Microsoft").join("WinGet").join("Packages"));
        }
    }
    for r in roots {
        let direct = r.join("LibreHardwareMonitor.exe");
        if direct.is_file() {
            return Some(direct);
        }
        // One level of subdirectory, for the winget package layout.
        if let Ok(entries) = std::fs::read_dir(&r) {
            for e in entries.flatten() {
                let c = e.path().join("LibreHardwareMonitor.exe");
                if c.is_file() {
                    return Some(c);
                }
            }
        }
    }
    None
}

/// Turn on LHM's local web server by editing its config.
///
/// The keys are *menu item* names — `runWebServerMenuItem`, not `runWebServer` —
/// because LHM derives each setting's key from the UI control that owns it. A
/// key it does not recognise is silently dropped on the next save, which is how
/// the wrong name was found: the value simply vanished.
///
/// Bound to `127.0.0.1` explicitly. LHM's own default listens on all interfaces,
/// and a QC tool has no business opening a sensor feed to the network.
pub fn configure(exe: &Path, port: u16) -> std::io::Result<()> {
    let cfg = exe.with_file_name("LibreHardwareMonitor.config");
    let mut text = std::fs::read_to_string(&cfg).unwrap_or_else(|_| {
        String::from("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<configuration>\n  <appSettings>\n  </appSettings>\n</configuration>")
    });

    for (key, value) in [
        ("runWebServerMenuItem", "true".to_string()),
        ("listenerPort", port.to_string()),
        ("listenerIp", "127.0.0.1".to_string()),
        // Start out of the way: this is a background sensor source, not
        // something an operator should have to minimise on every bench machine.
        ("startMinMenuItem", "true".to_string()),
        ("minTrayMenuItem", "true".to_string()),
    ] {
        text = set_key(&text, key, &value);
    }
    std::fs::write(&cfg, text)
}

/// Replace or insert one `<add key="…" value="…" />` entry.
///
/// A targeted text edit rather than an XML parse: the file is machine-written
/// with one entry per line, and carrying an XML parser to change five values
/// would be the same trade this crate declines everywhere else.
fn set_key(text: &str, key: &str, value: &str) -> String {
    let entry = format!("    <add key=\"{key}\" value=\"{value}\" />");
    let needle = format!("<add key=\"{key}\" ");
    if let Some(pos) = text.find(&needle) {
        // Replace the whole existing line, whatever its current value.
        let start = text[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let end = text[pos..].find('\n').map(|i| pos + i).unwrap_or(text.len());
        return format!("{}{}{}", &text[..start], entry, &text[end..]);
    }
    // Otherwise insert just inside <appSettings>.
    match text.find("<appSettings>") {
        Some(p) => {
            let ins = p + "<appSettings>".len();
            format!("{}\n{}{}", &text[..ins], entry, &text[ins..])
        }
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hwinfo::select;

    /// Two real lines from this bench's `/metrics`, trimmed of nothing.
    const SAMPLE: &str = r#"# HELP lhm_cpu_power_watts power
lhm_cpu_power_watts {"sensorName"="CPU Package", "sensorAlias"="CPU Package (/power/0)", "hardwareName"="Intel Core i9-10850K", "hardwareId"="/intelcpu/0", "host"="DESKTOP-TMT37GF"} 166.080917358398
lhm_cpu_power_watts {"sensorName"="CPU Cores", "hardwareName"="Intel Core i9-10850K"} 162.974884033203
lhm_cpu_temperature_celsius {"sensorName"="CPU Package", "hardwareName"="Intel Core i9-10850K"} 88.5
lhm_cpu_temperature_celsius {"sensorName"="Core Max", "hardwareName"="Intel Core i9-10850K"} 91.25
lhm_cpu_temperature_celsius {"sensorName"="CPU Core #3", "hardwareName"="Intel Core i9-10850K"} 87
lhm_motherboard_temperature_celsius {"sensorName"="Temperature 2", "hardwareName"="ASUS ROG STRIX Z490-I GAMING"} 47
lhm_memory_temperature_celsius {"sensorName"="DIMM #1", "hardwareName"="Generic Memory"} 43.5
lhm_motherboard_fan_rpm {"sensorName"="Fan #1", "hardwareName"="Nuvoton NCT6798D"} 1123
lhm_cpu_clock_megahertz {"sensorName"="CPU Core #1", "hardwareName"="Intel Core i9-10850K"} 4800
"#;

    #[test]
    fn the_real_metrics_format_parses() {
        let r = parse_metrics(SAMPLE);
        // Clocks are skipped: this plane reports power, temperature and fans.
        assert_eq!(r.len(), 8, "got {r:#?}");
        let pkg = r
            .iter()
            .find(|x| x.kind == TYPE_POWER && x.label == "CPU Package")
            .expect("package power");
        assert!((pkg.value - 166.08).abs() < 0.01);
        assert_eq!(pkg.sensor, "Intel Core i9-10850K");
        assert_eq!(pkg.unit, "W");
    }

    #[test]
    fn the_selector_picks_the_right_readings_out_of_a_live_table() {
        // Same selection logic as the HWiNFO backend, exercised against LHM's
        // naming — the two daemons label things differently and both have to
        // land on the same answer.
        let s = select(&parse_metrics(SAMPLE));
        assert_eq!(s.package_power_w, Some(166.080917358398), "CPU Cores must not win");
        assert_eq!(s.package_temp_c, Some(88.5), "package, not Core Max");
        assert_eq!(s.core_max_c, Some(87.0), "hottest numbered core");
        assert_eq!(s.dimm_c, vec![43.5], "DIMM temperature, which is the point");
        assert_eq!(s.fan_rpm, vec![1123.0]);
    }

    #[test]
    fn a_comment_or_a_malformed_line_is_skipped_not_guessed() {
        let junk = "# HELP something\n\nnot a metric at all\nlhm_cpu_power_watts {} notanumber\n";
        assert!(parse_metrics(junk).is_empty());
    }

    #[test]
    fn a_missing_label_yields_an_empty_name_rather_than_a_panic() {
        let line = "lhm_cpu_power_watts {} 12.5\n";
        let r = parse_metrics(line);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].label, "");
        assert_eq!(r[0].value, 12.5);
    }

    #[test]
    fn config_keys_are_replaced_in_place_and_inserted_when_absent() {
        let before = "<?xml version=\"1.0\"?>\n<configuration>\n  <appSettings>\n    <add key=\"listenerPort\" value=\"9999\" />\n    <add key=\"other\" value=\"keepme\" />\n  </appSettings>\n</configuration>";
        let after = set_key(before, "listenerPort", "8085");
        assert!(after.contains("<add key=\"listenerPort\" value=\"8085\" />"));
        assert!(!after.contains("9999"), "the old value must be gone");
        assert!(after.contains("keepme"), "other settings must survive");

        let added = set_key(&after, "runWebServerMenuItem", "true");
        assert!(added.contains("<add key=\"runWebServerMenuItem\" value=\"true\" />"));
        assert!(added.contains("keepme"));
    }

    #[test]
    fn the_key_name_is_the_menu_item_one() {
        // Measured the hard way: `runWebServer` is silently dropped by LHM on
        // its next save, and the server never starts. Guard the name so nobody
        // "tidies" it back.
        let cfg = set_key("<appSettings>\n</appSettings>", "runWebServerMenuItem", "true");
        assert!(cfg.contains("runWebServerMenuItem"));
        assert!(!cfg.contains("\"runWebServer\" "));
    }

    #[test]
    fn a_dead_port_is_none_rather_than_a_stall() {
        // Nothing listens here; this must fail fast, because it runs inside the
        // sampling loop.
        let t0 = std::time::Instant::now();
        assert!(read(1).is_none());
        assert!(t0.elapsed() < Duration::from_secs(5), "took {:?}", t0.elapsed());
    }
}

/// Start LibreHardwareMonitor if it is not already running.
///
/// Returns whether a launch was attempted. It needs Administrator to talk to
/// PawnIO, so an unelevated launch produces a process that runs and reports no
/// CPU sensors — which the caller sees as an absent reading rather than a zero.
pub fn launch(exe: &Path) -> bool {
    use std::process::{Command, Stdio};
    Command::new(exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}
