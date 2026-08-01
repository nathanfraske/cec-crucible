// SPDX-License-Identifier: MIT
//! The per-machine package — every run for one box, on one page.
//!
//! A run leaves six or seven files behind: a JSON report, a marker log, two
//! CSVs, a chart page, PNGs, an event archive, sometimes an ETL and a crash
//! record. After a morning on the bench that is a directory of a hundred files
//! whose names are timestamps. The evidence exists and nobody can find it.
//!
//! This gathers everything belonging to one **device short-id** — the stable
//! machine identity already stamped into every filename — into a single index
//! page: a verdict table newest-first, the charts inline, and a link to every
//! artifact. Then it opens in the operator's browser.
//!
//! Deliberately built by **scanning the directory, not by remembering what this
//! process wrote.** The runs that matter most are the ones that crashed, and a
//! crashed run never got to record itself anywhere. If the file is on disk it
//! appears on the page.
//!
//! The charts are **embedded as `data:` URIs rather than linked**. A linked
//! image only renders while the page sits next to its PNGs — email the HTML on
//! its own, or open it through a viewer, and every chart is a broken-image icon.
//! Since the whole point is handing this to somebody, the page has to survive
//! being moved. Chart PNGs compress to tens of KB (see [`crate::deflate`]), so a
//! self-contained page for a day of runs stays a few MB.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

/// Everything found on disk for one run, keyed by its timestamp stem.
#[derive(Debug, Default, Clone)]
pub struct RunFiles {
    /// `crucible-<device>-<unix>-<pid>` — the shared prefix of every artifact.
    pub stem: String,
    pub report: Option<PathBuf>,
    pub markers: Option<PathBuf>,
    pub results_csv: Option<PathBuf>,
    pub telemetry_csv: Option<PathBuf>,
    pub charts_html: Option<PathBuf>,
    pub eventlog: Option<PathBuf>,
    pub etl: Option<PathBuf>,
    pub crash: Option<PathBuf>,
    pub pngs: Vec<PathBuf>,
    /// Unix seconds parsed out of the stem, for ordering.
    pub when: u64,
}

/// Parse `crucible-<device>-<unix>-<pid>` out of a file name, returning the
/// stem, the device id and the timestamp.
fn split_stem(name: &str) -> Option<(String, String, u64)> {
    let rest = name.strip_prefix("crucible-")?;
    let mut it = rest.split('-');
    let device = it.next()?.to_string();
    let unix: u64 = it.next()?.parse().ok()?;
    // The pid segment ends at the first '.', where the artifact suffix begins.
    let pid = it.next()?.split('.').next()?.to_string();
    Some((format!("crucible-{device}-{unix}-{pid}"), device, unix))
}

/// Collect every run in `dir`, grouped by device short-id.
pub fn scan(dir: &Path) -> io::Result<BTreeMap<String, Vec<RunFiles>>> {
    let mut by_device: BTreeMap<String, BTreeMap<String, RunFiles>> = BTreeMap::new();

    for entry in std::fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((stem, device, when)) = split_stem(name) else {
            continue;
        };

        let run = by_device
            .entry(device)
            .or_default()
            .entry(stem.clone())
            .or_insert_with(|| RunFiles {
                stem: stem.clone(),
                when,
                ..Default::default()
            });

        // Longest suffixes first: `.report.csv` must not be claimed by a naive
        // `.csv` test, and `.telemetry.csv` must not be claimed by `.report.csv`.
        if name.ends_with(".report.json") {
            run.report = Some(path);
        } else if name.ends_with(".markers.jsonl") {
            run.markers = Some(path);
        } else if name.ends_with(".telemetry.csv") {
            run.telemetry_csv = Some(path);
        } else if name.ends_with(".report.csv") {
            run.results_csv = Some(path);
        } else if name.ends_with(".telemetry.html") {
            run.charts_html = Some(path);
        } else if name.ends_with(".eventlog.jsonl") {
            run.eventlog = Some(path);
        } else if name.ends_with(".crash.json") {
            run.crash = Some(path);
        } else if name.ends_with(".etl") {
            run.etl = Some(path);
        } else if name.ends_with(".png") {
            run.pngs.push(path);
        }
    }

    Ok(by_device
        .into_iter()
        .map(|(device, runs)| {
            let mut v: Vec<RunFiles> = runs.into_values().collect();
            // Newest first: the run you just did is the one you want to see.
            v.sort_by(|a, b| b.when.cmp(&a.when).then(b.stem.cmp(&a.stem)));
            for r in &mut v {
                r.pngs.sort();
            }
            (device, v)
        })
        .collect())
}

/// Pull the headline fields out of a report JSON without a JSON parser.
///
/// This crate writes JSON and has never needed to read it. A handful of scalar
/// lookups by key is a fair trade against carrying a parser for a summary line
/// — and anything it cannot find simply does not appear, rather than failing the
/// page. The authoritative copy is always the linked report itself.
fn peek(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else {
        let end = rest.find([',', '\n', '}'])?;
        Some(rest[..end].trim().to_string())
    }
}

/// Read a PNG and return it as a `data:` URI, or `None` if it cannot be read.
fn embed_png(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("data:image/png;base64,{}", base64(&bytes)))
}

/// Standard base64. Small enough to write out rather than take a dependency for.
fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        // Pad to a multiple of four: a truncated group must be marked, not
        // silently short, or decoders disagree about the trailing bytes.
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Link to a file relative to the page, which sits in the same directory.
fn href(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Build the package page for one device and write it into `dir`.
///
/// Returns the page path. Self-contained apart from the artifacts it links,
/// which are its neighbours — so the whole directory zips up and still works on
/// a machine that has never seen this tool.
pub fn render(dir: &Path, device: &str, runs: &[RunFiles]) -> io::Result<PathBuf> {
    let mut h = String::new();
    h.push_str(&format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>cec-crucible · {}</title><style>{}</style></head><body>",
        esc(device),
        STYLE
    ));

    h.push_str(&format!(
        "<header><h1>cec-crucible</h1><p class=\"sub\">machine <code>{}</code> · {} run(s) on this page</p></header>",
        esc(device),
        runs.len()
    ));

    // Summary table first: the question is almost always "did anything fail".
    h.push_str("<table><thead><tr><th>run</th><th>verdict</th><th>errors</th><th>duration</th><th>GPU peak</th><th>artifacts</th></tr></thead><tbody>");
    for r in runs {
        let report_text = r
            .report
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();
        let verdict = peek(&report_text, "verdict").unwrap_or_else(|| "—".into());
        let errors = peek(&report_text, "error_count").unwrap_or_else(|| "—".into());
        let secs = peek(&report_text, "duration_seconds").unwrap_or_else(|| "—".into());
        let power = peek(&report_text, "power_peak_w");
        let temp = peek(&report_text, "temp_peak_c");
        let gpu = match (power, temp) {
            (Some(p), Some(t)) => format!("{p} W / {t} °C"),
            _ => "—".into(),
        };
        // A crash file present means the run died; that outranks whatever
        // verdict a partial report happens to carry.
        let cls = if r.crash.is_some() {
            "crash"
        } else if verdict == "FAIL" {
            "fail"
        } else if verdict == "PASS" {
            "pass"
        } else {
            "part"
        };
        let shown = if r.crash.is_some() {
            "CRASHED".to_string()
        } else {
            verdict
        };

        let mut links = Vec::new();
        let mut add = |label: &str, p: &Option<PathBuf>| {
            if let Some(p) = p {
                links.push(format!("<a href=\"{}\">{label}</a>", esc(&href(p))));
            }
        };
        add("report", &r.report);
        add("charts", &r.charts_html);
        add("telemetry", &r.telemetry_csv);
        add("results", &r.results_csv);
        add("markers", &r.markers);
        add("events", &r.eventlog);
        add("etl", &r.etl);
        add("crash", &r.crash);

        h.push_str(&format!(
            "<tr><td><code>{}</code></td><td class=\"{cls}\">{}</td><td>{}</td><td>{} s</td><td>{}</td><td class=\"links\">{}</td></tr>",
            esc(&r.stem),
            esc(&shown),
            esc(&errors),
            esc(&secs),
            esc(&gpu),
            links.join(" ")
        ));
    }
    h.push_str("</tbody></table>");

    // Charts inline, newest run first — the point of the page is not having to
    // open anything to see what happened.
    for r in runs {
        if r.pngs.is_empty() {
            continue;
        }
        h.push_str(&format!("<section><h2>{}</h2><div class=\"grid\">", esc(&r.stem)));
        for p in &r.pngs {
            match embed_png(p) {
                Some(uri) => h.push_str(&format!(
                    "<figure><img src=\"{uri}\" alt=\"chart\" loading=\"lazy\"></figure>"
                )),
                // Unreadable: link it rather than dropping it silently, so the
                // page still points at something that exists on disk.
                None => h.push_str(&format!(
                    "<figure><img src=\"{}\" alt=\"chart\" loading=\"lazy\"></figure>",
                    esc(&href(p))
                )),
            }
        }
        h.push_str("</div></section>");
    }

    h.push_str("<footer>Built by Critical Error Computing · every artifact linked here is a plain file in this folder.</footer></body></html>");

    // The device id reaches this function from SMBIOS, and it lands in two
    // places: the page body (escaped above) and this filename. In practice it is
    // a hex short-id, but "in practice" is not a guarantee — a string with a
    // path separator or a reserved character in it would either escape the
    // output directory or fail the write. Reduce it to characters that are safe
    // in a filename on every platform.
    let safe: String = device
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let out = dir.join(format!("crucible-{safe}.index.html"));
    std::fs::write(&out, h)?;
    Ok(out)
}

const STYLE: &str = "\
:root{color-scheme:dark}\
body{margin:0;background:#070711;color:#f5f5f8;font:15px/1.55 system-ui,Segoe UI,sans-serif}\
header{padding:28px 32px 16px;border-bottom:1px solid #2a2a44}\
h1{margin:0;font-size:22px;letter-spacing:.02em;color:#ed2398}\
.sub{margin:6px 0 0;color:#a9a9b7}\
code{font:13px ui-monospace,Consolas,monospace;color:#9f9cff}\
table{width:calc(100% - 64px);margin:24px 32px;border-collapse:collapse;font-size:14px}\
th{text-align:left;padding:8px 10px;color:#9f9cff;border-bottom:1px solid #2a2a44;font-weight:600}\
td{padding:8px 10px;border-bottom:1px solid #191930;vertical-align:top}\
.pass{color:#41d9f8;font-weight:700}.fail{color:#ee0b2a;font-weight:700}\
.crash{color:#ee0b2a;font-weight:700}.part{color:#f2a618;font-weight:700}\
.links a{color:#8c67f2;margin-right:10px;text-decoration:none;white-space:nowrap}\
.links a:hover{text-decoration:underline}\
section{margin:32px}\
h2{font:13px ui-monospace,Consolas,monospace;color:#a9a9b7;font-weight:500;margin:0 0 12px}\
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}\
figure{margin:0;background:#161526;border:1px solid #2a2a44;border-radius:8px;overflow:hidden}\
img{display:block;width:100%;height:auto}\
footer{padding:24px 32px 40px;color:#6f6f86;font-size:13px;border-top:1px solid #2a2a44;margin-top:24px}\
@media(max-width:600px){table{width:calc(100% - 24px);margin:16px 12px}section{margin:16px 12px}}\
";

/// Open a path in the operator's default browser.
///
/// `cmd /c start` rather than `ShellExecute`: it is one process spawn with no
/// FFI, and it is the same mechanism a shortcut uses. The empty first argument
/// is `start`'s window-title parameter — omit it and a path containing spaces is
/// taken as the title and nothing opens.
pub fn open_in_browser(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        std::process::Command::new("cmd")
            .args(["/c", "start", "", &path.display().to_string()])
            .spawn()
            .map(|_| ())
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_stem_is_recovered_from_every_artifact_suffix() {
        for name in [
            "crucible-50e8caec-1785278039-21696.report.json",
            "crucible-50e8caec-1785278039-21696.telemetry.csv",
            "crucible-50e8caec-1785278039-21696.eventlog.jsonl",
            "crucible-50e8caec-1785278039-21696.gpu-power.png",
        ] {
            let (stem, device, when) = split_stem(name).expect(name);
            assert_eq!(stem, "crucible-50e8caec-1785278039-21696");
            assert_eq!(device, "50e8caec");
            assert_eq!(when, 1785278039);
        }
        // Anything else in the directory is ignored rather than mis-grouped.
        assert!(split_stem("notes.txt").is_none());
        assert!(split_stem("crucible-only-two.json").is_none());
    }

    #[test]
    fn artifacts_are_matched_by_their_longest_suffix() {
        // `.report.csv` and `.telemetry.csv` both end in `.csv`, and a naive
        // check would file the telemetry log as the results CSV.
        let dir = std::env::temp_dir().join("crucible-pkg-test-suffix");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "crucible-abc123-1700000000-1.report.csv",
            "crucible-abc123-1700000000-1.telemetry.csv",
            "crucible-abc123-1700000000-1.report.json",
        ] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        let found = scan(&dir).unwrap();
        let runs = &found["abc123"];
        assert_eq!(runs.len(), 1);
        assert!(runs[0].results_csv.is_some(), "results CSV missing");
        assert!(runs[0].telemetry_csv.is_some(), "telemetry CSV missing");
        assert_ne!(runs[0].results_csv, runs[0].telemetry_csv);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runs_group_by_machine_and_sort_newest_first() {
        let dir = std::env::temp_dir().join("crucible-pkg-test-group");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in [
            "crucible-aaa-1700000000-1.report.json",
            "crucible-aaa-1700000900-2.report.json",
            "crucible-bbb-1700000500-3.report.json",
        ] {
            std::fs::write(dir.join(f), "{}").unwrap();
        }
        let found = scan(&dir).unwrap();
        assert_eq!(found.len(), 2, "two machines");
        let a = &found["aaa"];
        assert_eq!(a.len(), 2);
        assert!(a[0].when > a[1].when, "newest run must come first");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_crashed_run_outranks_whatever_verdict_it_managed_to_write() {
        let dir = std::env::temp_dir().join("crucible-pkg-test-crash");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("crucible-ccc-1700000000-1.report.json"),
            "{\"verdict\":\"PASS\",\"error_count\":0,\"duration_seconds\":3.0}",
        )
        .unwrap();
        std::fs::write(dir.join("crucible-ccc-1700000000-1.crash.json"), "{}").unwrap();

        let found = scan(&dir).unwrap();
        let page = render(&dir, "ccc", &found["ccc"]).unwrap();
        let html = std::fs::read_to_string(&page).unwrap();
        assert!(
            html.contains("CRASHED"),
            "a run with a crash record must not be presented as a PASS"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base64_matches_the_standard_including_padding() {
        // The RFC 4648 test vectors. A padding mistake here shows up as charts
        // that fail to decode in some browsers and not others.
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn charts_are_embedded_so_the_page_survives_being_moved() {
        let dir = std::env::temp_dir().join("crucible-pkg-test-embed");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("crucible-ddd-1700000000-1.report.json"), "{}").unwrap();
        std::fs::write(dir.join("crucible-ddd-1700000000-1.gpu-power.png"), b"\x89PNG-fake").unwrap();

        let found = scan(&dir).unwrap();
        let page = render(&dir, "ddd", &found["ddd"]).unwrap();
        let html = std::fs::read_to_string(&page).unwrap();
        assert!(
            html.contains("src=\"data:image/png;base64,"),
            "the chart must be inline, not a link that breaks when the page moves"
        );
        assert!(
            !html.contains("src=\"crucible-ddd-1700000000-1.gpu-power.png\""),
            "an embedded chart must not also be linked"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_fields_are_read_without_a_json_parser() {
        let j = "{\"verdict\":\"FAIL\",\"error_count\":3,\"duration_seconds\":12.50,\"gpu\":{\"power_peak_w\":221.5}}";
        assert_eq!(peek(j, "verdict").as_deref(), Some("FAIL"));
        assert_eq!(peek(j, "error_count").as_deref(), Some("3"));
        assert_eq!(peek(j, "duration_seconds").as_deref(), Some("12.50"));
        assert_eq!(peek(j, "power_peak_w").as_deref(), Some("221.5"));
        assert_eq!(peek(j, "not_present"), None);
    }

    #[test]
    fn a_device_name_cannot_inject_markup() {
        // Device ids come from SMBIOS, which is not a trusted source of HTML.
        let dir = std::env::temp_dir().join("crucible-pkg-test-esc");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let page = render(&dir, "<script>alert(1)</script>", &[]).unwrap();
        let html = std::fs::read_to_string(&page).unwrap();
        assert!(!html.contains("<script>alert"), "device id was not escaped");
        assert!(html.contains("&lt;script&gt;"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
