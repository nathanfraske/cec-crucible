# cec-crucible v0.0.6 — Alpha 6 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

## Read this first if you used Alpha 5

**Every event-log archive Alpha 5 wrote is about 99% noise.** A 60-second run's
archive spanned fourteen hours. The events are real, but most of them predate the
run they are filed under, so do not attribute them to it. Re-capture on 0.0.6.

The same broken time filter sat under the WHEA/TDR scan that decides verdicts, so
a machine-check from hours earlier could in principle have failed a clean run. No
capture we have shows that happening, but the door was open. Details below.

---

## Two things to download

| | |
|---|---|
| `cec-crucible-0.0.6-win-x64.zip` | **Installer.** Double-click `INSTALL.cmd`. Per-user, no admin. Puts itself on PATH, makes shortcuts, offers the CPU-sensor driver, and uninstalls all of it cleanly. |
| `cec-crucible-0.0.6-win-x64-portable.zip` | **Portable.** Extract and run. Adds nothing to PATH, creates no shortcuts, installs no driver. For a customer site, or a machine that must be left exactly as it was found. |

Same payload either way. The portable archive deliberately contains no installer,
so it cannot half-install itself by accident.

---

## Everything is now bundled

`LibreHardwareMonitor` and `PresentMon` ship inside both archives, with their
licence texts in `licenses/`. Nothing to install, nothing to configure —
cec-crucible writes LibreHardwareMonitor's config and starts it itself.

**One piece is downloaded rather than shipped: PawnIO.** CPU package power and
die temperature live in model-specific registers only a kernel module can read.
PawnIO's driver *source* is GPL-2.0-or-later, but the **signed** setup — the only
one that will actually load — lives in a separate repository with no licence file
at all, and winget classifies it as proprietary freeware. Unstated redistribution
terms are not permissive ones, so the installer fetches it from the official
release and verifies a pinned SHA-256 instead. From the bench it is one prompt;
nothing is manual.

`licenses/THIRD-PARTY-BOUNDARY.md` documents the boundary in full: we never link
to any of these. LibreHardwareMonitor is spoken to over HTTP on loopback,
PresentMon by reading the CSV it writes, and PawnIO only ever by
LibreHardwareMonitor via device IOCTL — which PawnIO's licence explicitly
excepts. That makes all of them independent programs aggregated with ours, not
derived works.

### Uninstall actually uninstalls

```
powershell -ExecutionPolicy Bypass -File Install-Crucible.ps1 -Uninstall
```

Removes the install directory, the PATH entry, both shortcuts, the settings in
`%APPDATA%`, and any ETW session a killed run left recording in the kernel. It
stops the bundled sensor daemon first — and leaves alone a copy you installed
yourself. It then *asks* about PawnIO rather than assuming, because FanControl
and a separately-installed LibreHardwareMonitor use it too. `-KeepPawnIO` skips
the question.

Verified by round trip on the bench: install → confirm every artifact present →
uninstall → confirm nothing remains.

---

## Fixed: event archives full of events from other days

A field capture of seven runs made it obvious. Seven channels contributed
*exactly* 512 events each — the per-channel cap — meaning they had returned their
entire contents. One channel supplied 512 events spanning six seconds, twelve
hours away from the run.

Cause: `timediff(@SystemTime) <= ms` is accepted by the event-log query engine
and then **silently ignored** on many channels. The query is now built with an
absolute timestamp, `TimeCreated[@SystemTime>='<ISO-8601>']`, which is honoured
everywhere — and records outside the window are rejected in code afterwards, so a
channel that ignores the filter can no longer poison the archive. The number
rejected is reported rather than hidden.

Archives now span exactly the run window. On that machine they go from ~3 MB of
mostly-history to tens of KB of what actually happened.

## Fixed: 47 GHz CPU clocks

The same capture logged seven impossible effective clocks, from 6.7 GHz to
**47,663 MHz**. Every one at `t=0.001s` — the first row of the file.
`% Processor Performance` is a rate counter measured between collects, and the
first sample landed microseconds after the priming collect, so the divisor was
almost zero.

The first sample is now discarded and re-primes instead, with an 8 GHz
plausibility ceiling behind it. One bad sample did disproportionate damage: it
rescaled every chart it appeared on until the real data was a flat line along the
bottom.

## Fixed: a memory test that refused to run

The working-set reserve added in Alpha 5 could cap an explicit `--mb` request to
*zero* on a machine already inside its reserve — refusing to test at all on
exactly the machine most worth testing. The cap now has a floor.

---

## Also in this release

Everything from the Alpha 5 line that has not shipped before:

* **CPU package power, die temperature, hottest core, VRM** — and **DIMM
  temperatures** on boards that expose them. Measured on the bench: 178.5 W peak,
  96 °C peak during a CPU run.
* **NVMe drive health** — composite temperature, endurance used, spare, power-on
  hours, unsafe shutdowns, media errors, decoded critical warnings. Works
  unelevated.
* **ACPI board thermal zones**, reported as what they are — a chassis sensor, not
  the CPU die.
* **`sensors` command** — what each plane can and cannot report on this machine,
  and what to install for the ones it cannot.
* **PresentMon no longer stalls a run.** It waited for frames from tests that
  never present; a `worst-case --presentmon` went from 46.2 s to 26.1 s against a
  25.5 s floor. It also leaked an ETW session per run, which is fixed.

---

## Known gaps

* **AMD sensors are untested.** The load side runs (a field capture on a B550 /
  Ryzen / RTX 4070 Ti SUPER came back clean), but no AMD machine has been
  available to validate the CPU sensor path.
* **The elevated ETW path is still unverified** — the non-elevated refusal is
  tested; an actual `.etl` has not been produced here.
* **The ReBAR PCIe fix is unproven end-to-end** — the ceiling check is tested
  against the exact 370 GB/s figure from the field, but no ReBAR machine
  exhibiting the fault has been available.
* **iGPU coverage is Intel-only** — validated against a UHD 630; AMD APUs
  untested, and the CPU↔iGPU contention test APUs need does not exist yet.
* Five defects in `docs/game-realism.md` §1 remain unfixed.
* No network test; no display/scanout, cable or EDID validation.
* Not code-signed, so SmartScreen will warn on first run.

Full roadmap and design docs are in `docs/`.

---

MIT licensed. Built by Critical Error Computing.
