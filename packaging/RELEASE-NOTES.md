# cec-crucible v0.0.4 — Alpha 4 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

## Install

1. Download `cec-crucible-0.0.4-win-x64.zip` and extract it.
2. Double-click **`INSTALL.cmd`**.

Per-user, **no admin required**. Installs to `%LOCALAPPDATA%\Programs\cec-crucible`,
adds it to your user PATH, and creates Start Menu + Desktop shortcuts.

**Launch the TUI:** double-click the **CEC Crucible** shortcut, or open a *new*
terminal and run `cec-crucible` with no arguments.

The main executable is self-contained (statically linked CRT — no Visual C++
Redistributable needed, so it runs on a freshly imaged machine). PresentMon is
bundled so `--presentmon` works with no setup.

---

## What's new in Alpha 4

### Power and thermal telemetry — on every run, no flags

The suite has always been able to tell you *that* a card underperformed. It
could never tell you **why**. Now it can: GPU power draw, core temperature,
memory-junction temperature, fan speed, SM/memory clocks and NVIDIA's own
throttle-reason bits are sampled at 4 Hz for the whole run and land in the
report, the results CSV, the telemetry CSV and the live dashboard.

```
gpu: NVIDIA GeForce RTX 3070 — power avg 200 W, peak 222 W (limit 240 W),
     peak 77 °C, fan 81%  THROTTLED: SW power cap (at the board's power limit)
```

That last clause is the point. A score that comes in low with zero errors is
either a bad part or a part that was never allowed to run — and until now those
two looked identical in a report. The throttle reasons are read straight from
the driver, so `HW power brake` (an **external** power-brake assertion, typically
the PSU pulling the board back) is distinguishable from an ordinary thermal or
power-limit clamp. Nothing to install: NVML is driver-resident and needs no
admin, no SDK, and no vendor tool.

Sensors that a board does not have are written **blank, never zero**. Most
consumer cards have no memory-junction sensor; recording `0` there would assert a
0 °C junction, which reads as a measurement and drags a shared temperature axis
to the floor. A blank cell means "no sensor", which is the truth.

### Charts, rendered automatically

Every run with `--telemetry-csv` now also writes a **self-contained HTML page**
next to it — GPU power and temperature, CPU effective clock (mean with a
min–max band), utilisation, per-lane work rate, and red rules at the exact
moments an error count rose. Hand-written inline SVG: no JavaScript, no CDN, no
dependency, opens by double-clicking on a machine with nothing installed.

A CSV is an archive. A chart is what somebody actually looks at — and asking a
tester to import a file into a spreadsheet to see a power curve means the curve
never gets looked at. `--no-graph` opts out.

### `--etw` — the operating system's own account of the run

Everything else this tool reports is something *we* measured. An ETW trace is the
opposite: context switches, DPC/ISR latency, GPU work packets, disk queues, power
state transitions — written by providers we could never instrument ourselves.
When a machine stutters or dies without leaving a WHEA entry, that trace is
usually the only artifact that still contains the answer.

```
cec-crucible run worst-case --etw
cec-crucible render --seconds 120 --etw-profiles CPU,GPU,Power,Thermal
```

The `.etl` lands in `--out` and opens directly in Windows Performance Analyzer.
It is driven through the in-box Windows Performance Recorder (`wpr.exe`), so the
provider sets and keyword masks are Microsoft's own rather than a worse set
hand-derived by us.

* **Needs an elevated shell.** Arming system-wide ETW is privileged. Run
  non-elevated and you get told exactly that, in words — a capture that could not
  run is reported as *unavailable*, never as a clean empty result.
* **Traces are large** — order 100 MB per minute under load in file mode.
* **A crash no longer loses the trace.** If a run dies with a session still
  armed, the kernel keeps buffering it. The next run detects that and flushes it
  to `*.recovered.etl` — the trace covering the crash, salvaged.

`wpr -profiles` lists everything available; custom `.wprp` files are accepted too.
The Settings screen carries an ETW ring (off / triage / cpu+gpu / power+thermal /
everything) so it can be driven without touching the command line.

### Fixed: the border animation could overwrite live values

The reactive border sparks painted their glyphs onto whatever cell they landed
on. Where the animation's path crossed a panel frame carrying real output, a
spark could replace a character of it — and a *missing digit is invisible*: the
number just reads wrong, with nothing to indicate anything was lost.

The FX now decides which cells it may write **before** it paints anything, and
that set contains only blank cells and box-drawing glyphs. Decoration can cover
its own trail (an error bolt still overdraws a spark) but it can never cost the
operator a digit. Two tests pin the property directly: one fills every cell with
content and requires it back byte for byte after a fully warmed FX pass, the
other proves the guard did not simply switch the animation off.

### Also

* The dashboard has a **POWER · THERMAL strip**: live watts and °C traces on a
  shared time axis, run peaks, fan and clocks, with the board's enforced power
  limit as the power trace's full scale — so trace height means "fraction of the
  power budget", not an autoscale that makes idle look like full load.
  Temperature colour is on an absolute scale (60 °C looks the same on every
  machine, and the ramp turns at 83 °C where a GeForce board starts clamping).
* Report JSON gains a `gpu` block and an `etw` block; the results CSV gains nine
  `gpu_*` columns; the telemetry CSV gains six per-sample sensor columns.

---

## Quick start

```
cec-crucible                        # interactive TUI
cec-crucible info                   # system + device identity
cec-crucible run quick              # ~15s CPU/RAM/storage QC
cec-crucible uncore                 # interconnect / FCLK verification
cec-crucible run gpu-recovery       # GPU load/idle/load recovery test
cec-crucible benchmark              # graphics composite score
cec-crucible run worst-case --ui --telemetry-csv    # everything at once, charted
```

---

## Known gaps

* **GPU sensors are NVIDIA-only.** They come from NVML. On AMD or Intel Arc the
  power/thermal columns are blank and the dashboard strip does not appear —
  correctly reported as absent, but absent.
* **The elevated ETW path is unverified on our bench.** The non-elevated refusal
  path is tested and reports correctly; the actual `.etl` capture has not been
  run end-to-end here. Treat `--etw` as alpha until it has produced a trace on
  your machine.
* **Single-bench validation.** AMD GPUs and Intel Arc untested — no hardware.
* **The PCIe test may not cross PCIe on systems with Resizable BAR.** A field
  capture reported `H2D ~370 GB/s`, which is physically impossible (PCIe Gen5 ×16
  is ~64 GB/s) — the staging buffer is likely landing in device-local memory,
  making the "upload" a VRAM→VRAM copy. Treat H2D figures as unproven until this
  is fixed; D2H looks correct.
* Five defects documented in `docs/game-realism.md` §1 remain unfixed — notably
  burst shapes overshooting their commanded duty cycle, and shape fidelity never
  being verified against the marker log.
* No network test; no display/scanout, cable or EDID validation.
* Not code-signed, so SmartScreen will warn on first run.
* Benchmark scores are calibrated against one RTX 3070 — treat cross-machine
  comparisons as provisional.

Full roadmap and design docs are in `docs/`.

---

## Read this if you used Alpha 1 or 2

A bug in those builds made healthy machines look like they were crashing:
`WM_DESTROY` called `PostQuitMessage`, so a destroyed preview window left a
`WM_QUIT` on the *thread's* queue that killed the **next** preview run in the
same process, instantly, with one dispatch and no verifications. Fixed in Alpha
3. Re-test anything diagnosed off Alpha 1/2 GPU crash data.

---

MIT licensed. Built by Critical Error Computing.
