# cec-crucible v0.0.1 — Alpha 1 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

> **Alpha.** Validated on one bench (i9-10850K + RTX 3070 + UHD 630, Windows 11).
> Treat findings as real but treat *absence* of findings as unproven on hardware
> that differs from that. Known gaps are listed at the bottom — they are listed
> because they are known, not because they are excused.

---

## Install

1. Download `cec-crucible-0.0.1-win-x64.zip` and extract it.
2. Double-click **`INSTALL.cmd`**.

Per-user, **no admin required**. It installs to
`%LOCALAPPDATA%\Programs\cec-crucible`, adds that to your user PATH, and creates
Start Menu + Desktop shortcuts.

**To launch the interactive TUI:** double-click the **CEC Crucible** shortcut, or
open a *new* terminal and run `cec-crucible` with no arguments.

Uninstall:
```
powershell -ExecutionPolicy Bypass -File Install-Crucible.ps1 -Uninstall
```

### Portability

The executable is fully self-contained — every import is a Windows system DLL.
There is **no** Visual C++ Redistributable dependency (the CRT is statically
linked), so it runs on a freshly imaged machine. Vulkan, CUDA and NVML are loaded
from the GPU driver at runtime and degrade gracefully when absent.

Requirements: Windows 10/11 x64. A GPU driver is needed for the GPU tests; the
CPU/RAM/storage tests need nothing.

---

## What's in it

**Interactive TUI** — branded launcher for every test and profile, per-test config
screens, and a live dashboard: per-core heatmap with effective clock and
utilization (driverless, via PDH), per-domain panels showing live patterns, values
and verification checksums, and a reactive border that sparks with activity,
pulses cyan on each verification and cracks red lightning on a miscompare.

**Tests** — CPU (AVX2/FMA, recompute-verified) · RAM (moving-inversion, walking
ones/zeros, checkerboard, March C−, modulo-20) · storage (uncached
write/read-verify, multi-SSD cross-load) · GPU thrasher (watts) · VRAM integrity ·
PCIe transfer+verify (incl. CUDA full-duplex) · raster render · tensor cores ·
ray tracing · path tracer (8 materials incl. fibre fur) · OptiX.

**Composable runs** — `mix` composes an arbitrary run from any tests with any
parameters, concurrently, with per-test duration and phase offset:

```
cec-crucible mix --seconds 120 -- cpu --shape burst --burst-on 20 --burst-off 20 \
                                -- gpu --shape burst --burst-on 20 --burst-off 20 --at 20ms \
                                -- mem --mb 4096
```

**Profiles** — quick · soak · cross · power · storage-cross · worst-case · chaos ·
game-load · core-cycle · c-states · in-phase · anti-phase · beat.

**Benchmark** — `cec-crucible benchmark` scores each graphics engine on a
calibrated scale and combines them via geometric mean. Error-gated: any
miscompare invalidates the score. Reference RTX 3070 ≈ 20,000 composite.

**Reports** — device-ID'd JSON + JSONL markers (QPC-timestamped, for correlation
with an external power rig), optional per-stage results CSV and ~4 Hz time-series
telemetry CSV including per-core clock and utilization. Optional `--presentmon`
ETW capture for true displayed-frame pacing (GPU busy, CPU busy/wait, display
latency, dropped frames).

**Every test verifies its output.** Timings alone are never trusted — a dead
kernel once reported 1.65 TFLOP/s while the GPU sat idle, and that lesson is
baked into the design.

---

## Quick start

```
cec-crucible                        # interactive TUI
cec-crucible info                   # system + device identity
cec-crucible run quick              # ~15s CPU/RAM/storage QC
cec-crucible benchmark              # graphics composite score
cec-crucible run worst-case --ui    # everything at once, live dashboard
cec-crucible help                   # everything else
```

---

## Known gaps in this alpha

* **Single-bench validation.** AMD GPUs and Intel Arc are untested — no hardware.
  Per-vendor behaviour is not claimed.
* **Six defects are documented but not yet fixed** (see `docs/game-realism.md` §1),
  the notable ones being that burst shapes overshoot their commanded duty cycle by
  up to one work chunk, and that GPU transient errors between verification
  intervals can be overwritten before they are seen.
* **The uncore is not exercised** — no cross-core coherence traffic, so
  FCLK/Infinity-Fabric instability is not currently detected.
* **No network test.** No display/scanout, cable or EDID validation.
* **`--presentmon` needs an installed PresentMon 2.x** and elevation for the ETW
  session; it is opt-in and never fails a run when absent.
* Not code-signed, so SmartScreen will warn on first run.
* The benchmark score is calibrated against one RTX 3070; treat cross-machine
  comparisons as provisional until a fleet baseline exists.

Full roadmap and design docs are in `docs/` in the repository.

---

MIT licensed. Built by Critical Error Computing.
