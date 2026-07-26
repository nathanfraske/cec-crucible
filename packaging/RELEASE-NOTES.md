# cec-crucible v0.0.2 — Alpha 2 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

This release is driven almost entirely by one field capture. A client machine
crashing in games produced data that (a) exposed a **false PASS in this tool**,
and (b) showed a fault pattern we had no detector for. Both are fixed here.

---

## Install

1. Download `cec-crucible-0.0.2-win-x64.zip` and extract it.
2. Double-click **`INSTALL.cmd`**.

Per-user, **no admin required**. Installs to `%LOCALAPPDATA%\Programs\cec-crucible`,
adds it to your user PATH, and creates Start Menu + Desktop shortcuts.

**Launch the TUI:** double-click the **CEC Crucible** shortcut, or open a *new*
terminal and run `cec-crucible` with no arguments.

The main executable is self-contained (statically linked CRT — no Visual C++
Redistributable needed, so it runs on a freshly imaged machine). **PresentMon is
now bundled** so `--presentmon` works with no setup.

---

## What's new in Alpha 2

### The report now carries WHEA and driver-level events — by default

Every run scans the Windows System log across its own window and reports WHEA
machine-checks, display TDRs, bugchecks and disk resets:

```
event log: 1 hardware fault(s), 0 warning(s) in the run window:
  [FAIL] Microsoft-Windows-WHEA-Logger id=19  corrected bus/interconnect error —
         on AMD typically Infinity Fabric ECC (suspect FCLK / SoC voltage)
```

**This is the only plane that can see a fault the hardware already corrected.**
A corrected machine-check is invisible to every checksum we compute — the data
came back right *because* the hardware fixed it. A machine quietly correcting
errors is not stable, and now a logged hardware fault **fails the run** even when
every checksum matched. A scan that could not run is reported as *unavailable*,
never as clean. Opt out with `--no-eventlog`.

### `uncore` — cross-core / Infinity-Fabric verification

Closes the largest hole in the suite. Every other CPU test here is
register-resident, which means the L3, the ring/mesh and the Infinity Fabric were
**idle in every run this tool has ever done**. Marginal FCLK / SoC voltage is one
of the most common causes of "passes every stress test, crashes in games".

`cec-crucible uncore` walks core pairs with single-producer/single-consumer rings
and verifies every record arrives exactly once, in order, intact. A fault names
the **core pair** — cross-CCD points at FCLK / SoC voltage, same-CCD at the L3 or
ring. That distinction is the diagnosis.

### `run gpu-recovery` — reproduces a real field failure

On the client machine, every ray-tracing run started within ~5–8 s of a sustained
GPU load died instantly; every run started after ≥30 s of idle was fine. It did
**not** track workload weight — the heaviest configuration survived after a long
idle, the lightest died after a short one. What predicted the crash was purely
*how recently the GPU had been loaded*.

So this profile makes the **idle gap** the variable:

```
cec-crucible run gpu-recovery --seconds 30 --gap 5 --cycles 6 --preview
cec-crucible run gpu-recovery --engine render --gap 5      # the present path
```

It reports which cycle failed, and the event-log line will show a TDR if the
display driver reset. `--engine render|rt|pathtrace` because the client saw the
crash on **both** render and pathtrace previews — which points at the shared
presentation path rather than at ray tracing.

### Fixed: a false PASS on crashed ray-tracing runs

The field capture contained two runs that managed **one dispatch, zero
verifications**, and reported `PASS, ok=true, errors=0`. Closing the preview
window trips the stop flag, so the kernel exited before its first verification
and fell through to the success path. `render` had guarded this since it was
written; the RT engines never did.

A false PASS is the worst outcome a QC gate can produce. `rt` and `pathtrace` now
return **NOT VERIFIED — no conclusion can be drawn**.

### Priority escalation

At normal priority the tool's own coordination work competes with the workers it
just spawned — a laggy UI, and worse, burst edges landing late. The process now
runs at `ABOVE_NORMAL`, and the dashboard and telemetry threads are raised too so
sampling keeps its cadence. `--priority high` goes further; `--no-priority` opts
out. (Deliberately not `HIGH` by default: starving the desktop and the drivers
we are measuring would distort the measurement.)

### Also

* **PresentMon bundled** (Intel, MIT, unmodified — see `THIRD-PARTY-NOTICES.md`),
  found next to the executable, so `--presentmon` works out of the box.
* `mix` is now documented in `--help` (it shipped working but undocumented).

---

## Quick start

```
cec-crucible                        # interactive TUI
cec-crucible info                   # system + device identity
cec-crucible run quick              # ~15s CPU/RAM/storage QC
cec-crucible uncore                 # interconnect / FCLK verification
cec-crucible run gpu-recovery       # GPU load/idle/load recovery test
cec-crucible benchmark              # graphics composite score
cec-crucible run worst-case --ui    # everything at once, live dashboard
```

---

## Known gaps

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

MIT licensed. Built by Critical Error Computing.
