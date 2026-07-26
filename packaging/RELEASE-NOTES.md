# cec-crucible v0.0.3 — Alpha 3 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

## Read this first if you used Alpha 1 or 2

**A bug in this tool made healthy machines look like they were crashing.** If you
have reports where a GPU test ended after a fraction of a second with one
dispatch and no verifications, that was us, not the hardware under test.

`WM_DESTROY` called `PostQuitMessage`, which posts `WM_QUIT` to the *thread's*
message queue rather than to the window. Once one preview window had been
destroyed, the leftover quit message sat there until the next preview opened on
that thread — whose first message pump found it, concluded the operator had
closed the window, and stopped the run immediately.

The signature: any preview test (`render`, `rt`, `pathtrace`) that ran *after
another preview test in the same process* would die at once. A fresh process
always worked. In one field capture that was 3 crashes out of 3 opportunities,
with zero false positives — and it produced reports that read as hardware faults.

Fixed: window closure is now per-window state, so one window's teardown cannot
end another window's run. **Re-test anything you diagnosed off Alpha 1/2 GPU
crash data.**

---

## Install

1. Download `cec-crucible-0.0.3-win-x64.zip` and extract it.
2. Double-click **`INSTALL.cmd`**.

Per-user, **no admin required**. Installs to `%LOCALAPPDATA%\Programs\cec-crucible`,
adds it to your user PATH, and creates Start Menu + Desktop shortcuts.

**Launch the TUI:** double-click the **CEC Crucible** shortcut, or open a *new*
terminal and run `cec-crucible` with no arguments.

The main executable is self-contained (statically linked CRT — no Visual C++
Redistributable needed, so it runs on a freshly imaged machine). **PresentMon is
now bundled** so `--presentmon` works with no setup.

---

## What's new in Alpha 3

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

### `run gpu-recovery` — load / idle / load cycling

Cycles a GPU engine with a configurable idle gap between runs, reporting which
cycle (if any) failed to verify:

```
cec-crucible run gpu-recovery --seconds 30 --gap 5 --cycles 6 --preview
cec-crucible run gpu-recovery --engine render --gap 5      # the present path
```

`--engine render|rt|pathtrace`. It was built to chase what looked like a
GPU recovery-time fault in the field; that turned out to be the `WM_QUIT` bug
described at the top of these notes, so treat the idle gap as one variable among
several rather than as a known failure mode. The profile is still the right shape
for exercising repeated device create/destroy cycles, which is a genuinely
under-tested path.

### Crash watchdog — a crash now leaves evidence

Four hard crashes in a field capture left nothing but a telemetry file that
stopped mid-write: no report, no verdict, no indication of what the tool had been
doing. For a QC tool that is the worst possible moment to learn nothing.

Every run now keeps a **breadcrumb** on disk, rewritten at each phase, and
installs an **unhandled structured-exception filter** plus a panic hook. An
access violation raised inside a GPU driver (which `catch_unwind` cannot catch)
now writes a `*.crash.json` naming the exception, the faulting address, and the
phase it died in. GPU kernels mark their own teardown, so a driver fault there is
attributable rather than silent.

Even an uncatchable `TerminateProcess` leaves the breadcrumb, and the next run
reports it: *"a previous run did not finish — last seen in running:cpu"*.
Verified against a real forced kill.

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

### Fill-the-hardware modes

`--mb max` on the memory test and `--vram-mb max` on the VRAM test size
themselves to the machine instead of taking a default slice.

`max` on VRAM needed a real answer, not a big number: **WDDM over-commits.** An
allocation past the end of dedicated VRAM succeeds and silently spills into
shared system memory, so the allocator never refuses — the failure arrives later,
during the fill, as a *device loss* that takes the run down. Measured on an RTX
3070 (8 GiB): 7168 MiB fills clean, 8192 MiB loses the device. So `max` now reads
the adapter's dedicated VRAM from DXGI and takes 85% of it, leaving the desktop
and compositor their working set. On that same 8 GiB card it fills 6784 MiB and
passes. `--mb max` takes 90% of *available* RAM for the same reason.

`uncore` is also now in the interactive menu, under COMPUTE.

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
