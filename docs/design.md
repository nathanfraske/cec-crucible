<!-- SPDX-License-Identifier: Apache-2.0 -->

# cec-crucible — design (ideation)

Design intent for CEC's in-house stress suite. Nothing here is built yet; this
is the plan of record.

## Principles

1. **Zero external dependencies.** Pure Rust std, no crates.io fetch. A tool
   that runs on customer machines and ships on a USB stick benefits from a nil
   supply-chain surface, reproducible builds, and offline-buildable benches —
   the same "no runtime deps" ethos as the PowerShell first-boot engine. JSON
   (reports/markers) and CLI parsing are small enough to hand-roll cleanly;
   the one exception under evaluation is the GPU backend (see GPU below).
2. **Load choreography over raw wattage.** The shop's field finding: steady
   100% load — even FurMark — misses real bugs. The differentiator is *how*
   the load moves: bursty duty cycles that hammer VRM transient response,
   cross-domain loading (CPU pinned while the GPU sees burst trains), sweeps,
   worst-case scenarios. Every kernel is scriptable in shape, not just
   intensity.
3. **Correlate, don't just measure.** The tool exists partly to feed the
   external 1kHz+ power rig. It emits QPC-precision markers at every load
   transition so the analog capture aligns to the exact load edges — the
   software can't (and needn't) sample at 1kHz; it needs to *timestamp* the
   edges precisely.
4. **Device-identified, retrievable.** Every run is keyed to the machine
   (SMBIOS UUID + board serial) so results are retrievable and diffable across
   the fleet, shipped over the same reports channel as the rest of QC.
5. **Error detection, not just crash detection.** A stress pass that doesn't
   crash but produced a wrong result is still a fail. Kernels checksum/verify
   their work; WHEA (watched by the PowerShell harness) catches corrected
   hardware errors that never surface as a crash.

## Planned crate layout (Cargo workspace)

```
cec-crucible/
  crates/
    crucible-core/     QPC markers, device-id, report model, LoadKernel trait,
                       telemetry traits, hand-rolled JSON — no deps
    crucible-cpu/       FMA/AVX burn, integer/cache load, per-core, recompute check
    crucible-mem/       pattern / moving-inversion RAM tests
    crucible-storage/   scratch-file read/write/verify (non-destructive)
    crucible-gpu/       CubeCL (or wgpu) compute: thrasher, VRAM, wattage servo
    crucible-cli/       orchestrator binary: subcommands + profile runner
```

The CLI binary doubles as the per-test tool the PowerShell harness calls
(`cec-crucible cpu --seconds 60 --device-id <uuid>`) *and* a standalone runner
(`cec-crucible run soak`). Single source of truth for device-id: the harness
passes it down; standalone falls back to a local read.

## LoadKernel model

```
trait LoadKernel {
    fn name(&self) -> &str;
    fn kind(&self) -> Kind;                 // Cpu | Mem | Storage | Gpu
    fn run(&self, budget: Budget, stop: &StopFlag, markers: &mut MarkerLog) -> LoadResult;
}
struct LoadResult { ok: bool, iterations, checksum, detail, error_count }
```

- `budget`: duration and/or a load-shape script (steady, burst {on_ms, off_ms},
  ramp, sweep, target-watts).
- `stop`: shared atomic so the orchestrator can end all kernels together (this
  is how cross-load works — N kernels, one stop, one timer).
- `markers`: kernels stamp their own transitions (burst edges especially) into
  the shared marker log.

## Per-domain designs

### CPU (`crucible-cpu`)
- **Power/heat:** wide FMA (AVX2/AVX-512 where present) accumulation loops,
  N threads = logical cores, pinned per-core so we can stress one core or all.
- **Error detection:** each block computes a deterministic reduction and
  recomputes it; mismatch = soft error on that core. (Deeper: cross-thread
  compare of identical blocks.)
- **Shapes:** steady, per-core round-robin (CoreCycler-style — isolate the one
  unstable core), bursty.
- Prior art to mirror in behavior: Prime95 small-FFT (heat) / blend (RAM+cache);
  CoreCycler's per-core orchestration.

### RAM (`crucible-mem`)
- Allocate a large buffer (configurable % of free RAM), run pattern tests:
  moving inversions, walking ones/zeros, own-address, random with re-seedable
  verification. Goal: TestMem5/Karhu-class fast instability catches for
  XMP/EXPO margin validation.
- Reports first-fail address + pattern; ties to the XMP-vs-BOM check in the
  first-boot verify stage.

### Storage (`crucible-storage`)
- Sustained sequential + random read/write/verify against a **scratch file**
  on the target drive (non-destructive; never raw-device). Configurable size,
  queue depth, R/W mix, O_DIRECT/FILE_FLAG_NO_BUFFERING for true device load.
- Verify written patterns on read-back (data-integrity, not just throughput).
- diskspd covers this today; owning it lets us unify markers + reports.

### GPU (`crucible-gpu`)
- **Backend:** custom compute via **CubeCL** (Burn ecosystem; wgpu/CUDA/ROCm
  backends → one codebase covers NVIDIA/AMD/Intel), shipped as a compiled
  binary. wgpu/Vulkan compute is the safe floor if a CubeCL backend lags on a
  vendor. This is the one place an external toolkit is justified — validate
  backend maturity per vendor at project start.
- **Thrasher:** persistent compute dispatch of FMA + VRAM-thrashing access
  patterns. On boost-limited modern GPUs a well-occupied FMA+bandwidth kernel
  pins the card at its power limit — FurMark-class by definition; the GPU's own
  limiter does the tuning.
- **VRAM:** memtest-style patterns across allocated VRAM (memtest_vulkan is the
  open reference to match/adopt for the VRAM stage).
- **Load shapes:** steady / bursty duty cycles / sweep / cross-load-with-CPU.
- **Wattage mode (closed loop):** read board power (NVML / ADLX / LHM), servo
  the load mix (occupancy, FMA:memory ratio, duty cycle) to hold a commanded
  power profile — hold-at-N-watts, staircase, ramp, sine sweep. **Self-optimizing:**
  a per-GPU-model calibration pass searches the load-mix space for the
  max-sustained-power point and stores that profile device-ID'd, so later runs
  start pre-tuned. Control-loop reality: the software servo reacts at ~100ms–1s
  (fine for *holding* a target); the ms-scale transient attacks are open-loop
  burst scenarios, judged by the 1kHz rig.
- Robustness is the real work: TDR/timeout handling, per-vendor telemetry,
  error detection, multi-arch validation (~1–3 weeks for a v1 + per-generation
  smoke tests).

### Cross-load (`crucible-cli` orchestration)
- Run multiple kernels concurrently under one stop flag + one timeline: e.g.
  CPU held at full FMA while the GPU runs a burst train; storage load under
  CPU+RAM pressure. These are the absolute-worst-case transients that pass
  steady-state tests but fail in real games. Scenario files define the
  choreography so QC grows a library of known-killer patterns.

### Wattage characterization
- The `power` profile: GPU wattage-targeting sweeps + CPU max-power, dense
  markers, for the 1kHz rig to profile the rails. Output is a power-vs-time
  trace keyed to the load markers.

## Markers + 1kHz correlation

- QPC via `QueryPerformanceCounter` (FFI, no dep) — the **exact clock** the
  PowerShell harness and the external rig align to. Each marker: `{ seq, event,
  kernel, mode, qpc_ticks, qpc_frequency, wall }` as JSONL.
- Events: run-start/stop, stage-start/stop, and per-kernel burst edges. The
  1kHz analog capture aligns on QPC ticks with sub-µs precision — no inference.

## Reporting

- Device-ID'd JSON report (SMBIOS UUID + board serial + short id) + the marker
  JSONL, written where the PowerShell harness collects them
  (`%ProgramData%\firstboot\logs\`), shipped over the reports channel designed
  in CEC-Autosetup's AllMyStuff plan.
- Verdict: Pass / Fail (any kernel error or WHEA event) / Partial. WHEA is
  watched by the PowerShell harness around the whole window; kernels report
  their own compute/verify errors.

## Integration boundary

- **cec-crucible owns:** load kernels, load shapes, fine GPU/CPU telemetry,
  load markers.
- **CEC-Autosetup stress harness owns:** profile orchestration, WHEA gating,
  report aggregation, device-ID'd JSONL, the CLI entrypoint operators run.
- Interface: the harness invokes `cec-crucible <subcommand>` per stage and
  reads its markers/report. The two can also run fully standalone.
