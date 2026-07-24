<!-- SPDX-License-Identifier: Apache-2.0 -->

# cec-crucible — roadmap (ideation)

Phased so each step is buildable, testable offline, and useful on its own. The
hard/long piece (GPU power-virus) is isolated to its own phase.

## Phase 0 — repo stand-up (this)

Folder + ideation docs. Decide the name, the license (Apache-2.0 to match
CEC-Autosetup, vs AGPL like cec-support-agent), and whether it's a GitHub repo
now or local until code lands.

## Phase 1 — core + CPU/RAM/storage (days)

- `crucible-core`: QPC markers, device-id, hand-rolled JSON, `LoadKernel`
  trait, `StopFlag`, report model. Unit-tested with tiny inputs (no real soak
  during `cargo test`).
- `crucible-cpu`: real FMA burn, per-core, recompute error-check.
- `crucible-mem`: real pattern/moving-inversion buffer test (bounded).
- `crucible-storage`: real scratch-file read/write/verify (non-destructive).
- `crucible-cli`: subcommands (`cpu`/`mem`/`storage`/`info`/`run`), profile
  runner, marker + device-ID'd report output.
- Wire the binary into CEC-Autosetup's `stress-tools/` + verified `argsTemplate`s
  so the PowerShell harness can drive it. **Genuinely working CPU/RAM/storage
  QC at the end of this phase.**

## Phase 2 — cross-load + telemetry

- Cross-load orchestration (concurrent kernels, one stop, one timeline,
  scenario files) — the CPU↔GPU-adjacent worst-case patterns (CPU + RAM +
  storage first, GPU joins in phase 3).
- CPU/board telemetry sampling (self-contained where possible; LibreHardwareMonitor
  interop only if needed — mind the HVCI/WinRing0 caveat from the research).

## Phase 3 — GPU (the long pole)

- `crucible-gpu` on CubeCL/wgpu: thrasher (steady + burst + VRAM), multi-vendor
  validation, TDR handling.
- Wattage closed-loop servo (NVML/ADLX/LHM readback) + per-GPU-model
  self-calibration, device-ID'd.
- Full cross-load with GPU in the mix; the `power` characterization profile for
  the 1kHz rig.
- ~1–3 weeks graphics work for a v1 + per-generation smoke tests; needs real
  GPUs of each vendor to validate.

## Phase 4 — polish

- Scenario library of known-killer patterns (grown from QC field data).
- Report dashboard on the designated server (alongside the rehearsal-report
  collection).
- Optional: wrap a licensed OCCT/BurnInTest as a certified customer-facing
  report layer, if ever wanted — our engine stays the QC gate.

## Open questions

1. Repo name (working: `cec-crucible`), license, GitHub-now-vs-local.
2. CubeCL vs plain wgpu for the GPU backend — decide after a backend-maturity
   spike on NVIDIA + AMD + Intel.
3. How much telemetry to own in-tool vs. lean on LibreHardwareMonitor
   (HVCI/WinRing0 blocklist risk).
4. Scratch-file storage test vs. optional raw-device (destructive, gated) mode
   for full-drive validation.
5. Does the standalone CLI need its own report retrieval, or always go through
   the PowerShell harness + AllMyStuff reports channel?
