<!-- SPDX-License-Identifier: MIT -->

# cec-crucible — roadmap

Phased so each step is buildable, testable offline, and useful on its own. The
hard/long piece (GPU power-virus) is isolated to its own phase.

## Phase 0 — repo stand-up ✅ (done)

Folder + design docs. **License: MIT** — maximally permissive, zero-liability
reuse for a tool that stress-tests other people's hardware. `LICENSE` at repo
root, SPDX headers throughout. Working name kept.

## Phase 1 — core + CPU/RAM/storage ✅ (built)

Cargo workspace, zero external dependencies, 44 unit tests, `cargo clippy` clean.

- [x] `crucible-core`: QPC markers (FFI + fallback), SMBIOS device-id,
  hand-rolled JSON, `LoadKernel`/`StopFlag`/`ShapeDriver`, report model +
  verdict. Unit-tested with tiny inputs (no real soak during `cargo test`).
- [x] `crucible-cpu`: AVX2+FMA burn (scalar fallback), per-core pinning,
  dual-accumulator recompute error-check. (~288 GFLOP/s on the Z490-I bench.)
- [x] `crucible-mem`: moving-inversion / own-address / seeded-random battery,
  disjoint per-thread chunks, first-fail (vaddr + pattern) reporting.
- [x] `crucible-storage`: non-destructive scratch-file write/sync/verify with
  seeded patterns and first-fail reporting.
- [x] `crucible-cli`: subcommands (`info`/`cpu`/`mem`/`storage`/`run`), built-in
  profiles (`quick`/`soak`/`cross`/`power`), Ctrl-C graceful stop, marker +
  device-ID'd report output. **Genuinely working CPU/RAM/storage QC.**
- [ ] Wire the binary into the companion QC harness's `stress-tools/` +
  verified arg templates so the PowerShell harness can drive it. *(cross-repo;
  the CLI contract — `--device-id`, `--out`, exit codes, report/marker
  filenames — is in place and ready to wire.)*

Basic concurrent cross-load (CPU+RAM+storage under one stop/timeline, the
`cross` profile) already landed here — the rest of Phase 2 remains below.

## Phase 2 — cross-load + telemetry

- Cross-load orchestration (concurrent kernels, one stop, one timeline,
  scenario files) — the CPU↔GPU-adjacent worst-case patterns (CPU + RAM +
  storage first, GPU joins in phase 3).
- CPU/board telemetry sampling (self-contained where possible; LibreHardwareMonitor
  interop only if needed — mind the HVCI/WinRing0 caveat from the research).

## Phase 3 — GPU (the long pole) — **in progress**

Backend decided: **CubeCL** (wgpu runtime by default). Full design and all
measurements in [`docs/gpu-plan.md`](gpu-plan.md).

- [x] **3a — backend spike.** Validated on NVIDIA + Intel, TDR-safe with chained
  short dispatches, ~92% of board power with an ALU+VRAM mix. Established that
  the CUDA runtime builds without a CUDA toolkit but will not *run* without one,
  so wgpu is the shipping default.
- [x] **3b — `crucible-gpu` as a real `LoadKernel`.** Steady + burst shapes, QPC
  marker emission, per-dispatch liveness + self-consistency verification, and the
  CPU↔GPU transient profiles (`in-phase` / `anti-phase` / `beat`) — all validated
  against marker timestamps. Ships in the single binary behind `--features gpu`;
  the default build stays zero-dependency.
- [x] **3c — VRAM integrity test + whole-platform worst-case.** Chained
  moving-inversion battery across VRAM (GPU-side verify, atomic first-fail
  reporting, fault-injection validated). Plus the `worst-case` profile: CPU
  transients anti-phase to the GPU under RAM + storage + VRAM-integrity load,
  every domain verifying its own data. A *separate* test from the wattage
  thrasher — watts are irrelevant to it.
- [ ] **3d — wattage servo.** Closed-loop power targeting (NVML/ADLX readback) +
  per-GPU-model self-calibration, device-ID'd.
- [ ] **3e — AMD validation.** No AMD hardware on the bench yet; no per-vendor
  claims until there is.

## Phase 3P — PCIe / motherboard link-integrity (bad-riser detection)

A distinct sub-thread from the compute tests. Scoped in
[`docs/pcie-plan.md`](pcie-plan.md): drive sustained verified traffic across the
x16 link as *stimulus*, and detect a marginal riser/cable/seating via link-train
checks and WHEA/AER error deltas — because a bad link RETRIES rather than slows.
Feasibility of the retry-detection is firmware-gated and needs Gen5 hardware +
a known-bad riser to validate; the link-training check and the transfer+verify
load are buildable now.

## The Gauntlet — QC burn-in campaign

With the domain kernels and cross-load profiles built, the
[`docs/gauntlet.md`](gauntlet.md) campaign sequences them into a full stress
screen: express (~2 h, the standing gate), standard (~12 h), and full (~24 h)
tiers, each a fault-mode-ordered phase list with per-phase WHEA attribution.
Delivered as [`scripts/gauntlet.ps1`](../scripts/gauntlet.ps1) — a per-phase
sequencer (one CLI process per phase = durable checkpoint, resumable manifest),
which is the safe form given reports/markers flush only at `finish()`. A native
`run gauntlet` profile is deferred until it can replicate checkpoint + flush +
per-phase WHEA (gauntlet.md §10).

## Phase 4 — polish

- Scenario library of known-killer patterns (grown from QC field data).
- Report dashboard on an internal collection server.
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
   the companion PowerShell harness's reports channel?
