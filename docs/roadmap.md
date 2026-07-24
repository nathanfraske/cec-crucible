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
