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

## Phase 3T — transient & light-load classes ✅ (built)

The mirror image of the heavy-load suite — the fault classes that hide in the
load *edges* and at *idle*, reached by two new load shapes (`Shape::Jitter`,
`Shape::Pulse`) plus per-core seed decorrelation:

- [x] **`chaos`** — CPU + GPU on independent seeded jitter (spike + `floor_pct`
  trickle) over steady RAM/storage/VRAM/PCIe: the never-settle superset of
  `beat`, hitting coincident-spike (OCP) and hand-off (VRM-chase) statistically.
  `--per-core` decorrelates each CPU core (CPU-VRM chaos) vs one system slam.
  `--seed` replays the commanded pattern.
- [x] **`game-load`** — frame-paced CPU→GPU handoff at moderate power; the game
  *electrical/thermal* signature (frame cadence, CPU-leads-GPU, VRAM streaming),
  explicitly **not** the graphics stack (see 3G).
- [x] **`core-cycle`** — single-core steady boost rotated over all cores: the
  weak-core-at-max-boost hunt masked in all-core runs.
- [x] **`c-states`** — single-core pulse + deep idle rotated: the idle /
  C-state / low-load-voltage class. Requires BIOS C-states + boost + a deep-idle
  power plan (no telemetry to confirm; idle-only faults show via WHEA/reboot).

## Phase 3G — GPU functional-unit coverage (the immediate follow-on)

Today the GPU coverage is general shader ALU (thrasher), VRAM, and PCIe — it
touches **none** of the specialized silicon. A card with a dead ROP, a bad
tensor MAC lane, or a defective RT intersection unit passes everything and fails
at the customer's game / DLSS / ray-traced workload. Scoped against current
primary docs and **cross-checked against the pinned `wgpu-types-29.0.4` source** —
full detail in [`docs/gpu-functional-units.md`](gpu-functional-units.md). Honest
reality: render is portable today; the portable tensor path is **Vulkan/SPIR-V
only** (CubeCL `cmma`, int8→int32 golden; the wgpu-*native* coop-matrix in 29 is
8×8-f32-only and near-useless — verified in source); RT is wgpu-29
`EXPERIMENTAL_RAY_QUERY`, **Vulkan-only and experimental**. Build order by
coverage-per-effort:

1. **`render` (procedural)** — headless wgpu graphics pipeline to an offscreen
   target: exercises the **rasterizer, TMU/texture units, and ROP/blend/depth**
   that compute bypasses, plus real draw calls and the driver command path.
   **Zero new deps** (raw wgpu, like `link.rs`), portable across all three
   vendors, verified by framebuffer pixel-checksum (same-device self-consistency;
   cross-vendor is not bit-identical). Highest value, lowest risk. Optional
   `render --preview` (opt-in `preview` feature, Windows) pops a live window that
   mirrors the render as it runs — a hand-rolled Win32 window (no winit) whose
   swapchain receives a copy of the already-finished offscreen frame, so what gets
   verified is byte-identical to a headless run and the ≤60 Hz mirror never
   throttles the load. Close the window to stop the test.
2. **`render --scene` (glTF/PBR)** — optional realism upgrade: a bundled
   permissively-licensed Khronos glTF scene + a PBR shader ported from the
   Apache-2.0 glTF-Sample-Viewer, and a `--scene <file>` override so a shop
   supplies its own licensed asset (we redistribute nothing proprietary; no EULA/
   download). Gated behind a new `gpu-gltf` feature (`gltf` + `image` crates) so
   `--features gpu` stays glTF-free.
3. **`tensor`** — a heavy verified low-precision GEMM (the "little ML test"):
   **int8→int32 gives an exact cross-vendor golden** (the strongest verification
   in the suite), fp16→fp32 self-consistency. Reaches tensor cores via CubeCL
   CMMA on the **CUDA** runtime — so it inherits the NVRTC/toolkit requirement
   (NVIDIA + `--features cuda`, bench-only, not ship-anywhere). wgpu can't reach
   tensor cores; the Vulkan/SPIR-V route is experimental.
4. **`rt`** ✅ **BUILT** (`rt.rs`, `--features rt`) — hardware BVH-traversal +
   ray/triangle-intersection stress. Built on the **stable** path, not wgpu's
   experimental one: raw Vulkan (`ash` 0.38) with the ratified `VK_KHR_ray_query`,
   portable across all three RT vendors in the single binary. The WGSL ray-query
   shader is compiled to SPIR-V **at runtime by naga** (already in the wgpu tree),
   so no shader compiler / Vulkan SDK is needed anywhere. Traces a deterministic
   camera fan over a fixed BLAS/TLAS and folds each hit's primitive index + `t`
   into a per-ray checksum; verified by same-device self-consistency + liveness.
   Validated on the RTX 3070: PASS, 0 errors, **~5.35 Gray/s** steady (~2.64 Gray/s
   burst). `--rt-iters` sets traces/ray (load + TDR knob). `gpu-allocator` handles
   device memory + the buffer-device-address allocate flag the AS build requires.
   Optional `rt --preview` (opt-in `preview` feature, Windows) pops a live window
   showing the ray-traced image: the shader additionally writes a shaded colour
   image (Lambertian from the grid's analytic normal + a *traced hard shadow ray*
   per pixel, so the surface self-shadows — visibly the RT cores doing secondary
   rays) which a small wgpu present path upscales into the window. Gated by a
   `shade` uniform so a plain `rt` run does zero extra ray work and the checksum is
   byte-identical.

All four implement `LoadKernel`, so they drop into the shared shape/stop/phase/
marker machinery with no orchestrator change and immediately gain the chaos /
game-load / gauntlet profiles. Carry-forward caveat (as elsewhere): render/RT
self-consistency has the deterministic-from-t0 blind spot; the int8 tensor golden
and WHEA are the backstops. Video encode/decode (NVENC/NVDEC) and display scanout
stay out of scope.

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
