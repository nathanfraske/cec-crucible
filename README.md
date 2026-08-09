<!-- SPDX-License-Identifier: MIT -->

# cec-crucible

**Status: alpha, v0.0.6** — CPU, RAM, storage, GPU compute, VRAM, PCIe, ray
tracing and the graphics pipeline all run, alone or simultaneously, with power,
temperature, clock and drive-health telemetry throughout. Windows x64. Licensed
[MIT](LICENSE). See the [roadmap](docs/roadmap.md) for what is still missing.

A crucible is the vessel you heat metal in past its limits to see what it's
really made of. This is that, for finished PC builds: CEC's in-house
stress/validation tooling — **CPU, RAM, storage, GPU, cross-load, and wattage
tests** — built to emit device-ID'd, QPC-timestamped telemetry that
cross-correlates with the shop's external 1kHz+ power-monitoring rig.

## Why our own (not OCCT/Prime95/FurMark)

Researched in [`docs/prior-art-and-licensing.md`](docs/prior-art-and-licensing.md).
Short version: the good tools are closed and/or commercially licensed, none
expose the **load-shape control** the shop actually needs, and steady 100%
load (even FurMark) demonstrably misses real transient bugs. Building our own
buys: arbitrary load choreography (bursty duty cycles, CPU↔GPU cross-load,
sweeps), closed-loop wattage targeting we can tune, self-emitted QPC markers
for the 1kHz rig, and license-clean commercial use on customer builds.

## Scope

| Domain | What it does |
| --- | --- |
| CPU | FMA/AVX power burn + integer/cache load; per-core; recompute error-detection |
| RAM | pattern/moving-inversion tests (fast instability catches, TestMem5-class goal) |
| Storage | sustained read/write/verify to a scratch file (non-destructive) |
| GPU | custom CubeCL compute thrasher: steady + bursty + VRAM thrash + wattage targeting |
| Cross-load | orchestrate multiple domains concurrently (CPU pinned while GPU bursts, etc.) — the worst-case transients that kill marginal builds |
| Wattage | closed-loop power targeting (hold/step/ramp/sweep) + self-optimizing per-GPU calibration |

## Install

Grab the installer from the [latest release](https://github.com/nathanfraske/cec-crucible/releases/latest)
and run it, or do the whole thing from a terminal:

```powershell
$u='https://github.com/nathanfraske/cec-crucible/releases/latest/download/cec-crucible-setup.exe';$f="$env:TEMP\crucible-setup.exe";irm $u -OutFile $f;Start-Process $f '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -Wait
```

`Start-Process -Wait` rather than plain `&`: Setup is a GUI-subsystem process,
so `&` returns the moment it launches and anything you run next would race it.

Per-user, no admin needed. It puts `cec-crucible` on your PATH, appears in
Add/Remove Programs, and bundles everything it needs —
[LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor)
and [PresentMon](https://github.com/GameTechDev/PresentMon), with their licence
texts. Add `/MERGETASKS="cpusensors"` to also fetch
[PawnIO](https://github.com/namazso/PawnIO), which is what makes CPU package
power and die temperature readable; everything else works without it. See
[`third_party/README.md`](third_party/README.md) for where our code ends and
theirs begins.

There is also a **portable** zip on the same release page: extract, run, delete.
It touches nothing outside its own folder and contains no installer.

Uninstalling removes the install directory, the PATH entry, the shortcuts, the
settings in `%APPDATA%`, and any ETW session a killed run left behind:

```powershell
& "$env:LOCALAPPDATA\Programs\cec-crucible\unins000.exe" /VERYSILENT
```

Not code-signed yet, so SmartScreen will warn on first run.

## Build & run

The core suite has **zero external dependencies** — a stock stable Rust
toolchain builds it offline. The GPU kernel is the one documented exception (it
uses CubeCL), so it lives behind a cargo feature; the default build never
compiles it.

```
cargo build --release                                  # core only, 0 deps
cargo test                                             # unit tests, 0 deps
cargo build --release -p crucible-cli --features gpu   # full shipped binary
```

Both produce the same single `cec-crucible` binary — `--features gpu` just adds
the `gpu` / `gpu-info` commands and the CPU↔GPU transient profiles. GPU commands
in a core build fail with a message telling you to rebuild with the feature.

```
cec-crucible info                          # device id (SMBIOS), CPU, RAM, QPC
cec-crucible cpu     --seconds 60          # FMA/AVX burn, per-core, recompute check
cec-crucible mem     --seconds 60 --mb 8192
cec-crucible storage --seconds 60 --path D:\ --size-mb 4096
cec-crucible run cross --seconds 120       # all domains concurrently
cec-crucible run power --seconds 60        # CPU burst, dense markers for the rig
```

With `--features gpu`:

```
cec-crucible gpu-info                      # list usable GPUs
cec-crucible gpu --seconds 60              # GPU thrasher (~92% of board power)
cec-crucible vram --seconds 60             # VRAM integrity test (find bad memory)
cec-crucible link --seconds 60             # PCIe host<->device transfer + verify
cec-crucible gpu --seconds 60 --shape burst --burst-on 20 --burst-off 20
cec-crucible run in-phase   --seconds 120  # CPU+GPU burst together -> peak draw
cec-crucible run anti-phase --seconds 120  # they alternate -> VRM/PSU chase load
cec-crucible run beat       --seconds 120  # drifting periods -> sweeps all phases
```

The transient profiles are the point: a steady 100% load cannot produce them.
All kernels share one phase origin, so the commanded offsets hold exactly —
verified from the marker timestamps ([`docs/gpu-plan.md` §13](docs/gpu-plan.md)).

Each run writes a device-ID'd `crucible-<id>-<ts>.report.json` and a
`…-markers.jsonl` to `--out` (default: the harness log dir if present, else the
working dir). Pass `--device-id <uuid>` to key a run to a specific machine (the
PowerShell harness supplies this; standalone auto-detects via SMBIOS). Exit code
is `0` for PASS/PARTIAL, `1` for FAIL. `cec-crucible help` lists all options.

### Workspace layout

| Crate | Role |
| --- | --- |
| `crucible-core` | QPC clock, JSON writer, SMBIOS device-id, `LoadKernel`/`StopFlag`/`ShapeDriver`, marker log, report model — all std-only |
| `crucible-cpu` | FMA/AVX burn kernel with dual-accumulator soft-error detection + per-core pinning |
| `crucible-mem` | moving-inversion / own-address / seeded-random RAM battery with first-fail reporting |
| `crucible-storage` | non-destructive scratch-file write/sync/verify, true-uncached I/O, multi-SSD cross-load |
| `crucible-gpu` | CubeCL GPU thrasher with load shapes, markers and per-dispatch verification — **the one crate with external dependencies** |
| `crucible-cli` | `cec-crucible` binary: arg parsing, profiles, orchestration, report + marker output |

`crucible-gpu` is a workspace member (so it ships inside the one binary) but is
excluded from `default-members`, so everyday `cargo build` / `cargo test` never
compile its dependency tree. `Cargo.lock` is committed to keep the default build
resolvable offline despite the optional dependency.

## How it fits the rest of the suite

`cec-crucible` produces **compiled binaries** designed to be driven by a
companion PowerShell QC orchestration harness (part of CEC's internal first-boot
tooling). The harness discovers these binaries, runs them per profile, and owns
the parts that sit *around* the load: WHEA (hardware-error) gating, report
aggregation, and the operator-facing entry point.

Clean split of responsibilities:

- **cec-crucible owns:** load generation, load-shape choreography, fine
  CPU/storage telemetry, and QPC load markers.
- **The orchestration harness owns:** profile sequencing, hardware-error watch,
  and report collection.

The two also run fully standalone — the CLI is self-sufficient (`--device-id`,
`--out`, exit codes, and a device-ID'd JSON report + JSONL markers), so nothing
here depends on the harness being present.

## Docs

- [`docs/design.md`](docs/design.md) — architecture, zero-dependency Rust rationale, crate layout, per-domain test designs, the load-shape philosophy, QPC markers + 1kHz correlation, device-ID'd reporting.
- [`docs/roadmap.md`](docs/roadmap.md) — phased build plan (Phase 1 built; GPU is Phase 3).
- [`docs/gpu-plan.md`](docs/gpu-plan.md) — Phase 3 GPU work: backend decision (CubeCL/wgpu), the thrasher (§13), the VRAM integrity test + whole-platform worst-case (§14), TDR handling, per-vendor telemetry, and the spike/measurement results.
- [`docs/pcie-plan.md`](docs/pcie-plan.md) — PCIe / motherboard **link-integrity** design: detecting a bad riser (which retries rather than slows) via link-training checks, verified-transfer load, and WHEA/AER error deltas. Honest verdict on what's feasible on Windows without a kernel driver.
- [`docs/gauntlet.md`](docs/gauntlet.md) — the recommended QC burn-in **campaign**: express (~2 h) / standard (~12 h) / full (~24 h) phase tables, the fault-mode → provocation → detector map, a coverage matrix, the reliability-engineering rationale (Arrhenius / Coffin-Manson / HALT-HASS / ESS), and an honest 80/20 on diminishing returns. Runnable via [`scripts/gauntlet.ps1`](scripts/gauntlet.ps1) — one CLI process per phase (durable checkpoint), WHEA-bracketed and resumable.
- [`docs/prior-art-and-licensing.md`](docs/prior-art-and-licensing.md) — existing tools, licensing reality, build-vs-buy.
- [`spikes/gpu-3a/`](spikes/gpu-3a/) — the throwaway CubeCL probe that produced the Phase 3 measurements (excluded from the workspace; it pulls external crates, the core does not).
