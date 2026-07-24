<!-- SPDX-License-Identifier: Apache-2.0 -->

# cec-crucible

**Status: ideation / planning.** No code yet — this repo currently holds the
design for CEC's own stress-testing suite. ("cec-crucible" is a working name;
rename freely.)

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

## How it fits the rest of the suite

`cec-crucible` produces **compiled binaries**. The QC stress orchestrator
already built in the first-boot tool
([CEC-Autosetup `docs/stress-harness.md`](../CEC-Autosetup/docs/stress-harness.md),
`tools/Invoke-StressTest.ps1`) discovers those binaries under its `stress-tools/`
directory and drives them per profile — WHEA gating, report aggregation, and
device-ID'd JSONL live there. This repo owns the **load generation + fine
telemetry + load markers**; the PowerShell harness owns **orchestration +
hardware-error watch + reporting**. Clean split.

## Docs

- [`docs/design.md`](docs/design.md) — architecture, zero-dependency Rust rationale, planned crate layout, per-domain test designs, the load-shape philosophy, QPC markers + 1kHz correlation, device-ID'd reporting.
- [`docs/roadmap.md`](docs/roadmap.md) — phased build plan.
- [`docs/prior-art-and-licensing.md`](docs/prior-art-and-licensing.md) — existing tools, licensing reality, build-vs-buy.
