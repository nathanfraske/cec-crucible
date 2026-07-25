<!-- SPDX-License-Identifier: MIT -->

# Making the tests behave like a real game

Synthesis of four parallel design reviews — GPU/graphics, CPU/threading, storage+network,
and whole-system/methodology — each of which audited the current code before proposing
anything. This is the design of record for closing the gap between "a synthetic loop that
makes the hardware hot" and "a workload that fails the way a game fails".

Companion docs: `composable-runs.md` (the `mix` mechanism most of this composes through),
`gpu-functional-units.md`, `pcie-plan.md`, `gauntlet.md`.

---

## 0. The one-paragraph version

The reviews agreed on an uncomfortable finding: **the most valuable work is not a new
kernel.** Four defects in the code we already ship make existing results less trustworthy
than they look, and three whole silicon blocks have zero coverage while we add more ways to
exercise the ones we already cover. Fix the defects first (days, not weeks), then close the
coverage holes, then build game-shaped workloads. And the single honest answer to "make it
like a real game" is to **run a real game and be the instrument around it** — which reuses
the PresentMon plumbing already built.

---

## 1. Confirmed defects — fix before building anything new

These were found by reading our own source, not by speculation. Each makes a result we
already report less true than it appears.

### D1. Burst shapes overshoot their commanded duty cycle
`CpuKernel` runs a fixed `CHUNK_ITERS = 200_000` FMA chunk (~0.35 ms at 4.5 GHz) and only
checks the shape driver *between* chunks. So the on-phase always overruns by up to one
chunk: **~10% duty overshoot for `game-load --fps 120`** (3.3 ms on-window), **~23% for a
400 fps shooter profile** (1.5 ms). The chunk must be sized from the commanded on-time
(target ~1/10 of it), derived from a startup calibration rather than a constant.

### D2. Our declared MSRV permits a build where every burst shape silently no-ops
`Cargo.toml` declares `rust-version = "1.74"`. Rust gained high-resolution waitable-timer
sleeps on Windows in **1.75**; below that the default timer resolution is ~15.6 ms and
`Burst`'s 2 ms `MAX_NAP` becomes fiction. We build with 1.97 today, so this is latent — but
the declared floor allows a build whose shapes don't run. Bump the MSRV with a comment
explaining *why* (shape fidelity), or call `timeBeginPeriod(1)` ourselves. Note that since
Windows 10 2004 `timeBeginPeriod` is per-process — it cannot be inherited.

### D3. Shape fidelity is never verified
We already stamp `BurstOn`/`BurstOff` edges into the marker log. **Post-process it and
assert the achieved period and duty are within tolerance of the commanded ones.** This costs
almost nothing, retroactively validates every existing profile, and is a direct application
of our own rule (never trust a timing, always verify). D1 and D2 both become self-detecting.
Same treatment for `c-states`, whose own doc comment admits it cannot confirm the core ever
reached a deep state.

### D4. The CPU verification is self-referential
`fma_chunk`'s dual-bank compare runs the same instructions, on the same core, from the same
seeds. It catches transient bit flips well and is structurally **blind to a systematically
wrong core** — both banks would be wrong identically. Two cheap fixes: bake a **golden
constant** for a fixed `(seed, N)`, and **compare per-core checksums against each other**
(today `run` sums them, discarding the equality check that is sitting right there).

### D5. GPU transient errors are overwritten before they're ever seen
`verify_every: 32` means a corrupted frame in frames 1–31 is replaced and never checked.
**Chain each frame's checksum into the next** (fold into a persistent `R32Uint` history
target) and any single-frame corruption becomes permanent and is caught by every subsequent
verification. This raises the effective error-detection duty cycle from ~3% to ~100% for
about twenty lines. Best change-per-line in the entire review.

### D6. Storage `--path` defaults to the current directory
So `storage` may be testing whichever volume the binary happens to be run from, not the
drive the technician means. Audit and report the resolved volume explicitly.

---

## 2. Coverage holes — silicon and subsystems with *zero* tests

| Hole | Why it matters | Cost |
|---|---|---|
| **The uncore is idle in every test we have** | Our CPU kernel is register-resident: no memory traffic, no cross-core traffic. The L3, ring/mesh, Infinity Fabric and memory controller are at idle for the whole run. FCLK/IF instability is one of the top real "passes every stress test, crashes in games" causes, and it logs **WHEA-19** rather than crashing. | S |
| **GPU atomic units** | Used by every real engine (light culling, GPU sort, visibility buffers). Also the strongest possible verification: the final counter **must** equal `threads × iterations` — exact, cross-vendor, bakeable. | XS |
| **GPU special-function units** | `rsqrt/exp2/log2/sin/rcp` are separate hardware from the FMA pipes and are used by every real shader. | XS |
| **We test the wrong DX12 shader path** | wgpu's `Dx12Compiler::Auto` falls back to **FXC → DXBC (SM 5.1)** when DXC isn't present — the legacy path. Modern DX12 games ship **DXIL/SM 6.x**, a different driver front-end entirely. | XS–S |
| **Throughput degradation is invisible** | GDDR EDC retries and thermal droop cost *bandwidth*, not correctness — our VRAM test passes a card that is 20% slow. Exactly the insight `pcie-plan.md` already records ("a bad link retries rather than slows"). Add per-model floors. | XS |
| **Random-access memory faults** | Our RAM battery is a cell-level pattern set run over *sequential* streams: prefetchable, row-buffer-friendly, few TLB entries. Marginal `tRC`/`tFAW`/`tRFC`/bank-group timings pass it and fail in-game. | M |
| **What the machine actually is** | No DIMM/slot enumeration, no PCIe link width/gen, no display mode vs EDID, no driver census, no event-log plane. Single-channel RAM, XMP not applied, a GPU at x8, a 165 Hz panel running at 60 Hz — all invisible, all common. | S |
| **Video encode (NVENC)** | A whole silicon block. Real gaming builds stream; a dead encoder is a genuine RMA. Reachable via Media Foundation MFTs — no vendor SDK. | L |

---

## 3. Ranked build order (cross-lane)

Effort: XS &lt;½ day · S ~1 day · M 2–5 days · L 1–3 wk.
Class: **STR**ess · **BEN**chmark · **TEL**emetry · **MET**hodology · **FIX**.

| # | Item | Lane | Effort | Class | What it buys |
|---|---|---|---|---|---|
| 1 | Shape self-verification from the marker log (D3) + adaptive chunk (D1) + MSRV (D2) | CPU | XS–S | FIX | Makes every existing profile trustworthy |
| 2 | Temporal checksum chaining (D5) | GPU | XS | FIX | Error detection ~3% → ~100% |
| 3 | Golden constant + cross-core compare (D4) | CPU | S | FIX | Catches a systematically wrong core |
| 4 | **Config/BOM audit** — DIMM slots+speed (SMBIOS T17), PCIe link width/gen, display mode vs EDID, ReBAR/power-plan/HAGS/TdrDelay, driver versions | System | S | TEL+MET | Catches more real complaints than any new stress test |
| 5 | **Event-log detector plane** — TDR 4101, Kernel-Power 41, BugCheck 1001, disk 153/129, WHEA, per-phase bracketed | System | S | TEL | The whole "black screen / random reboot / drive vanished" family |
| 6 | **Cross-core SPSC verify matrix** | CPU | S | STR+TEL | The uncore hole. Reports the failing *core pair* |
| 7 | Atomic-contention kernel | GPU | XS | STR+BEN | Dead silicon block + exact cross-vendor golden |
| 8 | SFU / transcendental kernel | GPU | XS | STR+BEN | Dead silicon block |
| 9 | Throughput-floor gates on render/vram/link | GPU | XS | BEN | Degradation-without-corruption |
| 10 | **NVML telemetry plane** — incl. `HwPowerBrakeSlowdown` = *external power-brake assertion, e.g. by the PSU* | System | S | TEL | **A marginal PSU asserting protection, in software, without a shutdown** |
| 11 | Latency/jitter passenger — 1 kHz timer, wake-latency histogram, runs under every phase | CPU/Sys | XS | TEL | DPC/ISR latency: the #1 cause of "random stutter" |
| 12 | Backend + shader-compiler matrix (`--backend dx12\|vulkan\|gl`, force DXC) | GPU | XS–S | STR+BEN | Tests the DXIL path games actually ship |
| 13 | Injected-module + kernel-driver census, plus overlay A/B | System | S | TEL+BEN | Quantifies overlay/audio-middleware stutter, hands over a one-line fix |
| 14 | NVMe SMART bracket (reuses `drives.rs` IOCTL pattern) | Storage | S | TEL | Thermal throttle, media errors, "new" drive with hours on it |
| 15 | `psostorm` — PSO / shader-compile storm | GPU | S–M | STR+BEN | The #1 real-world bad-frame source; doubles as a CPU/RAM test |
| 16 | `drawstorm` — draw-call + state-change storm | GPU | S | STR+BEN | The CPU-side driver cost that makes games CPU-bound |
| 17 | Random-access storage mode (QD sweep, per-thread handles) | Storage | M | STR+BEN | SLC write cliff, thermal throttle, controller stalls |
| 18 | Pointer-chase memory verify (cycle-closure invariant) | CPU | M | STR+BEN | Latency/TLB/bank-timing faults the sequential battery misses |
| 19 | **Thermal-equilibrium gating + headroom pass criteria** | System | M | MET | The "fails after 40 minutes" class |
| 20 | VRAM oversubscription / residency churn | GPU | M | STR+BEN | "Runs fine until VRAM fills, then stutters" |
| 21 | Mixed-ISA "game frame" CPU kernel | CPU | M | STR | "Passes Prime95, crashes in-game" |
| 22 | `shooter` profile — 1–2 cores at max boost, 300–800 Hz transients, rest idle | CPU | S | STR | A distinct electrical corner nothing else reaches |
| 23 | **`watch` — wrap a real game/benchmark** with our whole detector stack | System | M | MET+BEN | 100% fidelity; reuses `presentmon.rs` |
| 24 | Vsync-locked P-state churn | GPU | S | STR | GPU analogue of `c-states`; display provides the timebase |
| 25 | `frame` — a real deferred frame graph (depth prepass → shadows → G-buffer MRT → SSAO → volumetrics → lighting → transparency → post → UI) | GPU | L–XL | STR+BEN | Each pass loads different silicon; build incrementally |
| 26 | `session` profile — hours-long seeded varied timeline | System | M | STR+MET | State-transition and long-tail faults |
| 27 | Cold-boot / sleep-resume cycling (harness-side) | System | M | MET+STR | "Sometimes it won't POST" — marginal memory training |
| 28 | Memory methodology bundle — bit-signature classification, coverage accounting, contention grading | System | S–M | MET | Turns "mem FAIL" into "RMA this stick" vs "back off XMP" |
| 29 | Gateway ICMP with verified payload (idle-then-burst) | Network | M | STR+TEL | First thing that honestly deserves the word "network" |
| 30 | NVENC coverage via Media Foundation MFT | System | L | STR | "Streaming crashes" — untested silicon block |

---

## 4. Lane notes worth carrying forward

### 4.1 Verification upgrades that apply everywhere
- **Exact/analytic goldens beat self-consistency.** Integer shader math, `Rgba8Uint`/`R32Uint`
  targets, point sampling at mip 0, atomics — all spec-exact across vendors. This makes a
  **cross-vendor graphics golden achievable**, which `gpu-functional-units.md` currently says
  it is not. Keep triangle edges away from pixel centres to dodge fill-rule differences.
- **Per-pass checksums, not one final hash.** A multi-pass frame must name the failing pass,
  mirroring the existing first-fail model ("chunk 0 word 1000 [own-address]").
- **Structural invariants** need no reference at all: G-buffer normals unit-length, AO ∈ [0,1],
  bloom mip-chain energy conservation, analytically-known coverage counts.
- **Negative controls for every detector.** `vram` already fault-injects — generalize it. A
  detector that never fires on its own control is decoration.
- **Two new ways to time a no-op** (the 1.65 TFLOP/s lesson, restated): D3D12 drivers may
  legitimately **defer PSO compilation** and background-recompile at idle priority, and they
  may **coalesce redundant state changes**. So every pipeline created must be drawn with and
  verified, and every state change must be observable in the output.

### 4.2 Load-edge realism — our burst model covers one band of three
| Band | Source | Coverage today |
|---|---|---|
| 0.01–10 Hz | level load, alt-tab, shader compile, streaming hitch | **none** |
| 60–800 Hz | frame cadence | partial (`game-load` square wave) |
| 10–100 kHz | job completions, spin-wait exits, cache-miss stalls | **none** |

`Jitter` draws uniformly over ~4–50 ms (≈20–250 Hz) — a band VRM control loops are designed
for. A **pink-noise (1/f) draw** extending sub-Hz through kHz is ~20 lines inside
`jitter_interval`, preserves the pure-function-of-`(seed, index)` property that keeps threads
synchronized, and excites the VRM loop, the boost algorithm, the heatsink's thermal mass and
the PSU hold-up simultaneously. Highest value-per-line in the CPU lane. A **burst-frequency
sweep** (5 Hz → 2 kHz logarithmic, constant duty) is the natural companion for the power rig,
since VRM loop instability and bulk-cap resonance are frequency-dependent.

### 4.3 What the report should say
Not a score — a verdict, then one action:

```
VERDICT   PASS WITH CONCERNS      margin: B     device 7F3A-…
ACTION    Remount the CPU cooler and re-run phase E4.
WHY       CPU clock retention 74% after 30 min soak (cohort median 96%, n=41).
          Throttle onset at 00:11:42. No integrity errors, no WHEA.
EVIDENCE  phase E4, QPC 41,882,193,004 → markers.jsonl seq 18,442
```

Plus a fourth verdict state: **INVALID**. A run where PresentMon didn't attach, NVML was
unavailable, or the WHEA scan failed is not a pass. `gauntlet.md` already gets this right for
WHEA; generalize it, and have every report state which detector planes were live.

### 4.4 Grading instead of pass/fail
Sweep-to-failure (report the *rung reached*), error-rate curves rather than error presence,
and frame-time dispersion as a leading indicator (it degrades before hard failure — and
`frame_timer.rs` already computes the percentiles). A memory contention ladder — Class A
(passes hot, post-equilibrium, under full contention) down to Class D (fails cold solo) —
gives a real margin scale from stimuli we already have. **Do not** have the tool mutate
voltages or clocks to find margin.

---

## 5. Not worth doing, dishonest to claim, or needs hardware

**Not worth doing.** DLSS/NGX (closed SDK, and the tensor cores are already covered) ·
porting FSR2/3 (plain compute shaders — stresses nothing new) · FSR4 (RDNA4-only, closed
weights) · XeSS FFI (XMX already reached more directly by `tensor.rs`) · hardware
tessellation and geometry shaders (dead silicon, no wgpu support) · a from-scratch D3D11
renderer (the GL backend is the same *shape* of driver for ~2% of the cost, and is already in
the dependency tree) · SLI · emulating a specific engine's job graph (build the shape, not the
brand) · a GC-pressure simulator (C++ engines preallocate; it's just random access) · rowhammer
patterns (security research, not a QC screen) · in-tool ETW consumer (PresentMon + event log
cover it) · HWiNFO shared memory (**Pro licence required and the SDK terms conflict with a
public MIT repo**) · LibreHardwareMonitor as an external process (still installs a
WinRing0-class driver — same refusal as MSR) · emulating overlay/anti-cheat *injection*
(indistinguishable from malware; the census covers the real thing) · auto-clearing the driver
shader cache (user-data mutation — document it as a bench step or make it a loud opt-in).

**Dishonest to claim.** That software measures ms-scale transients (the rig does) · that a
failing virtual address identifies a DIMM (needs a kernel driver) · "we tested your PSU" (we
*stressed* it; only the rig — and an NVML power-brake assertion — says more) · "memory
validated" without stating the coverage fraction (kernel/nonpaged pool is untestable from user
mode; report `covered / GetPhysicallyInstalledSystemMemory()`) · vendor coverage without vendor
hardware · PASS on a run where a detector plane was unavailable · passing a build because it
"didn't throttle at 22 °C bench ambient" · that SMART counters prove anything on a *new* drive
(their value is the inverse test: nonzero hours on a supposedly-new drive) · that our render
pacing delta predicts a specific game's stutter (it's evidence of interference, not prediction).

**Needs hardware we don't have**, by value per dollar: ambient/intake temp probe (~$20 — makes
every thermal number comparable across days and seasons) · microphone (~$20 — coil-whine
frequency sweep) · instrumented 12VHPWR adapter (~$60–100 — per-pin imbalance is the actual
melting mechanism and most cards don't report it) · capture card or photodiode (the only honest
way to detect black-screen/scanout events) · thermal camera · a known-bad riser + Gen5 board
(already on the `pcie-plan.md` list) · an AMD GPU (Phase 3e is blocked without one).

---

## 6. Phased plan

**Phase A — trustworthiness (≈1 week).** Items 1–3 (the defect fixes), 7, 8, 9, 2. Everything
we already report becomes trustworthy, error detection goes to ~100%, and two dead silicon
blocks get covered. No new dependencies.

**Phase B — know the machine (≈1 week).** Items 4, 5, 10, 11, 13, 14. Nearly all telemetry, all
deterministic, no new kernels — and between them they catch more real customer complaints than
any stress profile we could add. Lands as a new report section rather than a new load.

**Phase C — game-shaped loads (2–3 weeks).** Items 6, 12, 15, 16, 17, 18, 21, 22, plus the
pink-noise shape. This is where the workloads start failing the way games fail.

**Phase D — session + fidelity (2–4 weeks).** Items 19, 20, 23, 24, 26, 27, 28. Thermal
equilibrium gating changes what "pass" means; `watch` gives 100% fidelity on the exact title a
customer says is crashing.

**Phase E — the long poles.** Item 25 (`frame`, built incrementally: depth prepass + G-buffer
first), 29, 30.

Most of Phase C onward composes through `mix` (`composable-runs.md`) rather than needing new
profile plumbing — a game-shaped run is a composition of a presenting render, a bursty CPU, a
random-access storage load and a latency passenger, which is exactly what `mix` expresses.
