<!-- SPDX-License-Identifier: MIT -->

# cec-crucible — GPU test plan (Phase 3)

The GPU power-virus is the one multi-week piece of the suite and the piece worth
owning most: **no open, license-clean Windows GPU power-virus exists** (FurMark
et al. are closed; gpu-burn is CUDA/Linux). This document is the design of
record for `crucible-gpu`, to be built once the Phase 1/2 CPU/RAM/storage engine
is in production.

Everything here plugs into the primitives already built in `crucible-core`: the
`LoadKernel` trait, `StopFlag`, `ShapeDriver` (steady/burst), the QPC `MarkerLog`,
and the device-ID'd `Report`. The GPU is "just another kernel" to the
orchestrator — it joins `run cross` under the same one-stop/one-timeline model.

## 1. Goals

1. **Thrasher / power-virus** — pin any modern GPU at its power limit with a
   custom compute kernel (FMA + VRAM-bandwidth mix). On boost-limited cards the
   board's own limiter does the tuning once the kernel is well-occupied.
2. **VRAM test** — memtest-style pattern coverage across allocated VRAM, with
   read-back verification (data-integrity, not just heat).
3. **Wattage servo** — closed-loop control that holds a commanded power profile
   (hold / step / ramp / sine sweep) by modulating the load mix, plus a
   self-calibration pass that finds and stores each GPU model's max-sustained
   power point.
4. **Cross-load** — GPU burst trains while the CPU is pinned, storage churns,
   etc. — the worst-case transients that kill marginal builds.
5. **License-clean, multi-vendor (NVIDIA / AMD / Intel), Windows-first**, so it
   can be used on paid customer QC without the licensing maze of the closed
   tools.

## 2. Backend decision — **CubeCL** (decided)

The rest of the suite is zero-dependency std-only. `crucible-gpu` is the
**documented exception** — it will pull external crates. Kept in its own crate
and out of the default build so the core suite stays dep-free and offline-buildable.

| Option | Vendors | Language | Notes |
| --- | --- | --- | --- |
| **CubeCL** (Burn ecosystem) — **chosen** | CUDA / ROCm / wgpu runtimes | Rust | Already CEC's multi-vendor GPU backend in another in-house project — shared toolchain and debugging experience. Adds a native CUDA path on top of wgpu. |
| **wgpu** (Vulkan/DX12 compute, WGSL) | NVIDIA/AMD/Intel/Apple | Rust | **Not an alternative — it is one of CubeCL's runtimes.** Remains the portable floor underneath. |
| Vendor-native (CUDA / ROCm / oneAPI direct) | one vendor each | C/Rust FFI | Max control, worst portability + redistribution friction. Not worth it. |

**Decision: build on CubeCL.** Rationale:

1. **Toolchain consistency** — it is already the multi-vendor GPU backend in
   another CEC project, so the kernel language, debugging workflow, and failure
   modes are known quantities rather than new risk.
2. **It is a superset, not a gamble** — wgpu is a CubeCL runtime. Choosing
   CubeCL keeps the portable floor *and* adds a native CUDA path. If a backend
   disappoints on some vendor, drop that vendor to the wgpu runtime; the
   downside is bounded to "what we would have had anyway."
3. **The CUDA path helps the hard part** — the open risk is not "does it run,"
   it is "does it actually pin the board at its power limit." A native CUDA
   runtime can push NVIDIA harder than portable WGSL, and NVIDIA is the majority
   of the builds this shop ships.

### Windows routing reality

On **Windows specifically**, CubeCL's multi-backend story narrows to
**CUDA + wgpu**, because ROCm/HIP on Windows is limited:

| Vendor | Windows runtime | Consequence |
| --- | --- | --- |
| NVIDIA | CUDA (native) | Best throughput; most likely to reach the true power limit. |
| AMD | wgpu (DX12/Vulkan) | Same characteristics as a plain-wgpu build. |
| Intel | wgpu (DX12/Vulkan) | Same; validate Arc first (weakest driver stack). |

So the wgpu-path caveats still apply to AMD/Intel: **prefer the DX12 backend**
(it is what Microsoft WHQL-certifies across vendors), **query adapter limits**
rather than hardcoding workgroup sizes, and expect naga/WGSL translation to be
the thing that occasionally surprises you. TDR (§6) is an OS watchdog and
applies to every backend equally.

**Deployment: ANSWERED by spike 3a (see §12).** The CUDA runtime **builds**
without a CUDA toolkit but **does not run** without one — CubeCL JIT-compiles
kernels through NVRTC, so `nvrtc.dll` (a *toolkit* component, not a driver
component) must be present at runtime. Consequence: **ship the wgpu runtime as
the default build**; CUDA is an opt-in cargo feature for bench machines, or
requires redistributing NVRTC (verify the NVIDIA EULA redistributable list and
the added binary size before going that way).

**Spike 3a — DONE.** Results and measurements in §12.

Every external dependency's license is audited before adoption (CubeCL and wgpu
are both permissive/MIT-compatible — confirm at the pinned version) — the whole
point is to *stay* license-clean for commercial QC.

## 3. Crate structure

```
crates/crucible-gpu/         (new workspace member; NOT default-built)
  Cargo.toml                 cubecl (wgpu runtime default, cuda behind a feature),
                             vendor telemetry
  src/
    lib.rs                   GpuKernel: LoadKernel impl (thrasher + shapes)
    adapter.rs               enumerate devices, pick discrete GPU, caps/limits
    thrasher.rs              FMA + bandwidth compute kernel (CubeCL #[cube], Rust)
    vram.rs                  VRAM pattern test (buffers, verify)
    power/                   per-vendor power readback (see §5)
    servo.rs                 closed-loop wattage controller + calibration
    tdr.rs                   device-lost detection / recovery
```

Kernels are written in **Rust** via CubeCL's `#[cube]` macro rather than in
WGSL, which keeps the thrasher and the VRAM battery in the same language as the
CPU/mem kernels they mirror.

`GpuKernel` implements `LoadKernel` (`kind() == Kind::Gpu`), so it flows through
the existing orchestrator, `ShapeDriver`, `StopFlag`, and `MarkerLog` unchanged.
The `crucible-gpu` dependency is added to `crucible-cli` behind a `gpu` cargo
feature so `cargo build` without the feature stays std-only.

## 4. Kernel designs

### 4.1 Thrasher (power-virus)
- Persistent compute: many back-to-back dispatches of a WGSL kernel doing wide
  FMA accumulation interleaved with strided VRAM reads/writes (bandwidth
  pressure). A tunable FMA:memory ratio moves the load between "ALU-bound" and
  "bandwidth-bound," which matters for hitting the true power ceiling.
- **Load shapes:** reuse `Shape` — steady (continuous dispatch) and burst
  (dispatch trains with idle gaps). Burst edges are stamped into the marker log
  exactly like the CPU kernel, for the 1kHz rig.
- **Error detection:** the kernel writes a reduction/checksum per workgroup;
  compare across dispatches (and optionally against a CPU reference for a small
  tile). A GPU that computes the wrong answer under load is a FAIL — WHEA does
  not see this.

### 4.2 VRAM test
- Allocate a configurable fraction of free VRAM as storage buffers; run the
  `crucible-mem`-style battery (moving inversions, own-address, seeded random)
  in compute shaders; verify on read-back. `memtest_vulkan` (open) is the
  reference to match.
- Report first-fail offset + pattern, mirroring the CPU/mem/storage first-fail
  reporting already in the report model.

### 4.3 Wattage servo (closed loop)
- Read board power (see §5) at ~100 ms–1 s and drive a controller (start with
  bang-bang, move to PID) that adjusts **occupancy, FMA:memory ratio, and duty
  cycle** to hold a commanded target: `hold@N W`, staircase, ramp, sine sweep.
- **Self-calibration:** a search over the load-mix space to find the
  max-sustained-power operating point for the specific GPU model, stored
  device-ID'd so later runs start pre-tuned.
- **Reality check:** the software servo reacts at ~100 ms–1 s — fine for
  *holding* a target. The millisecond-scale transient attacks stay open-loop
  burst scenarios, judged by the external 1kHz rig against our QPC markers.

### 4.4 Cross-load
- `GpuKernel` joins the existing concurrent orchestrator (the `cross` / `power`
  profiles) under one `StopFlag` and one marker timeline — CPU pinned at full
  FMA while the GPU runs a burst train, etc.

## 5. Power & telemetry (per-vendor)

| Vendor | API | Access | Notes |
| --- | --- | --- | --- |
| NVIDIA | **NVML** (`nvml.dll`, ships with driver) | `dlopen` at runtime, no redistribution | power draw, limits, temps, clocks, throttle reasons. Best-supported. |
| AMD | **ADLX** / ADL | driver SDK | power/temps; API churn — pin a version. |
| Intel | **IGCL** / Level Zero Sysman | driver | newer; validate per generation. |
| Any | **LibreHardwareMonitor** (MPL-2.0) | lib/WMI | fallback; loads a WinRing0-lineage driver — **HVCI/vulnerable-driver-blocklist risk**; validate on the shop's HVCI image before relying on it. |

Strategy: prefer the vendor-native runtime API via `dlopen` (no SDK
redistribution, degrade gracefully when absent). NVML first (highest ROI), then
AMD, then Intel. Power reading is *required* for the servo but *optional* for
the thrasher (the board's own limiter still pins it without us reading watts).

## 6. TDR & robustness (the real work)

Windows **Timeout Detection & Recovery** resets the GPU if a single command
takes longer than ~2 s (default `TdrDelay`). A naive long-running dispatch will
trip it and kill the test.

- **Keep individual dispatches short** (well under `TdrDelay`) and chain many of
  them to sustain load — a "persistent kernel" built from many small dispatches.
- **Detect device-lost** (`DXGI_ERROR_DEVICE_REMOVED` / wgpu `DeviceLost`) and
  report it as a specific, named failure rather than a crash.
- Optionally document (with caution, not auto-apply) `TdrDelay`/`TdrDdiDelay`
  registry tuning for dedicated bench machines.
- Per-vendor driver-timeout behavior differs — part of the validation matrix.

## 7. Marker integration

Same `MarkerLog` as the rest of the suite: stamp QPC markers at dispatch-train
burst edges, stage start/stop, and every wattage-target transition. The 1kHz
analog capture aligns on `qpc_ticks` — no change to the marker schema.

## 8. Validation matrix

Real hardware of each vendor is required. At minimum:

- NVIDIA (2× generations, e.g. Ampere + Ada), AMD (RDNA2 + RDNA3), Intel (Arc).
- Per card: thrasher pins power limit ✓, VRAM test clean ✓, servo holds target
  ✓, no TDR over a 30-min soak ✓, telemetry reads correctly ✓.
- Add a per-generation smoke test to CI-adjacent manual validation as new GPUs
  land in the shop.

## 9. Milestones (within Phase 3)

- **3a — backend spike:** CubeCL compute kernel pins power on all 3 vendors (CUDA on NVIDIA, wgpu on AMD/Intel) + runs with no toolkit installed.
- **3b — thrasher:** `GpuKernel: LoadKernel`, steady + burst, markers, TDR handling, error check.
- **3c — VRAM test:** pattern battery + read-back verify + first-fail.
- **3d — power + servo:** NVML readback → wattage servo (hold/step/ramp/sweep) → per-model self-calibration.
- **3e — integration:** GPU in `run cross`; the `power` characterization profile with dense markers for the rig.
- Rolling: per-generation smoke tests.

Estimate: ~1–3 weeks for a v1 thrasher + VRAM + basic servo on one vendor, plus
ongoing per-vendor/per-generation validation. The graphics robustness (TDR,
per-vendor telemetry, multi-arch) is the bulk of the effort, not the compute
kernel itself.

## 10. CLI surface (planned)

```
cec-crucible gpu-info                      # enumerate adapters, VRAM, power caps
cec-crucible gpu --seconds 60              # thrasher, steady
cec-crucible gpu --seconds 60 --shape burst --burst-on 30 --burst-off 30
cec-crucible gpu --vram                    # VRAM pattern test
cec-crucible gpu --watts 250               # hold 250 W (servo)
cec-crucible gpu --calibrate               # find + store max-power profile
cec-crucible run cross                     # CPU + RAM + storage + GPU concurrently
cec-crucible run power                     # CPU + GPU burst, dense markers for the rig
```

All device-ID'd and marker-emitting, consistent with the existing commands.

## 11. Open questions

1. ~~wgpu vs CubeCL~~ — **decided: CubeCL**. ~~Is the CUDA runtime usable on a
   machine with no toolkit?~~ **Answered by spike 3a: no** — it builds without a
   toolkit but needs `nvrtc.dll` at runtime. Ship wgpu by default. See §12.
2. How much power-telemetry to own via vendor `dlopen` FFI vs. leaning on
   LibreHardwareMonitor (HVCI/WinRing0 blocklist risk on the shop image).
3. TDR policy: chain-short-dispatches only, or also document registry tuning for
   bench machines?
4. Integrated GPUs and laptops — in scope, or discrete-only for QC?
5. Do we verify GPU compute correctness against a CPU reference tile, or
   GPU-internal recompute only?

## 12. Spike 3a results (measured 2026-07-24)

Run on the shop bench: **RTX 3070** (240 W limit, 264 W max) + **Intel UHD 630**
iGPU, Windows 11, CubeCL 0.10.0, driver 591.86, **no CUDA toolkit installed**.
Spike source: [`spikes/gpu-3a/`](../spikes/gpu-3a/). Idle board power 43.9 W.

### It works, on two vendors, unmodified

| Runtime | Device | Dispatch | Result |
| --- | --- | --- | --- |
| wgpu | RTX 3070 (discrete) | 27.0 ms | 5.58 TFLOP/s, output verified |
| wgpu | UHD 630 (integrated) | 26.3 ms | 0.23 TFLOP/s, output verified |

Same binary, same kernel, only a device flag differs. Kernel compile (naga →
DX12) was 58–124 ms. **CubeCL on Windows is confirmed for NVIDIA + Intel.** AMD
remains untested — no AMD hardware on this bench.

### TDR is a non-issue when dispatches are chained

Longest single dispatch **53.5 ms** against the ~2000 ms watchdog — a ~40×
margin. ~1900 dispatches across all runs, **zero device-lost events**. The
"chain many short dispatches" design in §6 is validated.

### Wattage test: memory traffic is what gets you to the limit

The decisive measurement. Identical FMA core, with and without a coalesced VRAM
stream (1 GiB buffer, stride = thread count):

| | pure ALU | ALU + VRAM |
| --- | --- | --- |
| Board power | 167–187 W | **208–221 W** |
| % of 240 W limit | ~75% | **~92%** |
| SM utilization | 71–95% | **100%** |
| Memory utilization | 4–5% | **100%** |
| Throughput | 5.58 TFLOP/s | 1.35 TFLOP/s |

Two conclusions:

1. **Pure ALU tops out around 75%.** GDDR and the memory controller are a large
   share of board power; a FLOP-maximizing kernel is *not* a watt-maximizing
   kernel. Throughput moved **inversely** to power (5.58 → 1.35 TFLOP/s while
   watts rose ~35 W). For the wattage test, TFLOP/s is not the objective
   function — watts are.
2. **The FMA:memory ratio knob (§4.3) is necessary, and 92% came from a naive
   first guess.** Closing the last ~8% is exactly the job of the per-model
   calibration pass. Also note coalesced streaming beats scattered access here:
   sustained bandwidth drives memory power, whereas a large scattered stride
   maximizes stalls but lowers achieved bandwidth.

This is the *wattage* test only. The **VRAM integrity test (§4.2) is a separate
test** with a different objective — pattern write/verify to find bad memory, not
to maximize watts. Do not conflate the two.

### fp32 is bit-identical across NVIDIA and Intel

With identical inputs both vendors produced exactly `1.0813978`. So a
**cross-vendor golden checksum is viable** for GPU error detection — it need not
be limited to same-device self-consistency. Caveat: this was a simple FMA
recurrence; transcendentals or any fast-math path would need re-checking.

### CUDA runtime: builds without a toolkit, does NOT run without one

- `cargo build --features cuda` **succeeded** on this toolkit-less machine.
- At runtime it **panics**: cannot load `nvrtc.dll`. CubeCL JIT-compiles kernels
  via NVRTC, which ships with the **CUDA toolkit**, not the driver.

**Decision: ship the wgpu runtime as the default build.** CUDA becomes an opt-in
cargo feature for bench machines, or requires redistributing NVRTC (check the
NVIDIA EULA redistributables and the size cost first). This does not weaken the
CubeCL choice — wgpu was always the floor — but it does mean the NVIDIA-native
fast path is **not** free for a ship-anywhere binary.

### The most important finding: the failure was silent

When NVRTC was missing, the panic occurred on a CubeCL **worker thread**. The
host loop kept going and reported a completely plausible result:

```
dispatches      : 772
elapsed         : 32.08 s
throughput      : 1.65 TFLOP/s (fp32 FMA)     <-- for work that never executed
```

The GPU sat at 43 W (idle) the entire time. Only the output read-back caught it
(`out[0] = 0`, nonzero = false).

**Design requirement for `crucible-gpu`:** every run must verify kernel output
and treat a device/compile error as a hard FAIL. Timing and dispatch counts can
report a confident lie. This is the GPU instance of the suite's existing
principle — *a pass that computed the wrong answer is still a FAIL* — and it is
now empirically justified rather than assumed. A QC gate that cannot tell "ran
and passed" from "never ran" is worse than no gate.

### What 3b should carry forward

- Keep the chained-short-dispatch schedule; it is TDR-safe with huge margin.
- Build the wattage test around a tunable FMA:memory ratio, defaulting to mixed,
  and implement calibration to close the last ~8%.
- Verify output every run; treat worker-thread panics and device-lost as FAIL.
- Default build = wgpu runtime. Gate CUDA behind a feature.
- Get AMD hardware in front of this before committing to per-vendor claims.

## 13. Phase 3b results — `crucible-gpu` built (2026-07-24)

The GPU is now a real [`LoadKernel`], so the orchestrator drives it with the same
`StopFlag`, `ShapeDriver` and QPC `MarkerLog` as every other domain. Measured on
the RTX 3070 bench.

### Packaging: one binary, still zero-dep by default

`crucible-gpu` is a workspace member (so it ships *inside* `cec-crucible`) but is
excluded from `default-members` and gated behind a `gpu` cargo feature:

```
cargo build --release                                   # core, 0 external deps
cargo build --release -p crucible-cli --features gpu    # full shipped binary
```

Verified: `cargo tree -p crucible-cli` lists **only** `crucible-*`; with
`--features gpu` it pulls 455 dependency entries. `Cargo.lock` is now committed
so the default build stays resolvable (and offline-buildable) even though an
optional CubeCL dependency exists in the manifest.

### Steady tuning is worth ~40 W

Steady and burst want *opposite* tuning, and getting it wrong silently costs a
fifth of the power budget:

| | small dispatches, sync every one | batched, large dispatches |
| --- | --- | --- |
| Board power | 166–184 W (~75%) | **215–221 W (~92%)** |
| SM / mem util | 70–91% / 60–79% | **98–100% / 96–100%** |
| Throughput | 0.70 TFLOP/s | **1.33 TFLOP/s** |

So the kernel now picks per shape: **steady** batches 4 dispatches per sync at
4096 iters (max sustained power), **burst** uses 1 dispatch at 256 iters (~6 ms,
fits inside a 20 ms ON window so the edge stays sharp). `--gpu-iters` overrides.

### Verification is live and clean

Every run checks liveness (finite + non-zero) and self-consistency (the kernel is
deterministic, so the checksum must reproduce bit-for-bit). Across all runs:
**0 errors**, e.g. 620 dispatches / 38 verifications on a 32 s steady run. This is
the guard against the 3a failure mode where a dead kernel reported 1.65 TFLOP/s.

### Phase control: a real defect, found by measuring

The first anti-phase implementation used a **thread start delay**. Marker
analysis showed it did not work — the commanded 20 ms offset came out ~10 ms, and
the CPU edges smeared across half the period. Two causes:

1. Each CPU worker thread started its **own** `ShapeDriver`, so 20 cores produced
   20 unsynchronized square waves rather than one sharp system-level step.
2. A start delay is measured from when each kernel's driver starts — but GPU
   setup (client init + shader compile) is ~100 ms+, dwarfing a 20 ms delay.

Fix: `Budget` gained `phase_epoch` + `phase_offset`, so **all kernels derive
burst phase from one shared origin**. The CPU kernel now shares a single epoch
across its worker threads; the orchestrator hands every kernel the same epoch.

`burst_on` phase distribution within the 40 ms period, from the marker file:

| | 0–5 ms | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU before fix | 46% | 18% | 22% | 8% | 4% | 1% | 0% | 0% |
| GPU before fix | 0% | 0% | **99.9%** | 0% | 0% | 0% | 0% | 0% |
| CPU after fix | **79%** | 13% | 6% | 3% | 0% | 0% | 0% | 0% |
| GPU after fix | 0% | 0% | 0% | 0% | **100%** | 0% | 0% | 0% |

The GPU now lands exactly in the commanded +20 ms bucket.

### All three scenarios validated

| scenario | CPU phase | GPU phase | reads as |
| --- | --- | --- | --- |
| `in-phase` | 80% @ 0–5 ms | 100% @ 0–5 ms | aligned — peak total draw |
| `anti-phase` | 79% @ 0–5 ms | 100% @ 20–25 ms | opposite — VRM/PSU chases load |
| `beat` | 65% @ 0–5 ms | 17/10/13/9/14/12/16/10% | drifts through every alignment |

The flat distribution under `beat` is the signature of a drifting phase
relationship — the whole phase space swept without enumerating it.

Average GPU board power under anti-phase settles ~114–127 W against ~218 W
steady, consistent with the ~50% duty cycle. Note that 1 Hz `nvidia-smi` sampling
can only show that average; the 20 ms oscillation is invisible to it. **That is
exactly what the QPC markers are for** — we timestamp the commanded edges, the
external 1 kHz rig measures the electrical reality, and the two are correlated
afterwards.

### Known limitations

- **CPU edge jitter remains.** ~21% of CPU burst edges land late (5–20 ms). Cause
  is scheduler oversubscription: 20 pinned worker threads plus the GPU and
  orchestrator threads contend for 20 logical cores, so some threads notice the
  phase transition late. Leaving a core free for orchestration, or shrinking the
  CPU work chunk, should tighten it. The GPU — the dominant transient source at a
  ~180 W swing — is already exact.
- **AMD still untested.** No AMD hardware on this bench; no per-vendor claims.
- **VRAM integrity test (§4.2) is still unbuilt.** It is a separate test with a
  different objective and must not be confused with the wattage knob used here.

## 14. Phase 3c results — VRAM integrity + whole-platform worst-case (2026-07-24)

### VRAM integrity test (`vram` command)

A **separate** test from the wattage thrasher, with a different objective: find
bad video memory. It runs a chained moving-inversion battery — own-address,
0x00/0xFF/0xAA/0x55 complements, and an index-derived random pattern — entirely
on the GPU.

Design decisions that matter:
- **Index-derived patterns.** Every expected value is a pure function of the
  element index (+ pattern id / seed), so each of the millions of GPU threads
  computes its own expectation with no shared PRNG state, and verification needs
  no golden copy on the host.
- **Verify on the GPU, read back only on failure.** A check kernel compares each
  word and reports via atomics into an 8-byte results buffer (error count +
  lowest failing index). Dragging gigabytes back over PCIe every pass would make
  the test bus-bound; the host reads a full chunk back *only* when an error is
  found, to recover the observed value for the report.
- **Grid-stride loop.** A one-element-per-thread launch needs 65,536 workgroups
  for a 64 MiB chunk at 256 threads — over the 65,535-per-dimension dispatch
  limit (this was a real bug, caught on first run). The grid-stride loop
  decouples dispatch size from buffer size.
- **Chunked allocation** under the storage-buffer binding limit (wgpu default
  128 MiB); all chunks stay resident so the whole requested span is under test.

Measured on the RTX 3070: **512 MiB across 8 chunks, 149 full passes, 447 GiB
verified at ~22 GiB/s, 0 errors** on healthy VRAM. Fault-injection test (corrupt
one word's expectation in the check kernel) → **FAIL**, first-fail correctly
reported as `chunk 0 word 1000 [own-address]: expected 0x000003e8 got …` — the
`0x3e8` = decimal 1000 confirms the own-address pattern and the host/GPU mirror
agree.

Inherent caveat, stated plainly: verification runs on the same GPU being tested,
so a sufficiently broken device could mis-verify. This is intrinsic to GPU-side
memory testing (memtest_vulkan has the same property) and is the price of not
being PCIe-bound.

### Whole-platform worst-case (`run worst-case`)

Everything at once: CPU transients **anti-phase** to the GPU (so the VRMs and
PSU never settle), under simultaneous RAM, storage, GPU-thrash and
VRAM-integrity load — all under one `StopFlag`, one shared phase epoch, one
marker timeline, **every domain verifying its own data.**

Measured on the bench (25 s, all five domains concurrent): **PASS, 0 errors,
20,590 markers.** Per-domain throughput drops under contention exactly as
expected and intended — VRAM ~4 GiB/s (vs ~22 solo), storage ~207 MiB/s (vs
~660 solo) — because they are now fighting for shared buses. The point is not
any single number; it is that a corruption appearing only under full-platform
contention has nowhere to hide. Notably the GPU thrasher (burst) and the VRAM
integrity test (steady) coexisted on the *same* discrete GPU without conflict.

### What worst-case does NOT yet cover

It does not load **PCIe** — the GPU kernels are on-card compute + VRAM traffic;
the x16 link sits nearly idle. A dedicated host↔device transfer test plus
link-integrity checking is scoped separately in
[`docs/pcie-plan.md`](pcie-plan.md).
