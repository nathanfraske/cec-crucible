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

## 2. Backend decision (the first spike)

The rest of the suite is zero-dependency std-only. `crucible-gpu` is the
**documented exception** — it will pull external crates. Kept in its own crate
and out of the default build so the core suite stays dep-free and offline-buildable.

| Option | Vendors | Language | License | Notes |
| --- | --- | --- | --- | --- |
| **wgpu** (Vulkan/DX12 compute, WGSL) | NVIDIA/AMD/Intel/Apple | Rust | MIT/Apache-2.0 | Mature, cross-vendor, MIT-compatible. **The safe floor.** |
| **CubeCL** (Burn ecosystem) | wgpu/CUDA/ROCm backends | Rust | permissive | Nicer compute ergonomics; backend maturity varies per vendor. |
| Vendor-native (CUDA / ROCm / oneAPI) | one vendor each | C/Rust FFI | mixed | Max control, worst portability + redistribution friction. |

**Recommendation:** start on **wgpu** (DX12 or Vulkan compute) — it is one
codebase across all three desktop vendors and its license (MIT/Apache-2.0) is
compatible with this repo's MIT. Evaluate CubeCL in parallel for kernel
ergonomics, but treat wgpu as the fallback that must always work.

**Spike 3a (1–3 days):** a WGSL compute "hello-thrash" that dispatches an
FMA-heavy kernel in a loop on each of NVIDIA/AMD/Intel and confirms (via vendor
tools) it pins the card at its power limit. Gate the backend choice on this.

Every external dependency's license is audited before adoption — the whole point
is to *stay* license-clean for commercial QC.

## 3. Crate structure

```
crates/crucible-gpu/         (new workspace member; NOT default-built)
  Cargo.toml                 wgpu (+ pollster or async runtime), vendor telemetry
  src/
    lib.rs                   GpuKernel: LoadKernel impl (thrasher + shapes)
    adapter.rs               enumerate adapters, pick discrete GPU, caps
    thrasher.wgsl            FMA + bandwidth compute shader
    vram.rs                  VRAM pattern test (buffers, verify)
    power/                   per-vendor power readback (see §5)
    servo.rs                 closed-loop wattage controller + calibration
    tdr.rs                   device-lost detection / recovery
```

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

- **3a — backend spike:** WGSL compute pins power on all 3 vendors → pick wgpu vs CubeCL.
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

1. wgpu vs CubeCL — decide after spike 3a on all three vendors.
2. How much power-telemetry to own via vendor `dlopen` FFI vs. leaning on
   LibreHardwareMonitor (HVCI/WinRing0 blocklist risk on the shop image).
3. TDR policy: chain-short-dispatches only, or also document registry tuning for
   bench machines?
4. Integrated GPUs and laptops — in scope, or discrete-only for QC?
5. Do we verify GPU compute correctness against a CPU reference tile, or
   GPU-internal recompute only?
