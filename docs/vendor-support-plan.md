# iGPU / APU and AMD support — findings and implementation plan

Status: **research complete, nothing implemented.** Everything in §1 was measured on
the bench (i9-10850K + Intel UHD 630 iGPU + RTX 3070) with the shipping 0.0.4
binary. Everything in §3–§6 is a plan.

---

## 1. What the current build does on an iGPU — measured, not assumed

This machine has an integrated Intel UHD 630 alongside the discrete 3070, and
`gpu-info` already lists it:

```
GPU devices:
  discrete0 : available
  integrated0 : available
  default : available
```

So the iGPU is *selectable today*. Running the suite against it produced four
defects, two of which are false results rather than crashes.

### 1.1 SEVERE — `vram` reports a PASS on a device with no VRAM

```
cec-crucible vram --seconds 6 --gpu-device integrated --vram-mb max
  [vram   ] PASS  8.4s  (integrity integrated0)  6784 MiB VRAM (106 x 64 MiB chunks),
                                                 7.9 GiB verified, ~1.3 GiB/s
```

**6784 MiB is 85% of the RTX 3070's 8 GiB.** The UHD 630 has no dedicated video
memory at all. The test allocated 6.6 GiB of buffers on the integrated adapter,
WDDM satisfied them out of system RAM, every checksum matched, and the run
reported `PASS … 6784 MiB VRAM`.

Root cause is an **index-space collision** in
[main.rs:752](../crates/crucible-cli/src/main.rs#L752):

```rust
let idx = match device {
    GpuDevice::Discrete(i) | GpuDevice::Integrated(i) => i as u32,
    GpuDevice::Default => 0,
};
crucible_gpu::vramsize::max_testable_vram_mb(idx)
```

`i` is a **wgpu device ordinal within a device class** — `Integrated(0)` means
"the first integrated adapter". It is then passed to `dedicated_vram_bytes`,
which uses it as a **DXGI adapter index**. DXGI adapter 0 on this machine is the
3070. The two index spaces are unrelated and are being treated as one.

This is the worst class of bug this tool can have: it does not fail, it answers
confidently and wrongly. A tester would come away believing they had verified
VRAM integrity on a part that has none.

### 1.2 SEVERE — `link` reports PCIe bandwidth across a bus that does not exist

```
cec-crucible link --seconds 6 --gpu-device integrated
running pcie for 6s (bidir integrated0)…
  [pcie   ] PASS  8.0s  integrated0 bidir 256MiB, 28 verified, 14.0 GiB moved,
                        H2D ~5.4 GB/s, D2H ~1.6 GB/s
```

An integrated GPU is on the same die as the CPU and shares its memory
controller. **There is no PCIe link to it.** The test measured a copy within
system RAM and labelled the result PCIe host↔device bandwidth, with a `pcie`
lane name in the telemetry and the report.

The data is not meaningless — memory-controller throughput under a
copy+verify load is a genuine thing to measure on an APU, and arguably a *better*
test than it is on a discrete card. But it must not be called PCIe.

### 1.3 MAJOR — `gpu` panics on a hardcoded allocation, and prints an impossible number

```
cec-crucible gpu --seconds 6 --gpu-device integrated
CRASH [panic] during phase 'running:gpu':
  called `Result::unwrap()` on an `Err` value: can't allocate buffer of size: 1073741824
  [gpu] FAIL  0.9s  integrated0 alu+vram threads=1048576 iters=4096,
                    4 dispatch(es), 0 verify, ~2.61 TFLOP/s; DEVICE LOST
```

Two things:

* The kernel asks for a 1 GiB buffer unconditionally. The iGPU's per-resource
  limit is lower, and the allocation `unwrap()`s. The crashguard caught it and the
  verdict is correctly FAIL, but a driver limit should be a graceful "size down or
  refuse", not a panic.
* It printed **~2.61 TFLOP/s for a UHD 630**. That part peaks near
  0.46 TFLOP/s FP32 (24 EUs × 8 lanes × 2 × ~1.2 GHz), so the figure is roughly
  6× over the theoretical maximum — computed from 4 dispatches with **0
  verifications**. A throughput number from a run that verified nothing should not
  be printed at all. `rt.rs` and `render.rs` already have the
  `verifications == 0 → NOT VERIFIED` guard; `gpu.rs` does not.

### 1.4 MAJOR — the sensor plane is not bound to the device under test

`gputel` opens NVML device index 0 unconditionally, so a run targeting the
**integrated** adapter still reports the **discrete** card's power and
temperature:

```
cec-crucible cpu --seconds 3
gpu: NVIDIA GeForce RTX 3070 — power avg 40 W, peak 40 W (limit 240 W), peak 58 °C
```

Harmless on a CPU test (it is honest ambient context). Actively misleading on an
iGPU test, where the strip would show the idle 3070 while the part actually under
load is invisible.

---

## 2. The root cause behind 1.1 and 1.4: there is no adapter identity

Every one of these bugs is the same shape — *"which adapter is this?"* answered
by an integer that means something different in each API. The suite currently
juggles at least four index spaces:

| Space | Where | `0` means |
|---|---|---|
| wgpu class ordinal | `GpuDevice::Integrated(0)` | first *integrated* adapter |
| DXGI adapter index | `EnumAdapters1(0)` | highest-performance adapter |
| NVML device index | `nvmlDeviceGetHandleByIndex_v2(0)` | first *NVIDIA* device |
| ADL adapter index | `ADL_Adapter_AdapterInfo_Get` | first *AMD* device |

**Windows already has the right answer: the adapter LUID.** It is a stable
64-bit identity for a display adapter, and it is the join key across every plane
we care about:

* DXGI — `DXGI_ADAPTER_DESC1::AdapterLuid` (we already call `GetDesc1` in
  [vramsize.rs](../crates/crucible-gpu/src/vramsize.rs); the LUID is in the struct
  we are already reading and currently ignoring).
* wgpu/DX12 — the HAL adapter exposes the same LUID; on Vulkan,
  `VkPhysicalDeviceIDProperties::deviceLUID` with `deviceLUIDValid`.
* PDH — **confirmed on this bench.** The vendor-neutral GPU counter sets are
  keyed by LUID:

  ```
  \GPU Adapter Memory(luid_0x00000000_0x0001b019_phys_0)\Dedicated Usage  →  1733238784
  \GPU Engine(pid_12072_luid_0x00000000_0x0001b019_phys_0_eng_0_engtype_3d)\Utilization Percentage
  ```

  Counter sets present: `GPU Engine`, `GPU Adapter Memory`,
  `GPU Local Adapter Memory`, `GPU Non Local Adapter Memory`,
  `GPU Process Memory`.

So step one of this work is not vendor SDKs at all. It is: **resolve the selected
device to a LUID + `DXGI_ADAPTER_DESC1` once, carry that record everywhere, and
delete the bare indices.** That alone fixes 1.1, fixes 1.4, and is the
precondition for everything below.

---

## 3. Sensor plane: one trait, four backends

`gputel::GpuSample` / `GpuSummary` are already the right shape. Generalise
`GpuTelemetry` into a trait and pick a backend from the resolved adapter's
`VendorId`.

| Vendor | Backend | Library | Admin? | SDK? |
|---|---|---|---|---|
| NVIDIA `0x10DE` | NVML | `nvml.dll` | no | no — **shipping** |
| AMD `0x1002`/`0x1022` | ADLX, fall back to ADL2 | `amdadlx64.dll` / `atiadlxx.dll` | no | no |
| Intel `0x8086` | Level Zero Sysman, fall back to IGCL | `ze_loader.dll` / `ControlLib.dll` | no | no |
| any | PDH + DXGI | in-box | no | no |

All four load by `LoadLibrary` + `GetProcAddress`, which is the ethos the rest of
the FFI in this project already follows. No vendored SDK, no build-time linkage,
no redistributables.

### 3.1 AMD

Two APIs, and it is worth taking both.

**ADL2 (legacy, `atiadlxx.dll`)** is the pragmatic first target. It is a flat C
API, it is what HWiNFO/GPU-Z/hashcat have used for a decade, and AMD's own
[PMLog sample](https://github.com/GPUOpen-LibrariesAndSDKs/display-library/blob/master/Sample/PMLog/PMLog.cpp)
gives the exact call sequence:

```
ADL_Main_Control_Create
ADL_Adapter_NumberOfAdapters_Get
ADL_Adapter_AdapterInfo_Get                 → match our LUID here
ADL2_Adapter_PMLog_Support_Get(ctx, idx, ADLPMLogSupportInfo*)
ADL2_Device_PMLog_Device_Create
ADL2_Adapter_PMLog_Support_Start(ctx, idx, ADLPMLogStartInput*, ADLPMLogStartOutput*, hDevice)
  … poll ADLPMLogData: ulValues[i][0] = sensor type, ulValues[i][1] = value,
    terminated by ADL_SENSOR_MAXTYPES …
ADL2_Adapter_PMLog_Support_Stop
ADL2_Device_PMLog_Device_Destroy
ADL_Main_Control_Destroy
```

The sensor enum covers everything `GpuSample` carries: edge and hotspot
temperature, memory temperature, ASIC/board power, fan RPM and percent, GFX/MCLK/SOC
clocks, and activity. `ADL2_New_QueryPMLogData_Get` is the older single-shot form
and is **deprecated** in favour of the shared-memory path
(`ADL2_Overdrive8_PMLog_ShareMemory_Read` where
`..._ShareMemory_Support` says so) — worth using the modern path, since the
deprecated one is exactly the symbol that
[goes missing on newer drivers](https://github.com/NebuTech/NBMiner/issues/218).

**ADLX (`amdadlx64.dll`)** is the supported modern library, with
`ADLXInitialize` / `ADLXQueryVersion` as the exported entry points and a C API
alongside the C++ one. It has a cleaner metrics surface
(`GetGPUTotalBoardPower`, `GetGPUHotspotTemperature`, and
`GetSupportedGPUMetrics` to ask what the part actually exposes — which matters a
lot on APUs, where the answer is "much less"). It also carries an
[ADL interop path](https://gpuopen.com/manuals/adlx/adlx-c_sample__work_with_a_d_l/)
so both can coexist.

Recommendation: **ADLX first, ADL2/PMLog as the fallback**, since ADLX is what
AMD supports going forward but ADL2 reaches older Adrenalin installs.

**APU caveat.** Radeon iGPU sensor coverage arrived with the Adrenalin 19.2.3
driver family and is *partial*: expect GFX clock and activity, often a
temperature, and frequently **no board power at all** — on an APU the power
budget belongs to the package (PPT), not the graphics block. `GetSupportedGPUMetrics`
must be honoured, and anything unsupported must come through as **blank**, which
is already the rule the CSV writer follows.

### 3.2 Intel

`zesInit()` then `zesDeviceEnumPowerDomains` / `zesPowerGetEnergyCounter`,
`zesDeviceEnumTemperatureSensors` / `zesTemperatureGetState`, plus frequency and
engine domains. Loaded through `ze_loader.dll`, which is designed to
[fail gracefully when no driver is present](https://github.com/oneapi-src/level-zero/issues/142)
— exactly the shape we want. Arc reports properly; older iGPUs like the UHD 630
on this bench are likely to report little or nothing, and the honest result there
is blank columns and no dashboard strip, not zeros.

### 3.3 Vendor-neutral fallback — do this one first

Confirmed working on this bench with no SDK of any kind: PDH `GPU Engine`
utilisation per engine type (3D, compute, copy, video) and `GPU Adapter Memory`
dedicated/shared usage, both keyed by LUID. **We already have PDH FFI plumbing
for the per-core CPU counters**, so this is largely reuse.

It gives no power and no temperature — but it gives *utilisation and memory
residency on every vendor including the iGPU*, which is enough to answer "was the
part actually loaded?" That question is currently unanswerable on anything but
NVIDIA, and it is the one that matters most for a QC gate.

---

## 4. UMA / APU semantics — what each test *means* changes

This is the part that is design, not FFI. Detect UMA authoritatively via
`D3D12_FEATURE_DATA_ARCHITECTURE::UMA` / `CacheCoherentUMA` (or Vulkan
`VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`), then:

| Test | On a discrete card | On a UMA adapter |
|---|---|---|
| `gpu` | ALU + VRAM thrash | same, but size buffers from real limits (§1.3) |
| `vram` | dedicated VRAM integrity | **there is no VRAM.** Re-target at the shared aperture, cap against *system* RAM headroom, and rename the lane. Must never claim a VRAM size it did not get. |
| `link` | PCIe H2D/D2H | **no PCIe.** Relabel as memory-controller copy bandwidth, or refuse. |
| `render` / `rt` | as-is | as-is; `rt` needs a ray-query capability check, not an assumption |
| `uncore` | CPU interconnect | as-is, and *more* relevant: on an APU the iGPU shares that fabric |

`--vram-mb max` must resolve differently per architecture: 85% of *dedicated*
VRAM on a discrete part, versus a much more conservative slice of *available
system RAM* on a UMA part — where over-allocating does not lose the device, it
pages the whole machine.

### The APU test that does not exist yet, and should

On an APU the CPU and the iGPU contend for **one memory controller** and **one
package power budget**. That contention is the actual failure mode of laptop and
SFF gaming: not a bad part, but a part that cannot sustain both loads at once.
Nothing in the suite exercises it, because on a discrete rig it is not a thing.

A UMA cross-load — CPU burst kernel and iGPU thrasher in anti-phase, with
`uncore` running underneath, watching effective clocks on both sides collapse —
is the single highest-value new test this work would unlock, and it needs no new
FFI beyond §3.3.

---

## 5. AMD discrete: less work than it looks

The **load** side is already vendor-neutral, because everything goes through wgpu
(DX12/Vulkan). What needs doing is honest gating rather than new kernels:

* `cuda`, `optix`, `tensor` — NVIDIA-only by construction. Already feature-gated;
  needs to *say so* at runtime on an AMD box rather than looking absent.
* `rt` / `pathtrace` — `VK_KHR_ray_query` is supported on RDNA2+ (RX 6000 and
  newer). The Vulkan path in [rt.rs](../crates/crucible-gpu/src/rt.rs) already
  enumerates physical devices and checks device type; it needs an explicit
  extension/feature check with a clear "this GPU has no ray-query support"
  message rather than a failure.
* `benchmark` — scores are calibrated against one RTX 3070. Cross-vendor
  comparison is not meaningful until there is a second calibration point. Say so
  in the output.

**This cannot be validated without AMD hardware.** The cheapest useful coverage
is one RDNA2/3 discrete card (exercises ADLX, ray-query, and the discrete path)
plus one Ryzen APU laptop or desktop APU (exercises UMA, partial sensors, and the
contention test). A discrete AMD card alone leaves the entire UMA half unproven.

---

## 6. Phasing

Ordered so that each phase is shippable and the false results die first.

**Phase 0 — stop lying (no new dependencies, testable on this bench today)**
1. Adapter identity: resolve to LUID + `DXGI_ADAPTER_DESC1` once; thread it
   through; delete the bare indices. Fixes §1.1 and §1.4.
2. UMA detection; `vram` and `link` refuse or relabel on UMA. Fixes §1.2.
3. `gpu` sizes buffers from device limits instead of a hardcoded 1 GiB, and
   gains the `verifications == 0 → NOT VERIFIED` guard. Fixes §1.3.

**Phase 1 — vendor-neutral sensors (reuses existing PDH plumbing)**
4. PDH GPU Engine + Adapter Memory by LUID → utilisation and memory residency on
   every vendor. Dashboard strip appears for the iGPU.

**Phase 2 — vendor sensor backends**
5. `GpuTelemetry` becomes a trait; NVML moves behind it unchanged.
6. Intel Level Zero Sysman (validatable on this bench, partially).
7. AMD ADLX with ADL2/PMLog fallback (**needs AMD hardware to validate**).

**Phase 3 — APU semantics**
8. UMA-aware `--vram-mb max` sizing against system RAM headroom.
9. The CPU↔iGPU memory-controller and package-power contention cross-load.

**Phase 4 — AMD discrete**
10. Ray-query capability check; runtime messages for NVIDIA-only features;
    benchmark calibration caveat.

Phases 0–1 are worth doing regardless of whether AMD hardware ever appears: they
remove two false PASSes and light up the iGPU that half the machines walking into
a shop are running on.

---

## Sources

* [AMD Device Library eXtra (ADLX) SDK](https://gpuopen.com/adlx/) —
  [Quick Start](https://gpuopen.com/manuals/adlx/adlx-page_guide_use__a_d_l_x),
  [PerfGPUMetrics (C)](https://gpuopen.com/manuals/adlx/adlx-c__perf_g_p_u_metrics/),
  [WorkWithADL interop](https://gpuopen.com/manuals/adlx/adlx-c_sample__work_with_a_d_l/)
* [AMD Display Library (ADL)](https://gpuopen.com/archived/adl/) —
  [PMLog example](https://gpuopen-librariesandsdks.github.io/adl/PMLog-example.html),
  [PMLog.cpp sample](https://github.com/GPUOpen-LibrariesAndSDKs/display-library/blob/master/Sample/PMLog/PMLog.cpp),
  [deprecated list](https://gpuopen-librariesandsdks.github.io/adl/deprecated.html),
  [NBMiner #218 — missing symbol on newer drivers](https://github.com/NebuTech/NBMiner/issues/218)
* [Level Zero Sysman API spec](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/sysman/api.html),
  [Intel compute-runtime SYSMAN guide](https://github.com/intel/compute-runtime/blob/master/programmers-guide/SYSMAN.md),
  [pti-gpu Level Zero system management](https://github.com/intel/pti-gpu/blob/master/chapters/system_management/LevelZero.md),
  [ze_loader graceful-failure discussion](https://github.com/oneapi-src/level-zero/issues/142)
* [D3D12_FEATURE_DATA_ARCHITECTURE](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture),
  [UMA optimizations / default texture mapping](https://learn.microsoft.com/en-us/windows/win32/direct3d12/default-texture-mapping),
  [D3D12 GPU Upload Heaps (UMA vs CacheCoherentUMA)](https://microsoft.github.io/DirectX-Specs/d3d/D3D12GPUUploadHeaps.html)
* [windows_exporter GPU collector — PDH GPU counter names](https://github.com/prometheus-community/windows_exporter/blob/master/docs/collector.gpu.md)
* [AMD Adrenalin 19.2.3 — APU / Ryzen Mobile Vega support](https://overclock3d.net/news/gpu-displays/amds-radeon-software-adrenalin-19-2-3-drivers-packs-apu-and-ryzen-mobile-support/)
* [wgpu #683 — Intel iGPU misreported as discrete on the DX12 backend](https://github.com/gfx-rs/wgpu/issues/683)
