<!-- SPDX-License-Identifier: MIT -->
# GPU functional-unit coverage — tensor & RT scoping (documentation-verified)

Extends the GPU tests beyond the general shader ALU / VRAM / PCIe the suite
already covers, to the **rasterizer/TMU/ROP** (done — see `render.rs`), the
**tensor/matrix cores**, and the **ray-tracing cores**. Every version-sensitive
claim below was researched against **current primary documentation** and the
key ones **cross-checked against the pinned source in this repo's own
`Cargo.lock`** (wgpu/naga 29.0.4, CubeCL 0.10, cudarc 0.19.8). Tags: **[FACT]**
primary-source-confirmed · **[VERIFIED]** checked against the pinned crate source
· **[INFER]** reasoning · **[GATED]** portability/maturity caveat.

## 0. Status

| Unit | Test | State |
|---|---|---|
| Rasterizer / TMU / ROP | `render` | ✅ built (`render.rs`) |
| Tensor / matrix cores | `tensor` | ✅ built (`tensor.rs`) — f16→f32 cmma, ~22 TFLOP/s on the RTX 3070 |
| RT cores | `rt` | ✅ built (`rt.rs`) — raw `ash` `VK_KHR_ray_query`, **~5.35 Gray/s on the RTX 3070** |

## 1. Ray-tracing cores (`rt`) — BUILT

**Built via the stable path, not the wgpu experimental one.** `rt.rs` uses raw
Vulkan (`ash` 0.38) with `VK_KHR_ray_query`: it builds a BLAS/TLAS over a fixed
displaced-grid mesh and traces a deterministic camera fan from a compute shader,
folding each committed hit's primitive index + distance `t` into a per-ray
checksum. The WGSL ray-query shader is compiled to SPIR-V **at runtime by naga**
(already in the wgpu tree), so no shader compiler / Vulkan SDK is required on the
build or target machine. Verified on the RTX 3070: PASS, 0 errors, 79
self-consistency read-backs at ~5.35 Gray/s (steady) / ~2.64 Gray/s (burst, 50%
duty) — a throughput only possible with hardware RT-core acceleration. The scoping
that led here is preserved below.



**[VERIFIED, `wgpu-types-29.0.4/src/features.rs`]** wgpu 29 exposes exactly one
usable RT capability: `EXPERIMENTAL_RAY_QUERY` (bit `1<<32`). Its doc-comment,
verbatim from the pinned source:

> ***THIS IS EXPERIMENTAL:*** Features enabled by this may have major bugs in it
> and are expected to be subject to breaking changes … Allows for the creation of
> ray-tracing queries within shaders. **Supported platforms: - Vulkan.** This is a
> native-only feature.

So: **inline ray queries + acceleration-structure build, Vulkan backend only, no
RT pipelines, flagged experimental.** [FACT] Inline ray query drives the *same*
BVH-traversal + ray/triangle-intersection hardware as a full raygen/hit/miss
pipeline — the pipeline only changes shader scheduling — so inline query is a
**complete** way to stress the RT cores, not a compromise.

- **[FACT]** All three desktop vendors expose DXR 1.1 + Vulkan ray-query on
  Windows; the constraint is wgpu (Vulkan-only in 29), not the drivers. NVIDIA
  needs Turing+ (RTX); AMD RDNA2+; Intel Arc. Pascal/GCN/RDNA1 have no RT HW.
- **Design:** raw wgpu (like `render.rs`/`link.rs`; CubeCL can't do RT), force
  `Backends::VULKAN`, request `EXPERIMENTAL_RAY_QUERY`, build a BLAS/TLAS over a
  fixed opaque mesh (reuse `render::build_mesh`), trace deterministic rays from a
  compute shader (`enable wgpu_ray_query;` … `rayQueryInitialize/Proceed/
  GetCommittedIntersection`), record `{hit, t, primitive, instance}`, checksum.
  Model on the wgpu `ray_cube_compute` example.
- **Verify:** liveness (not all-miss) + same-device self-consistency. **[FACT,
  DXR spec]** committed closest-hit over opaque triangles is deterministic on
  fixed hardware; **not** cross-vendor bit-identical (watertight, but `t`/
  barycentric LSBs differ) — same stance as `render`.
### RT path decision — four options (both agents, cross-checked)

Deeper research (NVIDIA OptiX + AMD/Intel native) reframed this. The wgpu path is
the *easiest* but the *least stable*; there are three stable alternatives:

| Path | Vendors | OS | Stable? | Runtime | Rust | Effort |
|---|---|---|---|---|---|---|
| **wgpu `EXPERIMENTAL_RAY_QUERY`** | all 3 | Win+Linux | ❌ "major bugs / breaking" | driver | in-tree (wgpu) | **low** (a kernel) |
| **raw `ash` `VK_KHR_ray_query`** | all 3 | Win+Linux | ✅ ratified 2020 | driver (ship SPIR-V) | `ash` 0.38 | high (hand Vulkan) |
| **DXR 1.1 inline RayQuery** | all 3 | **Win only** | ✅ stable | driver (ship DXIL) | `windows` crate | high (d3d12 FFI) |
| **OptiX** | **NVIDIA only** | Win+Linux | ✅ battle-tested | driver (`nvoptix.dll`) | hand-roll FFI | high (~1.5–3 wk) |

- **OptiX [FACT, verified]: GO for a public-MIT, driver-only, runtime-dispatched
  kernel — with one rule.** Since OptiX 7 the runtime lives *in the driver*
  (`nvoptix.dll`, min R435; use OptiX 9 / R570 for Blackwell); device programs
  AOT-compile to PTX at build (nvcc) and the driver JITs to SASS at runtime — **no
  NVRTC, no toolkit, no SDK on the target**, exactly the cudarc philosophy. Ship a
  committed `compute_50` PTX (`include_bytes!`). **License:** a *compiled binary*
  built with OptiX is royalty-free redistributable (commercial or OSS), **but you
  may not commit NVIDIA's SDK headers as source** — so **hand-roll the FFI**
  (~15 fns / ~20 structs; declare the function table, `LoadLibrary` the driver) and
  the repo carries zero NVIDIA source, sidestepping the license question entirely.
  No maintained Rust crate exists (`optix` is 2019/OptiX-5; `optix-sys`
  unpublished). 3070 (Ampere, RT v2.0) and 5090 (Blackwell, 4th-gen RT) both fine.
- **AMD/Intel native [FACT]:** AMD **HIPRT** is real + MIT + hits the Ray
  Accelerators + runtime-dispatchable, but has **no Rust bindings** (hand FFI +
  bundled bitcode) and DXR/Vulkan already drive the same units — not worth it
  unless stressing AMD's own traversal stack is a goal. Intel **Embree-GPU** is
  **unreachable from Rust** (SYCL-C++ only). So AMD/Intel RT = DXR or `ash`-Vulkan.
- **Determinism [FACT]:** no formal OptiX/RT bitwise guarantee, but the *committed
  closest hit* is deterministic on fixed HW+driver. Design the check as: each ray
  writes `{primitive_id, bitcast(t)}` to a per-ray buffer → reduce in **fixed host
  order** → hash; **self-consistency, not a repo-baked golden** (AS layout + SASS
  vary across driver/GPU). Avoid any-hit-order-dependent logic and cross-thread
  float atomics. Same stance as `render`.

**Recommendation:** the strategic winner is **raw `ash` `VK_KHR_ray_query`** — one
*stable*, cross-vendor, cross-OS, driver-only implementation covers the whole bench
(NVIDIA now, AMD/Intel later) for effort comparable to OptiX, whereas OptiX is
NVIDIA-only (choosing it means writing a *second* RT path for other vendors
anyway). Use **wgpu** only for a throwaway prototype (in-tree, but pin it and
expect breakage); add **OptiX** only if stressing NVIDIA's OptiX driver-compiler
stack is itself a QC goal. **Cross-check status:** the wgpu claims were verified
verbatim against the pinned `wgpu-types-29.0.4` source; the OptiX/`ash`/HIPRT/Embree
facts are agent-cited to current primary docs (OptiX 8.1 guide, Khronos, crates.io)
and one agent self-corrected the HIPRT license by reading the raw `license.txt` —
but the OptiX EULA + `ash` ray-query API should get a direct read before we commit
code (moot for OptiX if we hand-roll and vendor nothing).

## 2. Tensor / matrix cores (`tensor`)

**The core problem [FACT]:** the current DX12/HLSL path reaches tensor cores on
**no** vendor. Both portable cooperative-matrix paths (CubeCL and wgpu-native)
are **Vulkan(SPIR-V) + Metal only** — a real tension with `gpu-plan.md`'s DX12
preference. Four distinct routes:

| Route | You write | Reaches TC via | Runtime toolkit? |
|---|---|---|---|
| **A. CubeCL `cmma` on Vulkan** | `#[cube]` Rust | cubecl-spirv → `SPV_KHR_cooperative_matrix` | **No** (driver Vulkan) |
| **B. CubeCL `cmma` on CUDA** | `#[cube]` Rust | cubecl-cuda → wmma PTX → NVRTC | **Yes** (toolkit — the known blocker) |
| **C. Native WMMA PTX + cudarc** | CUDA C++, precompiled to PTX | `cuModuleLoadData` (driver API) | **No** (driver's own PTX→SASS JIT) |
| **D. wgpu-native WGSL coop-matrix** | hand WGSL | naga → SPIR-V/MSL | **No** |

**[FACT, CubeCL docs+source]** CubeCL 0.10 reaches tensor cores through its
Vulkan/SPIR-V compiler (`cubecl-spirv`) — **not** WGSL. Its CMMA matrix: CUDA ✔,
ROCm ✔, WGPU-WGSL ❌, **WGPU-SPIR-V ✔**; and **int8→int32 is supported on CUDA +
ROCm + SPIR-V**. That is the ship-anywhere, driver-only, keeps-`#[cube]`-Rust
route — **primary recommendation (Route A)**.

**[FACT] The NVRTC escape (Route C):** WMMA/`mma.sync` are ordinary device
intrinsics — `nvcc` AOT-compiles them to PTX at *build* time; the **driver's**
own PTX-JIT (`libnvidia-ptxjitcompiler`, not NVRTC) runs them via cudarc's
`cuModuleLoadData` — exactly how our CUDA *link* path already runs toolkit-free.
NVIDIA-only "push harder" fast path, gate behind the `cuda` feature.

**Verification [FACT/INFER]:** **int8→int32 CMMA is the cross-vendor golden** —
integer multiply-accumulate is exact and associative, so the K-reduction is
bit-identical across vendors *provided* no int32 overflow and non-saturating
accumulation (Vulkan exposes a `saturatingAccumulation` flag per combo). Every
float mode (fp16/bf16/tf32/fp8) is vendor-order-dependent → same-device
self-consistency or a tolerance band only. (No vendor sentence promises
"cross-vendor bit-identical"; the integer-exactness argument is sound and
testable — establish the golden on first run, then assert reproducibility.)

### ⚠ Cross-check correction to the research (found in the pinned source)

**[VERIFIED, `wgpu-types-29.0.4/src/features.rs`]** The wgpu-*native*
cooperative-matrix fallback (**Route D**, `EXPERIMENTAL_COOPERATIVE_MATRIX`, bit
`1<<57` — confirmed) is **much more limited** than "f16/f32/mixed". Its
doc-comment, verbatim:

> **Current limitations:** The implementation currently only supports **8x8 f32
> matrices**. On Vulkan, support is determined by querying
> `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` for configurations matching
> 8x8x8 f32. **Most Vulkan implementations (NVIDIA, AMD) primarily support f16
> inputs at larger sizes (e.g., 16x16), so Vulkan support may be limited.**
> Supported platforms: Metal (MSL 2.3+, Apple7+), Vulkan (VK_KHR_cooperative_matrix
> if 8x8 f32 is supported). Native-only.

So Route D in our pinned wgpu is **8×8 f32 only and may not even be exposed on
NVIDIA/AMD** — it is *not* an int8-golden path and is a weak reserve, not a
portable tensor solution. **The viable portable path is Route A (CubeCL-SPIR-V),
whose int8 support must be confirmed by a spike** (force the Vulkan backend, check
whether `cubecl-wgpu` 0.10 needs a SPIR-V-passthrough feature and that the adapter
exposes `VK_KHR_cooperative_matrix` int8 combos).

### Per-vendor reality [FACT/GATED]

| Unit | NVIDIA | AMD | Intel |
|---|---|---|---|
| Tensor via CubeCL-SPIR-V (Vulkan) | ✔ (RTX/TC) | ✔ RDNA3+ (WMMA) | ◐ Arc (XMX) — **Windows coop-matrix TDR/detection driver bugs**, gate off until fixed |
| Tensor via native PTX+cudarc | ✔ driver-only | — | — |
| Tensor via CUDA/NVRTC | ✔ but toolkit-gated | — | — |

Intel `joint_matrix`/DPAS needs the DPC++/SYCL toolchain (not reachable from
Rust/CubeCL); AMD rocWMMA/MFMA is Linux-centric — on Windows both are reached via
Vulkan cooperative-matrix, same as the CubeCL-SPIR-V route.

### Tensor spike recipe — [VERIFIED against `cubecl-*-0.10.0` source in-registry]

The exact ingredients, confirmed by reading the pinned CubeCL 0.10 source:

- **The switch to tensor cores is the `cubecl` `vulkan` feature** (`vulkan =
  ["wgpu-spirv"]` → `cubecl-wgpu/spirv` → pulls `cubecl-spirv` + `ash` 0.38 +
  `tracel-ash`). With it, `cubecl-wgpu`'s runtime reports `wgpu<spirv>` and the mma
  path is enabled — `cubecl-wgpu/src/runtime.rs:87` states outright *"no wgsl
  backends currently support manual mma."* Gate it behind a new `crucible-gpu`
  `tensor` feature so the default `gpu` build is unchanged (it adds `ash`).
- **cmma API** (`cubecl-core/src/frontend/cmma.rs`): `cmma::Matrix::<T>::new` /
  `from_slice(ident, m, n, k, layout)` → `cmma::fill` / `cmma::load(&mat, slice,
  stride)` → `cmma::execute::<A,B,C,D>(&a,&b,&c,&d)` (D = A·B + C) → `cmma::store`.
  16×16×16 tile. Launch like the thrasher: `WgpuRuntime::client(&dev)` then
  `kernel::launch::<T, WgpuRuntime>(...)`; **force the Vulkan backend** via
  `init_setup`/`create_setup_for_device` (the runtime routes `wgpu::Backend::Vulkan`
  → SPIR-V, `runtime.rs:48`).
- **dtypes:** `I8`/`I32`/`U8` exist in the type system, **but CubeCL's own cmma
  runtime tests (`runtime_tests/cmma.rs`) cover only f16→f32.** ⚠ So the spike must
  *prove* int8→int32 compiles + runs on the SPIR-V backend; if it doesn't, fall
  back to f16→f32. **int8→int32 = the cross-vendor golden; f16→f32 =
  self-consistency only.**
- **Spike plan:** (1) `tensor` feature + Vulkan-forced client on the 3070, first
  with the upstream-tested **f16→f32** cmma to confirm SPIR-V + tensor cores light
  up at all; (2) switch to **i8→i32** and confirm it compiles/runs (the load-bearing
  unknown); (3) checksum the result buffer, wire `LoadKernel` + a `tensor` command.
  Dep cost lands only under the feature.

### Tensor spike RESULT — [RAN on the RTX 3070] (`spikes/gpu-tensor`)

```
runtime: wgpu<spirv>
f16->f32 cmma: max_err=0.0000  PASS (tensor cores reached)
i8->i32  cmma: gpu[0]=0 ref[0]=-1  MISMATCH
```

- ✅ **The portable path is proven.** `init_setup::<Vulkan>` forces `wgpu<spirv>`,
  and a 16×16×16 **f16→f32** `cmma` is **bit-exact vs a CPU reference** — the tensor
  cores are genuinely reached through CubeCL-SPIR-V/Vulkan, driver-only, one binary.
- ❌ **int8→int32 does NOT work here** — it ran without error but returned all
  zeros (the untested-upstream risk, confirmed). Almost certainly the 3070's Vulkan
  `VK_KHR_cooperative_matrix` doesn't expose an int8 combo (the wgpu-source note
  said NVIDIA/AMD "primarily support f16"), or CubeCL's SPIR-V int8 emit is a gap.
- **Consequence:** the shippable `tensor` kernel uses **f16→f32 with same-device
  self-consistency** (like `render`), NOT the int8 cross-vendor golden. The
  bit-exact int8 golden remains available only on the **CUDA / native-WMMA-PTX**
  path (NVIDIA, toolkit-at-build) — a separate optional route if a cross-vendor
  golden is ever needed. To build the kernel: promote `spikes/gpu-tensor` into a
  `crucible-gpu` `tensor` feature (adds `cubecl/vulkan` → `ash`+`cubecl-spirv`),
  scale from one tile to a sustained GEMM loop, wire `LoadKernel`.

## 3. Recommended build order

1. **`render`** — ✅ done (portable, zero-dep, all three vendors).
2. **`render --scene` glTF/PBR** — realism upgrade, `gpu-gltf` feature.
3. **`tensor` spike → kernel** — Route A (CubeCL `cmma`, Vulkan/SPIR-V, int8→int32
   golden), with Route C (native PTX+cudarc) as the NVIDIA fast path. Confirm the
   int8 combo lands on the Vulkan backend before committing. NVIDIA-first;
   Intel-Arc gated on driver fixes.
4. **`rt`** — wgpu 29 `EXPERIMENTAL_RAY_QUERY` (Vulkan-only, experimental); stage
   in `spikes/`, promote when wgpu de-experimentalises or adds DX12.

All implement `LoadKernel`, so each drops into the shape/stop/phase/marker
machinery with no orchestrator change and immediately gains the chaos / gauntlet
profiles. Same verification bar as everything else: **verified-nothing ⇒ FAIL**,
same-device self-consistency (int8 tensor golden is the one cross-vendor golden).

## 4. Implementation plans (scoped, ready to execute)

- **`render` + glTF/PBR — plan ready (~1–1.5 days).** New `gpu-gltf` feature adds
  `gltf` 1.4 (which pulls `image` 0.25 trimmed to png+jpeg already) — all behind
  the feature, default/`gpu` builds untouched. Loads a bundled `.glb` into
  per-primitive draws with metallic-roughness material bind groups; PBR fragment
  shader ported from the Khronos glTF-Sample-Renderer (Apache-2.0); `--scene
  <file>` override; verification unchanged (framebuffer checksum, self-consistency).
  ⚠ **LEGAL [FACT, per-asset READMEs]:** the usual defaults are BLOCKED for a
  commercial public-MIT repo — **DamagedHelmet is CC-BY-NC** (NonCommercial) and
  **Khronos Sponza is CryEngine-licensed**. Bundle **BoomBox (CC0-1.0, Microsoft)**
  instead (full metal-rough PBR set + tangents; downscale textures + embed → ~1–2 MB).
- **`rt` via `ash` `VK_KHR_ray_query` — plan ready (~few days, ~700–1000 Rust +
  40 GLSL).** New `rt` feature adds `ash` 0.38 (+ `gpu-allocator`). All ash objects
  built inside `run()` under `catch_unwind` like the wgpu kernels; a separate
  `VkInstance` alongside wgpu's is fully supported. BLAS/TLAS over fixed geometry,
  a committed prebuilt `.spv` inline-ray-query compute shader (`glslangValidator
  --target-env vulkan1.2`), per-ray `{prim, bitcast(t)}` → fixed-order FNV hash →
  self-consistency. Verified against ash-0.38 docs; pin `ash = "=0.38.0"` (the
  `push_next` API differs on ash master).

## Sources

Primary docs (with dates) captured in the research: wgpu v29/v30 `features.rs`,
`ray_tracing.md`, `cooperative_matrix.md`, CHANGELOG, `ray_cube_compute` example;
Microsoft DXR functional spec v1.45; Khronos `VK_KHR_ray_query` /
`VK_KHR_cooperative_matrix` (rev 2, Ratified 2023-05-03) + `VkComponentTypeKHR`;
NVIDIA CUDA C++ Programming Guide + Ampere/Ada tuning guides + Driver API
`cuModuleLoadData`; AMD GPUOpen RDNA3 WMMA + rocWMMA + ROCm precision tables;
Intel SYCL `joint_matrix` spec (Rev 12); CubeCL `core-features/features.md` +
`cubecl-spirv`. **Cross-checked in-repo:** `wgpu-types-29.0.4/src/features.rs`
(both `EXPERIMENTAL_RAY_QUERY` bit `1<<32` Vulkan-only and
`EXPERIMENTAL_COOPERATIVE_MATRIX` bit `1<<57` 8×8-f32-only confirmed verbatim).
