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
| Tensor / matrix cores | `tensor` | scoped — needs a Vulkan/SPIR-V spike |
| RT cores | `rt` | scoped — wgpu 29 experimental, Vulkan-only |

## 1. Ray-tracing cores (`rt`)

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
- **[GATED]** Recommendation: stage in `spikes/` (as with `gpu-3a`), promote to a
  shipped kernel when the trigger fires — **wgpu drops the `EXPERIMENTAL_` prefix,
  or `EXPERIMENTAL_RAY_QUERY` gains DX12** (neither has happened as of wgpu v30).
  No Cargo change needed to prototype (the `vulkan` wgpu feature is already on).

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
