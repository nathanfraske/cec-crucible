<!-- SPDX-License-Identifier: MIT -->
# Path-tracing GPU test — scoping (NVIDIA-native + portable)

Scopes a **path-tracing** stress/QC test: multi-bounce Monte-Carlo global
illumination, the real modern RT workload (path-traced games, DLSS-RR), as a
step beyond the existing `rt` test (which is a *shallow, coherent, single-bounce*
ray-query check). Researched against **current primary documentation** (OptiX
9.1 / ABI 118, Vulkan RT spec, wgpu/naga, NVIDIA driver notes) and cross-checked;
tags: **[FACT]** primary-source-confirmed · **[INFER]** reasoning · **[GATED]**
hardware/maturity/licensing caveat or needs-a-spike.

**Status:** Phase 1 (portable inline multi-bounce, §2) is **BUILT** — `pathtrace`
command + `pathtrace.wgsl`, commit 8ac029c; determinism validated on the RTX 3070
(PASS, 0 errors, 423 self-consistency verifications). **Phase 2 (OptiX) host FFI +
denoiser DE-RISKED** on the bench (`spikes/optix-ffi`, no CUDA toolkit): a
hand-rolled FFI (zero NVIDIA SDK source) reached a live `OptixDeviceContext` on
driver 591.86, read `RTCORE_VERSION=20` (Ampere RT gen 2) from the hardware, and
instantiated the AI Denoiser — whose own log reports `buffers: fp16, xmma/xmma-jit
convolution`, i.e. **tensor-core** execution confirmed. The ABI-118 function table
is exactly 60 entries; the brute-force size probe + the full offset map (module 12,
program-group 24, pipeline 27, accel 33, sbt 48, launch 49, denoiser 52-54) are
recorded for the build. **Phase 2 device kernel BUILT** —
CUDA 13.3 installed; `crates/crucible-gpu/src/optix/path_tracer.cu` (the CUDA/OptiX
megakernel counterpart of `pathtrace.wgsl`) compiles to `path_tracer.ptx` (37 KB,
all three OptiX entry points + `_optix_*` intrinsics, sm_86 → JITs to Blackwell)
via nvcc + the public OptiX headers; `build_ptx.ps1` reproduces it. So **both hard
Phase-2 unknowns are resolved**: the hand-rolled host FFI reaches a live OptiX
context + tensor denoiser (spike `d7a87f6`), and the device `.cu` → correct PTX
(`25a778c`). **Remaining Phase-2 work: the host integration** —
`optixModuleCreate` (from the committed PTX) → program groups → pipeline → SBT →
accel build (BLAS/TLAS over the knot) → `optixLaunch` → readback/verify → a
`LoadKernel` + CLI. The full 60-entry function-table offset map is recorded (§3
+ memory). Phase 3 (SER, §4) remains scoped-not-built.

## 0. TL;DR

- A path tracer's distinctive QC value is **coverage**, not wattage: incoherent
  secondary rays give **divergent, deep BVH traversal** + sustained **SM↔RT**
  hand-off that the current coherent `rt` cannot reach. Honest caveat: incoherent
  rays are *less* cache-efficient, so peak Gray/s (and maybe RT power) may be
  **lower** than `rt`'s 5.35 Gray/s — sell coverage/realism, not watts. [INFER]
- **The "NVIDIA" reason to go native is tri-unit co-stress**: OptiX can drive
  **RT + SM + Tensor cores in one kernel** — RT/SM from the trace, Tensor from
  **Cooperative Vectors** (`optixCoopVecMatMul`, a tensor-core matmul callable
  *inside* the shader) and/or the **AI Denoiser**. No portable API can do that.
- **Determinism is the one hard constraint** (our self-consistency model needs
  bit-exact reproducibility across dispatches). It is solved by a **megakernel**
  (per-pixel loop over fixed samples×bounces in registers, index-seeded RNG,
  write once) — never float-atomic accumulation. Same construction `rt` already
  uses. [FACT+INFER]
- **Recommended phasing:** (1) a **portable inline multi-bounce** path tracer
  first — fast, deterministic, keeps the single-binary/no-shader-compiler
  property, de-risks the model; then (2) the **OptiX NVIDIA-native tri-unit**
  test — the headline "NVIDIA path tracing" deliverable. A Vulkan RT-pipeline /
  SER test is a separate, Blackwell-specific angle (§4).

## 1. Three candidate paths

| | A. Inline multi-bounce (extend `rt`) | B. OptiX (NVIDIA-native) | C. Vulkan RT-pipeline / SER |
|---|---|---|---|
| API | `ash` VK_KHR_ray_query (have it) | OptiX 9.1 (driver-resident) + `cudarc` | `ash` VK_KHR_ray_tracing_pipeline |
| Vendors | **all (NV/AMD/Intel)** | **NVIDIA only** | all (NV/AMD/Intel) |
| RT-core traversal stress | ✅ full (divergent) | ✅ full (divergent) | ✅ full (same silicon) |
| SM co-stress | ✅ (BRDF/RNG per bounce) | ✅ | ✅ |
| **Tensor co-stress (in-loop)** | ❌ | ✅ **CoopVec + denoiser** | ❌ |
| SER / HitObject (Ada/Blackwell reorder HW) | ❌ | ✅ (OptiX has it) | ✅ **only here on Vulkan** |
| Shader compiler | **naga at runtime, none committed** | nvcc→PTX **at build**, commit `.ptx` | glslang/Slang **at build**, commit `.spv` |
| Rust reach | trivial (extend `rt.wgsl`) | **hand-rolled FFI** (~15 fns) | ash native (+SBT) |
| New LOC | **~150–300** | ~1.5–2.4k Rust + ~300 CUDA | ~+250–400 |
| Effort | **~1–1.5 wk** | **~3–5 wk** (+denoiser 1–2, +CoopVec ~1) | ~2–4 wk |
| Keeps single-binary / no-compiler | ✅ | ✅ target (build needs toolkit) | ✅ target (build needs glslang) |

## 2. Path A — portable inline multi-bounce (the fast win)

Extend `rt.wgsl`'s verified loop from a coherent primary-ray fan into a real
**megakernel path tracer**: per pixel, loop `samples × bounces` entirely in
registers — cosine/GGX-sample a BSDF, spawn the next `rayQuery` from the
(incoherent) hit, carry `radiance`/`throughput`/`rng` in locals, fold the final
accumulated radiance (bitcast) into the existing FNV checksum. Miss = environment.

- **Keeps everything good about `rt`:** naga compiles it (still inline ray-query,
  no pipeline), single binary, **no shader compiler anywhere**, cross-vendor,
  and it reuses `rt.rs`'s whole ash/BLAS/TLAS/verify/`--preview` scaffolding.
- **New coverage now:** divergent/deep traversal + per-bounce SM work — the real
  gap vs today's coherent single-bounce `rt`. [INFER]
- **Determinism:** trivial — same register-accumulation, index-seeded (`hash2`
  from `crucible-core/src/rng.rs`) construction the suite already relies on.
- **Verification:** pre-(no-)denoise radiance self-consistency + liveness,
  exactly like `rt`. Bounce/sample counts are the TDR-granularity knob.
- **Cost:** ~150–300 LOC, ~1–1.5 wk. **This also de-risks the whole path-tracing
  determinism/verification model before the big OptiX build.**

## 3. Path B — OptiX (the NVIDIA-native tri-unit test)  **[verdict: GO]**

The literal "NVIDIA path-tracing test", and the only path that co-stresses **RT +
SM + Tensor** in one launch. Re-confirmed against OptiX **9.1 (ABI 118)** primary
sources.

- **Driver-resident, ship nothing.** [FACT] The binary links no OptiX lib;
  `optixInit` `LoadLibrary`/`dlopen`s `nvoptix.dll`/`libnvoptix.so.1`, resolves
  `optixQueryFunctionTable`, fills a function-pointer table. Runtime lives in the
  driver — no SDK/toolkit on target. Same "NVIDIA-driver-only" precedent as our
  `cudarc` link path.
- **Min driver:** OptiX 9.0 = **R570**, 9.1 = **R590**. [FACT] Blackwell 5090
  needs R570+ regardless; Ampere 3070 fine. Both bench GPUs covered.
- **Structure.** [FACT] Iterative path loop **in raygen** (not recursion —
  `maxTraceDepth 1–2`), 2 payload words pointing at a per-ray-data struct
  (radiance/throughput/next-ray/seed/depth), `tea`/LCG per-pixel RNG. Program
  groups raygen/miss/hitgroup + a 32-byte-header SBT.
- **Tensor path — Cooperative Vectors.** [FACT, the key find] OptiX 9 exposes
  `optixCoopVecMatMul` — a tensor-core matrix-multiply callable **inside** shader
  programs. This is a *cleaner* tri-unit primitive than the denoiser: it puts
  guaranteed Tensor-core work in the same launch already hammering RT + SM, and
  its (deterministic) output folds straight into the checksum. **[GATED]**
  CoopVec targets Ada/Blackwell tensor cores — verify it isn't a slow fallback
  (or feature-gated) on the Ampere 3070; it shines on the 5090.
- **AI Denoiser.** [FACT] Standalone-invokable (needs only an `OptixDeviceContext`
  — no pipeline/SBT/AS): `Create → ComputeMemoryResources → Setup → Invoke`,
  tensor-accelerated recurrent autoencoder over beauty(+albedo+normal) buffers.
  But NN output has **no bit-exact guarantee** → **liveness-only**, never a
  checksum target.
- **Rust.** [FACT] No maintained OptiX-9 crate → **hand-rolled FFI**: ~15 host
  functions + ~12–15 structs + ~10 enums + a ~40-line loader. **Reuse the
  existing `cudarc`** for `CUcontext`/stream/`CUdeviceptr` — the FFI is only the
  OptiX-specific entry points. Device `.cu` compiled **nvcc→PTX at build on the
  bench**, committed as an `include_bytes!` `.ptx` blob (text, no NVIDIA binary
  in-repo); the **target JIT-compiles it in-driver, toolkit-free**.
- **Licensing.** [FACT] Compiled binaries ship **royalty-free, commercial or
  OSS**; only redistributing the *SDK* is barred. Hand-rolling (zero NVIDIA
  header source committed; PTX not OptiX-IR) keeps the MIT repo clean. One
  low-stakes courtesy: NVIDIA *requests* an online notification for commercial
  releases — flag to owner, not a fee/copyleft. **[GATED — worth a two-line
  courtesy notice; confirm with counsel if commercializing.]**
- **Effort.** [INFER] ~1.5–2.4k Rust LOC + ~300 CUDA, **~3–5 wk** (front-loaded
  by ABI struct-layout debugging + first green `optixLaunch` on 3070 **and**
  5090). Denoiser +1–2 wk; CoopVec +~1 wk.
- New `optix` cargo feature in `crucible-gpu` (NVIDIA-only), alongside `cuda`.

## 4. Path C — Vulkan RT-pipeline / SER (separate Blackwell angle)

A full raygen/closest-hit/miss **pipeline** (vs our inline ray-query) is **not a
harder RT-core stress** — inline `rayQuery` and pipeline `traceRay` hit the *same*
traversal silicon. [FACT] Its unique value is being the **only** door on Vulkan to
**Shader Execution Reordering** (`VK_EXT_ray_tracing_invocation_reorder`, the
Ada/Blackwell ray-scheduling HW, via the HitObject model — impossible from inline
compute ray-query), plus cluster-AS (Mega Geometry) and Blackwell's Linear-Swept-
Spheres primitive. [FACT]

- **SER is a no-op on the Ampere 3070, full HW on the Blackwell 5090.** [GATED]
  Clean QC hook: reordering must **not** change results → *"same checksum with
  reorder on vs off"* is a correctness test of the 5090's scheduling silicon.
- **Cost:** loses naga (needs glslang/Slang at build + committed `.spv`), ~+250–
  400 LOC ash/SBT (SBT is the fault-prone part), ~2–4 wk.
- **Verdict:** worthwhile *only* as an explicit **"stress the newest 5090
  silicon"** test, later — not as the general path-tracing test. **Skip DXR**
  (Windows-only, dxc toolchain, no capability gain over `VK_EXT_...reorder`).
  **Drop displacement-micromaps** (`VK_NV_displacement_micromap` is deprecated).

## 5. The determinism constraint (applies to A and B)  **[the load-bearing risk]**

Our verification is same-device self-consistency: every read-back must reproduce
the first bit-for-bit. Monte-Carlo path tracing is compatible **iff**:

- **Do** [FACT+INFER]: index-seeded per-pixel RNG (`hash(pixel, sample, bounce)`,
  reuse `rng.rs::hash2`); **fixed** sample + bounce counts; accumulate each
  pixel's samples in **registers/PRD in fixed order** and write once; bitcast the
  accumulated radiance into the checksum (exactly like `rt.wgsl`'s `bitcast(hit.t)`
  today, already proven bit-exact over 79 read-backs on the 3070).
- **Avoid** [FACT]: **float-atomic accumulation** (reduction order is
  scheduler-dependent → different rounding → checksum drift) and **wavefront /
  ray-reordering** architectures (order-dependent). Use a **megakernel** — which
  *also* maximizes divergence stress, so it's the right call twice.
- **FMA contraction + transcendentals** (`sin/cos/sqrt/pow` in sampling) differ
  *across* devices/drivers, **not** on the same device — so the checksum stays
  **same-device self-consistency only**, never a repo-baked golden. This is
  already the stated stance for `rt`/`render`/`tensor`; no change. [FACT]
- **[GATED]** Validate bit-exactness empirically on **both** 3070 and 5090 early
  — it's the item most likely to force a kernel redesign, independent of API.

## 6. Verification design (recommended)

1. **Checksum = raw pre-denoise accumulated radiance** (fixed camera + seed +
   sample/bounce counts, register accumulation, bitcast→FNV). Verifies **RT + SM**
   correctness — a wrong intersection or wrong BRDF/RNG math changes radiance bits
   → FAIL.
2. **Denoise / CoopVec display = separate.** A vendor denoiser (OptiX AI) →
   **liveness-only** (non-empty, finite). A **deterministic** tensor pass
   (CoopVec matmul, or a hand-rolled `cmma` filter reusing `tensor.rs`) → can get
   its **own second self-consistency checksum** = genuine in-kernel **Tensor-core
   verification**. Spike-first (like the int8 tensor spike): confirm it reproduces
   dispatch-to-dispatch before trusting it, else fall back to liveness.
3. **Live `--preview`** reuses `preview.rs::PixelPresenter` verbatim (CPU-side
   RGBA8 upload + upscale) — an orbiting, denoised path-traced view, gated by a
   `shade`-style uniform so plain runs stay byte-identical. Near-free, and the
   most visually compelling test in the suite.

## 7. Suite fit

- **`LoadKernel` is trivial** — same shape as `rt.rs` (`catch_unwind` setup,
  probe+liveness, `ShapeDriver::tick()` loop, `verify_every` → readback+FNV).
  Inherits Steady/Burst/Pulse/Jitter + chaos/gauntlet with no orchestrator change.
- **New `pathtrace` command, not an `rt` flag** [INFER]: the repo convention is
  one named kernel = one silicon identity + one validated load. `rt` is
  deliberately the *shallow, coherent, fast* RT gate (express gauntlet tier); a
  divergent multi-bounce tracer is a materially different failure-mode profile
  (standard/full tiers). Share the Vulkan/BLAS/TLAS/preview plumbing via a small
  refactor (extract `rt.rs`'s context setup into a shared module). Same torus-knot
  BVH is already a fine divergent target. (Stopgap if the refactor isn't wanted:
  `rt --bounces N --samples M`, N=0 = today — honest but conflates identities.)
- **Complementary, not duplicate:** `rt`+`tensor` already catch a *dead*
  intersection unit or tensor lane. The path tracer's marginal value is
  **divergent-traversal scheduling faults**, **depth/grazing-angle intersection
  errors** the coherent fan never samples, and **combined-unit power/thermal/
  cache-contention** failures — the GPU-internal analog of the suite's own
  cross-load thesis. Real, incremental coverage. [INFER]

## 8. Recommendation

**Build it — phased, portable-first.**

1. **Phase 1 — `pathtrace` (portable inline multi-bounce).** ~1–1.5 wk. Highest
   value/lowest risk: real divergent RT + SM coverage now, keeps naga/single-
   binary/no-compiler, reuses all of `rt`'s plumbing + `--preview`, and **proves
   the determinism/verification model** before the OptiX epic.
2. **Phase 2 — OptiX NVIDIA-native tri-unit (`--features optix`).** ~3–5 wk. The
   headline "NVIDIA path tracing" test: OptiX path tracer + **CoopVec tensor-core
   co-stress** (+ optional AI denoiser), RT+SM+Tensor in one launch, driver-only,
   MIT-clean via hand-rolled FFI reusing `cudarc`. Spike the CoopVec-on-Ampere
   and determinism questions first.
3. **Phase 3 — (optional) Vulkan RT-pipeline SER test.** A distinct, 5090-focused
   *scheduling-silicon* correctness test (reorder-invariance checksum), only if
   stressing Blackwell's SER hardware becomes an explicit goal. Not the general
   path tracer.

## 9. Open decisions for the owner

- **Scope of "NVIDIA":** portable path tracer that also runs on the 5090 (Phase
  1), or the OptiX tri-unit test that is NVIDIA-only but stresses the tensor cores
  in-loop (Phase 2), or both phased?
- **Tri-unit tensor path:** CoopVec (in-shader, deterministic, checksummable) vs
  the AI Denoiser (liveness-only) vs a hand-rolled `cmma` denoise (deterministic,
  reuses `tensor.rs`, cross-vendor)?
- **Commercial-notification courtesy** for OptiX — send the two-line notice?

## Sources
OptiX 9.0/9.1 Programming Guide + host API (`raytracing-docs.nvidia.com/optix9`);
`github.com/NVIDIA/optix-dev` headers (`optix.h`/`optix_types.h`/`optix_stubs.h`/
`optix_function_table.h`/`license_info.txt`); NVIDIA forums 201549 (path loop),
324417 (R570), 196368 (licensing); `developer.nvidia.com/optix-denoiser`;
[wgpu ray_tracing.md](https://github.com/gfx-rs/wgpu/blob/trunk/docs/api-specs/ray_tracing.md)
(naga = ray-query only); [Vulkan raytracing chapter](https://docs.vulkan.org/spec/latest/chapters/raytracing.html);
[ash 0.38 ray_tracing_pipeline](https://docs.rs/ash/0.38.0/ash/khr/ray_tracing_pipeline/struct.Device.html);
[VK_EXT_ray_tracing_invocation_reorder](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_ray_tracing_invocation_reorder.html);
[PBRT wavefront/PCG seeding](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Path_Tracer_Implementation);
[FP non-associativity (arXiv 2408.05148)](https://arxiv.org/html/2408.05148v3);
[CUDA FP / FMA contraction](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
