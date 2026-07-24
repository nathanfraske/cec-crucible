<!-- SPDX-License-Identifier: MIT -->

# Spike 3a — CubeCL GPU thrasher probe

Throwaway probe that answered the Phase 3 backend questions before committing to
`crucible-gpu`. Findings are written up in
[`docs/gpu-plan.md` §12](../../docs/gpu-plan.md); this is the code that produced
them, kept so the measurements are reproducible and so 3b has a starting point.

**This is NOT part of the workspace.** It is listed under `exclude` in the root
`Cargo.toml`, because it pulls ~386 external crates (CubeCL + wgpu) and the core
suite's whole point is zero external dependencies. Building the workspace does
not build this.

## Scope

This probe covers the **wattage test only** — can a CubeCL kernel pin a board at
its power limit. The **VRAM integrity test is a separate test** and is not
implemented here; memory traffic appears only as a power-tuning knob, because
GDDR and the memory controller are a large share of board watts.

## Run

```
cd spikes/gpu-3a
cargo run --release -- --runtime wgpu --device discrete --mode mix --seconds 30
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--runtime` | `wgpu` | `wgpu` or `cuda` (CUDA needs a toolkit at runtime — see below) |
| `--device` | `discrete` | `discrete`, `integrated`, `default` (wgpu only) |
| `--mode` | `fma` | `fma` = pure ALU, `mix` = ALU + coalesced VRAM streaming |
| `--seconds` | `10` | Duration |
| `--threads` | `1048576` | Total GPU threads |
| `--iters` | `8192` | Inner loop iterations per dispatch (runtime scalar, not unrolled) |
| `--workgroup` | `256` | Workgroup size |
| `--data-mb` | `1024` | VRAM buffer size in `mix` mode |
| `--stride` | `= threads` | Element stride; the default keeps accesses coalesced |

Measure power alongside it with `nvidia-smi dmon -c 30`.

## Headline results (RTX 3070, 240 W limit, idle 43.9 W)

| | pure ALU | ALU + VRAM |
| --- | --- | --- |
| Board power | 167–187 W (~75%) | **208–221 W (~92%)** |
| SM / mem util | 71–95% / 4–5% | **100% / 100%** |
| Throughput | 5.58 TFLOP/s | 1.35 TFLOP/s |

Throughput moves *inversely* to power: for a wattage test, watts are the
objective function, not FLOP/s.

Also confirmed: runs unmodified on NVIDIA **and** Intel UHD 630; longest dispatch
53.5 ms vs the ~2000 ms TDR watchdog; fp32 results bit-identical across the two
vendors.

## Gotchas this probe uncovered

- **`ABSOLUTE_POS` and `Array::len()` are `usize`** in CubeCL 0.10. Mixing `u32`
  into index math does not compile.
- **Keep the inner loop bound a runtime scalar**, not `#[comptime]` — a comptime
  bound risks unrolling a long loop into a shader-compiler explosion.
- **CUDA builds without a toolkit but does not run without one**: CubeCL
  JIT-compiles via NVRTC, so `nvrtc.dll` must be present at runtime.
- **A missing NVRTC panics on a CubeCL worker thread and the host loop keeps
  "succeeding"** — it reported 1.65 TFLOP/s for work that never ran while the GPU
  sat at 43 W. Only the output read-back caught it. Hence the probe always
  verifies output, and `crucible-gpu` must treat compile/device errors as a hard
  FAIL rather than trusting timings.
