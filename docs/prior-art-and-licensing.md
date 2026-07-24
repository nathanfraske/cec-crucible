<!-- SPDX-License-Identifier: MIT -->

# Prior art + licensing (researched 2026-07-23)

Why build our own instead of wrapping existing tools. Full sources live in
CEC's internal stress-harness research notes; summarized here as the
justification for cec-crucible.

## Existing tools + commercial-use reality (shop = paid use on customer builds)

| Tool | Domain | License reality | Automation |
| --- | --- | --- | --- |
| OCCT | all-in-one | free = personal only; **Pro** (CSV) / **Enterprise** (CLI) paid, price on request | CLI at Enterprise |
| Prime95 | CPU | free incl. commercial; no explicit redistribution grant | fully scriptable |
| FurMark 2 | GPU | freeware **excluding** commercial; Geeks3D commercial license exists | good CLI |
| y-cruncher / Linpack Xtreme / TestMem5 | CPU / RAM | commercial terms **unpublished** | varies |
| MemTest86 | RAM (boot) | free edition OK for business; automation is Pro/Site | Pro only |
| MemTest86+ | RAM (boot) | GPL-2.0 | boot USB |
| memtest_vulkan | VRAM | open | CLI |
| diskspd | storage | MIT | full CLI |
| FurMark/Kombustor/OCCT GPU | GPU power-virus | all closed | — |
| LibreHardwareMonitor | telemetry | MPL-2.0 (WinRing0 driver — HVCI risk) | lib/WMI |
| gpu-burn | GPU | CUDA/**Linux** only | — |

## Findings

1. **No open Windows GPU power-virus exists.** The FurMark-class tools are all
   closed; gpu-burn is CUDA/Linux. A custom compute burner is the only open
   path — and it's the piece worth owning most.
2. **Steady 100% misses bugs.** Field experience + community consensus: FurMark
   / steady loads pass builds that fail on real transient/mixed workloads. The
   value is in **load-shape control** no off-the-shelf tool exposes.
3. **Licensing is a maze for a paid shop.** OCCT's automation is paywalled to
   Enterprise; FurMark bars commercial use without a license; several tools
   have unpublished commercial terms. A clean-licensed in-house tool removes
   the whole question for customer-facing QC.
4. **The parts that are easy are easy.** CPU burn (FMA loops), RAM patterns,
   storage (scratch-file R/W/verify), QPC markers, JSON reports — all
   straightforward Rust. diskspd (MIT) and memtest_vulkan (open) are references
   to match, not blockers.
5. **Prior-art architecture to copy:** CoreCycler (open PowerShell orchestrator
   driving external stress EXEs, per-core, output-parsing) — its structure is
   what CEC's companion PowerShell harness already mirrors; cec-crucible
   provides the native kernels it orchestrates.

## Conclusion

Build the load engine in-house (license-clean, load-shape-controllable,
marker-emitting, device-ID'd), adopt memtest_vulkan for the VRAM stage as a
reference, and keep a licensed OCCT/BurnInTest only as an optional certified
customer-facing report layer if ever wanted. The GPU power-virus is the one
multi-week piece; everything else is phase-1/2 straightforward.

## Verification caveats carried forward

- **LibreHardwareMonitor** loads a WinRing0-lineage kernel driver that
  Microsoft's vulnerable-driver blocklist / HVCI can block — validate the
  pinned build on the shop's HVCI-enabled image before relying on its telemetry.
- **CubeCL** backend maturity varies per GPU vendor — spike NVIDIA/AMD/Intel
  before committing; plain wgpu/Vulkan compute is the safe floor.
