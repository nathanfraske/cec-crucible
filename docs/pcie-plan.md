<!-- SPDX-License-Identifier: MIT -->

# cec-crucible — PCIe / motherboard link-integrity plan

**Goal:** detect a **bad or marginal PCIe link** to the GPU — a bad riser
(vertical mounts are common in these builds), a marginal cable, poor seating, or
dirty/damaged contacts. To our knowledge no Windows tool does this, which is why
it is worth building.

This document is the scoped design of record. It was researched in depth
(Windows FFI feasibility verified empirically on the bench); the honest verdict
and its limits are stated up front so nobody builds on a false premise.

## The one insight that reframes everything

**A marginal PCIe link does not primarily get *slower*. It RETRIES.** When a TLP
arrives with a bad CRC, the data-link layer replays it. The data still arrives
correct; the throughput barely moves; but the correctable-error counters climb.

Two consequences follow, and they drive the whole design:

1. **A bandwidth benchmark will happily pass a genuinely bad riser.** Saturation
   is therefore *stimulus*, not the measurement.
2. **End-to-end data verification also passes it** — because the retry delivered
   correct data. Verification only catches *uncorrected* corruption, which is
   the rarer, already-catastrophic case.

So the actual detector is **error counters and link-state**, not bytes/sec and
not checksums. That is the heart of this test and the part that is hardest on
Windows.

## Honest verdict (read this before building)

Split into three planes by how certain each is:

| Plane | What it catches | Feasible on Windows, no kernel driver? |
| --- | --- | --- |
| **A. Link-training check** | A link that trained *low* (x8 not x16, Gen3 not Gen5) — mis-seat, bad riser, dead lanes | ✅ **Yes, verified.** Non-admin, zero-dep FFI. Cheap and reliable. |
| **B. Transfer + verify load** | *Uncorrected* corruption crossing the link; keeps the link hot as stimulus for A and C | ✅ **Yes.** Buildable now on wgpu; optional CUDA fast path. Cannot saturate Gen5 (see below), but does not need to. |
| **C. Error-delta detector** | The retry signature (Bad TLP/DLLP, replays, receiver errors) — *the actual bad-riser signal* | ⚠️ **Conditionally.** Only via WHEA event-log deltas, and **only if the board's firmware exposes PCIe AER via ACPI `_OSC`** — many consumer boards do not, in which case Windows logs nothing. |

**Bottom line:** we can *definitely* ship a link-training checker (B/A catch a
mis-trained or grossly broken link — a real, cheap QC win). We can *probably*
catch the marginal-retry case via WHEA AER deltas, **but that is gated on board
firmware and must be validated on real Gen5 hardware with a known-bad riser** —
neither of which exists on the current bench. We will not ship a kernel driver
to read raw config space (the only alternative), because the WinRing0-lineage
approach is CVE-flagged and blocked by HVCI / the Windows vulnerable-driver
blocklist by default.

## Plane A — link-training check (build first; certain)

Read each PCIe device's **negotiated vs maximum** link width and speed. A GPU
that trained to x8 Gen3 in an x16 Gen5 slot is an instant catch.

**Mechanism (empirically verified, non-admin, zero external crates):** enumerate
PCI devnodes via `cfgmgr32` (`CM_Get_Device_ID_List*`, `CM_Locate_DevNodeW`) and
read `CM_Get_DevNode_PropertyW` for the `DEVPKEY_PciDevice_*` properties (GUID
`{3AB22E31-8264-4B4E-9AF5-A8D2D8E33E62}`):

| Property | PID | Meaning |
| --- | --- | --- |
| CurrentLinkSpeed | 9 | code 1–6 = Gen1–6 (2.5/5/8/16/32/64 GT/s) |
| CurrentLinkWidth | 10 | raw lane count (1,2,4,8,12,16,32) |
| MaxLinkSpeed | 11 | as above |
| MaxLinkWidth | 12 | as above |

All `UINT32`. Decode the speed with the PCIe-spec table, **not** the header enum
(the SDK header's enum is stale, stops at Gen2). Devices that are not discrete
PCIe endpoints return `CR_NO_SUCH_VALUE` / `ERROR_NOT_FOUND (1168)` — handle
gracefully. Attribute topology (which physical slot) via `DEVPKEY_Device_LocationPaths`
(the CPU x16 slot hangs off root-complex device 1 → `PCI(0100)` / ACPI `PEG*`;
chipset ports are `RP*`). This mirrors the existing SMBIOS/DeviceIoControl FFI
style already in `crucible-core` and `crucible-storage`.

**Critical limitation (verified):** these values are **cached at device
enumeration**, not re-read from config space per query. So Plane A catches a link
that trained low **at boot**, but polling it *during load* will very likely
**miss** a link that downtrains dynamically under stress — that would need a PnP
re-enumeration (admin, disruptive) or a live config-space read (kernel driver,
blocked). Report it as a boot-state check, not a live one, and do not over-claim.

## Plane B — transfer + verify load (build now; the stimulus)

The existing GPU thrasher does **not** touch PCIe — its compute and VRAM traffic
are on-card. This plane moves real bytes host↔device and verifies them.

**New module** `crates/crucible-gpu/src/link.rs` → `LinkKernel`, implementing the
existing `LoadKernel` trait so it joins `run cross` / `worst-case` under one
`StopFlag` / `Budget` / `MarkerLog` like every other domain. Add `Kind::Pcie` to
the core enum (or reuse `Kind::Gpu` with `name()=="link"` for a zero-core-change
first cut).

**Transfer engine (wgpu default path):** pre-allocate a small pool of upload
(`MAP_WRITE|COPY_SRC`), device (`COPY_DST|COPY_SRC`), and readback
(`MAP_READ|COPY_DST`) buffers (64–256 MiB, queue depth 2–4); fill each upload
buffer **once** with a verifiable SplitMix64 pattern and reuse it across the run
(this amortization keeps host-RAM cost ~1× instead of ~3×). Each iteration
records H2D and D2H copies, submits, polls, and periodically checksums the
readback against the known pattern. Any mismatch = uncorrected corruption across
PCIe = FAIL. Keep the same "verified nothing ⇒ FAIL" guard the other kernels
have (a dead transfer engine must not report a clean pass — the spike-3a lesson).

**Two verified facts that shape it:**
- **wgpu exposes one queue per device** — it cannot address the GPU's dedicated
  copy engines and cannot do true full-duplex. Uploads and downloads serialize
  on the one queue. So wgpu keeps the link continuously busy in both directions
  but will not hit line rate.
- **An optional CUDA path deploys without the toolkit.** A *transfer-only*
  cudarc workload (pinned host memory via `cuMemAllocHost`, multiple explicit
  streams, async copies) never touches NVRTC — NVRTC is only loaded to *compile
  kernels*, which transfers do not do. Verified from the cubecl/cudarc sources
  and this toolkit-less bench. So unlike the compute thrasher, the CUDA fast path
  here is ship-anywhere (driver-only). Gate it behind the existing `cuda`
  feature; NVIDIA-only, wgpu stays the portable floor.

**Saturation is not the goal, and cannot be reached anyway:**
- **Host-RAM ceiling.** 63 GB/s one-way is most of a dual-channel DDR5-6000
  budget (~90 GB/s theoretical, ~60–75 sustained); bidirectional ~126 GB/s
  exceeds it outright. On the shop's typical dual-channel builds, "saturated
  bidirectional" is physically a *host-RAM* statement, not a link statement. The
  test must therefore report a **three-point baseline** — host-RAM copy ceiling,
  on-card VRAM bandwidth, and achieved PCIe throughput — so a RAM-limited result
  is never misread as a bad link.
- **Payload patterns don't reach the wire.** PCIe Gen3+ scrambles every lane
  with an LFSR, so a software-chosen `0xAA`/alternating payload is whitened to
  pseudo-random on the physical lanes. You cannot target the serial eye from a
  memcpy buffer. The real levers are **volume, duration, continuity, and
  bidirectionality** — high-entropy data kept flowing so the link stays pinned
  in L0 at top speed (idle gaps let ASPM downshift and mask the test). This is
  the opposite of the wattage thrasher's bursty design — a deliberate difference.
- **~40 GB/s of sustained bidirectional traffic is sufficient stimulus.** The
  link signals at full rate whenever it is trained and busy regardless of payload
  throughput; integrity is a property of the analog channel and the signaling
  rate, not of hitting line rate. More bits/sec raises statistical sensitivity
  marginally; it is not required.

## Plane C — error-delta detector (the real signal; firmware-gated)

This is what actually catches a retrying-but-alive bad riser. On Windows,
**without a kernel driver, the only path is WHEA event-log deltas.** There are no
user-mode hardware error counters, and no user-mode config-space access (verified
— the AER capability lives in extended config space, reachable only from a kernel
driver, and the WinRing0-class shim is HVCI-blocked / CVE-flagged).

**Mechanism (verified, non-admin):** via `wevtapi` (`EvtQuery`/`EvtSubscribe` +
`EvtRender`), watch **Microsoft-Windows-WHEA-Logger** in the **System** log.
Count **Event ID 17 (corrected)** and **18 (fatal)** across the bracketed load
window (delta = post − pre). For each event, pull the raw `WHEA_ERROR_RECORD`
(CPER) from the event's `RawData`, confirm it is PCIe (`ErrorSource == 4`
`WheaErrSrcTypePCIe`, or the PCIe CPER section GUID
`{CF93C01F-1A16-4DFC-B8BC-9C4DAF67C104}`), read the Segment:Bus:Device.Function to
attribute it to a device, and decode the AER **Correctable Error Status** bits to
see *which* fired:

- **Bad TLP (bit 6), Bad DLLP (bit 7), Receiver Error (bit 0), Replay Num
  Rollover (bit 8), Replay Timer Timeout (bit 12)** = the marginal-signal-
  integrity family. A healthy link under sustained load shows zero of these; a
  bad riser shows them climbing. This confirms the premise exactly.

Event-log timestamps default to QPC, so these correlate directly to the run's
existing QPC markers. Reading the System log needs no admin (unlike an ETW
session or the Kernel-WHEA/Errors channel).

**The make-or-break gate (verified):** PCIe AER is **optional** on Windows —
Windows only enables it if platform firmware grants control via ACPI `_OSC` and
is not firmware-first. **Many consumer/gaming boards do not**, in which case
Windows logs *nothing* no matter how bad the link is. Whether this works on the
shop's boards is the single biggest unknown and must be validated on real
hardware before relying on Plane C.

**Complements the existing harness:** the PowerShell QC harness already watches
WHEA around the whole run. This plane's contribution is the *bracketed delta*
(errors attributable to the PCIe load window specifically) and the *AER-bit
decode* (which correctable error, on which device) — not a duplicate global watch.

## Interface between the planes

All planes run in one orchestrated window under the shared `StopFlag` / phase
epoch / `MarkerLog`. Plane B (the load) owns the brackets and emits QPC-stamped
markers: a quiescent `baseline_read_point` before load, then
`pcie_load_start` / `pcie_load_stop`. Planes A and C read at those points; the
verdict is the delta. No marker-schema change is needed. Load must be
**continuous** between the brackets so a dynamic ASPM speed change is not
mistaken for a fault downtrain.

## Milestones

- **P0 — Plane A link-training check.** cfgmgr32 FFI, negotiated-vs-max per
  device, topology attribution, a `link-info` / `platform` report section.
  Buildable and testable now (reports the 3070's Gen3 x16 here). Cheap, certain,
  independently useful.
- **P1 — Plane B transfer+verify load (wgpu).** `LinkKernel`, round-trip
  verification, three-point baseline, marker brackets. Validate on the 3070
  (Gen3/4): keeps the link busy, and the verifier catches an injected byte flip.
- **P2 — Plane B CUDA fast path (optional).** cudarc-direct pinned + multi-stream
  full-duplex, driver-only. Confirm toolkit-less deploy; measure uplift.
- **P3 — Plane C WHEA/AER delta detector.** `wevtapi` FFI, Event 17/18 bracketed
  delta, CPER parse, AER-bit decode, device attribution.
- **P4 — HARDWARE SPIKE (cannot be done on the current bench).** Requires a
  **PCIe 5.0 x16 platform** *and* a **known-bad riser + known-good control**.
  This is the only way to validate the thesis: error/replay/downtrain deltas fire
  on the bad riser while a throughput-only test passes it. Also validates whether
  the target boards even expose AER via `_OSC` (the Plane C gate).

## Risks / open questions

- **Plane C may be blind on consumer boards** if firmware withholds AER. Highest
  risk; validate on real hardware first.
- **Dynamic downtrain is hard to observe** from user mode (cached link props;
  no live config-space). Plane A is a boot-state check; catching under-load
  downtraining without a driver is likely not possible.
- **No trustworthy public wgpu PCIe GB/s number exists** — must self-benchmark;
  do not cite the common "WebGPU bandwidth" figure (it measures on-card VRAM).
- **Neither Gen5 hardware nor a bad riser is on the bench** — every line-rate and
  bad-riser-detection claim here is unvalidated until P4.

## What this buys even in the worst case

Even if Plane C proves board-gated, Planes A + B still ship something no known
Windows tool offers: a device-ID'd check that the GPU (and NVMe, NIC) trained to
full width and speed, plus a sustained verified-transfer load that catches
uncorrected corruption across the link — run as part of the same one-command,
marker-emitting QC pass as everything else.
