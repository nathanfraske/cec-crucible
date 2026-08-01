# cec-crucible v0.0.5 — Alpha 5 (Windows x64)

PC-build stress, validation and benchmark suite.
**Mission: if something is ever going to fail, make it fail in the shop.**

## Read this first

**Two tests could report a confident PASS on hardware that does not exist.** If
you have run anything with `--gpu-device integrated`, re-test it. Details below.

## Install

1. Download `cec-crucible-0.0.5-win-x64.zip` and extract it.
2. Double-click **`INSTALL.cmd`**.

Per-user, **no admin required**. Installs to `%LOCALAPPDATA%\Programs\cec-crucible`,
adds it to your user PATH, and creates Start Menu + Desktop shortcuts.

**Launch the TUI:** double-click the **CEC Crucible** shortcut, or open a *new*
terminal and run `cec-crucible` with no arguments.

Self-contained (statically linked CRT — no Visual C++ Redistributable), and
PresentMon is bundled.

---

## Fixed: two false PASSes on integrated graphics

Running Alpha 4 against an Intel UHD 630 turned up four defects, two of which
answered wrongly rather than failing.

**`vram` passed on a device with no VRAM.** `vram --gpu-device integrated
--vram-mb max` reported `PASS … 6784 MiB VRAM` — which is 85% of the *discrete*
card's 8 GiB. A wgpu class ordinal (`Integrated(0)` = "the first integrated
adapter") was being handed to DXGI as an *adapter index*, and DXGI adapter 0 is
the high-performance card. It filled system RAM through the iGPU and called it
VRAM integrity.

**`link` reported PCIe bandwidth across a bus that does not exist.** An
integrated GPU is on the same die as the CPU and shares its memory controller.
There is no PCIe link. The test measured a copy inside system RAM and printed
`H2D ~5.4 GB/s`.

**`gpu` crashed and printed an impossible number.** It asked for a hardcoded
1 GiB buffer; the iGPU's storage-binding limit is 1023 MiB against the 3070's
2047, so the default overshot by exactly one megabyte. The run panicked — and
still printed `~2.61 TFLOP/s` for a part that peaks near 0.46, computed from four
dispatches and **zero verifications**.

**The sensor plane reported the wrong GPU.** NVML was opened at index 0
regardless of which adapter was under test.

### The cause was one thing: no adapter identity

Four unrelated index spaces — wgpu class ordinal, DXGI adapter index, NVML device
index, AMD ADL index — were being treated as interchangeable integers. Identity
is now resolved **once**, from wgpu (name, vendor, and authoritatively
`device_type`, so UMA is *known* rather than guessed from memory sizes) joined
with DXGI (the **adapter LUID**, dedicated VRAM, shared memory).

`gpu-info` now shows exactly what each selector resolves to, because that is the
question that used to be answered wrong:

```
adapters (as this machine presents them):
  NVIDIA GeForce RTX 3070 [NVIDIA] — discrete, 8018 MiB VRAM, Vulkan,
      max buffer 4094 MiB / binding 2047 MiB  [luid_0x00000000_0x0001b019]
  Intel(R) UHD Graphics 630 [Intel] — integrated (UMA), 16289 MiB shared, Vulkan,
      max buffer 4096 MiB / binding 1023 MiB  [luid_0x00000000_0x00021525]

selectors:
  integrated0 : available
      -> Intel(R) UHD Graphics 630 [Intel] — integrated (UMA), …
```

On a UMA adapter the tests now say what they actually measured:

```
vram max: 4021 MiB — 50% of 8043 MiB available system RAM (UMA adapter: it has
                     no video memory of its own to fill)
  [vram] PASS  3968 MiB shared (UMA, system RAM) …
  [pcie] PASS  up ~5.6 GB/s, down ~1.9 GB/s
               (UMA adapter: NO PCIe LINK EXISTS — this is memory-controller bandwidth)
```

Full findings and the AMD roadmap: `docs/vendor-support-plan.md`.

---

## Fixed: the PCIe test on Resizable BAR systems

A field capture reported **H2D ~370 GB/s** — about six times faster than the
fastest PCIe link that exists. With ReBAR the whole of video memory becomes
host-visible, so a Vulkan allocator asked for a `HOST_VISIBLE` upload buffer can
return memory that is also `DEVICE_LOCAL`. The "upload" was a VRAM→VRAM copy at
memory-bus speed.

Two defences:

* The link test now **forces DX12 on Windows**, where `MAP_WRITE` means
  `D3D12_HEAP_TYPE_UPLOAD` — system memory by definition. ReBAR's device-local
  host-visible heap is the separate opt-in `GPU_UPLOAD` type, which wgpu does not
  use for this.
* The measured rate is checked against the **negotiated link**, read from the
  driver, and a result above it **fails the stage**:

```
  [pcie] PASS  H2D ~6.8 GB/s, D2H ~1.8 GB/s; link Gen3 x16 (~15.8 GB/s one-way)
```

An impossible figure can no longer be printed as bandwidth, whatever future
backend produces it. **Caveat:** the DX12 change is reasoned from the D3D12 heap
model and cannot be proven here — this bench has no ReBAR system that exhibited
the fault. The ceiling check is tested directly against the 370 GB/s figure.

---

## Memory: never page, never OOM, always reclaim

**`--mb max` used to take 90% of available RAM.** Available memory is not spare
memory — Windows holds a file cache, the GPU driver has pinned pages, and the
operator's session needs a working set. Commit past that and the machine starts
swapping, at which point a memory test is measuring the disk while the desktop
stops responding.

One rule now governs both RAM and UMA video memory: reserve **2 GiB or an eighth
of total RAM, whichever is larger**. On the UMA path the test also re-reads real
availability between chunks rather than predicting it, and stops while there is
still headroom.

**VRAM is now released, and the release is measured.** Dropping the buffer
handles was not enough: CubeCL pools device allocations against a process-wide
client, so memory stayed committed for the life of the process — which is why a
later stage could hit OOM on a card that looked empty. Both GPU kernels now run
an explicit cleanup, and report what it achieved rather than assuming it worked:

```
  [vram] PASS  3968 MiB shared (UMA, system RAM) … ; reclaimed 3984 MiB (0 MiB still reserved)
```

A pool that does not shrink is reported as a warning, because the next stage is
the one that pays for it.

---

## Every Windows event log, not a curated list

The detector plane still decides the verdict from WHEA / TDR / bugcheck / disk
events. Alongside it, every run now archives **everything the machine logged in
the run window, across every channel it will let us read**:

```
events:  crucible-…​.eventlog.jsonl  (789 event(s) from 474/1289 channel(s), 815 not readable)
```

The counts are the point: "we read everything" and "we read what we were allowed
to" are different claims, and most of a Windows machine's channels are simply not
enabled (Security needs elevation). One JSON object per line, so `findstr` works
on it directly. The provider nobody has written a classifier for yet is exactly
the one that matters when a machine fails in a new way.

---

## Charts as PNGs, and one page per machine

Every run with `--telemetry-csv` now also writes **PNG charts** — GPU power, GPU
temperature, CPU effective clock, CPU utilisation — with peak and mean on the
title row and a red rule wherever an error count rose. Rendered by a hand-written
rasteriser and PNG encoder (no dependencies), and small enough to paste into a
ticket: about 20–60 KB each.

And **one page per machine**, gathering every run for one box:

```
cec-crucible package --open
```

Verdict table newest-first, charts inline, every artifact linked. It is built by
**scanning the folder, not by remembering what this process wrote** — the runs
worth finding are the ones that crashed, and a crashed run recorded nothing. A
run with a crash file shows `CRASHED` regardless of what verdict it managed to
write.

Charts are embedded as data URIs, so the page still renders after you email it.
It is on the menu under **Diagnostics → Open reports**, and there is a Settings
toggle to open it automatically when a run finishes.

---

## Settings that survive a restart

The Settings screen reset on every launch, so the option you forgot to re-arm was
the one you needed. Settings now persist to
`%APPDATA%\cec-crucible\settings.conf` — plain `key = value`, safe to edit by
hand, and a corrupt file falls back to defaults rather than stopping the tool.

They apply to command-line runs too, as **defaults**: an explicit flag always
wins, so what you type now can never be overridden by what the menu was set to
last week. New rows: **Charts** and **Open in browser**, alongside ETW, PresentMon
and priority.

---

## Quick start

```
cec-crucible                        # interactive TUI
cec-crucible gpu-info               # what each GPU selector actually resolves to
cec-crucible run quick              # ~15s CPU/RAM/storage QC
cec-crucible uncore                 # interconnect / FCLK verification
cec-crucible run worst-case --ui --telemetry-csv    # everything at once, charted
cec-crucible package --open         # this machine's report page
```

---

## Known gaps

* **AMD is untested.** The load side already works (everything runs through
  wgpu), but there are no AMD sensors yet and nothing has been validated on AMD
  hardware, because we have none. `docs/vendor-support-plan.md` specifies the
  ADLX / ADL2 work and what hardware it needs.
* **GPU sensors are NVIDIA-only** (NVML). On AMD and Intel the power/thermal
  columns are blank and the dashboard strip does not appear — correctly reported
  as absent, but absent.
* **The elevated ETW path is still unverified.** The non-elevated refusal is
  tested; an actual `.etl` capture has not been produced on this bench.
* **The ReBAR fix is unproven end-to-end** — see above.
* **iGPU coverage is Intel-only so far.** Validated against a UHD 630; AMD APUs
  are untested, and the CPU↔iGPU contention test that APUs actually need does not
  exist yet.
* Five defects documented in `docs/game-realism.md` §1 remain unfixed.
* No network test; no display/scanout, cable or EDID validation.
* Not code-signed, so SmartScreen will warn on first run.
* Benchmark scores are calibrated against one RTX 3070 — cross-machine
  comparisons are provisional.

Full roadmap and design docs are in `docs/`.

---

MIT licensed. Built by Critical Error Computing.
