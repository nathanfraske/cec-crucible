<!-- SPDX-License-Identifier: MIT -->
# The Gauntlet — cec-crucible QC burn-in campaign

**Mission:** *if a unit is ever going to fail, make it fail in the shop, not at the
customer's desk* — even if that costs a full day of every worst-case scenario we
can produce.

This document is the **recommended stress campaign**: which loads to run, in what
order, for how long, what each one is *for*, and — just as importantly — what it
**cannot** catch. It synthesizes two pieces of scoping research (a fault-mode →
provocation map and a full reliability-engineering campaign design) into a
runnable plan. The runnable form is [`scripts/gauntlet.ps1`](../scripts/gauntlet.ps1).

Two honesty rules are kept throughout:

- **[FACT]** = established reliability/test engineering, or documented tool behavior.
- **[INFER]** = reasoned from the code + physics. All specific *durations* and
  *yield percentages* below are **[INFER]** — engineering estimates for this shop's
  assembled-desktop mix, to be **recalibrated against real field-return data** once
  a few dozen campaigns have run. The per-phase reports the campaign emits are
  exactly the dataset for that calibration.

---

## 1. Why a gauntlet, and why *this* shape

**[FACT]** This is a textbook **Environmental Stress Screening (ESS) / burn-in**
problem: precipitate latent and infant-mortality defects into detectable failures
before ship, driving the population past the *infant-mortality knee* of the
bathtub curve so shipped units sit on the flat part.

The suite already has every *stimulus* primitive — CPU/RAM/storage/GPU-thrash/
VRAM-integrity/PCIe-link kernels, steady + burst shapes, in-phase/anti-phase/beat
choreography, one shared `StopFlag` + phase-epoch + QPC timeline, and a device-ID'd
verdict rollup where **any** kernel error → campaign FAIL. What the gauntlet adds
is a **day-scale sequencer with per-phase attribution and crash-resumability.**

### The one architectural decision that matters

**Run the gauntlet as a series of short, self-contained `cec-crucible` invocations
— one process per phase — not as a single day-long process.** This is forced by
three facts in the current code, not by style:

1. **Reports and markers are written only at `finish()`.** There is no periodic
   flush. A single 24 h process that BSODs or loses power at hour 23 produces
   **zero** output. Per-phase invocations make every completed phase a durable
   checkpoint.
2. **The marker log accumulates the whole run in RAM** (`Mutex<Vec<Marker>>`) and
   serializes once at the end. Transient profiles emit ~800 markers/s ⇒ ~70 M
   markers over 24 h, each holding four strings — multiple GB of RAM and a
   multi-GB one-shot write. Per-phase files stay bounded. (Steady phases emit no
   burst edges, so this pressure is specific to the short transient phases — which
   per-phase invocation caps anyway.)
3. **WHEA is read by the harness around each window.** One window over 24 h gives
   one corrected/fatal delta with **no attribution**. One window *per phase* pins
   a corrected-error storm to the stimulus that caused it.

So the gauntlet is an ordered **manifest** (device-id, campaign-id, phase list,
completed-set) that the harness walks, invoking the CLI per phase, bracketing each
with a WHEA read, and AND-ing the exit codes. That is precisely what
`scripts/gauntlet.ps1` does, and why a native `run gauntlet` profile is deferred
(§7) — it would have to replicate checkpoint + flush + per-phase-WHEA or it
reintroduces the day-long-single-process failure mode.

---

## 2. Reliability foundation — why hot, why cycle, why long

**[FACT] Established practice (canonical references):**

| Principle | What it says | Drives which phase |
|---|---|---|
| **Bathtub / infant mortality** (MIL-HDBK-217F; O'Connor & Kleyner) | Hazard rate is high early (workmanship/DOA), flat in useful life, rising at wear-out. Burn-in drives units past the early knee. | The whole campaign |
| **Semiconductor burn-in** (MIL-STD-883 M1015) | Steady elevated temp + voltage precipitates silicon-level latents. | Steady hot soak (P4) |
| **ESS** (MIL-STD-2164; IEST-RP-PR001) | **Thermal *cycling* is the single most effective screen for workmanship** (solder, connectors, seating) — CTE mismatch flexes joints. Steady soak is better for silicon latents. Both needed. | Macro-cycle (P5) + soak (P4) |
| **HALT / HASS** (Hobbs) | Stress beyond the customer's real workload, within the precipitation window (above spec, below destruct), to precipitate + detect fast. **Warns against over-screening** — excess stress consumes useful life. | Whole campaign; the *stop* at ~2 h / ~12 h |
| **Arrhenius** (JEDEC JEP122) | Failure rate ∝ exp(−Ea/kT); latent mechanisms accelerate exponentially with temperature (rule of thumb ≈ 2× per ~10–11 °C). **Needs time-at-temperature.** | Steady hot dwell (P4) |
| **Coffin-Manson** (JEDEC JESD22-A104, JESD47) | Thermal-cycle fatigue life Nf ∝ ΔT^−c (c≈2 for solder). Fatigue is a power law in the **swing amplitude**, not the dwell. | Macro-cycle (P5) |

**Our gauntlet is a software HASS:** electrical + thermal stress well beyond a
customer workload, with integrity-verification and WHEA detection catching the
precipitated faults.

Three load timescales flex three different things **[FACT]**:

| Load shape | Timescale | What it flexes | How we produce it |
|---|---|---|---|
| **Burst micro-cycle** | ms (20/20 ms) | Rails / VRM (die thermal mass filters ms — *not* thermal) | `--shape burst`, in/anti/beat |
| **Steady soak** | hours | Nothing mechanical; pure Arrhenius | steady `cross` / `gpu` / `mem` / `vram` |
| **Macro thermal cycle** | minutes | Solder / socket / connector / mount via CTE ΔT | harness alternates hot block ↔ idle block (§5) |

**One honest limit [FACT]:** we cannot chamber-cool. Macro-cycle ΔT is
**ambient-bounded** — a real CTE flex, but smaller than a thermal chamber's. And
the burst half-period is clamped to 60 s in-kernel, so a *native* burst macro-cycle
is only a partial excursion; the full swing comes from harness-sequenced
hot-block/idle-block alternation, which `gauntlet.ps1` does.

---

## 3. Fault-mode → provocation → detector (the core map)

Every transient failure mechanism lives in the **load edges** and in the
**simultaneous squeeze of multiple margins**. A steady 100 % load has settled
control loops, constant current, one hot component at a time, and headroom on every
axis it isn't testing — so by construction it cannot excite VRM transient response,
PSU OCP, or the cross-margin faults that pass each test in isolation.

| Fault mode | Best stimulus | Command today | Detector |
|---|---|---|---|
| **PSU OCP / multi-rail trip** | Largest **coincident** load-ON edge (CPU+GPU slam ON together) | `run in-phase` | 1 kHz rig (rail sag), operator (shutdown), WHEA |
| **VRM transient response / weak bulk caps / ring-back** | Continuous hand-off so the loop **never settles** | `run anti-phase` | 1 kHz rig (over/undershoot), WHEA |
| **The one resonant phase alignment** | **Sweep** the CPU↔GPU phase relationship | `run beat` | rig + WHEA, correlate failure time → phase via markers |
| **XMP/EXPO instability on a hot IMC** (the #1 modern-build instability) | Full mem battery **while the CPU die is hot** | `run cross` / `run worst-case` | mem first-fail vaddr+pattern, WHEA |
| **VRAM errors under heat** (GDDR temp-dependent) | Heat VRAM *while* verifying it | `run worst-case` (see G5 for a hotter variant) | `vram` first-fail |
| **GPU core instability at max boost** | Max **clock** (not max watts) | `gpu --shape steady --gpu-alu-only` | self-consistency checksum |
| **NVMe thermal throttle / controller errors** | Sustained unbuffered write→verify | `storage`, `run storage-cross` | throughput collapse, first-fail offset |
| **DMI / chipset-uplink saturation** | Many drives concurrent vs solo | `run storage-cross` | >15 % per-drive slowdown (diagnostic) |
| **PCIe signal integrity** (bad riser/cable/seat) | Sustained verified link traffic under contention | `link`, now folded into `run worst-case` | uncorrected corruption (retry detection is gated — see G2) |
| **Whole-system cross-margin** | Union of all contention | `run worst-case` | every domain self-verifies while contended |

**[FACT] Two operator insights worth stating on the bench:**

- **max boost ≠ max watts.** The wattage-max GPU mix throttles clocks at the power
  ceiling; `--gpu-alu-only` runs cooler → clocks *higher* → catches boost-clock
  instability the mix hides. VRM/PSU want the mix; core-OC stability wants ALU-only.
  **Use both.**
- **in-phase vs anti-phase is [DISPUTED] board-dependent.** In-phase maximizes
  *magnitude* (OCP, Vdroop depth); anti-phase maximizes *never-settling* (loop
  stability, ripple). **Run both**, and `beat` to sweep the middle.

---

## 4. Coverage matrix — {subsystem} × {stress type}

`●` covered · `◐` partial / only-under-contention · `○` **gap** · `—` n/a

| Subsystem \ Stress | Transient (di/dt) | Soak (Arrhenius) | Cycle (Coffin-Manson) | Data-integrity | Contention |
|---|---|---|---|---|---|
| **CPU cores** | ● cpu burst | ● cpu steady | ◐ macro-cycle (§5) | ● dual-bank recompute | ● worst-case |
| **CPU/SoC VRM + PSU** | ● in/anti/beat (±180 W GPU step) | ● steady rail current | ◐ macro-cycle | ◐ WHEA + rig | ● worst-case |
| **System RAM (cells+IMC+XMP)** | ◐ under cross | ● mem steady | — | ● 7-pattern battery | ● worst-case |
| **Storage (NVMe/SATA)** | — | ● storage steady | — | ● scratch verify | ● storage-cross |
| **GPU compute + board power** | ● gpu burst | ● gpu steady ~92 % | ◐ macro-cycle | ● per-dispatch self-consistency | ● worst-case |
| **VRAM cells** | — | ◐ (G5: only ~50 % duty in worst-case) | — | ● vram battery | ◐ worst-case |
| **PCIe link (x16)** | — | ● `link` kernel | — | ● uncorrected-corruption | ● **now in worst-case** |
| **Cooling / thermal** | — | ● throttle/runaway on soak | ● macro-cycle (pump-out) | — | ● worst-case |
| **Board DMI / interconnect** | — | ◐ | — | — | ● storage-cross / worst-case |

Note the PCIe row: the fault-mode research flagged "PCIe in **no** profile" as the
top gap (G1). **That is now closed** — `link` runs steady underneath `run worst-case`
(commit folding PCIe into the profile), so the x16 link is exercised while the RAM
controller and DMA engines are already saturated. A dedicated `link` phase still
runs in the gauntlet for isolation.

---

## 5. The recommended campaign

Three tiers. **The default standing gate is the express run.** The longer tiers are
justified *per build*, not as a blanket default (§8).

### Ordering principle
Cheapest-and-grossest first (fail fast) → precipitate assembly defects while fresh
and cold → Arrhenius hot dwell → Coffin-Manson cycling → re-attack hot → cooldown
re-verify. **Hour 1 ≠ hour 6:** hour 1 precipitates + detects *gross/assembly*
faults on a cold, fresh unit (verdict policy: abort-on-fail — pull a hard-faulted
unit before investing the soak). Hour 6 detects *temperature-marginal latents* — the
**same** integrity batteries run against a hotter, more-stressed DUT. The decisive
comparison is *hot integrity vs the cold baseline*: identical stimulus, different
temperature → a delta localizes an Arrhenius-activated latent.

### Express (~2 h) — the standing QC gate

Best assembly-defect yield per hour: keeps the four highest-yield screens
(transients, one contention+ramp, a full integrity sweep, a short macro-cycle) and
drops only the long Arrhenius dwell.

| # | Wall | Invocation | Purpose | Gate |
|---|---|---|---|---|
| E0 | ~6 m | `info`; `drives`; `gpu-info`; `run quick --seconds 60`; `gpu`/`vram`/`link` 60 s | smoke / DOA / mis-seat / mis-train | **HARD — any FAIL aborts** |
| E1 | 20 m | `run anti-phase --seconds 600`; `run beat --seconds 600` | VRM/PSU electrical margins | soft |
| E2 | 20 m | `run worst-case --seconds 1200` | contention + first heat ramp | soft |
| E3 | 35 m | `mem --seconds 1200`; `vram --seconds 900` | cold-ish integrity baseline | soft |
| E4 | 30 m | steady soak (`run cross --seconds 1800`) | one Arrhenius dwell | soft |
| E5 | 20 m | 4× [`run worst-case --seconds 180` → idle 120 s] | macro-cycle (top workmanship yield) | soft |
| E6 | 15 m | `run worst-case --seconds 600` (hot); cooldown `mem`/`vram` 150 s | hot latent + down-cycle | soft |

### Standard (~12 h) — premium / new-part builds

The express phases, plus the long blocks that need *dwell*: extend the Arrhenius
soak and the macro-cycle. Roughly: P0 smoke (10 m, hard gate) → P1 cold transient
trio (45 m) → P2 cold integrity baseline: mem+vram+storage-cross (75 m) → P3
worst-case + first ramp (30 m) → **P4 steady-max hot soak, ~5 h**, breaking every
~90 min for a hot `mem`+`vram` re-verify → **P5 macro-cycle, ~2 h** → P6 hot
transient re-attack (45 m) → P7 late worst-case + final hot integrity (90 m) → P8
cooldown re-verify (30 m).

### Full (~24 h) — SLA / mission-critical / adverse field history only

Identical structure; the elastic blocks stretch to **P4 ~10–12 h** and **P5 ~4 h**.
Fixed phases ≈ 5.7 h + P4 + P5 ⇒ ~22–23 h wall. **[INFER]** Read §8 before choosing
this tier — hours 12–24 are largely burn-past-the-knee insurance and begin
consuming useful life. Reserve and justify per build.

### Full-day phase reference (for `gauntlet.ps1 -Profile full`)

| # | Phase | Wall (full) | Invocation(s) — one process each | New? |
|---|---|---|---|---|
| P0 | Smoke gate | 10 m | `info`;`drives`;`gpu-info`;`run quick 60`;`gpu`/`vram`/`link` 60 | no |
| P1a/b/c | Cold anti-phase / in-phase / beat | 45 m | `run anti-phase 900`; `run in-phase 600`; `run beat 1200` | no |
| P2a/b/c | Cold mem / vram / storage-cross baseline | 75 m | `mem 1800 --mb ~90%free`; `vram 1200 --vram-mb ~90%`; `run storage-cross 1500` | no |
| P3 | Worst-case + ramp (now incl. PCIe) | 30 m | `run worst-case 1800` | no |
| P4 | **Steady-max hot soak** + interleaved hot integrity | **6–12 h** | `run cross <block>` soak; every ~90 m: `mem 900`+`vram 900` hot | ◐ cross lacks vram (G5) |
| P5 | **Macro thermal cycle** | **2–4 h** | repeat: [`run worst-case 240`] → [idle 180 s] | ◐ harness-sequenced |
| P6 | Hot transient re-attack | 45 m | `run anti-phase 1500`; `run beat 1200` | no |
| P7 | Late worst-case + final hot integrity | 90 m | `run worst-case 1800`; `mem 1800`; `vram 1800` | no |
| P8 | Cooldown re-verify | 30 m | `mem 600`; `vram 600`; `run storage-cross 600` | no |

---

## 6. Top-10 nastiest cross-loads (ranked by assembled-desktop fault-yield)

✅ runnable today · ⚠️ needs a small addition/manual step · 🔒 detection gated

1. **Hot-IMC memory hunt ✅** — `run cross`, 2–8 h. All-core FMA heats the on-die
   IMC while the full mem battery runs. The #1 real-world build instability
   (XMP/EXPO that passes a cool solo memtest, fails hot + contended). Longest block.
2. **In-phase CPU+GPU burst ✅** — `run in-phase`, 20–40 m. Biggest coincident
   dI/dt (~180 W GPU step + CPU step on one edge) → multi-rail OCP / PSU collapse.
3. **Anti-phase hand-off ✅** — `run anti-phase`, 20–40 m. VRMs chase load every
   20 ms, never settle → weak bulk caps, poor compensation, ring-back.
4. **Full-platform worst-case soak ✅** — `run worst-case`, 8–24 h. Union of
   power+thermal+IMC+GPU+**PCIe** contention, every domain self-verifying. The
   mission centerpiece.
5. **Beat phase-sweep ✅** — `run beat`, 30–60 m. Drifts through every phase
   alignment (flat marker distribution = proof it swept the resonance).
6. **VRAM-hot integrity ⚠️** — steady `gpu` mix + `vram` on one card, 30 m–2 h.
   GDDR errors are temp-dependent; worst-case only heats VRAM ~50 % (burst). Needs
   a "vram-hot" profile (G5) or a manual concurrent launch.
7. **GPU boost-stability probe ✅** — `gpu --shape steady --gpu-alu-only`, 30–60 m.
   ALU-only → higher boost clocks → catches marginal core OC the mix hides.
8. **Multi-SSD fabric saturation ✅** — `run storage-cross`, 30–60 m. DMI/uplink
   contention + all NVMe controllers heat together.
9. **Full-fabric worst-case + PCIe ✅** — `run worst-case` (now includes `link`),
   1–4 h. Adds host↔device DMA contending PCIe/DMI *and* host-RAM bandwidth the mem
   battery already fights for. **This session closed the gap that made this ⚠️.**
10. **Cold-start → long-hot soak bracket ⚠️** — `run quick` at genuine cold boot,
    then `run worst-case` for the day. Cold and hot faults are disjoint populations;
    warm-only testing misses the cold half. Operator-driven (tool can't power-cycle).

---

## 7. Error channels & what catches what

- **In-kernel verify** (strongest, immediate, device-ID'd): CPU dual-bank
  miscompare; mem/storage/VRAM first-fail; GPU self-consistency; link
  uncorrected-corruption. Catches *wrong answers* even with no crash.
- **WHEA** (harness, per-phase bracket, **zero = pass**, any event = FAIL, a
  *failed scan* is also a FAIL — never a silent pass): corrected/uncorrected
  machine-check, memory ECC-adjacent, PCIe AER (if firmware exposes it).
- **1 kHz analog rig** (aligns on QPC `qpc_ticks` markers at every burst edge):
  rail sag/overshoot/ringing on load edges — the only channel that sees the ms-scale
  transient (1 Hz `nvidia-smi`-class sampling cannot).
- **Operator**: shutdown, reboot, TDR/black-screen, throttle.

**Blind spots [FACT] — disclose on the QC sign-off sheet:**
- Deterministic-from-t0 GPU miscompute (self-consistency needs a golden reference).
- **PCIe retry / marginal riser:** `link` only catches *uncorrected* corruption. A
  marginal link **retries** and delivers correct data — throughput *and* checksum
  both pass a genuinely bad riser. Real detection needs WHEA/AER deltas (Plane C,
  unbuilt, **firmware-gated** on `_OSC`) + link-training check (Plane A, unbuilt).
  See [`pcie-plan.md`](pcie-plan.md).
- A hard-stuck CPU FMA bit corrupting *both* accumulator banks identically (recompute
  compares within-thread, not against a golden value — WHEA is the backstop).

---

## 8. Diminishing returns — the honest 80/20

**[INFER, to be calibrated against shop field data]:**

- **~80 % of *catchable* defects** for an assembled desktop (DOA, mis-seat, gross
  bad cell, marginal VRM/PSU/OCP, bad cooler mount, bad NVMe, weak core, gross VRAM,
  mis-trained link) precipitate in the **first ~2 hours** *if* that time is spent on
  smoke + transients + one contention/ramp + one full integrity sweep + a short
  macro-cycle. That is the entire thesis of the express gauntlet, and where the shop
  should set its default.
- **The extra hours buy** temperature-marginal latents that need *dwell* — a
  cell/core/joint that only errs after the die has held Tj_max for hours; slow
  cooling degradation (paste pump-out); low-duty intermittents whose detection
  probability rises with time-on-test. **Real, but lower-yield.**
- **Hours 6–24 are strongly diminishing.** Hours 2–6 catch most latents hour-1
  misses; 6–12 add a small-but-real increment (premium/mission-critical builds);
  **12–24 is largely insurance** and begins consuming useful life. **[FACT]** ESS
  practice explicitly warns against over-screening.

**Operating posture:** **express (~2 h) as the standing gate; ~12 h standard for
premium builds; the literal ~24 h reserved and justified per build** — an SLA
demand, new/unvetted parts, or the shop's own field data showing a late-appearing
mechanism. Then close the loop: the per-phase reports this campaign emits are the
dataset to *replace* these estimates with the shop's measured infant-mortality
curve, and re-tune P4/P5 to the real knee.

---

## 9. Known gaps (ranked by fault-yield lost)

| Gap | Status | Yield |
|---|---|---|
| **G1 — PCIe under contention** | ✅ **CLOSED** — `link` folded into `run worst-case` | high |
| **G2 — PCIe marginal-riser detection** | ○ unbuilt + firmware-gated; needs Gen5 + known-bad-riser spike | high (gated) |
| **G3 — No ramp/sweep/servo** | ○ only square-wave bursts; can't slew dI/dt or sweep burst frequency for VRM resonance (`beat` sweeps *phase*, not frequency) | medium |
| **G4 — No native thermal-cycling mode** | ◐ approximated by harness hot/idle alternation in `gauntlet.ps1`; native needs a macro-burst shape or the §7 sequencer | medium |
| **G5 — worst-case heats VRAM at only ~50 % duty** | ○ a steady-thrash "vram-hot" variant would raise GDDR yield | medium |
| **G6 — CPU is AVX2-only** | ○ no AVX-512 (materially more power/heat on Zen4/5); no integer/cache-thrash workload | medium on Zen4/5 |
| **G7 — No cold-boot orchestration** | ○ operator must launch at a genuine cold boot | low-med |
| **G8 — No data-retention dwell in mem test** | ○ back-to-back passes; retention/charge-leak uncovered | low |
| **iGPU not in worst-case** | ○ shares system RAM (extra IMC pressure); can't run discrete + iGPU together today | medium |

---

## 10. Native `run gauntlet` — deferred, and why

A single-command gauntlet needs one new orchestrator primitive: a **sequence of
`concurrent_phased` groups** that **checkpoints after each** (flush markers, write
an interim report, check `StopFlag`, stamp the phase name), because today
`sequential()` runs only single kernels and `concurrent_phased()` runs only one
group. The phase specs in §5 transcribe directly into a `Vec<GauntletPhase>`.

**Non-negotiable** or it reintroduces the §1 failure mode: `checkpoint()` must flush
the marker log per phase and write an interim report, and the harness must still
bracket WHEA per phase (the CLI can't read WHEA — that's the harness's job by
design). Given those constraints, **the harness-sequenced form
(`scripts/gauntlet.ps1`) is strictly safer and needs zero core changes** — ship that
first; treat native `gauntlet` as later convenience.

---

## Sources

**[FACT] Established:** bathtub/infant-mortality (MIL-HDBK-217F; O'Connor & Kleyner);
semiconductor burn-in (MIL-STD-883 M1015); ESS & thermal-cycling-catches-workmanship
(MIL-STD-2164; IEST-RP-PR001); HALT/HASS precipitation window and over-screening
caution (Hobbs, *Accelerated Reliability Engineering: HALT and HASS*); Arrhenius
temperature acceleration (JEDEC JEP122); Coffin-Manson / ΔT fatigue (JEDEC
JESD22-A104, JESD47); March-test DRAM fault models (van de Goor). **[INFER] Ours, to
calibrate against shop field data:** all specific durations, the phase ordering's
time budget, the ~80 %/2 h figure, and the hours-6–24 yield assessment.
