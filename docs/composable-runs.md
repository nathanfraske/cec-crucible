<!-- SPDX-License-Identifier: MIT -->

# cec-crucible — composable runs (`mix`)

**Goal:** let a user compose an **arbitrary** run — *"this test with these
parameters, AND that test with those parameters, at the same time"* — any number
of tests, each independently configured, from **both** the CLI/API and the
interactive menu. The existing curated cross-loads (`worst-case`, `chaos`,
`in-phase`, …) become **presets over that same mechanism**, not hand-wired
special cases.

This is the design of record for that work. It is written against the code as it
stands, and it is honest about the two profiles that genuinely cannot be
expressed as a composition and must stay bespoke.

## Executive summary

**The orchestration engine for arbitrary composition already exists and is
correct.** `Runner::concurrent_phased` (`crates/crucible-cli/src/main.rs:2219`)
already takes a `Vec<PhasedStage>` where every stage carries **its own kernel,
its own `Budget` (duration *and* shape), and its own phase offset**
(`main.rs:2836-2844`), and runs them all under one `StopFlag`, one `MarkerLog`,
and one shared phase epoch stamped at `main.rs:2243`. Every hardcoded profile is
just a different literal `Vec<PhasedStage>`.

What is missing is **a way to build that vector from user input**. Nothing more.

**Recommendation:**

1. **Syntax: a `--`-separated chain.** `cec-crucible mix [run opts] -- <test>
   [opts] -- <test> [opts] …`. Tokenizing is `argv.split(|t| t == "--")` — about
   five lines, no new escaping layer, no quoting hazard, and it survives the
   PowerShell harness's naive `$Command.Split(' ')`
   (`scripts/gauntlet.ps1:130`), which the alternatives do not.
2. **Refactor: a `kernel_from_argv` registry** that reuses the existing
   `*_kernel_from(&Parsed)` builders verbatim by extracting the top half of each
   `cmd_*` into a `spec_*(&Parsed) -> KernelSpec`. The eight GPU builders need
   **no signature change at all**.
3. **v1 scope: concurrent only**, with per-test `--seconds` and per-test `--at`
   (phase offset). Both are already free — `PhasedStage` carries them today.
4. **Presets: `run <profile>` keeps working**, but the expressible profiles
   delegate to the mix builder so a preset and the equivalent hand-typed `mix`
   are provably the same run.

**Two profiles cannot be expressed and stay bespoke:** `game-load` (its frame
windows are microseconds and `--burst-on/--burst-off` are integer milliseconds;
plus `--bound` and the auto-sized `gpu.iters` are cross-parameter derivations)
and `storage-cross` (enumerates drives at runtime and computes solo-vs-concurrent
throughput deltas). Both are stated with code references in §5.

---

## 1. What already exists (and why v1 is small)

| Capability | Where it lives | Status for composition |
| --- | --- | --- |
| N kernels, one `StopFlag`, one `MarkerLog` | `Runner::concurrent_phased`, `main.rs:2219` | ✅ done |
| **One shared phase origin** for every kernel | `let epoch = Instant::now()` → `budget.phased(epoch, offset)`, `main.rs:2243-2247`; `Budget.phase_epoch`, `kernel.rs:119-137` | ✅ done — this is the bug that was already fixed once |
| **Per-test load shape** | `PhasedStage.budget`, `main.rs:2839` | ✅ done |
| **Per-test duration** | `PhasedStage.budget.duration` | ✅ done (one reporting bug, §7) |
| **Per-test phase offset** | `PhasedStage.phase_offset`, `main.rs:2843` | ✅ done |
| Feature-gated "rebuild with `--features X`" errors | the `#[cfg]` arms in each `cmd_*`, e.g. `main.rs:559`, `926`, `999`, `1184` | ✅ reusable verbatim |
| Per-test **parameter parsing** from a sub-argv | — | ❌ **the actual work** |
| A **surface** to express the composition | — | ❌ **the actual work** |
| Menu multi-select | `menu.rs` is single-select (`App.sel`, `menu.rs:585-598`) | ❌ new screen |

The load-bearing invariant to protect (`menu.rs:19`, asserted by
`argv_is_identical_to_a_cli_run`, `menu.rs:1332`): **a menu launch is
byte-identical to the hand-typed CLI command**, because the menu literally builds
an argv and calls `crate::run(&argv)` (`menu.rs:1223`). Any composition syntax
must therefore be expressible as a plain `Vec<String>`.

---

## 2. Composition syntax on the CLI

### 2.1 The three candidates

**(a) Repeated grouped flags** — `mix --with "cpu --seconds 60 --shape burst"
--with "mem --mb 2048"`.

Requires an in-process tokenizer with quoting and escaping, because the shell
has already stripped one layer of quotes and the group arrives as a single argv
element. Concretely:

- `--with "storage --path C:\Program Files\scratch"` breaks unless we invent a
  *second* escaping convention on top of the shell's, and then document it.
- It **breaks the existing harness.** `scripts/gauntlet.ps1:130` runs
  `$tokens = $Command.Split(' ')` over each phase-step string. A quoted group is
  shredded into `"cpu`, `--seconds`, `60`, `--shape`, `burst"`. Every phase in
  `New-Phases` (`gauntlet.ps1:158-205`) is written as a plain space-delimited
  string; a syntax that cannot be written that way is a syntax the QC campaign
  cannot use.
- It **breaks the menu's argv preview.** `menu.rs:925` renders the command as
  `test.build_argv(&settings).join(" ")` — quotes are gone, so the printed line
  is no longer a command you can copy and re-run, which is the whole point of
  showing it.
- Parsing cost: ~120-180 LOC of tokenizer plus a real test matrix for
  backslashes, embedded quotes, and Windows paths. That is the one class of
  hand-rolled parser that reliably grows bugs forever.

**(b) `--`-separated chain** — `mix --seconds 120 -- cpu --shape burst -- mem
--mb 2048`.

The shell has already tokenized; we split on the literal token `--`:

```rust
let groups: Vec<&[String]> = argv.split(|t| t == "--").collect();
```

No new escaping layer, ever. Paths with spaces work because the shell handles
them, exactly as they do today for `storage --path "C:\My Drive"`. Survives
`$Command.Split(' ')` intact. The menu emits it as ordinary argv elements and the
space-joined preview is a genuinely runnable command line.

**Verified:** PowerShell 5.1 (the harness's shell) passes a bare `--` through to
a native executable verbatim — probed with `& cmd.exe /c echo alpha -- beta`,
which echoed `alpha -- beta`. `--` is not special to `cmd.exe` or `bash` either.

One gotcha, and it is real: `Parsed::parse` (`args.rs:19-46`) strips the `--`
prefix, gets an empty option name, finds it is not in `bool_flags`, and
**consumes the following token as its value**. So the split *must* happen before
`Parsed::parse` is ever called. That is the natural structure anyway, but it
belongs in a test.

**(c) A job file** — `mix --from run.toml`.

There is a hand-rolled JSON *writer* (`crucible-core/src/json.rs`) and **no
reader**, and no TOML anywhere. A restricted INI-subset reader (sections,
`key = value`, comments, quoted values, useful error positions) is ~200-300 LOC
plus tests; anything people would actually call "TOML" is several times that. It
also **breaks the menu invariant**: the menu would have to write a temp file and
launch `mix --from <tmp>`, at which point a menu launch is no longer an argv you
can retype, and the printed preview becomes a lie.

### 2.2 Recommendation: (b), the `--` chain

```
cec-crucible mix [RUN OPTIONS] -- <test> [TEST OPTIONS] [-- <test> [TEST OPTIONS]]…
```

- **Group 0** holds run-level options only, and must contain **no positional**.
  If it starts with a bare word, error: `mix: global options come first — put
  each test after a '--' (e.g. mix --seconds 60 -- cpu -- mem)`.
- **Every later group** starts with a command name (`cpu`, `mem`, `storage`,
  `gpu`, `vram`, `link`, `render`, `tensor`, `rt`, `pathtrace`, `optix`)
  followed by that command's own options.
- At least one test group is required.

| Scope | Options |
| --- | --- |
| **Run** (group 0) | `--seconds`, `--out`, `--device-id`, `--no-report`, `--json`, `--csv`, `--telemetry-csv`, `--ui`, `--seed`, `--dry-run`, `--help` |
| **Test** (later groups) | everything the solo command accepts *except* run-level flags, plus `--seconds <N>` (this participant's own duration; default = the run's), `--at <DUR>` (phase offset from the shared epoch; default 0), `--as <NAME>` (report/marker label; default `<cmd>#<slot>`) |

`<DUR>` grammar: `20ms`, `1500us`, `2s`, or a bare integer meaning milliseconds.
One shared parser, reused later for sub-millisecond burst units if we ever want
them (§5.3).

### 2.3 Worked examples

```
# Two domains, run defaults.
cec-crucible mix --seconds 120 -- cpu -- mem --mb 4096

# anti-phase, spelled out. Identical to `run anti-phase --seconds 120`.
cec-crucible mix --seconds 120 \
  -- cpu --shape burst --burst-on 20 --burst-off 20 \
  -- gpu --shape burst --burst-on 20 --burst-off 20 --at 20ms

# Per-test durations differing from the run duration.
cec-crucible mix --seconds 300 -- cpu -- vram --seconds 60 -- storage --size-mb 4096

# Two storage tests, different drives, different settings.
cec-crucible mix --seconds 180 \
  -- storage --path C:\ --size-mb 2048 --as sysdrive \
  -- storage --path D:\ --size-mb 8192 --unbuffered --as datadrive

# Three GPU engines at once (see R2 — this is legitimate but oversubscribing).
cec-crucible mix --seconds 60 --ui -- rt -- pathtrace --material fur -- vram

# Show what would run, run nothing.
cec-crucible mix --dry-run --seconds 120 -- cpu --shape burst -- gpu --at 20ms
```

`--dry-run` prints the resolved participant table (kernel, mode, duration, phase
offset) and exits 0. It is the cheapest possible safety net for a syntax whose
whole risk is "the user thought they asked for something else", and it costs
about ten lines.

### 2.4 A later, cheap concession to job files

If a stored scenario library is ever wanted (`docs/design.md:131` anticipates
"scenario files"), add `mix --from <file>` where **each non-comment line is one
`-- <cmd> [opts]` group, split on whitespace only**. No quoting, no sections, no
types — ~40 LOC, reusing the same group parser, and it degrades to exactly the
argv form so the menu invariant is untouched. Do not build a TOML reader.

---

## 3. Per-test parameter parsing — the registry refactor

The rule: **never duplicate a knob**. `mix` must parse `--pt-samples` through the
exact same code path `pathtrace` uses, or the two will drift within a release.

### 3.1 The shape of the refactor

Today each `cmd_*` does four things in one function: parse, allowlist, build the
kernel + mode label, then call `Runner`. Split the first three off:

```rust
/// Everything the CLI can resolve from one test's own argv, before a Budget or
/// a Runner exists. Both the solo `cmd_*` path and `mix` build one of these.
struct KernelSpec {
    kernel: Box<dyn LoadKernel>,
    /// The report/marker label the solo command already computes today, e.g.
    /// "burst core=all 20/20ms", "integrity discrete", "steady render 1280x720 x48".
    mode: String,
    /// Shape parsed from --shape / --burst-* / --jit-* / --pulse-*; Steady for
    /// kernels that ignore shape (mem, storage, vram, link).
    shape: Shape,
    /// Duration used when neither the test nor the run supplies --seconds.
    default_seconds: u64,
}

/// The option keys this command accepts *as a mix participant* — its own knobs
/// only, no run-level flags.
fn spec_opts(cmd: &str) -> Option<Vec<&'static str>>;

/// The registry. One arm per command, with the same #[cfg] shape the cmd_* fns
/// already use, so an unavailable kernel yields the identical
/// "rebuild with --features X" error.
fn kernel_from_argv(cmd: &str, args: &[String]) -> Result<KernelSpec, String>;
```

`kernel_from_argv` parses `args` with `Parsed::parse(args, COMMON_BOOLS)`,
allowlists with `spec_opts(cmd) ∪ PER_TEST_OPTS`, then dispatches to
`spec_cpu` / `spec_mem` / … which are the **top half of the existing `cmd_*`,
moved verbatim** — including the `format!` that builds `mode`, so labels stay
byte-identical to a solo run.

### 3.2 What actually changes

| Item | File:line | Change |
| --- | --- | --- |
| `gpu_kernel_from` | `main.rs:569` | **none** — already `(&Parsed, Shape)`; `spec_gpu` calls `shape_from(&p)?` first exactly as `cmd_gpu` does at `main.rs:605-641` |
| `vram_kernel_from` | `main.rs:645` | **none** |
| `link_kernel_from` | `main.rs:668` | **none** |
| `render_kernel_from` | `main.rs:725` | **none** |
| `tensor_kernel_from` | `main.rs:902` | **none** |
| `rt_kernel_from` | `main.rs:972` | **none** |
| `pathtrace_kernel_from` | `main.rs:1046` | **none** |
| `optix_kernel_from` | `main.rs:1155` | **none** |
| `mem_size_from`, `storage_cfg_from/_for` | `main.rs:2788`, `2809`, `2818` | **none** |
| `cmd_cpu` / `cmd_mem` / `cmd_storage` / `cmd_gpu` / … | `main.rs:459`, `498`, `518`, `605`, … | shrink to: parse → allowlist → `spec_*` → `budget_with` → `runner.single_stage`. **No behaviour change** |
| `Parsed::reject_unknown` | `args.rs:67-79` | add a strict variant. Today it unconditionally allows `ui\|csv\|telemetry-csv` (`args.rs:71-73`); those must be *rejected* inside a per-test group. ~8 LOC |
| `Runner::concurrent_phased` | `main.rs:2219` | per-stage timing fix, §7.1. ~6 LOC |

**That is the whole refactor.** The eight GPU builders are untouched; the work is
mechanical extraction plus one new dispatch function.

### 3.3 Feature gating (constraint 4)

Each registry arm carries the same `#[cfg]` pair the corresponding `cmd_*`
already has:

```rust
"rt" => {
    #[cfg(not(feature = "rt"))]
    { Err("the rt test needs a build with `--features rt`".to_string()) }
    #[cfg(feature = "rt")]
    { spec_rt(&Parsed::parse(args, COMMON_BOOLS)?) }
}
```

so `mix -- cpu -- rt` on a non-`rt` build fails with the **identical** message
`rt` already produces (`main.rs:999`), and — critically — it fails during
participant construction, i.e. **before any load starts**, at exit 2. Because
`mix` builds every participant before spawning any thread, a typo or a missing
feature never burns a 20-minute run.

---

## 4. Scheduling semantics

### 4.1 v1 — concurrent, with per-test duration and per-test phase

`mix` builds `Vec<PhasedStage>` and calls `concurrent_phased` **unchanged**. That
gives, for free:

- **One `StopFlag`, one `MarkerLog`, one shared phase epoch** (`main.rs:2243`) —
  constraint 3 is satisfied by construction, not by new code.
- **Per-test duration.** Each `PhasedStage.budget.duration` is independent. A
  participant that finishes early simply exits; the rest keep running. Run wall
  time = `max(duration_i)`.
- **Per-test phase offset** via `--at`, which is exactly what in-phase /
  anti-phase / beat *are* (`main.rs:1681-1691`, `1751`, `1946`).

**Be precise about what `--at` means**, because the code already is
(`main.rs:2840-2842`): it shifts the **burst phase**, not the thread start. That
is deliberate — a genuine start delay is swamped by per-kernel setup time (GPU
init is ~100 ms, which would eat a 20 ms offset). So:

- `--at 20ms` on a 20/20 ms burst = anti-phase. Exact, setup-jitter-immune.
- `--at 30s` does **not** mean "start 30 s in". For a `Steady` participant it
  does nothing at all. v1 must **reject `--at` on a participant whose shape is
  `Steady`** with that explanation, rather than silently no-op.

### 4.2 v2 — genuine start delay, sequential, staged

- **`--after <DUR>`** — a real start delay: the spawned closure sleeps, rechecks
  `stop`, then runs. Distinct from `--at` and documented as such. ~15 LOC in the
  `thread::scope` block at `main.rs:2252-2265`.
- **`mix --sequential`** — run participants back to back through
  `Runner::sequential` (`main.rs:2144`) instead. Expresses `quick` and `soak`.
- **Staged composition** — a second separator (`++`) that splits the chain into
  ordered *groups*, each run as its own `concurrent_phased` pass:

  ```
  cec-crucible mix --seconds 200 \
    -- storage --path C:\ ++ \
    -- storage --path C:\ -- cpu -- gpu
  ```

  A solo baseline phase, then the same test under contention. This is the
  `storage-cross` *shape*, and it is also how frame-pacing-under-contention gets
  measured (run `render --benchmark` alone, then again with the box loaded, and
  compare). It is additionally **the exact primitive `docs/gauntlet.md` §10 says
  a native `run gauntlet` needs** — "a sequence of `concurrent_phased` groups
  that checkpoints after each". Build it with a `checkpoint()` that flushes
  markers and writes an interim report, or gauntlet.md §1's failure mode returns.

### 4.3 Later

`--repeat <N>` on a group; `--at anti` sugar (offset = the participant's own
burst `on`); sub-millisecond burst units (`--burst-on 3333us`), which is one of
the three things blocking `game-load` (§5.3).

---

## 5. Existing profiles as presets

Keep `run <profile>` exactly as it is on the surface — it is baked into
`scripts/gauntlet.ps1`, the README, and the menu (`menu.rs:456-486`). Change what
is *underneath*: each expressible profile becomes

```rust
fn preset_argv(name: &str, p: &Parsed) -> Option<Vec<String>>
```

returning a **mix argv**, which `cmd_run` then feeds to `cmd_mix`. Two payoffs:
`run worst-case --explain` prints the exact `mix` command that would produce it
(the best possible documentation of the new syntax), and preset-vs-hand-typed
equivalence holds *by construction* rather than by hoping. Preserve each
profile's operator-facing `runner.note(...)` rationale (`main.rs:1779-1786`,
`1863-1872`, `1950-1955`) in a small table keyed by preset name — those strings
are why an operator picks the profile.

### 5.1 Exactly expressible

**`worst-case`** (`main.rs:1708-1786`) — six `PhasedStage` entries, 1:1:

```
cec-crucible mix --seconds 120 \
  -- cpu     --shape burst --burst-on 20 --burst-off 20 \
  -- mem     --mb 2048 \
  -- storage --size-mb 512 \
  -- gpu     --shape burst --burst-on 20 --burst-off 20 --at 20ms \
  -- vram \
  -- link
```

The `--at 20ms` on the GPU is `phase_offset: ms(on)` at `main.rs:1751` — the
anti-phase hand-off that is the whole point of the profile.

**`anti-phase` / `in-phase` / `beat`** (`run_transient_scenario`,
`main.rs:1624-1692`):

```
# anti-phase
mix --seconds 60 -- cpu --shape burst --burst-on 20 --burst-off 20 \
                 -- gpu --shape burst --burst-on 20 --burst-off 20 --at 20ms
# in-phase      : the same, without --at
# beat          : the same, GPU at --burst-on 23 --burst-off 23  (on+3/off+3, main.rs:1663-1664)
```

**`cross`** (`main.rs:1543-1570`):
`mix -- cpu -- mem --mb 1024 -- storage --size-mb 512 -- gpu`

**`power`** (`main.rs:1572-1587`): `mix -- cpu --shape burst`

### 5.2 Expressible with one small addition

**`chaos`** (`main.rs:1796-1873`) derives decorrelated per-domain seeds from one
base: `hash2(base, 1)` for the CPU (`main.rs:1801`), `hash2(base, 2)` for the GPU
(`main.rs:1833`). Written by hand, the user would have to pass two explicit
`--seed`s and the "re-run with `--seed 0x…` to replay the pattern" property
(`main.rs:1866-1868`) is lost.

Fix with ~5 lines: a run-level `mix --seed <N>` that assigns
`hash2(base, slot + 1)` to any participant that did not set its own. Then:

```
cec-crucible mix --seconds 120 --seed 0xC0FFEE \
  -- cpu --shape jitter -- mem --mb 2048 -- storage --size-mb 512 \
  -- gpu --shape jitter -- vram -- link
```

### 5.3 Genuinely NOT expressible — keep bespoke

**`game-load`** (`main.rs:1881-1959`). Three independent blockers:

1. **Units.** The frame windows are microseconds — `frame_us = 1_000_000 / fps`
   (`main.rs:1884`), `cpu_on = Duration::from_micros(frame_us * cpu_frac)`
   (`main.rs:1895`). At 120 fps that is 3333 µs. `--burst-on/--burst-off` are
   integer **milliseconds** (`shape_from_burst`, `main.rs:2762-2769`, via `ms()`
   at `main.rs:1615`). 3.33 ms is not representable.
2. **Cross-parameter derivation.** One `--bound` knob sets *both* the CPU duty
   fraction (`main.rs:1889-1893`) and the GPU's (`main.rs:1914-1918`), which are
   different numbers. No per-test flag expresses "these two participants share a
   derived duty split".
3. **Auto-sizing.** `gpu.iters` is computed from the GPU's own on-window —
   `((gpu_on.as_micros() / 50).clamp(32, 512))` (`main.rs:1934`) — so one
   dispatch fits inside a frame. A user composing by hand would have to
   reproduce that arithmetic or silently overrun the frame.

Sub-ms units would dissolve (1). (2) and (3) are model logic, not parameters.
**Keep `run game-load` as a bespoke command.**

**`storage-cross`** (`all_drives_storage`, `main.rs:2286-2436`). Four blockers:

1. Enumerates `crucible_storage::drives::discover()` **at runtime**
   (`main.rs:2287`) — the participant list is not knowable at parse time.
2. Two-phase schedule with a derived split: `solo_secs = seconds * 2/5`, rest
   concurrent (`main.rs:2304-2305`).
3. Uses `StorageKernel::run_measured` for `StorageStats` and computes per-drive
   solo-vs-concurrent slowdown with a >15 % contention flag
   (`main.rs:2412-2433`) — an analysis pass, not a load.
4. Treats an unwritable drive as **SKIP, not FAIL** (`main.rs:2334-2339`).

v2's staged composition covers only (2). **Keep `run storage-cross` bespoke.** A
future `--all-drives` *participant expansion* could cover (1), but (3) and (4)
would still need bespoke code.

**`core-cycle` / `c-states`** (`main.rs:1965`, `1994`) — sequential rotations
sized from `logical_cpus()` × passes, not concurrent compositions. Keep bespoke.

**`quick` / `soak`** (`main.rs:1511`, `1529`) — sequential; land them under
`mix --sequential` in v2.

---

## 6. Menu UX for multi-select composition

### 6.1 Flow

Add one screen and one button; do not rebuild the menu.

- **Main menu** gains a row under a new `COMPOSE` category: **"Compose a run"** →
  opens `Screen::Mix`.
- **Setup screen** (`draw_setup`, `menu.rs:863`) gains a second button beside
  ▶ FIRE: **`+ ADD TO MIX`**. `field_sel == nf` is FIRE (as today,
  `menu.rs:931`); `field_sel == nf + 1` is ADD. Pressing ADD snapshots the
  configured test into the mix and returns to `Screen::Mix`. Key `a` does the
  same from anywhere on the setup screen.
- **`Screen::Mix`** — the composed set:

  ```
   ⚠ CRUCIBLE                                     v0.1.0 · CEC · cec.direct
   PC-BUILD STRESS & VALIDATION                                  COMPOSE RUN

     Duration                                                  ‹ 120s ›

     #1  CPU burn      60s   burst    @0ms                     [e]dit [x]
   ▸ #2  GPU thrash    inherit burst  @20ms                    [e]dit [x]
     #3  Memory test   inherit steady @0ms                     [e]dit [x]

     + Add test

   ▶ cec-crucible mix --seconds 120 --ui -- cpu --seconds 60 --shape burst
     -- gpu --shape burst --at 20ms -- mem

           ┌───────────────────┐
           │   ▶  F I R E  ◀   │
           └───────────────────┘
   ↑/↓ move  e edit  x remove  +/a add  Enter ▶ FIRE  Esc back
  ```

  The argv preview is the same "what you see is what runs" line the setup screen
  already shows (`menu.rs:923-929`), wrapped over two lines when long.

- **Editing** `#2` re-enters the setup screen bound to *that entry's own* field
  state, so two participants of the same test configure independently.
- Long mixes scroll with the same whole-panel technique as `menu_top_group`
  (`menu.rs:644-658`).

### 6.2 New `App` state

```rust
struct MixEntry {
    gi: usize, ti: usize,        // catalog coordinates → label, desc, launch cmd
    fields: Vec<Field>,          // this entry's OWN ring state (cloned at add time)
    at: Field,                   // phase-offset ring: 0ms / 10 / 20 / 50 / 100ms
}

struct App {
    …                            // unchanged (menu.rs:585-598)
    mix: Vec<MixEntry>,
    mix_sel: usize,
    mix_duration: Field,         // the run-level --seconds ring
    adding_to_mix: bool,         // Setup screen entered from the Mix screen
    editing: Option<usize>,      // Setup screen is editing mix[i]
}
```

Two mechanical prerequisites:

- **`#[derive(Clone)]` on `Opt` and `Field`** (`menu.rs:126`, `134`). Both are
  `String` / `Vec<String>` / `usize` — trivially cloneable. Cloning is what lets
  each entry own independent ring state without aliasing the catalog.
- **A mix-mode Duration ring** whose head option is `inherit` emitting nothing,
  so a fresh three-test mix produces the tidy
  `mix --seconds 120 -- cpu -- gpu -- mem` rather than three redundant
  `--seconds`. Same trick `mem_size_field` already uses for `auto`
  (`menu.rs:285-291`).

### 6.3 `build_argv`

Extract the field-args loop that `Test::load_argv` (`menu.rs:195-202`) and
`Test::build_argv`'s `Bench` arm (`menu.rs:178-187`) already duplicate:

```rust
fn field_args(fields: &[Field]) -> Vec<String>;
```

Then the mix emitter is:

```rust
fn mix_argv(&self) -> Vec<String> {
    let mut argv = vec!["mix".into()];
    argv.extend(self.mix_duration.args().iter().cloned());   // --seconds N
    self.settings.append_flags(&mut argv);                   // menu.rs:565
    argv.push("--ui".into());
    for e in &self.mix {
        argv.push("--".into());
        argv.push(self.cmd_of(e).into());                    // Launch::Load(cmd)
        argv.extend(field_args(&e.fields));
        argv.extend(e.at.args().iter().cloned());            // --at 20ms
    }
    argv
}
```

**Every per-test fragment comes from the same `Field::args()` the solo path
uses** — that is precisely what keeps the invariant true rather than merely
tested.

Only `Launch::Load` rows are mixable. `Info` (no load), `Profile` (a preset, not
a kernel) and `Bench` (its own scoring harness, and deliberately no `--ui`,
`menu.rs:116-120`) are not offered on the compose screen.

### 6.4 How the argv-identity test extends

Add to the existing `argv_is_identical_to_a_cli_run` block (`menu.rs:1332`):

```rust
#[test]
fn mix_argv_is_a_cli_command() {
    let mut app = App::new();
    app.add_to_mix_for_test("cpu");   // Duration ring left at `inherit`
    app.add_to_mix_for_test("mem");
    app.mix[1].fields[1].right();     // mem Size: auto -> 512 MB
    assert_eq!(app.mix_argv(), strs(&[
        "mix", "--seconds", "30", "--ui",
        "--", "cpu",
        "--", "mem", "--mb", "512",
    ]));
}
```

And — stronger, and worth the extra twenty lines — a **round-trip** test that
feeds the emitted argv back through the mix parser and asserts the resulting
participant list (kernel names, modes, durations, offsets) matches what the menu
displayed. That tests the invariant rather than a string.

---

## 7. Reporting and attribution

Kernels **can repeat**, so `kernel.name()` is no longer a unique key. Three
concrete collisions exist in the code today:

1. **Stage rows.** `StageReport::new(kernel.name(), …)` (`main.rs:2272-2278`)
   with `name()` hardcoded per kernel — every `StorageKernel` returns `"storage"`
   (`crucible-storage/src/lib.rs:94-96`). Two storage participants yield two
   rows distinguishable only by the `mode` string.
2. **Live lanes.** `MarkerLog::register_lane` **dedups by label**
   (`markers.rs:246-257`), so two participants with the same lane name share one
   `Arc<LiveLane>`. `--ui` and `--telemetry-csv` would silently *merge* their
   work counters. Lane names are hardcoded: `"mem"`
   (`crucible-mem/src/lib.rs:143`), `"storage"`
   (`crucible-storage/src/lib.rs:185`), `"pcie"` (`crucible-gpu/src/link.rs:362`
   and `571`), `"vram"` (`crucible-gpu/src/vram.rs:376`), `"core N"` via
   `relabel_lane` (`crucible-cpu/src/lib.rs:218`).
3. **Markers.** A `Marker` carries `kernel` / `mode` / `detail`
   (`markers.rs:184-207`); `ShapeDriver` stamps burst edges with the kernel name
   it was constructed with (`kernel.rs:443-452`, `495-497`). Two *bursting*
   participants of the same kernel at different `--at` offsets are ambiguous in
   the JSONL the 1 kHz rig aligns on — which defeats the purpose of composing
   them.

### 7.1 v1 — report-level, zero core change

- **Slot-tag the mode.** Every participant's `mode` becomes
  `format!("#{slot} {mode}")`, or `format!("{name} {mode}")` when `--as` was
  given: `#2 unbuffered D:\`, `datadrive unbuffered D:\`. Flows automatically
  into the JSON report (`report.rs:71-84`) and the CSV `mode` column
  (`report.rs:208-245`).
- **Record the recipe.** Write the full resolved mix argv into `report.notes`
  via `Report::note` (`report.rs:130`) so any report is replayable verbatim.
- **Fix per-stage seconds.** `concurrent_phased` computes one shared
  `secs = t0.elapsed()` (`main.rs:2267`) and applies it to *every* stage
  (`main.rs:2270-2278`). Harmless today because all stages share a duration;
  **wrong the moment per-test `--seconds` exists**. Change the spawned closure to
  time itself and return `(LoadResult, f64)`. ~6 LOC, and a genuine latent bug.

### 7.2 v2 — lane and marker attribution

Add `pub label: Option<Arc<str>>` to `Budget` (`kernel.rs:119-137`).
`ShapeDriver::start` (`kernel.rs:323-352`) uses
`budget.label.as_deref().unwrap_or(kernel)` for both `register_lane`
(`kernel.rs:350`) and the stamped `kernel` field; the five direct
`register_lane` call sites listed above read the same label.

Cost and compatibility:

- Adding a field breaks all **12** `Budget { … }` struct-literal constructions in
  `main.rs` (lines 491, 637, 817, 964, 1038, 1147, 1218, 1317, 1580, and the
  helpers at 2023/2035). Fix by first routing them through the existing
  `budget_with()` helper (`main.rs:2029`) — a cleanup worth doing regardless.
- Solo runs set `label: None`, so marker JSONL, lane names and the rig's
  kernel-keying are **byte-identical to today**. Only composed runs emit
  `storage#2`. That is a backwards-compatible schema change (existing consumers
  key on `kernel`; they simply see a new value in a new run mode).

---

## 8. Risks and failure modes

**R1 — Two storage participants on one directory will corrupt each other.**
`StorageKernel::scratch_path()` is `crucible-scratch-<pid>.tmp`
(`crucible-storage/src/lib.rs:85-90`) — unique per **process**, not per kernel.
Two composed storage tests in the same directory open, write and verify **the
same file**, producing guaranteed miscompares: a **false FAIL**, the worst
possible outcome for a QC gate. (`all_drives_storage` is safe only because every
kernel gets a distinct `primary_root`.) **Fix both ends, and it is not optional
for v1:** add a slot discriminator to `StorageConfig` folded into the filename,
*and* have the mix validator reject two storage participants that resolve to the
same directory.

**R2 — Unbounded resource oversubscription.** Five GPU participants means five
wgpu devices and contexts (VRAM exhaustion, higher TDR risk). Two `mem`
participants each default to 50 % of free RAM (`MemSize::default()`,
`main.rs:2795`) → OOM. N `cpu` participants each spawn one thread per logical
core (`CoreSel::All`) → oversubscription that *flattens the burst edges the test
exists to produce*, so the run silently stops testing what it claims to. v1:
`note()` a warning on more than one participant of the same `Kind`, and require
an explicit `--mb` when composing more than one `mem`. Warn, do not block —
"three GPU engines at once" is a legitimate thing to want to try, and the point
of this feature is to permit it.

**R3 — Verdict semantics.** `Report::verdict()` already FAILs if any stage fails
(`report.rs:136-144`), which is correct for a composed run: a mix is a
conjunction. Two refinements: (a) build every participant **before** spawning
anything, so a missing feature or a typo exits 2 with no load run; (b) do **not**
copy `storage-cross`'s SKIP-on-unwritable behaviour (`main.rs:2334-2339`) into
`mix` — a silently skipped participant in a QC gate is worse than a failure,
because the operator believes coverage they did not get.

**R4 — Arg-parsing ambiguity.** The bare `--` must be split *before*
`Parsed::parse` (`args.rs:19-46`) or it eats the following token as the value of
an empty-named option. A group with no command name; a run-level flag inside a
test group; a duplicated option within a group (`values` is a `BTreeMap`, so the
last wins **silently**, `args.rs:37`). All catchable with explicit errors, all
worth a test each.

**R5 — Menu/CLI drift.** The invariant lives entirely in "mix argv is assembled
from the same `Field::args()` fragments as the solo path". Guard it with the
round-trip test in §6.4, not just a string comparison.

**R6 — `--preview` in a mix.** Two participants with `--preview` pop two Win32
windows in one process (`crucible-gpu/src/preview.rs`), and closing either stops
the whole run. v1: reject more than one `--preview` participant.

**R7 — Marker volume.** `docs/gauntlet.md` §1 already notes ~800 markers/s for
transient profiles, accumulated in a `Mutex<Vec<Marker>>` and serialized once at
`finish()`. A six-participant bursting mix multiplies that. Not this design's
problem to fix, but `mix` makes it easier to hit — flag it in `--dry-run` output
when more than three participants use a bursty shape.

---

## 9. Phased build plan

### v1 — minimal and genuinely useful

| Work | Where | ~LOC |
| --- | --- | --- |
| `mix` group splitter + validator + `--dry-run` | new `crucible-cli/src/mix.rs` | 180 |
| `KernelSpec` + `spec_*` extraction (11 commands) | `main.rs` (moved, not new) | 150 |
| `kernel_from_argv` registry + `spec_opts` | `mix.rs` | 120 |
| `<DUR>` parser (`20ms` / `1500us` / `2s`) | `args.rs` | 30 |
| `reject_unknown` strict variant | `args.rs:67` | 10 |
| Per-stage timing fix in `concurrent_phased` | `main.rs:2267` | 6 |
| Slot-tagged stage labels + mix argv in notes | `mix.rs` | 20 |
| Storage same-directory guard + slot in scratch name | `mix.rs`, `crucible-storage` | 25 |
| Tests (grammar, gating, round-trip, guards) | | 150 |
| **Menu:** `Screen::Mix`, `MixEntry`, ADD button, `mix_argv`, scrolling | `menu.rs` | 350 |
| Menu tests | `menu.rs` | 80 |
| **Total v1** | | **≈ 1 100** |

Most of it is mechanical extraction, and roughly a third is the menu screen.

### v1.5 — presets become presets

`preset_argv` + `run <profile> --explain`; delete the bodies of
`run_transient_scenario` and `run_worst_case` (~150 LOC removed); mix-level
`--seed` derivation; optional `mix --from <file>`. Net ≈ **+120 / −150 LOC**.

### v2 — attribution and scheduling

`Budget.label` plumbed to lanes and markers (~40 LOC across core + 5 kernel
crates, plus routing the 12 `Budget` literals through `budget_with`);
`--after` real start delay (~15); `mix --sequential` (~30); staged `++`
composition with `checkpoint()` (~120 — this is also the native-gauntlet
unlock, `gauntlet.md` §10). ≈ **250 LOC**.

### Later

`--repeat`; `--at anti`; sub-millisecond burst units (which would let
`game-load`'s *timing* be expressed, though not its derivations); `--all-drives`
participant expansion.

### Harness follow-up (one line, do not forget)

`scripts/gauntlet.ps1:124` — add `'mix'` to `$script:LoadCmds`, or `--out` and
`--device-id` will not be appended to composed phase steps and the campaign will
write reports to the wrong directory.

---

## 10. Open questions

1. **Does `mix` supersede `run`, eventually?** Recommendation: no. `run <profile>`
   is a curated, named, *documented* stimulus with an operator-facing rationale
   note. `mix` is the general mechanism. Keeping both, with presets delegating,
   is the honest structure.
2. **Should `--at` be rejected or warned on a `Steady` participant?** Rejected
   in v1 (it is silently meaningless), revisited if `--after` lands and users
   conflate the two.
3. **Does the 1 kHz rig's ingest tolerate `kernel: "storage#2"` in the marker
   JSONL?** Needs confirming with whoever owns the capture side before v2 ships
   the label change.
4. **Should `mix` cap total participants?** Leaning no — warn, and let
   `--dry-run` plus the oversubscription note do the teaching.
