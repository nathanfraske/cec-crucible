// SPDX-License-Identifier: MIT
//! # uncore — cross-core coherence / interconnect verification
//!
//! Every other CPU test in this suite is **register-resident**: the FMA
//! recurrence in [`crate::CpuKernel`] never touches memory and never talks to
//! another core, so the L3, the ring/mesh and (on AMD) the Infinity Fabric sit
//! *idle* for the entire run. Marginal FCLK / SoC voltage / IF training is one
//! of the top real-world causes of "passes every stress test, crashes in
//! games" — it logs a WHEA-19 rather than crashing outright — and nothing this
//! suite shipped before this kernel could see it. See `docs/game-realism.md`
//! §2, the "the uncore is idle in every test we have" row.
//!
//! ## What it does
//!
//! For an ordered pair of logical cores (**producer P → consumer C**) it puts a
//! single-producer / single-consumer ring buffer in shared memory, pins P and C
//! to their cores, and streams self-describing records across it:
//!
//! * `seq` — a strictly increasing counter,
//! * `payload` — a pure function of `seq` ([`crucible_core::rng::hash2`]),
//! * `check` — a hash over `(seq, payload)`.
//!
//! The consumer verifies three independent things about every record it reads:
//! the sequence is monotonic with **no gaps and no duplicates**, the payload is
//! what that `seq` must produce, and the check word matches the payload it
//! actually observed. Because `payload` is derived from `seq` alone, the
//! consumer needs no side channel to know the right answer — it re-derives it.
//!
//! Every record handed over forces a modified cache line to migrate from P's
//! cache hierarchy into C's, and the head/tail index lines ping-pong the other
//! way. On a dual-CCD Ryzen a pair that spans the two dies drives that traffic
//! over the Infinity Fabric; on a monolithic part it drives the L3/ring. That
//! is the traffic no other kernel here generates.
//!
//! ## Why the ring uses atomics for the *data*, not just the indices
//!
//! This is a QC instrument: a fault it reports must be attributable to the
//! hardware, never to a race we wrote. Slot fields are [`AtomicU64`] accessed
//! `Relaxed`, published behind an `Acquire`/`Release` pair on the head/tail
//! indices. On x86-64 relaxed loads/stores and acquire/release ordering are
//! plain `mov`s — identical machine code to raw reads and writes, zero cost —
//! but the program has **no data race by definition**, so there is no unsafe
//! code and no UB to explain away when a miscompare shows up. The ring is the
//! textbook SPSC design (producer-owned head, consumer-owned tail, cached
//! opposite index); nothing clever, deliberately.
//!
//! ## Pair sweep
//!
//! Testing all N² ordered pairs is unbounded on a 128-thread Threadripper, so
//! the sweep is organised into **rounds**. A round picks a *distance* `d` and
//! builds a matching that pairs core `i` with core `i+d`, so every core is in
//! exactly one pair and all pairs run *concurrently* — maximum fabric traffic,
//! while each ring keeps its own counters so attribution stays exact.
//! Distances are drawn from `{1, 2, 4, N/8, N/4, N/2}` (deduped) and each is run
//! in both directions, so the schedule is at most 12 rounds regardless of N:
//! neighbour / SMT sibling at `d=1`, near-cluster in the middle, and `d=N/2`,
//! which on a symmetric dual-CCD part is the cross-die pairing. Rounds cycle
//! until the budget runs out. Runtime is therefore `rounds × dwell` and does
//! **not** grow with core count.
//!
//! ## Reading a failure
//!
//! The pair is always in the message, because the pair *is* the diagnosis: a
//! failure confined to cross-die pairs (`d=N/2` on a dual-CCD chip) is the
//! FCLK / SoC-voltage signature, while failures on short-distance pairs point at
//! the L3 / ring / core itself.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;
use crucible_core::rng::hash2;
use crucible_core::sysinfo;

use crate::pin_current_thread;

/// Live-UI lane label (also the marker-log kernel name).
const LANE: &str = "uncore";

/// Slots per ring, per pair, unless overridden. 4096 × 32 B = 128 KiB: large
/// enough that index publication amortizes and the producer rarely stalls,
/// small enough to stay in the producer's L2 so the handoff is a genuine
/// cache-to-cache (cross-die) transfer rather than a trip through DRAM.
pub const DEFAULT_RING_SLOTS: usize = 4096;

/// How long one round of pairs runs before the sweep rotates to the next
/// distance. Long enough to be more than warm-up, short enough that a 30 s run
/// still visits several distances.
pub const DEFAULT_DWELL: Duration = Duration::from_secs(2);

/// Records moved per [`Tick::Work`] chunk. ~128 KiB of traffic — a few tens of
/// microseconds — so the shape driver keeps tight control of burst duty cycles
/// and the stop flag is honoured promptly.
const CHUNK_RECORDS: u64 = 4096;

/// Spin attempts on a full (producer) / empty (consumer) ring before yielding
/// and returning to the shape driver. Bounds how long a chunk can ignore the
/// stop flag when the other side is descheduled.
const SPIN_LIMIT: u32 = 4096;

/// Bytes a record occupies on the wire (one padded slot) — used for the
/// approximate cross-core throughput figure.
const SLOT_BYTES: u64 = 32;

/// Ring floor/ceiling, after rounding to a power of two.
const MIN_SLOTS: usize = 64;
const MAX_SLOTS: usize = 1 << 20;

/// Below this much remaining budget a fresh round is not worth its thread spawns
/// — unless nothing has run yet, in which case we always run one so a short run
/// still verifies something instead of reporting NOT VERIFIED.
const MIN_SLICE: Duration = Duration::from_millis(5);

/// Mixed into the check word so it is not a bare re-hash of the payload.
const CHECK_KEY: u64 = 0x0CEC_C0DE_51CE_0001;

/// Base salt for per-pair payload streams.
const SALT_BASE: u64 = 0x556E_C0F1_5EED_1234;

// ---------------------------------------------------------------------------
// Records and verification
// ---------------------------------------------------------------------------

/// One record handed across the interconnect. `payload` and `check` are pure
/// functions of `seq`, so the consumer re-derives the expected values instead of
/// being told them — there is no shared expectation state to get out of step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub seq: u64,
    pub payload: u64,
    pub check: u64,
}

impl Record {
    /// The one and only correct record for `(salt, seq)`.
    #[inline]
    pub fn make(salt: u64, seq: u64) -> Record {
        let payload = payload_word(salt, seq);
        Record {
            seq,
            payload,
            check: check_word(seq, payload),
        }
    }
}

#[inline]
fn payload_word(salt: u64, seq: u64) -> u64 {
    hash2(salt, seq)
}

/// Hash over the *payload the reader actually observed* plus its sequence, so a
/// corrupted payload invalidates the check word even though the check word
/// itself arrived intact.
#[inline]
fn check_word(seq: u64, payload: u64) -> u64 {
    hash2(payload ^ CHECK_KEY, seq)
}

/// What kind of corruption a fault was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// The sequence number was not the next one expected (gap, duplicate, or
    /// reorder) — the record stream itself did not arrive intact.
    Sequence,
    /// The payload was not the value its own sequence number requires.
    Payload,
    /// The check word did not match the payload as read.
    Check,
}

impl FaultKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaultKind::Sequence => "sequence fault",
            FaultKind::Payload => "payload miscompare",
            FaultKind::Check => "check-word miscompare",
        }
    }
}

/// The first fault a pair saw. Kept verbatim: for a QC report the *first*
/// divergence is the useful one; later ones are usually consequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub kind: FaultKind,
    pub seq: u64,
    pub expected: u64,
    pub got: u64,
}

impl Fault {
    /// One-line human description, as it appears in the run report.
    pub fn describe(&self) -> String {
        match self.kind {
            FaultKind::Sequence => {
                let what = if self.got > self.expected {
                    "seq gap"
                } else {
                    "seq regression/duplicate"
                };
                format!("{what}: expected {}, got {}", self.expected, self.got)
            }
            FaultKind::Payload => format!(
                "payload at seq {}: expected {:#018x}, got {:#018x}",
                self.seq, self.expected, self.got
            ),
            FaultKind::Check => format!(
                "check word at seq {}: expected {:#018x}, got {:#018x}",
                self.seq, self.expected, self.got
            ),
        }
    }
}

/// Consumer-side verification state for one ring.
#[derive(Debug, Clone)]
pub struct Verifier {
    salt: u64,
    /// Next sequence number this stream must produce.
    expect: u64,
    pub verified: u64,
    pub seq_faults: u64,
    pub payload_faults: u64,
    pub check_faults: u64,
    /// Order-independent fold over the data actually read, so combining several
    /// pairs (or several rounds) is stable regardless of who finished first.
    pub checksum: u64,
    pub first: Option<Fault>,
}

impl Verifier {
    pub fn new(salt: u64) -> Verifier {
        Verifier {
            salt,
            expect: 0,
            verified: 0,
            seq_faults: 0,
            payload_faults: 0,
            check_faults: 0,
            checksum: 0,
            first: None,
        }
    }

    /// Total faults of every kind.
    pub fn faults(&self) -> u64 {
        self.seq_faults + self.payload_faults + self.check_faults
    }

    /// Verify one record. All three checks always run — a record can be wrong in
    /// more than one way, and knowing which combination is diagnostic (payload
    /// alone is a data-path bit flip; sequence alone means a whole record was
    /// lost or replayed).
    pub fn check(&mut self, r: Record) {
        if r.seq != self.expect {
            self.seq_faults += 1;
            self.note(Fault {
                kind: FaultKind::Sequence,
                seq: self.expect,
                expected: self.expect,
                got: r.seq,
            });
            // Resynchronize to what actually arrived. Without this a single lost
            // record turns every subsequent one into a fault and the error count
            // becomes a meaningless "everything after here", hiding how many
            // *independent* events occurred.
            self.expect = r.seq;
        }

        // Both derivations use the sequence/payload as *read*, never as expected,
        // so a corrupted seq is caught by the ordering check above rather than
        // silently changing what we compare against.
        let want_payload = payload_word(self.salt, r.seq);
        if r.payload != want_payload {
            self.payload_faults += 1;
            self.note(Fault {
                kind: FaultKind::Payload,
                seq: r.seq,
                expected: want_payload,
                got: r.payload,
            });
        }

        let want_check = check_word(r.seq, r.payload);
        if r.check != want_check {
            self.check_faults += 1;
            self.note(Fault {
                kind: FaultKind::Check,
                seq: r.seq,
                expected: want_check,
                got: r.check,
            });
        }

        // Fold what was read (not what was expected) so the checksum reflects the
        // bytes that actually crossed the interconnect.
        self.checksum = self
            .checksum
            .wrapping_add(r.payload ^ r.check.rotate_left(17));
        self.verified += 1;
        self.expect = self.expect.wrapping_add(1);
    }

    #[inline]
    fn note(&mut self, f: Fault) {
        if self.first.is_none() {
            self.first = Some(f);
        }
    }
}

// ---------------------------------------------------------------------------
// The SPSC ring
// ---------------------------------------------------------------------------

/// One ring slot, padded to 32 bytes so a record never straddles a cache line —
/// a handoff is then a bounded, predictable number of coherence transactions
/// rather than sometimes one line and sometimes two.
#[repr(C, align(32))]
#[derive(Debug, Default)]
struct Slot {
    seq: AtomicU64,
    payload: AtomicU64,
    check: AtomicU64,
    _pad: AtomicU64,
}

/// A cache-line-isolated index. Head and tail must not share a line: false
/// sharing there turns the whole ring into one ping-ponging line and collapses
/// throughput to the coherence round-trip time. 128 bytes covers Intel's
/// adjacent-line prefetch and Apple's 128 B lines, not just the 64 B line size.
#[repr(align(128))]
#[derive(Debug, Default)]
struct Index(AtomicU64);

/// A single-producer / single-consumer ring buffer shared by exactly two pinned
/// threads.
///
/// Capacity is a power of two so the slot index is a mask, and `head`/`tail` are
/// free-running `u64` counters (never wrapped) so "how many are in flight" is a
/// plain subtraction with no ambiguous full/empty state.
#[derive(Debug)]
pub struct Ring {
    slots: Box<[Slot]>,
    mask: u64,
    capacity: u64,
    /// Written only by the producer, read by the consumer.
    head: Index,
    /// Written only by the consumer, read by the producer.
    tail: Index,
}

impl Ring {
    /// Allocate a ring with `slots` rounded up to a power of two and clamped to
    /// a sane range. The `collect` touches every slot, so the pages are faulted
    /// in before the timed run rather than during it.
    pub fn with_slots(slots: usize) -> Ring {
        let n = slots.clamp(MIN_SLOTS, MAX_SLOTS).next_power_of_two();
        let slots: Box<[Slot]> = (0..n).map(|_| Slot::default()).collect();
        Ring {
            slots,
            mask: (n - 1) as u64,
            capacity: n as u64,
            head: Index::default(),
            tail: Index::default(),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The producer handle. Call **exactly once** per ring, and use it from one
    /// thread only — that is the SPSC contract this ring is built on.
    pub fn producer(&self) -> Producer<'_> {
        Producer {
            ring: self,
            head: 0,
            cached_tail: 0,
        }
    }

    /// The consumer handle. Same contract as [`Ring::producer`].
    pub fn consumer(&self) -> Consumer<'_> {
        Consumer {
            ring: self,
            tail: 0,
            cached_head: 0,
        }
    }

    /// Overwrite a slot's payload out of band. **Test-only**: this deliberately
    /// breaks the SPSC contract to simulate a corrupted transfer, which is how
    /// the detection path itself is tested.
    #[cfg(test)]
    fn poke_payload(&self, index: u64, payload: u64) {
        self.slots[(index & self.mask) as usize]
            .payload
            .store(payload, Ordering::Relaxed);
    }
}

/// Producer half. Owns `head`; keeps a cached copy of the consumer's `tail` so
/// the common case never loads the consumer's (remote, contended) line.
#[derive(Debug)]
pub struct Producer<'a> {
    ring: &'a Ring,
    head: u64,
    cached_tail: u64,
}

impl Producer<'_> {
    /// Publish one record. Returns `false` if the ring is full (the consumer is
    /// behind); the caller backs off and retries.
    #[inline]
    pub fn push(&mut self, r: Record) -> bool {
        if self.head.wrapping_sub(self.cached_tail) >= self.ring.capacity {
            // Only now pay for the remote read of the consumer's index.
            self.cached_tail = self.ring.tail.0.load(Ordering::Acquire);
            if self.head.wrapping_sub(self.cached_tail) >= self.ring.capacity {
                return false;
            }
        }
        let slot = &self.ring.slots[(self.head & self.ring.mask) as usize];
        // Relaxed: these writes are made visible as a group by the Release store
        // on `head` below, which is the only thing the consumer synchronizes on.
        slot.seq.store(r.seq, Ordering::Relaxed);
        slot.payload.store(r.payload, Ordering::Relaxed);
        slot.check.store(r.check, Ordering::Relaxed);
        self.head = self.head.wrapping_add(1);
        // Release: everything written above happens-before the consumer's Acquire
        // load of `head` observing this value. This is the whole handoff.
        self.ring.head.0.store(self.head, Ordering::Release);
        true
    }

    /// Records published so far.
    pub fn published(&self) -> u64 {
        self.head
    }
}

/// Consumer half. Owns `tail`; caches the producer's `head`.
#[derive(Debug)]
pub struct Consumer<'a> {
    ring: &'a Ring,
    tail: u64,
    cached_head: u64,
}

impl Consumer<'_> {
    /// Take the next record, or `None` if the ring is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<Record> {
        if self.tail == self.cached_head {
            // Acquire: pairs with the producer's Release store, making its slot
            // writes visible before we read them below.
            self.cached_head = self.ring.head.0.load(Ordering::Acquire);
            if self.tail == self.cached_head {
                return None;
            }
        }
        let slot = &self.ring.slots[(self.tail & self.ring.mask) as usize];
        let r = Record {
            seq: slot.seq.load(Ordering::Relaxed),
            payload: slot.payload.load(Ordering::Relaxed),
            check: slot.check.load(Ordering::Relaxed),
        };
        self.tail = self.tail.wrapping_add(1);
        // Release: the reads above happen-before the producer's Acquire load of
        // `tail` sees this, so the producer cannot overwrite a slot we are
        // still reading.
        self.ring.tail.0.store(self.tail, Ordering::Release);
        Some(r)
    }

    /// Records consumed so far.
    pub fn consumed(&self) -> u64 {
        self.tail
    }
}

// ---------------------------------------------------------------------------
// Pair schedule
// ---------------------------------------------------------------------------

/// An ordered producer→consumer core pair. `distance` is `|producer - consumer|`
/// in logical-core index space — the only topology proxy available without a
/// dependency, but a good one: on every mainstream desktop layout SMT siblings
/// are adjacent and dies split the index range in half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorePair {
    pub producer: usize,
    pub consumer: usize,
    pub distance: usize,
}

/// A set of *disjoint* pairs run concurrently: every core appears at most once,
/// so one thread per core, and every ring has independent counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Round {
    pub distance: usize,
    /// `true` when the higher-numbered core is the producer — direction matters
    /// because a link can be marginal in one direction only.
    pub reversed: bool,
    pub pairs: Vec<CorePair>,
}

impl Round {
    /// Short label for markers and the live UI.
    pub fn label(&self) -> String {
        format!(
            "d={}{}",
            self.distance,
            if self.reversed { " rev" } else { "" }
        )
    }
}

/// The distances sampled for `cores` logical processors: near neighbours plus
/// fractions of the whole range, deduped and sorted. At most six values, so the
/// schedule stays bounded no matter how wide the part is.
///
/// * `1` — SMT sibling / immediately adjacent core (intra-core, intra-CCX).
/// * `2`, `4` — same cluster / same L3 slice neighbourhood.
/// * `N/8`, `N/4` — across the cluster.
/// * `N/2` — on a symmetric two-die part, straight across the fabric. This is
///   the pairing that isolates FCLK / SoC-voltage instability.
pub fn pair_distances(cores: usize) -> Vec<usize> {
    let mut d: Vec<usize> = [1, 2, 4, cores / 8, cores / 4, cores / 2]
        .into_iter()
        .filter(|&x| x >= 1 && x < cores)
        .collect();
    d.sort_unstable();
    d.dedup();
    d
}

/// The full deterministic sweep for `cores` logical processors: each distance in
/// both directions. A pure function of the core count — the same machine always
/// runs the same schedule, so two runs are comparable and a reported pair means
/// the same thing tomorrow.
///
/// Bounded by construction: at most 6 distances × 2 directions = 12 rounds, each
/// with at most `cores/2` pairs. On a 128-thread part that is ≤ 768 pairs versus
/// 16 256 for an exhaustive ordered sweep.
pub fn pair_rounds(cores: usize) -> Vec<Round> {
    if cores < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for distance in pair_distances(cores) {
        for reversed in [false, true] {
            let pairs = matching(cores, distance, reversed);
            if !pairs.is_empty() {
                out.push(Round {
                    distance,
                    reversed,
                    pairs,
                });
            }
        }
    }
    out
}

/// Build a disjoint matching at `distance`: within each block of `2*distance`
/// cores, pair `b+j` with `b+j+distance`. Every index in a whole block is used
/// exactly once, and blocks are disjoint, so the result is a matching by
/// construction. A partial trailing block leaves cores over; those are paired
/// off adjacently so no core sits idle while its neighbours are hammering the
/// fabric (an idle core is a core whose ring/fabric port is untested).
fn matching(cores: usize, distance: usize, reversed: bool) -> Vec<CorePair> {
    let mut pairs = Vec::new();
    let mut used = vec![false; cores];
    let block = distance * 2;
    let mut base = 0usize;
    while base < cores {
        for j in 0..distance {
            let lo = base + j;
            let hi = lo + distance;
            if hi >= cores {
                break;
            }
            pairs.push(order(lo, hi, distance, reversed));
            used[lo] = true;
            used[hi] = true;
        }
        base += block;
    }
    let leftover: Vec<usize> = (0..cores).filter(|&i| !used[i]).collect();
    for chunk in leftover.chunks(2) {
        if let [lo, hi] = chunk {
            pairs.push(order(*lo, *hi, hi - lo, reversed));
        }
    }
    pairs
}

#[inline]
fn order(lo: usize, hi: usize, distance: usize, reversed: bool) -> CorePair {
    let (producer, consumer) = if reversed { (hi, lo) } else { (lo, hi) };
    CorePair {
        producer,
        consumer,
        distance,
    }
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// The cross-core coherence / uncore load kernel.
#[derive(Debug, Clone)]
pub struct UncoreKernel {
    /// Ring slots per pair; rounded up to a power of two and clamped to
    /// `[64, 1<<20]`. See [`DEFAULT_RING_SLOTS`] for why the default is sized
    /// the way it is.
    pub ring_slots: usize,
    /// How long each round runs before rotating to the next distance.
    pub dwell: Duration,
    /// Pretend the machine has this many logical cores. `None` = ask the OS.
    /// Present so a technician can shrink a sweep (and so tests are
    /// topology-independent); it does not change which cores exist, only which
    /// pairings are generated.
    pub cores: Option<usize>,
}

impl Default for UncoreKernel {
    fn default() -> Self {
        UncoreKernel::new()
    }
}

impl UncoreKernel {
    pub fn new() -> UncoreKernel {
        UncoreKernel {
            ring_slots: DEFAULT_RING_SLOTS,
            dwell: DEFAULT_DWELL,
            cores: None,
        }
    }

    /// Resolved ring size in slots (power of two, clamped).
    pub fn resolved_slots(&self) -> usize {
        self.ring_slots
            .clamp(MIN_SLOTS, MAX_SLOTS)
            .next_power_of_two()
    }
}

impl LoadKernel for UncoreKernel {
    fn name(&self) -> &str {
        "uncore"
    }

    fn kind(&self) -> Kind {
        // No `Kind::Uncore` exists and adding one is a core-crate change; this is
        // a CPU-domain load either way (it runs on CPU threads and its verdict is
        // about the CPU package).
        Kind::Cpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let cores = self.cores.unwrap_or_else(sysinfo::logical_cpus);
        if cores < 2 {
            return LoadResult::setup_failure(format!(
                "uncore needs at least 2 logical cores to form a producer/consumer pair; found {cores}"
            ));
        }
        let rounds = pair_rounds(cores);
        if rounds.is_empty() {
            return LoadResult::setup_failure(format!(
                "no core pairings could be generated for {cores} logical core(s)"
            ));
        }
        let slots = self.resolved_slots();

        // One shared phase origin for every worker in every round, so burst
        // edges line up system-wide instead of smearing per round (same reason
        // CpuKernel does this). An orchestrator-supplied epoch always wins.
        let base = budget.clone().phased_if_unset(Instant::now());
        let start = Instant::now();
        let deadline = start + budget.duration;

        let mut results: Vec<PairOut> = Vec::new();
        let mut panics = 0u64;
        let mut rounds_run = 0usize;

        for round_index in 0.. {
            if stop.stopped() {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            // Always run at least one round: a run too short for a full slice
            // must still verify something rather than report NOT VERIFIED.
            if remaining < MIN_SLICE && rounds_run > 0 {
                break;
            }
            let round = &rounds[round_index % rounds.len()];
            let mut slice = base.clone();
            slice.duration = self.dwell.min(remaining);

            let (outs, p) = run_round(round, round_index, &slice, stop, markers, slots);
            results.extend(outs);
            panics += p;
            rounds_run += 1;
        }

        let seconds = start.elapsed().as_secs_f64();

        // Aggregate. Faulty pairs are merged across rounds so a link that fails
        // every time it is visited reads as one line with a total, not as N.
        let mut verified = 0u64;
        let mut produced = 0u64;
        let mut errors = 0u64;
        let mut checksum = 0u64;
        let mut all_pinned = true;
        let mut faulty: Vec<PairOut> = Vec::new();
        for o in &results {
            verified += o.verified;
            produced += o.produced;
            errors += o.faults();
            checksum = checksum.wrapping_add(o.checksum);
            all_pinned &= o.pinned;
            if o.faults() > 0 {
                match faulty
                    .iter_mut()
                    .find(|f| f.producer == o.producer && f.consumer == o.consumer)
                {
                    Some(f) => f.merge(o),
                    None => faulty.push(o.clone()),
                }
            }
        }
        // A panicked worker must not vanish into a zeroed result.
        errors += panics;

        // Never report a confident PASS on nothing. A run that ended before a
        // single record was checked has no verdict to give.
        if verified == 0 {
            let mut detail = format!(
                "uncore NOT VERIFIED — {rounds_run} round(s) over {cores} logical core(s) in \
                 {seconds:.1}s produced {produced} record(s) but verified none. The run ended \
                 before the first check (stopped early, budget too short, or every consumer \
                 starved); no conclusion can be drawn."
            );
            if panics > 0 {
                detail.push_str(&format!(" {panics} worker thread(s) PANICKED."));
            }
            return LoadResult::new(false, 0, checksum, errors, detail);
        }

        let mrec = verified as f64 / seconds.max(1e-9) / 1.0e6;
        let gbps = (verified * SLOT_BYTES) as f64 / seconds.max(1e-9) / 1.0e9;
        let pairs_run = results.len();
        let distances: Vec<String> = pair_distances(cores)
            .iter()
            .map(|d| d.to_string())
            .collect();

        let mut detail = format!(
            "{rounds_run} round(s) / {pairs_run} pair-run(s) over {cores} logical core(s), \
             distances [{}], {:.1}M records verified, ~{mrec:.1}M rec/s (~{gbps:.2} GB/s \
             cross-core, approx), ring {} KiB/pair, pinned={all_pinned}",
            distances.join(","),
            verified as f64 / 1.0e6,
            (slots as u64 * SLOT_BYTES) / 1024,
        );
        if panics > 0 {
            detail.push_str(&format!("; {panics} worker thread(s) PANICKED"));
        }
        if !faulty.is_empty() {
            detail.push_str("; COHERENCE FAULTS: ");
            // Cap the list so one systemically-bad fabric cannot produce a
            // report line per pair; the count still tells the whole story.
            const MAX_LISTED: usize = 8;
            let listed: Vec<String> = faulty
                .iter()
                .take(MAX_LISTED)
                .map(|f| f.describe())
                .collect();
            detail.push_str(&listed.join("; "));
            if faulty.len() > MAX_LISTED {
                detail.push_str(&format!(
                    "; ... and {} more faulty pair(s)",
                    faulty.len() - MAX_LISTED
                ));
            }
        }

        LoadResult::new(true, verified, checksum, errors, detail)
    }
}

// ---------------------------------------------------------------------------
// Per-round execution
// ---------------------------------------------------------------------------

/// One pair's outcome for one round.
#[derive(Debug, Clone)]
pub struct PairOut {
    pub producer: usize,
    pub consumer: usize,
    pub distance: usize,
    pub produced: u64,
    pub verified: u64,
    pub seq_faults: u64,
    pub payload_faults: u64,
    pub check_faults: u64,
    pub checksum: u64,
    pub first: Option<Fault>,
    /// Both threads were pinned to their intended cores. Unpinned means the
    /// distance (and therefore the topology conclusion) is not trustworthy.
    pub pinned: bool,
}

impl PairOut {
    pub fn faults(&self) -> u64 {
        self.seq_faults + self.payload_faults + self.check_faults
    }

    /// Fold another visit to the same pair into this one.
    fn merge(&mut self, o: &PairOut) {
        self.produced += o.produced;
        self.verified += o.verified;
        self.seq_faults += o.seq_faults;
        self.payload_faults += o.payload_faults;
        self.check_faults += o.check_faults;
        if self.first.is_none() {
            self.first = o.first;
        }
    }

    /// The diagnostic line. The core pair leads, because the pair *is* the
    /// diagnosis — a fault confined to long-distance pairs is the cross-die
    /// (FCLK / SoC voltage) signature, a fault on adjacent cores is not.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.payload_faults > 0 {
            parts.push(format!("{} payload miscompare(s)", self.payload_faults));
        }
        if self.check_faults > 0 {
            parts.push(format!("{} check-word miscompare(s)", self.check_faults));
        }
        if self.seq_faults > 0 {
            parts.push(format!("{} sequence fault(s)", self.seq_faults));
        }
        let mut s = format!(
            "core {} -> core {} (d={}): {}",
            self.producer,
            self.consumer,
            self.distance,
            parts.join(", ")
        );
        if let Some(f) = self.first {
            s.push_str(&format!(", first at seq {}: {}", f.seq, f.describe()));
        }
        if !self.pinned {
            s.push_str(" [UNPINNED — distance not trustworthy]");
        }
        s
    }
}

/// Everything a worker needs that is identical for every thread in a round.
#[derive(Clone, Copy)]
struct RoundCtx<'a> {
    budget: &'a Budget,
    stop: &'a StopFlag,
    markers: &'a MarkerLog,
    round: &'a Round,
    /// Pairs live in this round — shown in the live UI so the operator can see
    /// the whole machine is loaded, not just the one pair being reported.
    round_count: usize,
}

/// Run every pair of one round concurrently and collect their outcomes.
fn run_round(
    round: &Round,
    round_index: usize,
    budget: &Budget,
    stop: &StopFlag,
    markers: &MarkerLog,
    slots: usize,
) -> (Vec<PairOut>, u64) {
    // Rings are allocated up front (outside the thread scope) so both halves of
    // a pair can borrow the same one, and so allocation/first-touch cost is not
    // charged to the timed window.
    let rings: Vec<Ring> = round
        .pairs
        .iter()
        .map(|_| Ring::with_slots(slots))
        .collect();
    let ctx = RoundCtx {
        budget,
        stop,
        markers,
        round,
        round_count: round.pairs.len(),
    };

    std::thread::scope(|scope| {
        let handles: Vec<_> = round
            .pairs
            .iter()
            .zip(rings.iter())
            .enumerate()
            .map(|(i, (pair, ring))| {
                // A distinct data stream per pair, deterministic from the round
                // and pair index, so two rings never carry identical bytes (which
                // would let a cross-ring mix-up look like a pass).
                let salt = hash2(SALT_BASE ^ (round_index as u64), i as u64);
                let p = *pair;
                let ph = scope.spawn(move || produce(p, ring, salt, ctx));
                // Pair 0's consumer drives the live UI lane; the rest stay quiet
                // so the panel does not thrash from N writers.
                let ch = scope.spawn(move || consume(p, ring, salt, ctx, i == 0));
                (p, ph, ch)
            })
            .collect();

        let mut outs = Vec::new();
        let mut panics = 0u64;
        for (pair, ph, ch) in handles {
            let prod = ph.join();
            let cons = ch.join();
            match (prod, cons) {
                (Ok(p), Ok(c)) => outs.push(PairOut {
                    producer: pair.producer,
                    consumer: pair.consumer,
                    distance: pair.distance,
                    produced: p.produced,
                    verified: c.verifier.verified,
                    seq_faults: c.verifier.seq_faults,
                    payload_faults: c.verifier.payload_faults,
                    check_faults: c.verifier.check_faults,
                    checksum: c.verifier.checksum,
                    first: c.verifier.first,
                    pinned: p.pinned && c.pinned,
                }),
                _ => panics += 1,
            }
        }
        (outs, panics)
    })
}

struct ProdOut {
    produced: u64,
    pinned: bool,
}

struct ConsOut {
    verifier: Verifier,
    pinned: bool,
}

/// Producer half: pin, then stream records until the shape driver says stop.
fn produce(pair: CorePair, ring: &Ring, salt: u64, ctx: RoundCtx<'_>) -> ProdOut {
    let pinned = pin_current_thread(pair.producer);
    let detail = format!(
        "{} produce core {}->{}",
        ctx.round.label(),
        pair.producer,
        pair.consumer
    );
    let mut driver = ShapeDriver::start(ctx.budget, ctx.stop, ctx.markers, LANE, detail);
    let mut tx = ring.producer();
    let mut seq = 0u64;

    loop {
        match driver.tick() {
            Tick::Work => {
                let mut done = 0u64;
                while done < CHUNK_RECORDS {
                    if !push_with_backoff(&mut tx, Record::make(salt, seq)) {
                        // Ring full for a long time: the consumer is descheduled
                        // or idling. Give up this chunk so the shape driver and
                        // stop flag get a look in.
                        std::thread::yield_now();
                        break;
                    }
                    seq = seq.wrapping_add(1);
                    done += 1;
                }
            }
            Tick::Idle => {}
            Tick::Stop => break,
        }
    }

    ProdOut {
        produced: tx.published(),
        pinned,
    }
}

/// Consumer half: pin, then verify everything the producer hands over.
fn consume(pair: CorePair, ring: &Ring, salt: u64, ctx: RoundCtx<'_>, report: bool) -> ConsOut {
    let pinned = pin_current_thread(pair.consumer);
    let detail = format!(
        "{} consume core {}->{}",
        ctx.round.label(),
        pair.producer,
        pair.consumer
    );
    let mut driver = ShapeDriver::start(ctx.budget, ctx.stop, ctx.markers, LANE, detail);
    let mut rx = ring.consumer();
    let mut v = Verifier::new(salt);
    let start = Instant::now();
    let mut last_note = start - Duration::from_secs(1);

    loop {
        match driver.tick() {
            Tick::Work => {
                let mut done = 0u64;
                let mut stall = 0u32;
                while done < CHUNK_RECORDS {
                    match rx.pop() {
                        Some(r) => {
                            v.check(r);
                            done += 1;
                            stall = 0;
                        }
                        None => {
                            stall += 1;
                            if stall > SPIN_LIMIT {
                                std::thread::yield_now();
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                if report && driver.live() {
                    let now = Instant::now();
                    if now.duration_since(last_note) >= Duration::from_millis(90) {
                        last_note = now;
                        publish_status(&driver, &ctx, pair, &v, start, ring.capacity());
                    }
                }
            }
            Tick::Idle => {}
            Tick::Stop => break,
        }
    }

    // Bounded final drain: verify whatever the producer published just before
    // the deadline instead of throwing it away. Capped at one ring's worth so
    // this can never spin.
    let mut left = ring.capacity();
    while left > 0 {
        match rx.pop() {
            Some(r) => v.check(r),
            None => break,
        }
        left -= 1;
    }

    ConsOut {
        verifier: v,
        pinned,
    }
}

/// Push, spinning briefly on a full ring. `false` means "still full after
/// `SPIN_LIMIT` attempts" — the caller ends its chunk.
#[inline]
fn push_with_backoff(tx: &mut Producer<'_>, r: Record) -> bool {
    for _ in 0..SPIN_LIMIT {
        if tx.push(r) {
            return true;
        }
        std::hint::spin_loop();
    }
    false
}

/// Publish the live-UI status line (allocating; only called throttled, and only
/// when a UI is attached).
fn publish_status(
    driver: &ShapeDriver<'_>,
    ctx: &RoundCtx<'_>,
    pair: CorePair,
    v: &Verifier,
    start: Instant,
    capacity: u64,
) {
    let secs = start.elapsed().as_secs_f64().max(1e-6);
    let mrec = v.verified as f64 / secs / 1.0e6;
    let gbps = (v.verified * SLOT_BYTES) as f64 / secs / 1.0e9;
    driver.set_hash(v.checksum);
    driver.set_status(&format!(
        "round: {} ({} pair(s) live)\npair: core {} -> core {}  d={}\nrate: {mrec:.1}M rec/s ({gbps:.2} GB/s)\nverified: {:.2}M records\nerrors: {}\nring: {} KiB",
        ctx.round.label(),
        ctx.round_count,
        pair.producer,
        pair.consumer,
        pair.distance,
        v.verified as f64 / 1.0e6,
        v.faults(),
        (capacity * SLOT_BYTES) / 1024,
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crucible_core::markers::MarkerLog;

    // ---- ring correctness -------------------------------------------------

    #[test]
    fn ring_round_trips_cleanly() {
        // Single-threaded: everything pushed comes back in order, and the
        // verifier finds nothing wrong. If this ever fails, the harness is
        // broken, not the hardware.
        let ring = Ring::with_slots(64);
        let mut tx = ring.producer();
        let mut rx = ring.consumer();
        let mut v = Verifier::new(0xABCD);

        for seq in 0..1000u64 {
            assert!(tx.push(Record::make(0xABCD, seq)), "push {seq} failed");
            let r = rx.pop().expect("record available immediately after push");
            assert_eq!(r, Record::make(0xABCD, seq));
            v.check(r);
        }
        assert_eq!(v.verified, 1000);
        assert_eq!(v.faults(), 0, "clean ring produced a fault");
        assert_ne!(v.checksum, 0, "checksum should fold real data");
    }

    #[test]
    fn ring_layout_avoids_false_sharing() {
        // Load-bearing, and invisible if it regresses: a record must not
        // straddle a cache line, and head/tail must not share one. If they do,
        // every record costs a coherence round trip on a single ping-ponging
        // line and the kernel silently stops being an interconnect *bandwidth*
        // test — it still passes, it just measures almost nothing.
        assert_eq!(std::mem::size_of::<Slot>(), SLOT_BYTES as usize);
        assert_eq!(std::mem::align_of::<Slot>(), SLOT_BYTES as usize);
        let ring = Ring::with_slots(64);
        let head = &ring.head as *const Index as usize;
        let tail = &ring.tail as *const Index as usize;
        assert_eq!(head % 128, 0, "head index is not line-aligned");
        assert_eq!(tail % 128, 0, "tail index is not line-aligned");
        assert!(
            head.abs_diff(tail) >= 128,
            "head and tail share a cache line"
        );
    }

    #[test]
    fn ring_respects_capacity_and_empties() {
        let ring = Ring::with_slots(64);
        assert_eq!(ring.capacity(), 64);
        let mut tx = ring.producer();
        let mut rx = ring.consumer();
        for seq in 0..64u64 {
            assert!(tx.push(Record::make(1, seq)));
        }
        // Full: the producer must refuse rather than overwrite an unread slot.
        assert!(!tx.push(Record::make(1, 64)));
        for seq in 0..64u64 {
            assert_eq!(rx.pop().unwrap().seq, seq);
        }
        assert!(rx.pop().is_none(), "empty ring must yield None");
        // And it is usable again after draining.
        assert!(tx.push(Record::make(1, 64)));
        assert_eq!(rx.pop().unwrap().seq, 64);
    }

    #[test]
    fn ring_is_clean_across_two_real_threads() {
        // The actual configuration the kernel runs: one thread pushing, another
        // popping and verifying, concurrently. Any ordering bug in the ring
        // would show up here as a seq or payload fault.
        const N: u64 = 200_000;
        let ring = Ring::with_slots(256);
        let salt = 0x5EED_5EEDu64;
        let v = std::thread::scope(|scope| {
            let r = &ring;
            scope.spawn(move || {
                let mut tx = r.producer();
                let mut seq = 0u64;
                while seq < N {
                    if tx.push(Record::make(salt, seq)) {
                        seq += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });
            let h = scope.spawn(move || {
                let mut rx = r.consumer();
                let mut v = Verifier::new(salt);
                while v.verified < N {
                    match rx.pop() {
                        Some(rec) => v.check(rec),
                        None => std::hint::spin_loop(),
                    }
                }
                v
            });
            h.join().expect("consumer thread")
        });
        assert_eq!(v.verified, N);
        assert_eq!(
            v.faults(),
            0,
            "concurrent ring produced a fault: {:?}",
            v.first
        );
    }

    // ---- fault detection --------------------------------------------------

    #[test]
    fn corrupted_payload_is_detected_and_names_the_pair_and_seq() {
        // Simulate a bit flip in transit: the record is written correctly, then
        // a bit is knocked out of the payload before the consumer reads it.
        const BAD_SEQ: u64 = 37;
        let salt = 0xF00Du64;
        let ring = Ring::with_slots(64);
        let mut tx = ring.producer();
        let mut rx = ring.consumer();
        for seq in 0..64u64 {
            assert!(tx.push(Record::make(salt, seq)));
        }
        let good = Record::make(salt, BAD_SEQ);
        ring.poke_payload(BAD_SEQ, good.payload ^ (1 << 19));

        let mut v = Verifier::new(salt);
        while let Some(r) = rx.pop() {
            v.check(r);
        }

        assert_eq!(v.verified, 64);
        assert_eq!(v.seq_faults, 0, "sequence was intact");
        assert_eq!(v.payload_faults, 1, "exactly one corrupted payload");
        // The check word no longer matches the payload as read, so the
        // independent second check fires too.
        assert_eq!(v.check_faults, 1);
        let f = v.first.expect("a fault was recorded");
        assert_eq!(f.kind, FaultKind::Payload);
        assert_eq!(f.seq, BAD_SEQ);
        assert_eq!(f.expected, good.payload);
        assert_eq!(f.got, good.payload ^ (1 << 19));

        // And the report must name the pair — that is the whole diagnostic value.
        let out = PairOut {
            producer: 4,
            consumer: 12,
            distance: 8,
            produced: 64,
            verified: v.verified,
            seq_faults: v.seq_faults,
            payload_faults: v.payload_faults,
            check_faults: v.check_faults,
            checksum: v.checksum,
            first: v.first,
            pinned: true,
        };
        let msg = out.describe();
        assert!(msg.starts_with("core 4 -> core 12 (d=8):"), "msg: {msg}");
        assert!(msg.contains("1 payload miscompare(s)"), "msg: {msg}");
        assert!(msg.contains("first at seq 37"), "msg: {msg}");
        assert_eq!(out.faults(), 2);
    }

    #[test]
    fn seq_gap_is_detected_reported_and_resynced() {
        // A dropped record must be reported once, not turn every later record
        // into a fault (which would make the error count meaningless).
        let salt = 0x1234u64;
        let mut v = Verifier::new(salt);
        for seq in 0..5000u64 {
            v.check(Record::make(salt, seq));
        }
        for seq in 5002..6000u64 {
            v.check(Record::make(salt, seq));
        }
        assert_eq!(v.seq_faults, 1, "one gap, one fault");
        assert_eq!(v.payload_faults, 0);
        assert_eq!(v.check_faults, 0);
        let f = v.first.expect("gap recorded");
        assert_eq!(f.kind, FaultKind::Sequence);
        assert_eq!(f.expected, 5000);
        assert_eq!(f.got, 5002);
        assert_eq!(f.describe(), "seq gap: expected 5000, got 5002");
    }

    #[test]
    fn duplicate_seq_is_detected() {
        let salt = 7u64;
        let mut v = Verifier::new(salt);
        v.check(Record::make(salt, 0));
        v.check(Record::make(salt, 1));
        v.check(Record::make(salt, 1)); // replayed record
        assert_eq!(v.seq_faults, 1);
        let f = v.first.unwrap();
        assert_eq!(f.kind, FaultKind::Sequence);
        assert!(f.describe().starts_with("seq regression/duplicate"));
    }

    #[test]
    fn corrupted_check_word_alone_is_detected() {
        // Payload intact, check word flipped: only the check test may fire.
        let salt = 99u64;
        let mut good = Record::make(salt, 500);
        let mut v = Verifier::new(salt);
        for seq in 0..500u64 {
            v.check(Record::make(salt, seq));
        }
        good.check ^= 1;
        v.check(good);
        assert_eq!(v.payload_faults, 0);
        assert_eq!(v.seq_faults, 0);
        assert_eq!(v.check_faults, 1);
        assert_eq!(v.first.unwrap().kind, FaultKind::Check);
    }

    // ---- pair schedule ----------------------------------------------------

    #[test]
    fn pair_rounds_are_deterministic() {
        for n in [2usize, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256] {
            assert_eq!(
                pair_rounds(n),
                pair_rounds(n),
                "schedule must be pure for {n}"
            );
        }
        assert!(pair_rounds(1).is_empty(), "one core cannot form a pair");
        assert!(pair_rounds(0).is_empty());
    }

    #[test]
    fn pair_rounds_are_bounded_for_wide_parts() {
        // The whole point of the round scheme: a 128-thread part must not turn
        // into an N^2 sweep.
        for n in [64usize, 128, 256] {
            let rounds = pair_rounds(n);
            assert!(rounds.len() <= 12, "{n} cores -> {} rounds", rounds.len());
            let total: usize = rounds.iter().map(|r| r.pairs.len()).sum();
            assert!(total <= 6 * n, "{n} cores -> {total} pairs");
            assert!(
                total < n * n / 4,
                "{n} cores -> {total} pairs is not bounded"
            );
        }
    }

    #[test]
    fn every_round_is_a_disjoint_matching() {
        // Concurrency and attribution both depend on this: one thread per core,
        // one ring per pair.
        for n in [2usize, 3, 5, 6, 8, 9, 12, 16, 24, 32, 48, 64, 128] {
            for round in pair_rounds(n) {
                let mut seen = vec![false; n];
                for p in &round.pairs {
                    assert!(
                        p.producer < n && p.consumer < n,
                        "core out of range for {n}"
                    );
                    assert_ne!(p.producer, p.consumer, "a core cannot pair with itself");
                    for c in [p.producer, p.consumer] {
                        assert!(
                            !seen[c],
                            "core {c} appears twice in round {}",
                            round.label()
                        );
                        seen[c] = true;
                    }
                    assert_eq!(
                        p.distance,
                        p.producer.abs_diff(p.consumer),
                        "distance must match the pairing"
                    );
                }
                assert!(!round.pairs.is_empty());
            }
        }
    }

    #[test]
    fn distances_span_near_and_far() {
        // Adjacent (SMT sibling) and half-the-machine (cross-die on a symmetric
        // two-die part) must both be covered, in both directions.
        let d = pair_distances(16);
        assert_eq!(d, vec![1, 2, 4, 8]);
        assert_eq!(pair_distances(128), vec![1, 2, 4, 16, 32, 64]);
        assert_eq!(pair_distances(2), vec![1]);

        let rounds = pair_rounds(16);
        assert!(rounds.iter().any(|r| r.distance == 1 && !r.reversed));
        assert!(rounds.iter().any(|r| r.distance == 8 && r.reversed));
        // The far round pairs low half against high half.
        let far = rounds
            .iter()
            .find(|r| r.distance == 8 && !r.reversed)
            .unwrap();
        assert_eq!(far.pairs.len(), 8);
        assert!(far.pairs.iter().all(|p| p.producer < 8 && p.consumer >= 8));
    }

    #[test]
    fn leftover_cores_are_still_paired() {
        // A partial trailing block must not leave cores idle — an idle core is
        // an untested fabric port.
        let m = matching(12, 4, false);
        let mut seen = [false; 12];
        for p in &m {
            seen[p.producer] = true;
            seen[p.consumer] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "every core of 12 should be paired at d=4"
        );
    }

    // ---- kernel behaviour -------------------------------------------------

    #[test]
    fn short_run_verifies_and_passes() {
        let kernel = UncoreKernel {
            ring_slots: 256,
            dwell: Duration::from_millis(40),
            cores: Some(4),
        };
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_millis(200));
        let r = kernel.run(&budget, &stop, &markers);
        assert!(r.ok, "kernel failed to run: {}", r.detail);
        assert_eq!(r.error_count, 0, "false coherence fault: {}", r.detail);
        assert!(r.iterations > 0, "nothing verified: {}", r.detail);
        assert!(r.passed());
        assert!(
            r.detail.contains("records verified"),
            "detail: {}",
            r.detail
        );
        assert!(r.detail.contains("distances ["), "detail: {}", r.detail);
    }

    #[test]
    fn burst_shape_runs_and_stamps_edges() {
        // Composes into cross-load / mix: the shape driver must actually drive
        // this kernel, not just the FMA one.
        let kernel = UncoreKernel {
            ring_slots: 256,
            dwell: Duration::from_millis(60),
            cores: Some(2),
        };
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::burst(
            Duration::from_millis(180),
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        let r = kernel.run(&budget, &stop, &markers);
        assert!(r.ok, "{}", r.detail);
        assert_eq!(r.error_count, 0, "{}", r.detail);
        assert!(markers.len() >= 2, "burst edges should be stamped");
    }

    #[test]
    fn stop_flag_ends_run_early_without_claiming_a_pass() {
        // Pre-stopped: nothing is verified, so the result must NOT be a pass.
        // This is the false-PASS class the RT kernel had to fix; do not
        // reintroduce it here.
        let kernel = UncoreKernel::new();
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_secs(3600));
        stop.stop();
        let start = Instant::now();
        let r = kernel.run(&budget, &stop, &markers);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "stop must be prompt"
        );
        assert!(!r.ok, "a run that verified nothing must not report ok");
        assert!(!r.passed());
        assert!(r.detail.contains("NOT VERIFIED"), "detail: {}", r.detail);
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn single_core_machine_is_a_setup_failure() {
        let kernel = UncoreKernel {
            cores: Some(1),
            ..UncoreKernel::new()
        };
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let r = kernel.run(&Budget::steady(Duration::from_millis(50)), &stop, &markers);
        assert!(!r.ok);
        assert!(!r.passed());
        assert!(
            r.detail.contains("at least 2 logical cores"),
            "detail: {}",
            r.detail
        );
    }

    #[test]
    fn ring_slots_are_clamped_to_a_power_of_two() {
        let k = UncoreKernel {
            ring_slots: 1000,
            ..UncoreKernel::new()
        };
        assert_eq!(k.resolved_slots(), 1024);
        let tiny = UncoreKernel {
            ring_slots: 1,
            ..UncoreKernel::new()
        };
        assert_eq!(tiny.resolved_slots(), MIN_SLOTS);
        assert_eq!(Ring::with_slots(3).capacity(), MIN_SLOTS as u64);
    }

    #[test]
    fn checksum_fold_is_order_independent() {
        // Pairs finish in whatever order the scheduler decides; the run checksum
        // must not depend on that.
        let salt = 0xBEEFu64;
        let mut a = Verifier::new(salt);
        let mut b = Verifier::new(salt);
        let recs: Vec<Record> = (0..64).map(|s| Record::make(salt, s)).collect();
        for r in &recs {
            a.check(*r);
        }
        // Same set, folded from a different starting point via two verifiers.
        let mut b2 = Verifier::new(salt);
        for r in &recs[..32] {
            b.check(*r);
        }
        for (i, r) in recs[32..].iter().enumerate() {
            // b2 starts at seq 32; skip its ordering check by pre-seeding.
            if i == 0 {
                b2.expect = 32;
            }
            b2.check(*r);
        }
        assert_eq!(a.checksum, b.checksum.wrapping_add(b2.checksum));
        assert_eq!(b2.faults(), 0);
    }
}
