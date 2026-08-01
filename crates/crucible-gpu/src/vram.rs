// SPDX-License-Identifier: MIT
//! VRAM **integrity** test — find bad video memory.
//!
//! This is a different test from the wattage thrasher in [`crate`], with a
//! different objective. The thrasher maximizes watts and does not care what is
//! in the buffer; this walks patterns across VRAM and verifies every word to
//! find bad memory. Watts are irrelevant here. Do not conflate them.
//!
//! ## Design
//!
//! * **Index-derived patterns.** Every expected value is a pure function of the
//!   element index (plus a pattern id / seed), so each GPU thread computes its
//!   own expectation independently — no sequential PRNG state to thread through
//!   a massively parallel device, and verification needs no golden copy.
//! * **Verify on the GPU.** A check kernel compares each word and reports via
//!   atomics into an 8-byte results buffer. Dragging gigabytes of VRAM back over
//!   PCIe every pass would make the test bandwidth-bound and pointlessly slow;
//!   the host reads the full chunk back *only* when an error is found, to
//!   recover the observed value for the report.
//! * **Chained moving inversions.** Each pass verifies the pattern the previous
//!   pass wrote and lays down the next one in the same dispatch, so a cell is
//!   rewritten immediately after being read — the property moving-inversion
//!   testing depends on.
//! * **Battery:** own-address, constant 0x00/0xFF/0xAA/0x55, checkerboard +
//!   inverse, walking-ones / walking-zeros across every bit position, and a
//!   seeded index-hash — all pure functions of the index. (The ordered March
//!   tests are CPU-only: their fault sensitization needs strict address
//!   ordering, which massively-parallel GPU execution destroys.)
//! * **Chunked allocation.** VRAM is allocated as several buffers rather than
//!   one, because storage-buffer binding size is limited (wgpu's default is
//!   128 MiB) and because a failure read-back should cost one chunk, not the
//!   whole card. All chunks stay resident so the full requested VRAM is under
//!   test at once.
//!
//! Caveat worth stating plainly: the verification runs on the same GPU being
//! tested, so a sufficiently broken device could in principle mis-verify. That
//! is inherent to GPU-side memory testing (memtest_vulkan has the same
//! property) and is the price of not being PCIe-bound.

use std::time::{Duration, Instant};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, StopFlag};
use crucible_core::markers::{Event, MarkerLog, PHASE_DONE, PHASE_WORK};

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::GpuDevice;

/// Pattern generators. Kept as small integers so they can be passed as runtime
/// scalars and the same kernel serves every pattern. All are pure functions of
/// the element index, which is what lets each GPU thread verify independently.
const MODE_OWN_ADDRESS: u32 = 0;
const MODE_CONSTANT: u32 = 1;
const MODE_RANDOM: u32 = 2;
/// A single 1 walking through the 32-bit word; `value` carries the step.
const MODE_WALK_ONE: u32 = 3;
/// A single 0 walking through an all-ones word; `value` carries the step.
const MODE_WALK_ZERO: u32 = 4;
/// Adjacent cells held at 0xAAAA.. / 0x5555..; `value` carries the phase (0/1).
const MODE_CHECKER: u32 = 5;

const WORD_BITS: u32 = 32;

/// Expected content of element `i` under a given pattern — the single source of
/// truth, mirrored on the host by [`expected_host`].
#[cube]
fn expected_value(i: usize, mode: u32, value: u32, seed: u32) -> u32 {
    let idx = u32::cast_from(i);
    let mut out = value; // MODE_CONSTANT default
    if mode == MODE_OWN_ADDRESS {
        // Catches address-decode faults: every cell holds its own address.
        out = idx;
    } else if mode == MODE_RANDOM {
        // Index-derived hash (murmur-style finalizer): data-dependent coverage
        // without any cross-thread state.
        let mut z = idx ^ seed;
        z = (z ^ (z >> 16)) * 0x7feb_352du32;
        z = (z ^ (z >> 15)) * 0x846c_a68bu32;
        z = z ^ (z >> 16);
        out = z;
    } else if mode == MODE_WALK_ONE {
        // Isolated single set bit — the hardest bit to drive; targets data-line
        // coupling. The (idx + step) offset gives a diagonal so neighbours differ.
        out = 1u32 << ((idx + value) % WORD_BITS);
    } else if mode == MODE_WALK_ZERO {
        out = !(1u32 << ((idx + value) % WORD_BITS));
    } else if mode == MODE_CHECKER {
        let mut c = 0xAAAA_AAAAu32;
        if ((idx ^ value) & 1) != 0 {
            c = 0x5555_5555u32;
        }
        out = c;
    }
    out
}

/// Host-side twin of [`expected_value`], for reporting what a failing cell
/// should have held. Must stay bit-identical to the `#[cube]` version above.
fn expected_host(i: usize, mode: u32, value: u32, seed: u32) -> u32 {
    let idx = i as u32;
    match mode {
        MODE_OWN_ADDRESS => idx,
        MODE_RANDOM => {
            let mut z = idx ^ seed;
            z = (z ^ (z >> 16)).wrapping_mul(0x7feb352d);
            z = (z ^ (z >> 15)).wrapping_mul(0x846ca68b);
            z ^ (z >> 16)
        }
        MODE_WALK_ONE => 1u32 << ((idx.wrapping_add(value)) % WORD_BITS),
        MODE_WALK_ZERO => !(1u32 << ((idx.wrapping_add(value)) % WORD_BITS)),
        MODE_CHECKER => {
            if ((idx ^ value) & 1) != 0 {
                0x5555_5555
            } else {
                0xAAAA_AAAA
            }
        }
        _ => value,
    }
}

/// Fill a chunk with a pattern.
///
/// Grid-stride loop: each thread walks the buffer in steps of `stride` (the
/// total launched thread count). A one-element-per-thread launch would need
/// 65,536 workgroups for a 64 MiB chunk at 256 threads, which exceeds the
/// 65,535-per-dimension dispatch limit — this decouples dispatch size from
/// buffer size entirely.
#[cube(launch)]
fn vram_fill(buf: &mut Array<u32>, mode: u32, value: u32, seed: u32, stride: usize) {
    let n = buf.len();
    let mut i = ABSOLUTE_POS;
    while i < n {
        buf[i] = expected_value(i, mode, value, seed);
        i += stride;
    }
}

/// Verify the current pattern and immediately write the next one.
///
/// `results[0]` accumulates the error count; `results[1]` keeps the lowest
/// failing index (via `fetch_min`), which the host uses to report a first fail.
#[cube(launch)]
fn vram_check(
    buf: &mut Array<u32>,
    results: &mut Array<Atomic<u32>>,
    mode: u32,
    value: u32,
    seed: u32,
    next_mode: u32,
    next_value: u32,
    next_seed: u32,
    stride: usize,
) {
    let n = buf.len();
    let mut i = ABSOLUTE_POS;
    while i < n {
        let want = expected_value(i, mode, value, seed);
        let got = buf[i];
        if got != want {
            results[0].fetch_add(1);
            results[1].fetch_min(u32::cast_from(i));
        }
        // Rewrite immediately after reading — the moving-inversion property.
        buf[i] = expected_value(i, next_mode, next_value, next_seed);
        i += stride;
    }
}

/// One step of the battery: the pattern to verify now.
#[derive(Debug, Clone, Copy)]
struct Step {
    mode: u32,
    value: u32,
    seed: u32,
    label: &'static str,
}

/// The pattern chain. Each step verifies what the previous step wrote and lays
/// down the next, so consecutive complements (0x00/0xFF, 0xAA/0x55, checker
/// phases) form genuine moving inversions. Walking-bit steps march an isolated
/// 1 (then 0) through every bit position for data-line / isolated-bit coverage.
fn battery(seed: u32) -> Vec<Step> {
    let mut steps = vec![
        Step {
            mode: MODE_OWN_ADDRESS,
            value: 0,
            seed: 0,
            label: "own-address",
        },
        Step {
            mode: MODE_CONSTANT,
            value: 0x0000_0000,
            seed: 0,
            label: "const-00",
        },
        Step {
            mode: MODE_CONSTANT,
            value: 0xFFFF_FFFF,
            seed: 0,
            label: "const-ff",
        },
        Step {
            mode: MODE_CONSTANT,
            value: 0xAAAA_AAAA,
            seed: 0,
            label: "const-aa",
        },
        Step {
            mode: MODE_CONSTANT,
            value: 0x5555_5555,
            seed: 0,
            label: "const-55",
        },
        Step {
            mode: MODE_CHECKER,
            value: 0,
            seed: 0,
            label: "checker",
        },
        Step {
            mode: MODE_CHECKER,
            value: 1,
            seed: 0,
            label: "checker-inv",
        },
    ];
    // Walking ones then walking zeros across all bit positions.
    for step in 0..WORD_BITS {
        steps.push(Step {
            mode: MODE_WALK_ONE,
            value: step,
            seed: 0,
            label: "walk-one",
        });
    }
    for step in 0..WORD_BITS {
        steps.push(Step {
            mode: MODE_WALK_ZERO,
            value: step,
            seed: 0,
            label: "walk-zero",
        });
    }
    steps.push(Step {
        mode: MODE_RANDOM,
        value: 0,
        seed,
        label: "random",
    });
    steps
}

/// Details of the first bad word found.
#[derive(Debug, Clone, Copy)]
struct VramFail {
    chunk: usize,
    index: usize,
    expected: u32,
    observed: u32,
    pattern: &'static str,
}

/// The VRAM integrity kernel.
#[derive(Debug, Clone)]
pub struct VramKernel {
    pub device: GpuDevice,
    /// Total VRAM to place under test, in MiB. Allocation stops early if the
    /// card cannot satisfy it.
    pub vram_mb: usize,
    /// Size of each allocation, in MiB. Kept under the storage-buffer binding
    /// limit and small enough that a failure read-back is cheap.
    pub chunk_mb: usize,
    pub workgroup: u32,
}

impl Default for VramKernel {
    fn default() -> Self {
        VramKernel {
            device: GpuDevice::Discrete(0),
            vram_mb: 2048,
            chunk_mb: 64,
            workgroup: 256,
        }
    }
}

impl VramKernel {
    pub fn new(device: GpuDevice) -> Self {
        VramKernel {
            device,
            ..Default::default()
        }
    }
}

impl LoadKernel for VramKernel {
    fn name(&self) -> &str {
        "vram"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        let dev = self.device.to_wgpu();
        let client = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&dev)
        })) {
            Ok(c) => c,
            Err(_) => {
                return LoadResult::setup_failure(format!(
                    "no usable GPU for {} (adapter init failed)",
                    self.device.label()
                ))
            }
        };

        let chunk_bytes = self.chunk_mb.max(1) * 1024 * 1024;
        let chunk_elems = chunk_bytes / std::mem::size_of::<u32>();
        let target_bytes = self.vram_mb.max(self.chunk_mb) * 1024 * 1024;

        // On a UMA adapter every "VRAM" chunk is really system RAM, so filling
        // the requested span can starve the machine rather than merely the card.
        // There is no device to lose — Windows just starts paging, and a memory
        // test that has become a disk test tells you nothing while making the
        // operator's session unusable. So on UMA we watch real available memory
        // as we go and stop while there is still headroom.
        let uma = crate::adapter::resolve(self.device)
            .map(|a| a.uma)
            .unwrap_or(false);
        let host_reserve = crucible_core::sysinfo::memory()
            .map(|m| crucible_core::sysinfo::working_set_reserve_bytes(m.total_bytes))
            .unwrap_or(0);

        // Allocate every chunk up front so the whole requested span is resident
        // and genuinely under test — testing one buffer repeatedly would only
        // ever exercise the same physical memory.
        let mut chunks = Vec::new();
        let mut stopped_for_headroom = false;
        while chunks.len() * chunk_bytes + chunk_bytes <= target_bytes {
            if uma && host_reserve > 0 {
                // Re-read rather than predict: the only number that matters is
                // what the machine has left *now*, with this run's allocations
                // already counted against it.
                if let Some(m) = crucible_core::sysinfo::memory() {
                    if m.avail_bytes < host_reserve + chunk_bytes as u64 {
                        stopped_for_headroom = true;
                        break;
                    }
                }
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                client.empty(chunk_bytes)
            })) {
                Ok(h) => chunks.push(h),
                // Out of VRAM (or over a binding limit): test what we did get.
                Err(_) => break,
            }
        }
        if chunks.is_empty() {
            return LoadResult::setup_failure(format!(
                "could not allocate any VRAM chunk of {} MiB",
                self.chunk_mb
            ));
        }

        let tested_mb = chunks.len() * self.chunk_mb;
        markers.stamp(
            Event::Mark,
            "vram",
            "integrity",
            &format!(
                "{} chunk(s) x {} MiB = {} MiB under test",
                chunks.len(),
                self.chunk_mb,
                tested_mb
            ),
        );

        let wg = self.workgroup.clamp(1, 1024);
        let cube_dim = CubeDim::new_1d(wg);
        // Fixed grid well under the 65,535-per-dimension dispatch limit; the
        // grid-stride loop covers the rest of the buffer. `stride` is the total
        // launched thread count.
        let groups = (chunk_elems.div_ceil(wg as usize) as u32).min(4096);
        let stride = (groups as usize) * (wg as usize);
        let deadline = Instant::now() + budget.duration;
        let start = Instant::now();

        let mut passes: u64 = 0;
        let mut errors: u64 = 0;
        let mut words_checked: u64 = 0;
        let mut first_fail: Option<VramFail> = None;
        let mut io_error: Option<String> = None;

        // Live-UI lane for the dashboard (None unless a UI is attached), plus a
        // throttle so the (locking) status push is touched ~10x/s, not per chunk.
        let lane = markers.register_lane("vram");
        let mut last_note = Instant::now() - Duration::from_secs(1);

        // Seed the chain: lay down the first pattern in every chunk.
        let mut steps = battery(0xC0FF_EE01);
        for handle in &chunks {
            vram_fill::launch::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                cube_dim,
                unsafe { ArrayArg::from_raw_parts(handle.clone(), chunk_elems) },
                steps[0].mode,
                steps[0].value,
                steps[0].seed,
                stride,
            );
        }
        if cubecl::future::block_on(client.sync()).is_err() {
            return LoadResult::setup_failure("device lost during VRAM fill");
        }

        'outer: loop {
            if stop.stopped() || Instant::now() >= deadline {
                break;
            }
            // Re-seed the random step each pass so repeated cycles are not
            // identical: different data patterns, different coupling.
            steps = battery(
                0xC0FF_EE01u32
                    .wrapping_add(passes as u32)
                    .wrapping_mul(2654435761),
            );

            for s in 0..steps.len() {
                let cur = steps[s];
                let next = steps[(s + 1) % steps.len()];

                for (ci, handle) in chunks.iter().enumerate() {
                    if stop.stopped() || Instant::now() >= deadline {
                        break 'outer;
                    }

                    // Fresh results buffer: [error_count, min_failing_index].
                    let results = client.create_from_slice(u32::as_bytes(&[0u32, u32::MAX]));

                    vram_check::launch::<WgpuRuntime>(
                        &client,
                        CubeCount::Static(groups, 1, 1),
                        cube_dim,
                        unsafe { ArrayArg::from_raw_parts(handle.clone(), chunk_elems) },
                        unsafe { ArrayArg::from_raw_parts(results.clone(), 2) },
                        cur.mode,
                        cur.value,
                        cur.seed,
                        next.mode,
                        next.value,
                        next.seed,
                        stride,
                    );

                    if cubecl::future::block_on(client.sync()).is_err() {
                        io_error = Some("device lost during VRAM check".to_string());
                        break 'outer;
                    }

                    let raw = match client.read_one(results.clone()) {
                        Ok(r) => r,
                        Err(_) => {
                            io_error = Some("results read-back failed".to_string());
                            break 'outer;
                        }
                    };
                    let vals: &[u32] = u32::from_bytes(&raw);
                    let chunk_errors = vals[0] as u64;
                    words_checked += chunk_elems as u64;

                    // Live status for the UI (phase/work are cheap relaxed atomics;
                    // the detail push is throttled). All no-ops when headless.
                    if let Some(l) = &lane {
                        l.bump_work();
                        l.set_phase(PHASE_WORK);
                        if last_note.elapsed() >= Duration::from_millis(90) {
                            last_note = Instant::now();
                            let gib = words_checked as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
                            l.set_detail(&format!(
                                "pattern: {}\nregion: chunk {}/{}\nverified: {gib:.1} GiB",
                                cur.label,
                                ci + 1,
                                chunks.len()
                            ));
                        }
                    }

                    if chunk_errors > 0 {
                        errors += chunk_errors;
                        if first_fail.is_none() {
                            let idx = vals[1] as usize;
                            // Only now is a full read-back worth it: recover the
                            // observed value so the report names the bad word.
                            let observed = client
                                .read_one(handle.clone())
                                .ok()
                                .map(|b| {
                                    let words: &[u32] = u32::from_bytes(&b);
                                    words.get(idx).copied().unwrap_or(0)
                                })
                                .unwrap_or(0);
                            first_fail = Some(VramFail {
                                chunk: ci,
                                index: idx,
                                expected: expected_host(idx, cur.mode, cur.value, cur.seed),
                                observed,
                                pattern: cur.label,
                            });
                        }
                    }
                }
            }
            passes += 1;
        }
        if let Some(l) = &lane {
            l.set_phase(PHASE_DONE);
        }

        // ---- Reclaim -------------------------------------------------------
        //
        // Dropping the handles is not enough. CubeCL pools device allocations
        // and keeps them reserved against the (process-wide, cached) client, so
        // without an explicit cleanup the memory stays committed for the life of
        // the process — which is why a later stage in the same run, or a second
        // GPU test, could hit OOM on a card that looked empty.
        //
        // And because reclaim is the kind of thing that silently stops working,
        // it is *measured*: reserved bytes before and after, reported in the
        // detail. An assertion in a comment is not evidence.
        let chunk_count = chunks.len();
        let reserved_before = client
            .memory_usage()
            .map(|u| u.bytes_reserved)
            .unwrap_or(0);
        drop(chunks);
        client.memory_cleanup();
        // Cleanup is submitted, not blocking: sync so the pool has actually run
        // it before we look at the numbers.
        let _ = client.sync();
        let reserved_after = client
            .memory_usage()
            .map(|u| u.bytes_reserved)
            .unwrap_or(0);

        let seconds = start.elapsed().as_secs_f64();
        let gib_checked = (words_checked as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        let rate = if seconds > 0.0 {
            gib_checked / seconds
        } else {
            0.0
        };

        // On a UMA adapter there is no dedicated video memory: every chunk came
        // out of system RAM through the shared aperture. Calling that "VRAM"
        // would claim a test of hardware the machine does not have — the same
        // failure that let an integrated run report "6784 MiB VRAM" while
        // exercising system memory. What it *does* test is real and worth
        // saying: the path the iGPU reaches memory through.
        let memory_kind = if crate::adapter::resolve(self.device)
            .map(|a| a.uma)
            .unwrap_or(false)
        {
            "shared (UMA, system RAM)"
        } else {
            "VRAM"
        };
        let mut detail = format!(
            "{} MiB {memory_kind} ({} x {} MiB chunks), {passes} full pass(es), {:.1} GiB verified, ~{:.1} GiB/s",
            tested_mb,
            chunk_count,
            self.chunk_mb,
            gib_checked,
            rate
        );
        if stopped_for_headroom {
            detail.push_str(&format!(
                "; stopped short of the requested {} MiB to keep {} MiB of system RAM free \
                 (UMA: filling it would page the machine, not fail the card)",
                self.vram_mb,
                host_reserve / (1024 * 1024)
            ));
        }
        // Reclaim, stated as a measurement. A pool that did not shrink is a real
        // finding — the next stage is the one that will hit OOM because of it.
        let freed = reserved_before.saturating_sub(reserved_after);
        detail.push_str(&format!(
            "; reclaimed {} MiB ({} MiB still reserved)",
            freed / (1024 * 1024),
            reserved_after / (1024 * 1024)
        ));
        if reserved_after > reserved_before / 2 && reserved_before > 64 * 1024 * 1024 {
            detail.push_str("; WARNING: the device memory pool did not release — a later stage may hit OOM");
        }
        if let Some(e) = &io_error {
            detail.push_str(&format!("; {e}"));
            errors += 1;
        }
        if let Some(f) = first_fail {
            detail.push_str(&format!(
                "; FIRST FAIL chunk {} word {} [{}]: expected 0x{:08x} got 0x{:08x}",
                f.chunk, f.index, f.pattern, f.expected, f.observed
            ));
        }
        // A run that verified nothing must not look like a pass.
        if passes == 0 && words_checked == 0 && errors == 0 {
            detail.push_str("; NO WORDS VERIFIED - test did not run");
            errors += 1;
        }

        LoadResult::new(true, passes, words_checked, errors, detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_patterns_match_expectations() {
        assert_eq!(expected_host(1234, MODE_OWN_ADDRESS, 0, 0), 1234);
        assert_eq!(expected_host(7, MODE_CONSTANT, 0xAAAA_AAAA, 0), 0xAAAA_AAAA);
        // Random must depend on both index and seed, and be deterministic.
        let a = expected_host(99, MODE_RANDOM, 0, 42);
        assert_eq!(a, expected_host(99, MODE_RANDOM, 0, 42));
        assert_ne!(a, expected_host(100, MODE_RANDOM, 0, 42));
        assert_ne!(a, expected_host(99, MODE_RANDOM, 0, 43));
    }

    #[test]
    fn battery_chain_covers_complements() {
        let b = battery(1);
        let labels: Vec<&str> = b.iter().map(|s| s.label).collect();
        assert!(labels.contains(&"own-address"));
        // 0x00/0xFF and 0xAA/0x55 are complement pairs — moving inversions.
        assert!(labels.contains(&"const-00") && labels.contains(&"const-ff"));
        assert!(labels.contains(&"const-aa") && labels.contains(&"const-55"));
        assert!(labels.contains(&"random"));
    }

    #[test]
    fn defaults_are_chunked_under_binding_limit() {
        let k = VramKernel::default();
        // wgpu's default max storage buffer binding is 128 MiB.
        assert!(k.chunk_mb <= 128);
        assert_eq!(k.kind(), Kind::Gpu);
        assert_eq!(k.name(), "vram");
    }
}
