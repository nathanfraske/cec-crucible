// SPDX-License-Identifier: MIT
//! SplitMix64 — a tiny, fast, well-distributed PRNG, std-only.
//!
//! Used wherever the suite needs a deterministic, seedable byte/number stream:
//! memory/storage fill patterns, the PCIe link payload, and the jitter load
//! shape. A fixed seed yields a fixed sequence — that is what makes a randomized
//! run *reproducible* (re-run the seed, re-command the same pattern).
//!
//! SplitMix64 is specifically good at turning trivially-related seeds
//! (`0, 1, 2, …` or `base ^ index`) into well-separated streams, which is how
//! the jitter shape decorrelates one kernel from the next.

/// Advance `state` and return the next 64-bit value (the canonical SplitMix64).
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A stateless draw keyed purely by `(seed, index)` — the same `(seed, index)`
/// always yields the same value, with no mutable stream to thread between
/// callers. This is what lets every worker thread and every kernel that shares a
/// seed agree on a value for a given index without any synchronization.
#[inline]
pub fn hash2(seed: u64, index: u64) -> u64 {
    let mut s = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    splitmix64(&mut s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_seed_reproducible() {
        let mut a = 0xDEAD_BEEFu64;
        let mut b = 0xDEAD_BEEFu64;
        let seq_a: Vec<u64> = (0..8).map(|_| splitmix64(&mut a)).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| splitmix64(&mut b)).collect();
        assert_eq!(seq_a, seq_b, "same seed must reproduce the sequence");
        let mut c = 0x1234u64;
        let seq_c: Vec<u64> = (0..8).map(|_| splitmix64(&mut c)).collect();
        assert_ne!(seq_a, seq_c, "different seed must diverge");
    }

    #[test]
    fn hash2_is_pure_and_index_sensitive() {
        // Pure: same (seed, index) -> same value, no matter how often called.
        assert_eq!(hash2(0xABCD, 7), hash2(0xABCD, 7));
        // Adjacent indices decorrelate.
        assert_ne!(hash2(0xABCD, 7), hash2(0xABCD, 8));
        // Adjacent seeds decorrelate.
        assert_ne!(hash2(1, 100), hash2(2, 100));
    }
}
