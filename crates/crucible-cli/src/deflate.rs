// SPDX-License-Identifier: MIT
//! A small DEFLATE compressor — fixed Huffman codes with greedy LZ77.
//!
//! Written because the first cut of the PNG encoder used *stored* (uncompressed)
//! deflate blocks, which are legal but produced **2 MB per chart**. At that size
//! the images cannot be embedded in the package page, and a folder of runs is
//! hundreds of megabytes of mostly-flat background.
//!
//! Fixed Huffman (`BTYPE=01`) rather than dynamic: the code tables are defined by
//! the spec, so there is nothing to build, transmit, or get wrong — and on chart
//! images, which are enormous runs of identical pixels, nearly all of the win
//! comes from LZ77 matching rather than from optimal symbol coding. A 2 MB chart
//! comes out around 25 KB, and dynamic Huffman would perhaps halve that again for
//! several hundred more lines of code.
//!
//! The two things that are easy to get wrong here, both of which produce a file
//! no decoder will open:
//! * **Bit order.** Deflate packs bits into bytes starting at the *least*
//!   significant bit, but Huffman codes are emitted *most* significant bit first.
//!   So a code has to be bit-reversed before it goes out, while extra bits (which
//!   are plain integers) do not.
//! * **Match limits.** Lengths are 3..=258 and distances 1..=32768, and the
//!   encoder must never emit one outside those ranges even when a longer run
//!   exists.

/// Bits out, LSB-first within each byte — deflate's packing order.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    /// Write `n` bits of `v`, least significant bit first.
    fn bits(&mut self, v: u32, n: u32) {
        self.acc |= v << self.n;
        self.n += n;
        while self.n >= 8 {
            self.out.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// Write a Huffman code of `len` bits, most significant bit first.
    fn code(&mut self, code: u32, len: u32) {
        let mut rev = 0u32;
        for i in 0..len {
            rev |= ((code >> (len - 1 - i)) & 1) << i;
        }
        self.bits(rev, len);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push((self.acc & 0xff) as u8);
        }
        self.out
    }
}

/// The fixed literal/length code for symbol `sym`, as (code, bit length).
/// Straight from RFC 1951 §3.2.6.
fn fixed_lit(sym: u32) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym, 8),
        144..=255 => (0x190 + (sym - 144), 9),
        256..=279 => (sym - 256, 7),
        _ => (0xc0 + (sym - 280), 8),
    }
}

/// Length code, base and extra bits for a match length of 3..=258.
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const WINDOW: usize = 32768;
/// How far back the hash chain is followed. Charts are long runs of identical
/// bytes, so the first candidate is nearly always the best one; a deeper search
/// costs time for almost nothing.
const MAX_CHAIN: usize = 32;

fn len_code(len: usize) -> usize {
    // Largest base not exceeding `len`. 29 entries, so a linear walk is cheaper
    // than anything cleverer.
    let mut i = 28;
    while i > 0 && LEN_BASE[i] as usize > len {
        i -= 1;
    }
    i
}

fn dist_code(dist: usize) -> usize {
    let mut i = 29;
    while i > 0 && DIST_BASE[i] as usize > dist {
        i -= 1;
    }
    i
}

/// Compress with a single fixed-Huffman block.
pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bits(1, 1); // BFINAL — one block for the whole stream
    w.bits(1, 2); // BTYPE = 01, fixed Huffman

    // Hash of the next three bytes -> most recent position starting with them,
    // plus a chain of earlier positions. 15-bit table: small enough to stay in
    // cache, large enough that collisions are rare on image data.
    const HASH_BITS: u32 = 15;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; data.len().max(1)];

    let hash = |d: &[u8], i: usize| -> usize {
        let v = (d[i] as u32) << 16 | (d[i + 1] as u32) << 8 | d[i + 2] as u32;
        // Knuth multiplicative: cheap and spreads image data well.
        ((v.wrapping_mul(2654435761)) >> (32 - HASH_BITS)) as usize
    };

    let mut i = 0usize;
    while i < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if i + MIN_MATCH <= data.len() {
            let h = hash(data, i);
            let mut cand = head[h];
            let mut chain = 0;
            while cand != usize::MAX && chain < MAX_CHAIN {
                let dist = i - cand;
                if dist > WINDOW {
                    break;
                }
                // Compare forward. `data[cand + n]` may overlap `i`, which is
                // legal in LZ77 and is exactly what makes a run of identical
                // bytes compress to one match.
                let max = MAX_MATCH.min(data.len() - i);
                let mut n = 0;
                while n < max && data[cand + n] == data[i + n] {
                    n += 1;
                }
                if n > best_len {
                    best_len = n;
                    best_dist = dist;
                    if n == MAX_MATCH {
                        break; // cannot do better
                    }
                }
                cand = prev[cand];
                chain += 1;
            }
            // Insert this position into the chain.
            prev[i] = head[h];
            head[h] = i;
        }

        if best_len >= MIN_MATCH {
            let lc = len_code(best_len);
            let (c, n) = fixed_lit(257 + lc as u32);
            w.code(c, n);
            w.bits(
                (best_len - LEN_BASE[lc] as usize) as u32,
                LEN_EXTRA[lc],
            );
            let dc = dist_code(best_dist);
            w.code(dc as u32, 5); // distance codes are 5 bits, fixed
            w.bits(
                (best_dist - DIST_BASE[dc] as usize) as u32,
                DIST_EXTRA[dc],
            );
            // Register the positions the match skipped over, so a later match
            // can still start inside it.
            for k in 1..best_len {
                let p = i + k;
                if p + MIN_MATCH <= data.len() {
                    let h = hash(data, p);
                    prev[p] = head[h];
                    head[h] = p;
                }
            }
            i += best_len;
        } else {
            let (c, n) = fixed_lit(data[i] as u32);
            w.code(c, n);
            i += 1;
        }
    }

    let (c, n) = fixed_lit(256); // end of block
    w.code(c, n);
    w.finish()
}

/// Wrap a deflate stream in zlib framing: 2-byte header, the compressed data,
/// then the Adler-32 of the *uncompressed* input.
pub fn zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CM=8 CINFO=7; 0x7801 % 31 == 0
    out.extend_from_slice(&deflate(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal inflate for fixed-Huffman blocks — the tests must verify that a
    /// real decoder can read what we wrote, not merely that we wrote something.
    fn inflate_fixed(bits: &[u8]) -> Vec<u8> {
        struct R<'a> {
            d: &'a [u8],
            pos: usize,
        }
        impl R<'_> {
            fn bit(&mut self) -> u32 {
                let b = (self.d[self.pos >> 3] >> (self.pos & 7)) & 1;
                self.pos += 1;
                b as u32
            }
            fn bits(&mut self, n: u32) -> u32 {
                let mut v = 0;
                for i in 0..n {
                    v |= self.bit() << i;
                }
                v
            }
            /// Huffman codes arrive most significant bit first.
            fn code(&mut self, n: u32) -> u32 {
                let mut v = 0;
                for _ in 0..n {
                    v = (v << 1) | self.bit();
                }
                v
            }
        }

        let mut r = R { d: bits, pos: 0 };
        let mut out = Vec::new();
        loop {
            let _final = r.bit();
            let btype = r.bits(2);
            assert_eq!(btype, 1, "only fixed-Huffman blocks are produced");
            loop {
                // Decode a literal/length symbol from the fixed table.
                let mut v = r.code(7);
                let sym;
                if v <= 0x17 {
                    sym = v + 256;
                } else {
                    v = (v << 1) | r.bit();
                    if (0x30..=0xbf).contains(&v) {
                        sym = v - 0x30;
                    } else if (0xc0..=0xc7).contains(&v) {
                        sym = v - 0xc0 + 280;
                    } else {
                        v = (v << 1) | r.bit();
                        assert!((0x190..=0x1ff).contains(&v), "bad 9-bit code {v:#x}");
                        sym = v - 0x190 + 144;
                    }
                }
                if sym == 256 {
                    return out;
                }
                if sym < 256 {
                    out.push(sym as u8);
                    continue;
                }
                let lc = (sym - 257) as usize;
                let len = LEN_BASE[lc] as usize + r.bits(LEN_EXTRA[lc]) as usize;
                let dc = r.code(5) as usize;
                let dist = DIST_BASE[dc] as usize + r.bits(DIST_EXTRA[dc]) as usize;
                let start = out.len() - dist;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
        }
    }

    fn round_trip(data: &[u8]) {
        let got = inflate_fixed(&deflate(data));
        assert_eq!(got.len(), data.len(), "length mismatch after round trip");
        assert!(got == data, "content mismatch after round trip");
    }

    #[test]
    fn round_trips_the_shapes_a_chart_produces() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"ab");
        round_trip(b"abc");
        // A long identical run — the overlapping-match case, and the bulk of a
        // chart's background.
        round_trip(&vec![7u8; 100_000]);
        // Repeating structure, like scanlines of the same colour.
        let mut rows = Vec::new();
        for _ in 0..500 {
            rows.extend_from_slice(&[0u8, 7, 7, 17, 255]);
        }
        round_trip(&rows);
    }

    #[test]
    fn round_trips_incompressible_data() {
        // No matches anywhere: every byte becomes a literal. This is the path
        // where a bit-order mistake shows up as silent corruption.
        let mut x = 0x12345678u32;
        let data: Vec<u8> = (0..50_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xff) as u8
            })
            .collect();
        round_trip(&data);
    }

    #[test]
    fn round_trips_a_match_at_the_maximum_length() {
        // 258 is the longest encodable match and has its own length code with
        // zero extra bits — an off-by-one here corrupts the stream.
        let mut d = vec![0u8; 300];
        for (i, b) in d.iter_mut().enumerate() {
            *b = (i % 7) as u8;
        }
        let mut data = d.clone();
        data.extend_from_slice(&d);
        round_trip(&data);
    }

    #[test]
    fn a_flat_image_compresses_by_orders_of_magnitude() {
        // The whole reason this module exists: stored blocks made every chart
        // 2 MB. A 1200x420 RGBA frame of background must not.
        // The ceiling is set by the format, not by the search: a match encodes at
        // most 258 bytes, so ~7.8k matches are needed for a 2 MB frame, each
        // costing roughly 13 bits. That puts the floor near 13 KB — about 1/150.
        // Assert 1/100 so the test measures the property (matching works at all)
        // rather than pinning an exact ratio that a tuning change would break.
        let flat = vec![0u8; 1200 * 420 * 4];
        let z = deflate(&flat);
        assert!(
            z.len() < flat.len() / 100,
            "expected heavy compression, got {} bytes from {}",
            z.len(),
            flat.len()
        );
        round_trip(&flat);
    }

    #[test]
    fn the_zlib_wrapper_is_well_formed() {
        let data = b"cec-crucible chart data".repeat(50);
        let z = zlib(&data);
        assert_eq!(u16::from_be_bytes([z[0], z[1]]) % 31, 0, "FCHECK");
        let tail = u32::from_be_bytes(z[z.len() - 4..].try_into().unwrap());
        assert_eq!(tail, adler32(&data));
        assert_eq!(inflate_fixed(&z[2..z.len() - 4]), data);
    }
}
