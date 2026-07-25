// SPDX-License-Identifier: MIT
//! # crucible-storage
//!
//! Sustained storage load with **data-integrity verification**, not just
//! throughput. A scratch file on the target drive is written with a seeded
//! pattern, flushed to the device (`sync_all`), then read back and verified
//! block-for-block. On any miscompare the failing byte offset, expected, and
//! observed values are captured; the run is a FAIL.
//!
//! **Non-destructive by construction:** the kernel only ever touches its own
//! named scratch file inside a caller-provided directory — never a raw device,
//! never an existing file it did not create. The file is removed on completion
//! unless `keep` is set.
//!
//! **True uncached device load:** on Windows the scratch file is opened with
//! `FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`, so reads and writes hit
//! the device directly instead of the OS page cache — the read-back verify then
//! exercises the real storage read path, not RAM. This requires sector-aligned
//! buffers and I/O sizes, handled here with a 4 KiB-aligned block buffer (a
//! multiple of both 512e and 4Kn logical sector sizes). If the aligned open
//! fails, or on non-Windows targets, it falls back to buffered I/O + `sync_all`.
//! See [`drives`] for multi-device cross-load.

pub mod drives;

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, StopFlag};
use crucible_core::markers::{Event, MarkerLog, PHASE_DONE, PHASE_WORK};

/// Default scratch-file size when the caller does not specify one.
pub const DEFAULT_FILE_BYTES: u64 = 1024 * 1024 * 1024; // 1 GiB
/// Default I/O block size.
pub const DEFAULT_BLOCK_BYTES: usize = 1024 * 1024; // 1 MiB

/// Sector alignment for unbuffered I/O. 4 KiB is a multiple of both 512-byte
/// (512e) and 4096-byte (4Kn) logical sector sizes, so it satisfies
/// `FILE_FLAG_NO_BUFFERING` on any common volume without querying the sector
/// size at runtime.
const SECTOR_ALIGN: usize = 4096;

/// Storage kernel configuration.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Directory on the target drive to place the scratch file in.
    pub dir: PathBuf,
    /// Total scratch-file size in bytes.
    pub file_bytes: u64,
    /// I/O block size in bytes.
    pub block_bytes: usize,
    /// Keep the scratch file after the run (default: remove it).
    pub keep: bool,
    /// Request unbuffered (`FILE_FLAG_NO_BUFFERING`) device-direct I/O. Falls
    /// back to buffered if the aligned open fails or on non-Windows targets.
    pub unbuffered: bool,
}

impl StorageConfig {
    pub fn new(dir: impl Into<PathBuf>) -> StorageConfig {
        StorageConfig {
            dir: dir.into(),
            file_bytes: DEFAULT_FILE_BYTES,
            block_bytes: DEFAULT_BLOCK_BYTES,
            keep: false,
            // Default to true device load where the platform supports it.
            unbuffered: cfg!(windows),
        }
    }
}

/// The storage load kernel.
#[derive(Debug, Clone)]
pub struct StorageKernel {
    pub config: StorageConfig,
}

impl StorageKernel {
    pub fn new(config: StorageConfig) -> StorageKernel {
        StorageKernel { config }
    }

    fn scratch_path(&self) -> PathBuf {
        // Unique-per-process name so concurrent runs never collide, and so we
        // can be certain the file is ours to delete.
        let pid = std::process::id();
        self.config.dir.join(format!("crucible-scratch-{pid}.tmp"))
    }
}

impl LoadKernel for StorageKernel {
    fn name(&self) -> &str {
        "storage"
    }

    fn kind(&self) -> Kind {
        Kind::Storage
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        self.run_measured(budget, stop, markers).0
    }
}

/// Structured throughput from a storage run, for the multi-drive cross-load
/// comparison (solo baseline vs concurrent).
#[derive(Debug, Clone, Copy, Default)]
pub struct StorageStats {
    pub write_mibps: f64,
    pub read_mibps: f64,
    pub passes: u64,
    pub unbuffered: bool,
}

impl StorageKernel {
    /// Like [`LoadKernel::run`] but also returns structured throughput stats.
    pub fn run_measured(
        &self,
        budget: &Budget,
        stop: &StopFlag,
        markers: &MarkerLog,
    ) -> (LoadResult, StorageStats) {
        let want_unbuffered = self.config.unbuffered;
        // Unbuffered I/O needs sector-aligned block sizes; round up to 4 KiB.
        let mut block_bytes = self.config.block_bytes.max(512);
        if want_unbuffered {
            block_bytes = round_up(block_bytes, SECTOR_ALIGN);
        }
        let num_blocks = (self.config.file_bytes / block_bytes as u64).max(1);
        let effective_bytes = num_blocks * block_bytes as u64;
        let path = self.scratch_path();

        // Confirm the directory exists / is writable before committing.
        if let Err(e) = std::fs::create_dir_all(&self.config.dir) {
            return (
                LoadResult::setup_failure(format!(
                    "cannot create/access dir {}: {e}",
                    self.config.dir.display()
                )),
                StorageStats::default(),
            );
        }

        let (mut file, unbuffered) = match open_scratch(&path, want_unbuffered) {
            Ok(pair) => pair,
            Err(e) => {
                return (
                    LoadResult::setup_failure(format!(
                        "cannot create scratch file {}: {e}",
                        path.display()
                    )),
                    StorageStats::default(),
                )
            }
        };

        let mib = effective_bytes as f64 / (1024.0 * 1024.0);
        let io_mode = if unbuffered { "unbuffered" } else { "buffered" };
        markers.stamp(
            Event::Mark,
            "storage",
            io_mode,
            &format!("file={:.0}MiB block={}KiB", mib, block_bytes / 1024),
        );

        let deadline = Instant::now() + budget.duration;
        // Sector-aligned buffer so FILE_FLAG_NO_BUFFERING accepts it (harmless
        // in buffered mode).
        let mut block = AlignedBlock::new(block_bytes);

        let mut pass = 0u64;
        let mut bytes_written = 0u64;
        let mut bytes_read = 0u64;
        let mut write_secs = 0.0f64;
        let mut read_secs = 0.0f64;
        let mut errors = 0u64;
        let mut checksum = 0u64;
        let mut first_fail: Option<FailInfo> = None;
        let mut io_error: Option<String> = None;

        // Live-UI lane (None unless a UI is attached); fed verbosely from the
        // per-block checkpoints below so the dashboard shows the I/O in flight.
        let lane = markers.register_lane("storage");
        let mut last_note = Instant::now() - Duration::from_secs(1);
        let dir_label = self.config.dir.display().to_string();

        'passes: loop {
            if stop.stopped() || Instant::now() >= deadline {
                break;
            }

            // ---- Write phase ----
            if let Err(e) = file.seek(SeekFrom::Start(0)) {
                io_error = Some(format!("seek(write) failed: {e}"));
                break;
            }
            let w_start = Instant::now();
            for b in 0..num_blocks {
                let seed = block_seed(b, pass);
                fill_block(block.as_mut(), seed);
                if let Err(e) = file.write_all(block.as_slice()) {
                    io_error = Some(format!("write failed at block {b}: {e}"));
                    break 'passes;
                }
                bytes_written += block_bytes as u64;
                if (b & 0x3F) == 0 {
                    if let Some(l) = &lane {
                        l.set_phase(PHASE_WORK);
                        l.bump_work();
                        let now = Instant::now();
                        if now.duration_since(last_note) >= Duration::from_millis(90) {
                            last_note = now;
                            let secs = w_start.elapsed().as_secs_f64().max(1e-6);
                            let mbps = ((b + 1) * block_bytes as u64) as f64 / secs / (1024.0 * 1024.0);
                            let gib = bytes_written as f64 / (1024.0 * 1024.0 * 1024.0);
                            l.set_detail(&format!(
                                "mode: {io_mode}\nphase: WRITE\nrate: {mbps:.0} MB/s\nblock: {} KiB\npass: {pass}\nwritten: {gib:.2} GiB\ndir: {dir_label}",
                                block_bytes / 1024
                            ));
                        }
                    }
                    if stop.stopped() || Instant::now() >= deadline {
                        // Interrupted mid-write: no verify for a partial pass.
                        write_secs += w_start.elapsed().as_secs_f64();
                        break 'passes;
                    }
                }
            }
            // Flush to the device so the read-back reflects persisted data.
            if let Err(e) = file.sync_all() {
                io_error = Some(format!("sync_all failed: {e}"));
                break;
            }
            write_secs += w_start.elapsed().as_secs_f64();

            // ---- Read + verify phase ----
            if let Err(e) = file.seek(SeekFrom::Start(0)) {
                io_error = Some(format!("seek(read) failed: {e}"));
                break;
            }
            let r_start = Instant::now();
            for b in 0..num_blocks {
                if let Err(e) = file.read_exact(block.as_mut()) {
                    io_error = Some(format!("read failed at block {b}: {e}"));
                    break 'passes;
                }
                bytes_read += block_bytes as u64;
                let seed = block_seed(b, pass);
                match verify_block(block.as_slice(), seed) {
                    VerifyOutcome::Ok(sum) => checksum ^= sum,
                    VerifyOutcome::Mismatch {
                        offset_in_block,
                        expected,
                        got,
                        partial_sum,
                    } => {
                        checksum ^= partial_sum;
                        errors += 1;
                        if first_fail.is_none() {
                            first_fail = Some(FailInfo {
                                file_offset: b * block_bytes as u64 + offset_in_block as u64,
                                block: b,
                                expected,
                                got,
                            });
                        }
                    }
                }
                if (b & 0x3F) == 0 {
                    if let Some(l) = &lane {
                        l.bump_work();
                        let now = Instant::now();
                        if now.duration_since(last_note) >= Duration::from_millis(90) {
                            last_note = now;
                            let secs = r_start.elapsed().as_secs_f64().max(1e-6);
                            let mbps = ((b + 1) * block_bytes as u64) as f64 / secs / (1024.0 * 1024.0);
                            let gib = bytes_read as f64 / (1024.0 * 1024.0 * 1024.0);
                            l.set_hash(checksum);
                            l.set_detail(&format!(
                                "mode: {io_mode}\nphase: VERIFY\nrate: {mbps:.0} MB/s\nerrors: {errors}\npass: {pass}\nverified: {gib:.2} GiB\ndir: {dir_label}"
                            ));
                        }
                    }
                    if stop.stopped() || Instant::now() >= deadline {
                        read_secs += r_start.elapsed().as_secs_f64();
                        break 'passes;
                    }
                }
            }
            read_secs += r_start.elapsed().as_secs_f64();
            pass += 1;
        }
        if let Some(l) = &lane {
            l.set_hash(checksum);
            l.set_phase(PHASE_DONE);
        }

        // Clean up the scratch file unless asked to keep it.
        drop(file);
        if !self.config.keep {
            let _ = std::fs::remove_file(&path);
        }

        // A hard I/O error that prevented the test from running its check is a
        // setup failure; miscompares are reported as errors on a run that ran.
        if pass == 0 && errors == 0 {
            if let Some(e) = io_error {
                return (LoadResult::setup_failure(e), StorageStats::default());
            }
        }

        let w_mbps = throughput_mib(bytes_written, write_secs);
        let r_mbps = throughput_mib(bytes_read, read_secs);
        let mut detail = format!(
            "{:.0} MiB file, {} KiB blocks, {}, {} pass(es), write ~{:.0} MiB/s, read ~{:.0} MiB/s{}",
            mib,
            block_bytes / 1024,
            io_mode,
            pass,
            w_mbps,
            r_mbps,
            if self.config.keep {
                format!(" [kept {}]", path.display())
            } else {
                String::new()
            },
        );
        if let Some(e) = io_error {
            detail.push_str(&format!("; IO ERROR: {e}"));
            // An I/O error after at least one good pass still marks the run bad.
            errors += 1;
        }
        if let Some(f) = first_fail {
            detail.push_str(&format!(
                "; FIRST FAIL @ file offset {} (block {}): expected 0x{:02x} got 0x{:02x}",
                f.file_offset, f.block, f.expected, f.got
            ));
        }

        let stats = StorageStats {
            write_mibps: w_mbps,
            read_mibps: r_mbps,
            passes: pass,
            unbuffered,
        };
        (LoadResult::new(true, pass, checksum, errors, detail), stats)
    }
}

struct FailInfo {
    file_offset: u64,
    block: u64,
    expected: u8,
    got: u8,
}

enum VerifyOutcome {
    Ok(u64),
    Mismatch {
        offset_in_block: usize,
        expected: u8,
        got: u8,
        partial_sum: u64,
    },
}

/// Per-block seed: distinct per block and per pass so no two blocks ever hold
/// the same pattern (catches address/LBA mixups).
#[inline]
fn block_seed(block: u64, pass: u64) -> u64 {
    0x5EED_1234_ABCD_0000
        ^ block.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ pass.wrapping_mul(0xD1B5_4A32_D192_ED03)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fill a block with the SplitMix64 stream for `seed` (little-endian words).
fn fill_block(buf: &mut [u8], seed: u64) {
    let mut state = seed;
    for chunk in buf.chunks_mut(8) {
        let bytes = splitmix64(&mut state).to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&bytes[..n]);
    }
}

/// Verify a block against the stream for `seed`, returning the first mismatch
/// or a checksum of the (correct) data.
fn verify_block(buf: &[u8], seed: u64) -> VerifyOutcome {
    let mut state = seed;
    let mut sum = 0u64;
    for (ci, chunk) in buf.chunks(8).enumerate() {
        let want = splitmix64(&mut state).to_le_bytes();
        for (k, &got) in chunk.iter().enumerate() {
            let expected = want[k];
            sum = sum.wrapping_add(got as u64);
            if got != expected {
                return VerifyOutcome::Mismatch {
                    offset_in_block: ci * 8 + k,
                    expected,
                    got,
                    partial_sum: sum,
                };
            }
        }
    }
    VerifyOutcome::Ok(sum)
}

fn throughput_mib(bytes: u64, secs: f64) -> f64 {
    if secs > 0.0 {
        (bytes as f64 / (1024.0 * 1024.0)) / secs
    } else {
        0.0
    }
}

fn round_up(x: usize, align: usize) -> usize {
    x.div_ceil(align) * align
}

/// A heap block whose usable slice starts on a [`SECTOR_ALIGN`] boundary, as
/// `FILE_FLAG_NO_BUFFERING` requires. Over-allocates by one alignment unit and
/// offsets into it; the backing allocation address is stable for the lifetime.
struct AlignedBlock {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBlock {
    fn new(len: usize) -> AlignedBlock {
        let raw = vec![0u8; len + SECTOR_ALIGN];
        let base = raw.as_ptr() as usize;
        let off = (SECTOR_ALIGN - (base % SECTOR_ALIGN)) % SECTOR_ALIGN;
        AlignedBlock { raw, off, len }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }

    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

/// Open (create+truncate) the scratch file, preferring unbuffered device-direct
/// I/O when requested. Returns the file and whether unbuffered mode is active.
fn open_scratch(path: &Path, want_unbuffered: bool) -> std::io::Result<(std::fs::File, bool)> {
    #[cfg(windows)]
    if want_unbuffered {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        if let Ok(f) = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH)
            .open(path)
        {
            return Ok((f, true));
        }
        // else fall through to a buffered open below.
    }

    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    Ok((f, false))
}

/// Convenience: does this look like a writable directory we can scratch in?
pub fn dir_is_usable(dir: &Path) -> bool {
    dir.is_dir() || std::fs::create_dir_all(dir).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn scratch_dir() -> PathBuf {
        std::env::temp_dir().join("cec-crucible-storage-test")
    }

    #[test]
    fn fill_then_verify_roundtrips() {
        let mut buf = vec![0u8; 4096];
        fill_block(&mut buf, 0xABCD_1234);
        match verify_block(&buf, 0xABCD_1234) {
            VerifyOutcome::Ok(sum) => assert!(sum > 0),
            VerifyOutcome::Mismatch { .. } => panic!("clean block should verify"),
        }
    }

    #[test]
    fn verify_catches_corruption() {
        let mut buf = vec![0u8; 4096];
        fill_block(&mut buf, 7);
        buf[1234] ^= 0xFF; // flip a byte
        match verify_block(&buf, 7) {
            VerifyOutcome::Mismatch {
                offset_in_block, ..
            } => {
                assert_eq!(offset_in_block, 1234);
            }
            VerifyOutcome::Ok(_) => panic!("corruption should be detected"),
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = vec![0u8; 512];
        let mut b = vec![0u8; 512];
        fill_block(&mut a, block_seed(0, 0));
        fill_block(&mut b, block_seed(1, 0));
        assert_ne!(a, b);
    }

    #[test]
    fn short_end_to_end_run_passes_and_cleans_up() {
        let dir = scratch_dir();
        let cfg = StorageConfig {
            dir: dir.clone(),
            file_bytes: 2 * 1024 * 1024, // 2 MiB
            block_bytes: 64 * 1024,      // 64 KiB
            keep: false,
            unbuffered: true, // exercise the aligned/unbuffered path on Windows
        };
        let kernel = StorageKernel::new(cfg);
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_millis(300));
        let r = kernel.run(&budget, &stop, &markers);

        assert!(r.ok, "run failed: {}", r.detail);
        assert_eq!(r.error_count, 0, "unexpected error: {}", r.detail);
        assert!(r.iterations >= 1, "expected >=1 pass: {}", r.detail);
        assert!(r.detail.contains("MiB/s"), "detail: {}", r.detail);

        // Scratch file must be gone.
        let pid = std::process::id();
        let leftover = dir.join(format!("crucible-scratch-{pid}.tmp"));
        assert!(!leftover.exists(), "scratch file was not cleaned up");
    }

    #[test]
    fn setup_failure_on_unwritable_dir() {
        // A path under a file (not a dir) can't be created as a directory.
        let bogus = std::env::temp_dir().join("crucible-not-a-dir.tmp");
        let _ = std::fs::write(&bogus, b"x");
        let cfg = StorageConfig::new(bogus.join("subdir"));
        let kernel = StorageKernel::new(cfg);
        let stop = StopFlag::new();
        let markers = MarkerLog::new(crucible_core::Clock::new());
        let budget = Budget::steady(Duration::from_millis(50));
        let r = kernel.run(&budget, &stop, &markers);
        assert!(!r.ok, "expected setup failure on unwritable dir");
        let _ = std::fs::remove_file(&bogus);
    }
}
