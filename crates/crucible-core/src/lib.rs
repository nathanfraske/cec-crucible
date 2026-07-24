// SPDX-License-Identifier: MIT
//! # crucible-core
//!
//! Shared primitives for `cec-crucible`, CEC's in-house PC-build stress suite:
//!
//! * [`clock`] — QPC-precision timing (the clock the external 1kHz power rig
//!   aligns to).
//! * [`markers`] — the JSONL load-transition feed for that rig.
//! * [`device`] — SMBIOS-derived machine identity, so every run is keyed to the
//!   box under test.
//! * [`kernel`] — the [`kernel::LoadKernel`] trait, [`kernel::StopFlag`],
//!   load [`kernel::Shape`]s, and the burst-aware [`kernel::ShapeDriver`].
//! * [`report`] — the device-identified pass/fail report.
//! * [`json`] — a tiny, dependency-free JSON writer.
//! * [`sysinfo`] — CPU count / physical memory.
//!
//! **Zero external dependencies** — std only. Platform FFI (QPC, SMBIOS,
//! memory status) is hand-declared against `kernel32`; non-Windows targets get
//! working fallbacks so the workspace builds and tests everywhere.

pub mod clock;
pub mod device;
pub mod json;
pub mod kernel;
pub mod markers;
pub mod report;
pub mod rng;
pub mod sysinfo;

// Flat re-exports of the most-used types for ergonomic downstream `use`.
pub use clock::{Clock, Timestamp};
pub use device::DeviceId;
pub use json::Json;
pub use kernel::{Budget, Kind, LoadKernel, LoadResult, Shape, ShapeDriver, StopFlag, Tick};
pub use markers::{
    Event, LaneSnap, LiveLane, Marker, MarkerLog, PHASE_DONE, PHASE_IDLE, PHASE_WORK,
};
pub use report::{Report, StageReport, Verdict};
