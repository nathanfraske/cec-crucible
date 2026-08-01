// SPDX-License-Identifier: MIT
//! How much memory `--vram-mb max` should actually ask for.
//!
//! Needed because **you cannot find the ceiling by probing for it.** WDDM
//! happily over-commits: an allocation past the end of dedicated VRAM succeeds
//! and silently spills into shared system memory, so `create_buffer` never
//! refuses. The failure surfaces later, when the fill actually writes — and it
//! surfaces as a *device loss*, which takes the run down with it. Measured on an
//! RTX 3070 (8 GiB): 7168 MiB fills clean, 8192 MiB loses the device.
//!
//! On a **UMA adapter the answer is a different number from a different place.**
//! An integrated GPU has no dedicated VRAM at all; every allocation comes out of
//! system RAM. Over-allocating there does not lose the device, it pages the
//! whole machine — a worse outcome on a technician's bench than a failed test.
//! So the budget comes from *available* RAM, is far more conservative, and the
//! result is never described as VRAM.
//!
//! Identity comes from [`crate::adapter`], never a bare index. Handing a wgpu
//! class ordinal to DXGI as an adapter index is what previously sized an
//! integrated adapter's run to the discrete card's VRAM and reported
//! `PASS … 6784 MiB VRAM` on a part with none.

use crate::adapter::{self, AdapterRecord};
use crate::GpuDevice;

/// Fraction of **dedicated** VRAM used by `max` on a discrete card.
///
/// Deliberately not the whole card: the desktop, the compositor and our own
/// command buffers are already resident, so requesting 100% is what triggers the
/// over-commit that kills the device. 85% fills the card hard enough to be a
/// real test while leaving the driver room to breathe.
pub const VRAM_MAX_FRACTION: f64 = 0.85;

/// Fraction of **available system RAM** used by `max` on a UMA adapter.
///
/// Much lower than the discrete fraction, and against *available* rather than
/// total, because the failure mode is different: there is no device to lose, so
/// the driver keeps handing out memory until Windows starts paging and the
/// operator's whole session becomes unusable.
pub const UMA_MAX_FRACTION: f64 = 0.50;

/// What `max` resolved to, and on what basis.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxSpan {
    pub mb: usize,
    /// True when this came out of system RAM because the adapter has no VRAM.
    pub uma: bool,
    /// Human-readable derivation, printed with the run so the number is never
    /// unexplained.
    pub basis: String,
}

/// Dedicated VRAM in bytes for a resolved adapter. `None` when the adapter
/// reports none — which is exactly what a UMA or software adapter does, and is
/// information rather than an error.
pub fn dedicated_vram_bytes(a: &AdapterRecord) -> Option<u64> {
    match a.dedicated_vram {
        0 => None,
        b => Some(b),
    }
}

/// The span `--vram-mb max` should ask for on `device`.
///
/// Returns `Err` with an operator-facing reason rather than a silent fallback:
/// a wrong number here produces either a device loss or a test of the wrong
/// memory, and both have happened.
pub fn max_testable_span(device: GpuDevice) -> Result<MaxSpan, String> {
    let a = adapter::resolve(device).ok_or_else(|| {
        format!(
            "no adapter matches {} — run `cec-crucible gpu-info` to see what this machine presents",
            device.label()
        )
    })?;
    max_span_for(&a)
}

/// The span for an already-resolved adapter. Split out so it is testable
/// without hardware.
pub fn max_span_for(a: &AdapterRecord) -> Result<MaxSpan, String> {
    if a.uma {
        let avail = crucible_core::sysinfo::memory()
            .map(|m| m.avail_bytes)
            .filter(|b| *b > 0)
            .ok_or_else(|| {
                "this adapter has no dedicated VRAM, and available system memory could not be \
                 read to size a shared-memory run. Give an explicit size, e.g. --vram-mb 2048."
                    .to_string()
            })?;
        let avail_mb = avail / (1024 * 1024);
        let mb = (avail_mb as f64 * UMA_MAX_FRACTION) as usize;
        if mb < 128 {
            return Err(format!(
                "only {avail_mb} MiB of system memory is available; a shared-memory run needs at \
                 least 256 MiB to be worth anything. Close something, or give an explicit \
                 --vram-mb."
            ));
        }
        return Ok(MaxSpan {
            mb,
            uma: true,
            basis: format!(
                "{:.0}% of {avail_mb} MiB available system RAM (UMA adapter: it has no video \
                 memory of its own to fill)",
                UMA_MAX_FRACTION * 100.0
            ),
        });
    }

    let bytes = dedicated_vram_bytes(a).ok_or_else(|| {
        format!(
            "{} reports no dedicated VRAM and is not flagged as a unified-memory adapter, so \
             `max` has nothing to size against. Give an explicit size, e.g. --vram-mb 2048.",
            a.name
        )
    })?;
    let mb = (bytes / (1024 * 1024)) as f64 * VRAM_MAX_FRACTION;
    Ok(MaxSpan {
        mb: (mb as usize).max(64),
        uma: false,
        basis: format!(
            "{:.0}% of {} MiB dedicated VRAM on {}",
            VRAM_MAX_FRACTION * 100.0,
            bytes / (1024 * 1024),
            a.name
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discrete(vram_mb: u64) -> AdapterRecord {
        AdapterRecord {
            name: "Test Discrete".into(),
            dedicated_vram: vram_mb * 1024 * 1024,
            uma: false,
            ..Default::default()
        }
    }

    #[test]
    fn a_discrete_card_is_sized_to_a_fraction_of_its_own_vram() {
        let s = max_span_for(&discrete(8192)).expect("discrete span");
        assert!(!s.uma);
        assert_eq!(s.mb, (8192.0 * VRAM_MAX_FRACTION) as usize);
        // Measured on the bench: 7168 MiB fills clean on an 8 GiB 3070, 8192
        // loses the device. The chosen span must sit under that cliff.
        assert!(s.mb <= 7168, "span {} MiB is past the measured cliff", s.mb);
        assert!(s.basis.contains("dedicated VRAM"));
    }

    #[test]
    fn a_uma_adapter_is_never_sized_from_a_vram_figure() {
        // The bug this replaces: Integrated(0) was handed to DXGI as adapter
        // index 0 — the discrete card — and sized to 85% of ITS 8 GiB, then
        // reported PASS at "6784 MiB VRAM" on a part with no VRAM at all.
        let igpu = AdapterRecord {
            name: "Test iGPU".into(),
            uma: true,
            shared_system_memory: 16 * 1024 * 1024 * 1024,
            // Even when a UMA adapter reports a carve-out, it must not be
            // treated as dedicated VRAM to fill.
            dedicated_vram: 128 * 1024 * 1024,
            ..Default::default()
        };

        let s = max_span_for(&igpu).expect("uma span");
        assert!(s.uma, "a UMA adapter must be sized as UMA");
        assert!(
            s.basis.contains("system RAM"),
            "the basis must say where the memory came from: {}",
            s.basis
        );
        assert!(
            !s.basis.contains("dedicated VRAM"),
            "a UMA span must never be described as dedicated VRAM: {}",
            s.basis
        );
        // And it must bear no relation to the 128 MiB carve-out or to any
        // discrete card's size — it is a fraction of available RAM.
        assert_ne!(s.mb, (128.0 * VRAM_MAX_FRACTION) as usize);
    }

    #[test]
    fn an_adapter_with_no_memory_information_refuses_rather_than_guessing() {
        let blank = AdapterRecord {
            name: "Mystery".into(),
            ..Default::default()
        };
        let err = max_span_for(&blank).map(|_| ()).unwrap_err();
        assert!(
            err.contains("--vram-mb"),
            "the refusal must tell the operator what to do instead: {err}"
        );
    }

    #[test]
    fn the_live_machine_agrees_with_itself() {
        // Whatever this bench is, the resolved span must be consistent with the
        // adapter it resolved from — the property that was violated before.
        for d in [GpuDevice::Discrete(0), GpuDevice::Integrated(0)] {
            let Some(a) = adapter::resolve(d) else { continue };
            let Ok(s) = max_testable_span(d) else { continue };
            assert_eq!(s.uma, a.uma, "span basis disagrees with the adapter: {a:?}");
            if !a.uma {
                let total_mb = a.dedicated_vram / (1024 * 1024);
                assert!(
                    (s.mb as u64) < total_mb,
                    "the max span ({} MiB) must stay UNDER the card's {total_mb} MiB — asking \
                     for all of it is what loses the device",
                    s.mb
                );
            }
        }
    }
}
