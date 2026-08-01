// SPDX-License-Identifier: MIT
//! What the PCIe link is physically capable of — the ceiling a measured
//! transfer rate has to sit under to be a PCIe measurement at all.
//!
//! A field capture reported **H2D ~370 GB/s**. PCIe Gen5 ×16 is about 63 GB/s
//! one-way, so that figure was roughly six times faster than the fastest link
//! that exists. The transfer never crossed the bus: with **Resizable BAR** the
//! whole of video memory becomes host-visible, and a Vulkan allocator asked for
//! a `HOST_VISIBLE` upload buffer may hand back memory that is also
//! `DEVICE_LOCAL` — i.e. VRAM. The "upload" is then a VRAM→VRAM copy running at
//! memory-bus speed, and the test reports it as PCIe bandwidth.
//!
//! Two defences, and this module is the second one. The first is to allocate
//! the staging buffer where it cannot be device-local (see
//! [`crucible_gpu::link`]). The second is this: **compute the ceiling from the
//! negotiated link and refuse any result above it.** Even if a future driver or
//! backend finds a new way to put the staging buffer in VRAM, the number cannot
//! be printed as PCIe bandwidth, because it is not one.
//!
//! Per-lane rates are the PCIe base-spec signalling rates after line coding:
//! 8b/10b through Gen2, 128b/130b from Gen3 on.

/// One-way payload throughput per lane, in GB/s (10⁹ bytes/s), by generation.
///
/// * Gen1 2.5 GT/s, 8b/10b → 0.250
/// * Gen2 5.0 GT/s, 8b/10b → 0.500
/// * Gen3 8.0 GT/s, 128b/130b → 0.985
/// * Gen4 16 GT/s → 1.969
/// * Gen5 32 GT/s → 3.938
/// * Gen6 64 GT/s (PAM4, FLIT) → 7.877
const PER_LANE_GBPS: [f64; 7] = [0.0, 0.250, 0.500, 0.985, 1.969, 3.938, 7.877];

/// Absolute ceiling used when the negotiated link cannot be read, in GB/s.
///
/// Gen5 ×16 is ~63 GB/s and is the fastest link on any shipping consumer or
/// workstation part. 80 leaves generous headroom for a Gen6 ×16 machine we have
/// never seen while still catching the 370 GB/s case by a factor of four.
pub const UNKNOWN_LINK_CEILING_GBPS: f64 = 80.0;

/// Tolerance over the theoretical ceiling before a result is rejected.
///
/// Measured throughput should always sit *below* the theoretical line rate —
/// there is protocol overhead on top of the line coding. A little slack absorbs
/// timer granularity on short transfers; anything beyond it is not a PCIe
/// measurement.
pub const CEILING_TOLERANCE: f64 = 1.05;

/// The negotiated link, when something could tell us about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcieLink {
    pub generation: u32,
    pub width: u32,
}

impl PcieLink {
    /// Theoretical one-way payload ceiling for this link, GB/s.
    pub fn ceiling_gbps(&self) -> f64 {
        let per_lane = PER_LANE_GBPS
            .get(self.generation as usize)
            .copied()
            .unwrap_or(0.0);
        if per_lane == 0.0 || self.width == 0 {
            return UNKNOWN_LINK_CEILING_GBPS;
        }
        per_lane * self.width as f64
    }

    pub fn describe(&self) -> String {
        format!(
            "Gen{} x{} (~{:.1} GB/s one-way)",
            self.generation,
            self.width,
            self.ceiling_gbps()
        )
    }
}

/// The ceiling to judge a measurement against, and how it was arrived at.
#[derive(Debug, Clone, PartialEq)]
pub struct Ceiling {
    pub gbps: f64,
    /// `None` when the link could not be read and the absolute fallback is in
    /// use — which must be said out loud, because the check is then much weaker.
    pub link: Option<PcieLink>,
}

impl Ceiling {
    /// Read the negotiated link for NVIDIA adapter `index` via NVML, falling
    /// back to the absolute ceiling on anything else.
    ///
    /// AMD and Intel expose the same information through their own libraries;
    /// until those are wired up, the fallback still catches a result that is
    /// physically impossible, which is the property that matters.
    pub fn detect(index: u32) -> Ceiling {
        if let Some((generation, width)) = crate::gputel::pcie_link(index) {
            if generation > 0 && width > 0 {
                let link = PcieLink { generation, width };
                return Ceiling {
                    gbps: link.ceiling_gbps(),
                    link: Some(link),
                };
            }
        }
        Ceiling {
            gbps: UNKNOWN_LINK_CEILING_GBPS,
            link: None,
        }
    }

    /// Is `measured` a physically possible rate across this link?
    pub fn plausible(&self, measured_gbps: f64) -> bool {
        measured_gbps <= self.gbps * CEILING_TOLERANCE
    }

    /// The sentence to put in the run detail when a measurement is impossible.
    ///
    /// It names the cause, because "too fast" without an explanation reads as a
    /// broken tool rather than as the specific, well-understood ReBAR case.
    pub fn rejection(&self, what: &str, measured_gbps: f64) -> String {
        let link = match &self.link {
            Some(l) => l.describe(),
            None => format!(
                "link speed unknown, using the absolute {UNKNOWN_LINK_CEILING_GBPS:.0} GB/s ceiling"
            ),
        };
        format!(
            "{what} measured {measured_gbps:.1} GB/s, which EXCEEDS what the link can carry \
             ({link}). The transfer did not cross PCIe — with Resizable BAR the staging buffer \
             can land in host-visible VRAM, making the copy VRAM->VRAM. NOT A PCIe MEASUREMENT"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_ceilings_match_the_published_link_rates() {
        // x16 one-way, the configuration these cards actually run in.
        let g = |gen| PcieLink { generation: gen, width: 16 }.ceiling_gbps();
        assert!((g(3) - 15.76).abs() < 0.1, "Gen3 x16 ~15.75 GB/s, got {}", g(3));
        assert!((g(4) - 31.5).abs() < 0.2, "Gen4 x16 ~31.5 GB/s, got {}", g(4));
        assert!((g(5) - 63.0).abs() < 0.3, "Gen5 x16 ~63 GB/s, got {}", g(5));
    }

    #[test]
    fn the_field_capture_figure_is_rejected_on_every_real_link() {
        // 370 GB/s is the number that actually came back from a field run.
        for gen in 1..=6u32 {
            let c = Ceiling {
                gbps: PcieLink { generation: gen, width: 16 }.ceiling_gbps(),
                link: Some(PcieLink { generation: gen, width: 16 }),
            };
            assert!(
                !c.plausible(370.0),
                "370 GB/s must be rejected on Gen{gen} x16"
            );
        }
        // And with no link information at all, the absolute ceiling still
        // catches it — the case on an AMD or Intel card today.
        let unknown = Ceiling { gbps: UNKNOWN_LINK_CEILING_GBPS, link: None };
        assert!(!unknown.plausible(370.0));
    }

    #[test]
    fn a_real_measurement_is_not_rejected() {
        // Measured on the bench over a Gen3 x16 link: H2D ~12.8 GB/s, and the
        // 3070's own bidirectional runs sit around 5-13 GB/s. None of that may
        // trip the gate, or the gate is useless.
        let c = Ceiling {
            gbps: PcieLink { generation: 3, width: 16 }.ceiling_gbps(),
            link: Some(PcieLink { generation: 3, width: 16 }),
        };
        for measured in [1.0, 5.4, 12.84, 15.5] {
            assert!(c.plausible(measured), "{measured} GB/s must pass on Gen3 x16");
        }
    }

    #[test]
    fn a_narrow_link_is_judged_as_narrow() {
        // A card in a x4 slot must be measured against x4. Judging it against
        // x16 would let a 4x-too-fast result through.
        let x4 = Ceiling {
            gbps: PcieLink { generation: 4, width: 4 }.ceiling_gbps(),
            link: Some(PcieLink { generation: 4, width: 4 }),
        };
        assert!(x4.plausible(7.0), "7 GB/s fits in Gen4 x4 (~7.9)");
        assert!(!x4.plausible(25.0), "25 GB/s does not fit in Gen4 x4");
    }

    #[test]
    fn the_rejection_names_the_cause() {
        let c = Ceiling {
            gbps: 15.75,
            link: Some(PcieLink { generation: 3, width: 16 }),
        };
        let msg = c.rejection("H2D", 370.0);
        assert!(msg.contains("NOT A PCIe MEASUREMENT"));
        assert!(msg.contains("Resizable BAR"), "the operator needs the cause: {msg}");
        assert!(msg.contains("Gen3 x16"), "and what it was judged against: {msg}");
    }

    #[test]
    fn an_unreadable_link_falls_back_rather_than_disabling_the_check() {
        let c = Ceiling::detect(9999); // no such device
        assert!(c.gbps > 0.0, "the ceiling must never be zero, which would reject everything");
        assert!(
            !c.plausible(370.0),
            "the fallback must still catch a physically impossible rate"
        );
    }
}
