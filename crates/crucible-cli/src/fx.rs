// SPDX-License-Identifier: MIT
//! Reactive border FX — sparks that chase the edge of the UI, and lightning
//! that cracks when something goes wrong.
//!
//! This is deliberately *driven by the run*, not decoration on a timer:
//!
//! * **Spark rate + speed track live activity.** An idle suite shows a slow,
//!   dim drift; a suite with every lane hammering throws a fast, bright stream.
//!   You can read the machine's state from the edge of the screen across the room.
//! * **Colour tracks heat**, reusing the core-grid ramp — neutral slate when
//!   cold, through amber, to CEC pink at full tilt.
//! * **A verify pulse flashes cyan.** Every time a kernel publishes a fresh
//!   self-consistency checksum, a ring of cyan runs the border — visible proof
//!   that verification is still happening, not just load.
//! * **Errors fire red lightning.** A miscompare cracks a jagged bolt across the
//!   border and keeps cracking, so a failure is impossible to miss even if the
//!   panel that reported it has scrolled out of view.
//!
//! No dependencies and no RNG state: positions come from `crucible_core::rng`'s
//! stateless `hash2(seed, index)`, so the animation is reproducible for a given
//! tick sequence and the snapshot harness can render it deterministically.

#![cfg(feature = "tui")]

use ratatui::layout::{Position, Rect};
use ratatui::style::Color;
use ratatui::Frame;

use crucible_core::rng::hash2;

use crate::theme;

/// One spark travelling along the border, trailing behind itself.
#[derive(Clone, Copy)]
struct Spark {
    /// Position along the perimeter, in cells.
    pos: f32,
    /// Cells per tick; negative runs anticlockwise.
    vel: f32,
    /// 1.0 at birth → 0.0 dead.
    life: f32,
    /// 0.0 = cool (slate) … 1.0 = hot (pink). Sampled from activity at birth.
    heat: f32,
    /// Cyan verify-pulse spark rather than a heat spark.
    verify: bool,
}

/// A lightning bolt: a short jagged run of cells that flashes and dies.
#[derive(Clone, Copy)]
struct Bolt {
    /// Perimeter cell the bolt is centred on.
    at: u32,
    /// Half-length in cells.
    reach: u32,
    life: f32,
}

/// Border animation state. Cheap: two small vectors, no allocation per frame
/// once warmed.
pub struct Fx {
    sparks: Vec<Spark>,
    bolts: Vec<Bolt>,
    tick: u64,
    /// Error count at the previous update, to detect *new* errors.
    last_errors: u64,
    /// Mixed hash of every lane's checksum, to detect a fresh verification.
    last_verify_mix: u64,
}

/// Hard caps so a long run can never grow the FX state without bound.
const MAX_SPARKS: usize = 64;
const MAX_BOLTS: usize = 8;

impl Fx {
    pub fn new() -> Fx {
        Fx {
            sparks: Vec::new(),
            bolts: Vec::new(),
            tick: 0,
            last_errors: 0,
            last_verify_mix: 0,
        }
    }

    /// Advance one frame.
    ///
    /// * `activity` — 0.0..=1.0, how hard the box is working right now.
    /// * `errors` — running total across every lane (a rise fires lightning).
    /// * `verify_mix` — mixed checksum across lanes (a change fires a cyan ring).
    pub fn update(&mut self, activity: f32, errors: u64, verify_mix: u64) {
        self.tick = self.tick.wrapping_add(1);
        let act = activity.clamp(0.0, 1.0);

        // Age everything; drop the dead.
        for s in &mut self.sparks {
            s.pos += s.vel;
            // Hot sparks burn out faster, so a busy border keeps churning
            // instead of saturating into a solid ring.
            s.life -= 0.012 + 0.02 * s.heat;
        }
        self.sparks.retain(|s| s.life > 0.0);
        for b in &mut self.bolts {
            b.life -= 0.06;
        }
        self.bolts.retain(|b| b.life > 0.0);

        // Spawn rate follows activity: a trickle at idle, a stream at full tilt.
        let want = if act < 0.02 { 1 } else { 1 + (act * 4.0) as u32 };
        for i in 0..want {
            if self.sparks.len() >= MAX_SPARKS {
                break;
            }
            let h = hash2(self.tick ^ 0x5EED_1E55, i as u64);
            // Spawn only some ticks at low activity, so idle really is sparse.
            if act < 0.5 && (h >> 61) % 3 != 0 {
                continue;
            }
            let dir = if h & 1 == 0 { 1.0 } else { -1.0 };
            self.sparks.push(Spark {
                pos: ((h >> 8) % 4096) as f32,
                // Faster when busy — the edge visibly speeds up under load.
                vel: dir * (0.6 + 2.4 * act + ((h >> 20) % 32) as f32 / 64.0),
                life: 1.0,
                heat: act,
                verify: false,
            });
        }

        // A fresh verification anywhere → a cyan ring, several fast sparks
        // launched together from one point.
        if verify_mix != self.last_verify_mix {
            self.last_verify_mix = verify_mix;
            let start = (verify_mix % 4096) as f32;
            for k in 0..4 {
                if self.sparks.len() >= MAX_SPARKS {
                    break;
                }
                self.sparks.push(Spark {
                    pos: start,
                    vel: if k % 2 == 0 { 3.2 } else { -3.2 },
                    life: 1.0,
                    heat: 1.0,
                    verify: true,
                });
            }
        }

        // New errors → lightning, one bolt per error (capped), long-lived enough
        // to be unmissable.
        if errors > self.last_errors {
            let new = (errors - self.last_errors).min(MAX_BOLTS as u64);
            for i in 0..new {
                if self.bolts.len() >= MAX_BOLTS {
                    break;
                }
                let h = hash2(errors ^ 0xBAD_C0DE, i);
                self.bolts.push(Bolt {
                    at: (h % 4096) as u32,
                    reach: 2 + (h >> 32) as u32 % 4,
                    life: 1.0,
                });
            }
            self.last_errors = errors;
        }
    }

    /// Paint the border FX over the outer edge of `area`. Runs last, so it sits
    /// on top of the panel borders already drawn there.
    pub fn render(&self, f: &mut Frame, area: Rect) {
        let per = match perimeter(area) {
            Some(p) if p > 8 => p,
            _ => return, // too small to animate
        };
        let buf = f.buffer_mut();

        // Sparks first, then bolts on top (an error must never be hidden).
        for s in &self.sparks {
            let head = s.pos.rem_euclid(per as f32) as u32;
            // A short trail behind the head, fading out.
            for back in 0..3u32 {
                let t = if s.vel >= 0.0 {
                    (head + per - back) % per
                } else {
                    (head + back) % per
                };
                let fade = s.life * (1.0 - back as f32 * 0.33);
                if fade <= 0.05 {
                    continue;
                }
                let (glyph, col) = if s.verify {
                    (trail_glyph(back), dim_toward(theme::VALUE, fade))
                } else {
                    (trail_glyph(back), dim_toward(heat_color(s.heat), fade))
                };
                if let Some((x, y)) = perimeter_cell(area, t) {
                    let cell = &mut buf[Position::new(x, y)];
                    cell.set_symbol(glyph);
                    cell.set_fg(col);
                }
            }
        }

        for b in &self.bolts {
            // A jagged run: alternate glyphs so it reads as a crack, not a line.
            for k in 0..=(b.reach * 2) {
                let t = (b.at + per - b.reach + k) % per;
                let glyph = match k % 3 {
                    0 => "╱",
                    1 => "═",
                    _ => "╲",
                };
                // Flash white-hot at birth, settle to the brand red.
                let col = if b.life > 0.75 { theme::TEXT } else { theme::BAD };
                if let Some((x, y)) = perimeter_cell(area, t) {
                    let cell = &mut buf[Position::new(x, y)];
                    cell.set_symbol(glyph);
                    cell.set_fg(col);
                }
            }
        }
    }
}

/// Trail glyphs: a bright head fading to a faint tail.
fn trail_glyph(back: u32) -> &'static str {
    match back {
        0 => "✦",
        1 => "•",
        _ => "·",
    }
}

/// The core-grid heat ramp, reused so the border and the CPU grid agree on what
/// "hot" looks like: neutral slate → amber → CEC pink.
fn heat_color(f: f32) -> Color {
    let f = f.clamp(0.0, 1.0);
    let (a, b, t) = if f < 0.5 {
        ([78.0, 84.0, 104.0], [242.0, 166.0, 24.0], f * 2.0)
    } else {
        ([242.0, 166.0, 24.0], [237.0, 35.0, 152.0], (f - 0.5) * 2.0)
    };
    let lerp = |i: usize| (a[i] + (b[i] - a[i]) * t) as u8;
    Color::Rgb(lerp(0), lerp(1), lerp(2))
}

/// Fade a colour toward the background as a spark dies.
fn dim_toward(c: Color, amt: f32) -> Color {
    let (r, g, b) = match c {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (200.0, 200.0, 200.0),
    };
    // (7,7,17) is theme::BG.
    let k = amt.clamp(0.0, 1.0);
    let mix = |v: f32, bg: f32| (bg + (v - bg) * k) as u8;
    Color::Rgb(mix(r, 7.0), mix(g, 7.0), mix(b, 17.0))
}

/// Number of cells around the edge of `area` (corners counted once).
fn perimeter(area: Rect) -> Option<u32> {
    let (w, h) = (area.width as u32, area.height as u32);
    if w < 2 || h < 2 {
        return None;
    }
    Some(2 * (w + h) - 4)
}

/// Map a position along the perimeter to a cell, walking clockwise from the
/// top-left: across the top, down the right, back along the bottom, up the left.
fn perimeter_cell(area: Rect, t: u32) -> Option<(u16, u16)> {
    let per = perimeter(area)?;
    let (w, h) = (area.width as u32, area.height as u32);
    let (x0, y0) = (area.x as u32, area.y as u32);
    let t = t % per;

    let (x, y) = if t < w {
        (x0 + t, y0) // top, left → right
    } else if t < w + (h - 1) {
        (x0 + w - 1, y0 + (t - w + 1)) // right, top → bottom
    } else if t < w + (h - 1) + (w - 1) {
        (x0 + w - 1 - (t - (w + h - 1) + 1), y0 + h - 1) // bottom, right → left
    } else {
        // Left edge, bottom → top. Starts one ABOVE the bottom-left corner and
        // stops one BELOW the top-left corner: both corners already belong to
        // the bottom and top runs, so this segment is h-2 cells.
        (x0, y0 + h - 2 - (t - (2 * w + h - 2)))
    };
    Some((x as u16, y as u16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perimeter_walk_covers_every_edge_cell_once() {
        let area = Rect::new(2, 3, 10, 6);
        let per = perimeter(area).expect("perimeter");
        assert_eq!(per, 2 * (10 + 6) - 4);

        let mut seen = std::collections::BTreeSet::new();
        for t in 0..per {
            let c = perimeter_cell(area, t).expect("cell");
            assert!(seen.insert(c), "cell {c:?} visited twice at t={t}");
        }
        assert_eq!(seen.len() as u32, per, "every perimeter cell visited once");

        // Every visited cell is on an edge of the rect, never inside it.
        for (x, y) in seen {
            let on_edge = x == area.x
                || y == area.y
                || x == area.x + area.width - 1
                || y == area.y + area.height - 1;
            assert!(on_edge, "({x},{y}) is not on the border of {area:?}");
        }
    }

    #[test]
    fn tiny_areas_are_skipped_not_panicked() {
        assert!(perimeter(Rect::new(0, 0, 1, 5)).is_none());
        assert!(perimeter(Rect::new(0, 0, 5, 1)).is_none());
        assert!(perimeter_cell(Rect::new(0, 0, 0, 0), 3).is_none());
    }

    #[test]
    fn idle_stays_sparse_and_busy_streams() {
        let mut idle = Fx::new();
        let mut busy = Fx::new();
        for _ in 0..60 {
            idle.update(0.0, 0, 0);
            busy.update(1.0, 0, 0);
        }
        assert!(
            busy.sparks.len() > idle.sparks.len(),
            "busy border must be denser: busy={} idle={}",
            busy.sparks.len(),
            idle.sparks.len()
        );
        assert!(busy.sparks.len() <= MAX_SPARKS, "spark count is capped");
    }

    #[test]
    fn errors_fire_lightning_and_verifies_pulse() {
        let mut fx = Fx::new();
        fx.update(0.5, 0, 0);
        assert!(fx.bolts.is_empty());

        fx.update(0.5, 3, 0);
        assert!(!fx.bolts.is_empty(), "a new error must crack lightning");

        // Measure the pulse at zero activity so ambient spawning can't mask it:
        // a changed checksum launches a burst, an unchanged one must not.
        let before = fx.sparks.len();
        fx.update(0.0, 3, 0xdead_beef); // fresh verification
        let pulsed = fx.sparks.len() - before;
        assert!(
            pulsed >= 4,
            "a fresh checksum must launch a verify pulse, got {pulsed} new sparks"
        );

        let before = fx.sparks.len();
        fx.update(0.0, 3, 0xdead_beef); // same checksum → not a new verification
        let quiet = fx.sparks.len().saturating_sub(before);
        assert!(
            quiet <= 1,
            "an unchanged hash must not pulse, got {quiet} new sparks"
        );
    }
}
