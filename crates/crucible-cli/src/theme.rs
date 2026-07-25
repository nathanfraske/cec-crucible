// SPDX-License-Identifier: MIT
//! Shared colour palette for the ratatui menu + dashboard.
//!
//! This is the **Critical Error Computing** brand palette (criticalerrorcomputing.com):
//! the `oklch()` design tokens converted to sRGB, so the whole TUI reads as one
//! cohesive, on-brand surface — signature pink `#ed2398`, a deep near-black
//! background, and the site's category / status hues. Behind the `tui` feature.

#![cfg(feature = "tui")]
// A palette legitimately defines colours that only some screens use yet.
#![allow(dead_code)]

use ratatui::style::Color;

// Surface (site: --bg / --surface / --border) — a deep blue-black.
pub const BG: Color = Color::Rgb(7, 7, 17); // --bg, the full-screen backdrop
pub const SURFACE: Color = Color::Rgb(22, 21, 38); // --surface, panel fills
pub const BORDER: Color = Color::Rgb(48, 48, 73); // --border

// Text (site: --text / --dim / --faint).
pub const TEXT: Color = Color::Rgb(245, 245, 248);
pub const DIM: Color = Color::Rgb(169, 169, 183);
pub const FAINT: Color = Color::Rgb(144, 144, 164);

// Brand + emphasis.
pub const ACCENT: Color = Color::Rgb(237, 35, 152); // --accent #ed2398 (the CEC pink)
pub const ACCENT_DEEP: Color = Color::Rgb(187, 1, 118); // --accent-2
pub const LABEL: Color = Color::Rgb(159, 156, 255); // lavender (--vpop-label), panel titles
pub const VALUE: Color = Color::Rgb(94, 219, 129); // the number that matters (--ok green)
pub const HASH: Color = Color::Rgb(140, 103, 242); // --violet, verification hashes

// Status.
pub const GOOD: Color = Color::Rgb(94, 219, 129); // --ok / --g-gpu
pub const WARN: Color = Color::Rgb(242, 166, 24); // --warn amber
pub const BAD: Color = Color::Rgb(238, 11, 42); // --g-red
pub const IDLE_BAR: Color = Color::Rgb(48, 48, 73); // idle bar / gauge (= border)

// Category accents (site: --g-cpu / --g-mem / --g-gpu / --violet).
pub const CAT_DIAG: Color = Color::Rgb(65, 217, 248); // cyan (--vpop-cyan)
pub const CAT_CPU: Color = Color::Rgb(248, 139, 28); // orange (--g-cpu)
pub const CAT_GPU: Color = Color::Rgb(94, 219, 129); // green (--g-gpu)
pub const CAT_PROFILE: Color = Color::Rgb(140, 103, 242); // violet

// The FIRE button.
pub const FIRE: Color = Color::Rgb(238, 11, 42); // --g-red
pub const FIRE_BG: Color = Color::Rgb(58, 10, 22); // dark red fill when focused
pub const SEL_BG: Color = Color::Rgb(30, 30, 51); // selected-row bar (--surface-2)
