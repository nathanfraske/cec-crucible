// SPDX-License-Identifier: MIT
//! Shared colour palette for the ratatui menu + dashboard, so the whole TUI reads
//! as one cohesive, modern surface. Behind the `tui` feature.

#![cfg(feature = "tui")]
// A palette legitimately defines colours that only some screens use yet.
#![allow(dead_code)]

use ratatui::style::Color;

// Core surface + text.
pub const ACCENT: Color = Color::Rgb(96, 205, 255); // cyan — brand / highlights
pub const TEXT: Color = Color::Rgb(220, 228, 236); // primary text
pub const LABEL: Color = Color::Rgb(150, 200, 255); // panel/section titles
pub const VALUE: Color = Color::Rgb(190, 240, 205); // the number that matters
pub const DIM: Color = Color::Rgb(120, 135, 150); // secondary text
pub const FAINT: Color = Color::Rgb(88, 100, 116); // tertiary / field labels
pub const BORDER: Color = Color::Rgb(64, 86, 108); // panel borders

// Status.
pub const GOOD: Color = Color::Rgb(120, 230, 150); // working / pass
pub const BAD: Color = Color::Rgb(245, 104, 104); // error / fail
pub const HASH: Color = Color::Rgb(196, 168, 255); // verification hash
pub const IDLE_BAR: Color = Color::Rgb(58, 70, 86); // idle bar / gauge

// Category accents (menu).
pub const CAT_DIAG: Color = Color::Rgb(120, 200, 255);
pub const CAT_CPU: Color = Color::Rgb(255, 190, 100);
pub const CAT_GPU: Color = Color::Rgb(225, 130, 245);
pub const CAT_PROFILE: Color = Color::Rgb(180, 160, 255);
pub const FIRE: Color = Color::Rgb(240, 70, 60); // the FIRE button
