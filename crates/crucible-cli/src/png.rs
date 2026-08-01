// SPDX-License-Identifier: MIT
//! A minimal raster canvas and PNG encoder — zero dependencies.
//!
//! The chart page ([`crate::graph`]) is SVG, which is right for looking at and
//! wrong for pasting into a report, an email, or a ticket. Those want a PNG.
//!
//! Rasterising SVG would mean a renderer; instead the charts are drawn straight
//! into an RGBA buffer here and encoded as PNG by hand, over
//! [`crate::deflate`].
//!
//! Two choices do the work on chart images:
//! * **Filter type 2 (`Up`)** on every scanline. A chart is mostly a flat
//!   background, so each row is largely identical to the row above it and the
//!   filtered bytes are overwhelmingly zero.
//! * **LZ77 + fixed Huffman** over that. Together they take a 1200x420 frame
//!   from 2 MB down to tens of KB, which is what makes the images small enough
//!   to embed directly in the package page.
//!
//! Correctness details that are easy to get wrong and produce a file nothing
//! will open: the zlib stream carries an **Adler-32** of the *uncompressed*
//! data, and every chunk carries a **CRC-32** over its type and payload.

/// An RGBA8 image being drawn into.
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    px: Vec<u8>, // RGBA, row-major
}

/// 8-bit RGB. Alpha is always opaque — a chart pasted onto an unknown
/// background needs its own, or the dark theme's text lands on white.
pub type Rgb = (u8, u8, u8);

impl Canvas {
    pub fn new(w: usize, h: usize, bg: Rgb) -> Canvas {
        let mut px = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            px.extend_from_slice(&[bg.0, bg.1, bg.2, 255]);
        }
        Canvas { w, h, px }
    }

    #[inline]
    pub fn set(&mut self, x: i64, y: i64, c: Rgb) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 4;
        self.px[i] = c.0;
        self.px[i + 1] = c.1;
        self.px[i + 2] = c.2;
    }

    /// Blend `c` over the existing pixel at `a` (0.0–1.0) — used for the
    /// anti-aliased edges of a line, without which a chart looks like a fax.
    #[inline]
    pub fn blend(&mut self, x: i64, y: i64, c: Rgb, a: f64) {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 || a <= 0.0 {
            return;
        }
        let a = a.min(1.0);
        let i = (y as usize * self.w + x as usize) * 4;
        let mix = |o: u8, n: u8| (o as f64 * (1.0 - a) + n as f64 * a).round() as u8;
        self.px[i] = mix(self.px[i], c.0);
        self.px[i + 1] = mix(self.px[i + 1], c.1);
        self.px[i + 2] = mix(self.px[i + 2], c.2);
    }

    pub fn rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: Rgb) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.set(xx, yy, c);
            }
        }
    }

    pub fn hline(&mut self, x0: i64, x1: i64, y: i64, c: Rgb) {
        for x in x0.min(x1)..=x0.max(x1) {
            self.set(x, y, c);
        }
    }

    pub fn vline(&mut self, x: i64, y0: i64, y1: i64, c: Rgb) {
        for y in y0.min(y1)..=y0.max(y1) {
            self.set(x, y, c);
        }
    }

    /// Anti-aliased line (Xiaolin Wu). Widened by drawing parallel offsets,
    /// which is enough for chart traces and avoids a polygon rasteriser.
    pub fn line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: Rgb, width: f64) {
        let steep = (y1 - y0).abs() > (x1 - x0).abs();
        let half = ((width - 1.0) / 2.0).max(0.0);
        let mut off = -half;
        while off <= half + 1e-9 {
            if steep {
                self.wu(y0, x0 + off, y1, x1 + off, c, true);
            } else {
                self.wu(x0, y0 + off, x1, y1 + off, c, false);
            }
            off += 1.0;
        }
    }

    fn wu(&mut self, mut x0: f64, mut y0: f64, mut x1: f64, mut y1: f64, c: Rgb, swap: bool) {
        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        let dx = x1 - x0;
        let dy = y1 - y0;
        let grad = if dx.abs() < 1e-9 { 1.0 } else { dy / dx };
        let mut y = y0;
        let mut x = x0.round();
        let end = x1.round();
        while x <= end {
            let fl = y.floor();
            let frac = y - fl;
            let (a, b) = if swap { (fl as i64, x as i64) } else { (x as i64, fl as i64) };
            let (a2, b2) = if swap {
                (fl as i64 + 1, x as i64)
            } else {
                (x as i64, fl as i64 + 1)
            };
            self.blend(a, b, c, 1.0 - frac);
            self.blend(a2, b2, c, frac);
            y += grad;
            x += 1.0;
        }
    }

    /// Draw text at `(x, y)` (top-left) in the built-in 5x7 font, scaled.
    pub fn text(&mut self, x: i64, y: i64, s: &str, c: Rgb, scale: i64) {
        let mut cx = x;
        for ch in s.chars() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        self.rect(
                            cx + col as i64 * scale,
                            y + row as i64 * scale,
                            scale,
                            scale,
                            c,
                        );
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    /// Width in pixels a string will occupy at `scale`.
    pub fn text_width(s: &str, scale: i64) -> i64 {
        s.chars().count() as i64 * 6 * scale
    }

    /// Encode as a PNG byte stream.
    pub fn to_png(&self) -> Vec<u8> {
        // Scanlines with filter type 2 (`Up`): each byte is stored as the
        // difference from the byte directly above it. On a chart — long vertical
        // stretches of unchanged background — that turns almost the whole image
        // into zeros before the compressor even sees it. The first row has no
        // row above, so it filters against an implicit zero row, which `Up`
        // already does by definition.
        let stride = self.w * 4;
        let mut raw = Vec::with_capacity(self.h * (1 + stride));
        for y in 0..self.h {
            raw.push(2u8); // filter: Up
            let cur = y * stride;
            for x in 0..stride {
                let above = if y == 0 { 0 } else { self.px[cur - stride + x] };
                raw.push(self.px[cur + x].wrapping_sub(above));
            }
        }

        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(self.w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(self.h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter/interlace
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &crate::deflate::zlib(&raw));
        chunk(&mut out, b"IEND", &[]);
        out
    }
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// A 5x7 bitmap font, one `u8` per row with the low 5 bits used.
///
/// Only what a chart needs: digits, uppercase letters, and a handful of symbols.
/// Lowercase folds to uppercase rather than doubling the table — axis labels are
/// short and read fine in caps.
fn glyph(ch: char) -> [u8; 7] {
    let c = ch.to_ascii_uppercase();
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '%' => [0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '°' => [0x0C, 0x12, 0x12, 0x0C, 0x00, 0x00, 0x00],
        '·' => [0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        _ => [0; 7], // space, and anything the font does not carry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_png_signature_and_chunk_layout_are_well_formed() {
        let c = Canvas::new(8, 4, (7, 7, 17));
        let png = c.to_png();
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);

        // Walk the chunks the way a decoder would: length, type, data, CRC.
        let mut i = 8;
        let mut kinds = Vec::new();
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            let kind = String::from_utf8_lossy(&png[i + 4..i + 8]).to_string();
            let crc_at = i + 8 + len;
            assert!(crc_at + 4 <= png.len(), "chunk {kind} runs past the end");
            let stored = u32::from_be_bytes(png[crc_at..crc_at + 4].try_into().unwrap());
            assert_eq!(
                stored,
                crc32(&png[i + 4..crc_at]),
                "CRC mismatch in chunk {kind} — the file would not open"
            );
            kinds.push(kind);
            i = crc_at + 4;
        }
        assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"]);
        assert_eq!(i, png.len(), "trailing bytes after IEND");
    }

    #[test]
    fn a_chart_sized_image_stays_small_enough_to_embed() {
        // The package page carries these inline, so size is a correctness
        // property here rather than a nicety: at 2 MB apiece (which is what
        // stored deflate produced) a folder of runs is unshippable.
        let mut c = Canvas::new(1200, 420, (7, 7, 17));
        c.rect(110, 64, 1050, 300, (22, 21, 38));
        for i in 0..40 {
            c.line(110.0, 100.0 + i as f64 * 6.0, 1160.0, 90.0 + i as f64 * 6.0, (237, 35, 152), 2.5);
        }
        c.text(40, 26, "GPU POWER (W)", (245, 245, 248), 3);
        let png = c.to_png();
        assert!(
            png.len() < 200 * 1024,
            "a chart PNG must stay well under 200 KB, got {} bytes",
            png.len()
        );
    }

    #[test]
    fn drawing_stays_inside_the_canvas() {
        // Every primitive is fed out-of-range coordinates; a panic here would
        // take down a run at the very end, after all the work was done.
        let mut c = Canvas::new(16, 16, (0, 0, 0));
        c.set(-5, -5, (255, 0, 0));
        c.set(999, 999, (255, 0, 0));
        c.blend(-1, 8, (255, 0, 0), 0.5);
        c.rect(-4, -4, 40, 40, (10, 10, 10));
        c.line(-50.0, -50.0, 500.0, 500.0, (255, 255, 255), 3.0);
        c.hline(-20, 900, 8, (1, 2, 3));
        c.vline(4, -20, 900, (1, 2, 3));
        c.text(-10, -10, "OFF CANVAS", (255, 255, 255), 2);
        assert_eq!(c.px.len(), 16 * 16 * 4);
    }

    #[test]
    fn every_pixel_is_opaque() {
        // A chart pasted into a document must not come out with a transparent
        // background that renders as white-on-white.
        let mut c = Canvas::new(8, 8, (7, 7, 17));
        c.line(0.0, 0.0, 7.0, 7.0, (237, 35, 152), 2.0);
        for i in (3..c.px.len()).step_by(4) {
            assert_eq!(c.px[i], 255, "pixel {} is not opaque", i / 4);
        }
    }
}
