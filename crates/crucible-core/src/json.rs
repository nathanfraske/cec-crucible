// SPDX-License-Identifier: MIT
//! A tiny, dependency-free JSON writer.
//!
//! The suite only ever *emits* JSON (reports + JSONL markers), so this is a
//! writer, not a parser — deliberately small and correct rather than general.
//! Objects preserve insertion order (a `Vec` of pairs, not a map) so reports
//! read the same way every run and diff cleanly across the fleet.

use std::fmt::Write as _;

/// A JSON value. Construct with the helper constructors or the [`obj!`] /
/// [`arr!`] convenience via `From` impls below.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Signed integer, emitted without a decimal point.
    I64(i64),
    /// Unsigned integer, emitted without a decimal point.
    U64(u64),
    /// Floating point. Non-finite values (NaN/±Inf) are emitted as `null`,
    /// since JSON has no representation for them.
    F64(f64),
    Str(String),
    Array(Vec<Json>),
    /// Insertion-ordered object.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    /// Start an empty object to push key/value pairs onto.
    pub fn object() -> Json {
        Json::Object(Vec::new())
    }

    /// Push a `(key, value)` pair. Panics if `self` is not an object — this is
    /// a builder used on values we just constructed, so a misuse is a bug, not
    /// a runtime condition.
    pub fn push(&mut self, key: impl Into<String>, value: impl Into<Json>) -> &mut Json {
        match self {
            Json::Object(pairs) => pairs.push((key.into(), value.into())),
            _ => panic!("Json::push on a non-object"),
        }
        self
    }

    /// Serialize compactly (no insignificant whitespace) — the form used for
    /// one-object-per-line JSONL marker records.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, 0);
        out
    }

    /// Serialize with two-space indentation for human-readable reports.
    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(2), 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, depth: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::I64(n) => {
                let _ = write!(out, "{n}");
            }
            Json::U64(n) => {
                let _ = write!(out, "{n}");
            }
            Json::F64(f) => {
                if f.is_finite() {
                    // {} on f64 gives the shortest round-tripping form; ensure
                    // it always reads back as a float by appending .0 for
                    // integer-valued numbers.
                    let s = format!("{f}");
                    out.push_str(&s);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        out.push_str(".0");
                    }
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_escaped(out, s),
            Json::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline_indent(out, indent, depth + 1);
                    item.write(out, indent, depth + 1);
                }
                newline_indent(out, indent, depth);
                out.push(']');
            }
            Json::Object(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline_indent(out, indent, depth + 1);
                    write_escaped(out, k);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    v.write(out, indent, depth + 1);
                }
                newline_indent(out, indent, depth);
                out.push('}');
            }
        }
    }
}

fn newline_indent(out: &mut String, indent: Option<usize>, depth: usize) {
    if let Some(width) = indent {
        out.push('\n');
        for _ in 0..width * depth {
            out.push(' ');
        }
    }
}

/// Write `s` as a JSON string literal, escaping per RFC 8259.
fn write_escaped(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            // All other control characters must be \u-escaped.
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// Ergonomic conversions so `push("k", v)` accepts common Rust types directly.
impl From<bool> for Json {
    fn from(v: bool) -> Json {
        Json::Bool(v)
    }
}
impl From<i64> for Json {
    fn from(v: i64) -> Json {
        Json::I64(v)
    }
}
impl From<i32> for Json {
    fn from(v: i32) -> Json {
        Json::I64(v as i64)
    }
}
impl From<u64> for Json {
    fn from(v: u64) -> Json {
        Json::U64(v)
    }
}
impl From<u32> for Json {
    fn from(v: u32) -> Json {
        Json::U64(v as u64)
    }
}
impl From<usize> for Json {
    fn from(v: usize) -> Json {
        Json::U64(v as u64)
    }
}
impl From<f64> for Json {
    fn from(v: f64) -> Json {
        Json::F64(v)
    }
}
impl From<&str> for Json {
    fn from(v: &str) -> Json {
        Json::Str(v.to_string())
    }
}
impl From<String> for Json {
    fn from(v: String) -> Json {
        Json::Str(v)
    }
}
impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Json {
        match v {
            Some(x) => x.into(),
            None => Json::Null,
        }
    }
}
impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Json {
        Json::Array(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_strings() {
        let j = Json::str("a\"b\\c\nd\te\u{1}");
        // Quote/backslash/newline/tab use short escapes; other control chars
        // (here U+0001) use \uXXXX.
        assert_eq!(j.to_compact(), "\"a\\\"b\\\\c\\nd\\te\\u0001\"");
    }

    #[test]
    fn compact_object_is_ordered_and_tight() {
        let mut o = Json::object();
        o.push("z", 1i64).push("a", true).push("m", Json::Null);
        assert_eq!(o.to_compact(), r#"{"z":1,"a":true,"m":null}"#);
    }

    #[test]
    fn floats_round_trip_shape() {
        assert_eq!(Json::F64(1.0).to_compact(), "1.0");
        assert_eq!(Json::F64(1.5).to_compact(), "1.5");
        // Non-finite degrades to null rather than emitting invalid JSON.
        assert_eq!(Json::F64(f64::NAN).to_compact(), "null");
        assert_eq!(Json::F64(f64::INFINITY).to_compact(), "null");
    }

    #[test]
    fn pretty_indents_nested() {
        let mut inner = Json::object();
        inner.push("k", 2i64);
        let arr = Json::Array(vec![Json::I64(1), inner]);
        let pretty = arr.to_pretty();
        assert_eq!(pretty, "[\n  1,\n  {\n    \"k\": 2\n  }\n]");
    }

    #[test]
    fn empty_containers() {
        assert_eq!(Json::Array(vec![]).to_pretty(), "[]");
        assert_eq!(Json::object().to_pretty(), "{}");
    }
}
