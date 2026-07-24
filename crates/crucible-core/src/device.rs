// SPDX-License-Identifier: MIT
//! Device identity.
//!
//! Every run is keyed to the machine so results are retrievable and diffable
//! across the fleet. The primary key is the SMBIOS System UUID (+ baseboard
//! serial), read on Windows via `GetSystemFirmwareTable('RSMB', ...)` and
//! parsed here with zero dependencies.
//!
//! Two entry points:
//! * [`DeviceId::from_override`] — the PowerShell harness passes the canonical
//!   id down (`--device-id <uuid>`); this is the single source of truth when
//!   present.
//! * [`DeviceId::detect`] — standalone fallback that reads SMBIOS locally, and
//!   failing that derives a stable id from the hostname.

use crate::json::Json;

/// Machine identity attached to every report and marker file.
#[derive(Debug, Clone)]
pub struct DeviceId {
    /// Short, filesystem-safe 8-hex id derived from `uuid|board_serial` (or the
    /// hostname on the fallback path). Used in output filenames.
    pub short_id: String,
    /// SMBIOS System UUID (canonical), or `"unknown"`.
    pub uuid: String,
    /// SMBIOS baseboard serial, or `"unknown"`.
    pub board_serial: String,
    /// `"<system manufacturer> <product>"`, or `"unknown"`.
    pub system: String,
    /// `"<board manufacturer> <product>"`, or `"unknown"`.
    pub board: String,
    /// Machine hostname (always populated).
    pub host: String,
    /// How this id was obtained: `"override"`, `"smbios"`, or `"host-fallback"`.
    pub source: &'static str,
}

impl DeviceId {
    /// Build an id from a harness-supplied string (the authoritative path).
    pub fn from_override(id: &str) -> DeviceId {
        let id = id.trim();
        DeviceId {
            short_id: short8(id),
            uuid: id.to_string(),
            board_serial: "unknown".to_string(),
            system: "unknown".to_string(),
            board: "unknown".to_string(),
            host: hostname(),
            source: "override",
        }
    }

    /// Detect identity locally: SMBIOS first, hostname fallback second.
    pub fn detect() -> DeviceId {
        let host = hostname();

        #[cfg(windows)]
        {
            if let Some(sm) = win::read_smbios() {
                if let Some(uuid) = sm.uuid.clone() {
                    let serial = sm.board_serial.clone().unwrap_or_else(|| "unknown".into());
                    return DeviceId {
                        short_id: short8(&format!("{uuid}|{serial}")),
                        uuid,
                        board_serial: serial,
                        system: join_opt(&sm.system_manufacturer, &sm.system_product),
                        board: join_opt(&sm.board_manufacturer, &sm.board_product),
                        host,
                        source: "smbios",
                    };
                }
            }
        }

        DeviceId {
            short_id: short8(&host),
            uuid: "unknown".to_string(),
            board_serial: "unknown".to_string(),
            system: "unknown".to_string(),
            board: "unknown".to_string(),
            host,
            source: "host-fallback",
        }
    }

    pub fn to_json(&self) -> Json {
        let mut o = Json::object();
        o.push("short_id", self.short_id.as_str())
            .push("uuid", self.uuid.as_str())
            .push("board_serial", self.board_serial.as_str())
            .push("system", self.system.as_str())
            .push("board", self.board.as_str())
            .push("host", self.host.as_str())
            .push("source", self.source);
        o
    }
}

/// FNV-1a (64-bit), used only for the stable short id — not security-sensitive.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Top 32 bits of the FNV hash as 8 lowercase hex digits.
fn short8(s: &str) -> String {
    format!("{:08x}", (fnv1a64(s.as_bytes()) >> 32) as u32)
}

fn hostname() -> String {
    let raw = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_default();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "unknown-host".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(windows)]
fn join_opt(a: &Option<String>, b: &Option<String>) -> String {
    match (a.as_deref(), b.as_deref()) {
        (Some(a), Some(b)) => format!("{a} {b}"),
        (Some(a), None) => a.to_string(),
        (None, Some(b)) => b.to_string(),
        (None, None) => "unknown".to_string(),
    }
}

/// Parsed subset of the SMBIOS tables we care about. Public within the crate so
/// the pure parser can be unit-tested with synthetic buffers.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Smbios {
    pub uuid: Option<String>,
    pub system_manufacturer: Option<String>,
    pub system_product: Option<String>,
    pub board_manufacturer: Option<String>,
    pub board_product: Option<String>,
    pub board_serial: Option<String>,
}

/// Parse a raw `RSMB` firmware-table blob (the `RawSMBIOSData` header followed
/// by the packed SMBIOS structure table). Pure and platform-independent.
pub(crate) fn parse_smbios(raw: &[u8]) -> Option<Smbios> {
    // RawSMBIOSData: Used20CallingMethod(u8), Major(u8), Minor(u8),
    // DmiRevision(u8), Length(u32 LE), then the structure table.
    if raw.len() < 8 {
        return None;
    }
    let table_len = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let end = (8 + table_len).min(raw.len());
    let table = &raw[8..end];

    let mut out = Smbios::default();
    let mut i = 0usize;

    while i + 4 <= table.len() {
        let stype = table[i];
        let slen = table[i + 1] as usize;
        if slen < 4 {
            break; // malformed header length
        }
        let formatted_end = i + slen;
        if formatted_end > table.len() {
            break;
        }
        let formatted = &table[i..formatted_end];

        // The string-set runs from formatted_end to the first double-NUL.
        let mut m = formatted_end;
        let mut region_end = table.len();
        while m + 1 < table.len() {
            if table[m] == 0 && table[m + 1] == 0 {
                region_end = m + 2;
                break;
            }
            m += 1;
        }
        let strings: Vec<&[u8]> = table[formatted_end..region_end]
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .collect();
        let get = |idx: u8| -> Option<String> {
            if idx == 0 {
                return None;
            }
            strings
                .get(idx as usize - 1)
                .map(|s| String::from_utf8_lossy(s).trim().to_string())
                .filter(|s| !s.is_empty())
        };

        match stype {
            // Type 1: System Information.
            1 => {
                if formatted.len() >= 6 {
                    out.system_manufacturer = get(formatted[4]);
                    out.system_product = get(formatted[5]);
                }
                // UUID is 16 bytes at offset 8 (present since SMBIOS 2.1).
                if formatted.len() >= 24 {
                    out.uuid = format_uuid(&formatted[8..24]);
                }
            }
            // Type 2: Baseboard (Module) Information.
            2 => {
                if formatted.len() >= 6 {
                    out.board_manufacturer = get(formatted[4]);
                    out.board_product = get(formatted[5]);
                }
                if formatted.len() >= 8 {
                    out.board_serial = get(formatted[7]);
                }
            }
            127 => break, // end-of-table marker
            _ => {}
        }

        i = region_end;
    }

    Some(out)
}

/// Format a 16-byte SMBIOS UUID field to the canonical string, matching what
/// Windows WMI (`Win32_ComputerSystemProduct.UUID`) reports: per SMBIOS >= 2.6
/// the first three groups are stored little-endian, so they are byte-swapped
/// for display. Returns `None` for the all-zero / all-0xFF "not set" sentinels.
fn format_uuid(b: &[u8]) -> Option<String> {
    if b.len() != 16 {
        return None;
    }
    if b.iter().all(|&x| x == 0x00) || b.iter().all(|&x| x == 0xFF) {
        return None;
    }
    Some(format!(
        "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        b[3], b[2], b[1], b[0], // time-low  (LE)
        b[5], b[4],             // time-mid  (LE)
        b[7], b[6],             // time-high (LE)
        b[8], b[9],             // clock-seq (BE)
        b[10], b[11], b[12], b[13], b[14], b[15], // node (BE)
    ))
}

#[cfg(windows)]
mod win {
    use super::{parse_smbios, Smbios};

    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemFirmwareTable(provider: u32, table_id: u32, buffer: *mut u8, size: u32) -> u32;
    }

    pub(super) fn read_smbios() -> Option<Smbios> {
        // Provider signature 'RSMB' as a big-endian DWORD == 0x52534D42.
        let sig = u32::from_be_bytes(*b"RSMB");

        // First call sizes the buffer.
        // SAFETY: null buffer with size 0 is the documented sizing call.
        let needed = unsafe { GetSystemFirmwareTable(sig, 0, std::ptr::null_mut(), 0) };
        if needed == 0 {
            return None;
        }

        let mut buf = vec![0u8; needed as usize];
        // SAFETY: buf has `needed` bytes; the kernel writes at most `needed`.
        let got = unsafe { GetSystemFirmwareTable(sig, 0, buf.as_mut_ptr(), needed) };
        if got == 0 || got as usize > buf.len() {
            return None;
        }
        buf.truncate(got as usize);
        parse_smbios(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal RSMB blob: RawSMBIOSData header + one Type 1 and one
    // Type 2 structure, each with a small string set, then the type-127 end.
    fn synthetic_rsmb() -> Vec<u8> {
        let mut table = Vec::new();

        // --- Type 1: System Information, formatted length 0x1B (27) ---
        // offsets: 0 type,1 len,2..4 handle,4 manuf,5 product,6 version,
        // 7 serial, 8..24 uuid, 24 wakeup, 25 sku, 26 family
        let uuid = [
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
            0xFF, 0x00,
        ];
        let mut t1 = vec![1u8, 0x1B, 0x01, 0x00, 1, 2, 0, 0];
        t1.extend_from_slice(&uuid); // -> len now 8+16 = 24
        t1.push(0); // wakeup
        t1.push(0); // sku string
        t1.push(0); // family string
        assert_eq!(t1.len(), 0x1B);
        table.extend_from_slice(&t1);
        // strings for type 1: #1 manufacturer, #2 product
        table.extend_from_slice(b"ACME\0Model X\0\0");

        // --- Type 2: Baseboard, formatted length 8 ---
        // 4 manuf,5 product,6 version,7 serial
        let t2 = vec![2u8, 0x08, 0x02, 0x00, 1, 2, 0, 3];
        table.extend_from_slice(&t2);
        // strings: #1 manuf, #2 product, #3 serial
        table.extend_from_slice(b"ACME Boards\0Z490-I\0SN-DEADBEEF\0\0");

        // --- Type 127: end of table ---
        table.extend_from_slice(&[127u8, 0x04, 0x7F, 0x00, 0x00, 0x00]);

        // RawSMBIOSData header
        let mut blob = vec![0u8, 3, 4, 0];
        blob.extend_from_slice(&(table.len() as u32).to_le_bytes());
        blob.extend_from_slice(&table);
        blob
    }

    #[test]
    fn parses_uuid_manufacturer_and_serial() {
        let sm = parse_smbios(&synthetic_rsmb()).expect("parse");
        // First three groups are byte-swapped per SMBIOS >= 2.6.
        assert_eq!(
            sm.uuid.as_deref(),
            Some("44332211-6655-8877-99AA-BBCCDDEEFF00")
        );
        assert_eq!(sm.system_manufacturer.as_deref(), Some("ACME"));
        assert_eq!(sm.system_product.as_deref(), Some("Model X"));
        assert_eq!(sm.board_manufacturer.as_deref(), Some("ACME Boards"));
        assert_eq!(sm.board_product.as_deref(), Some("Z490-I"));
        assert_eq!(sm.board_serial.as_deref(), Some("SN-DEADBEEF"));
    }

    #[test]
    fn all_ff_uuid_is_none() {
        assert_eq!(format_uuid(&[0xFF; 16]), None);
        assert_eq!(format_uuid(&[0x00; 16]), None);
    }

    #[test]
    fn short_id_is_stable_and_8_hex() {
        let a = short8("same-input");
        let b = short8("same-input");
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(short8("a"), short8("b"));
    }

    #[test]
    fn override_roundtrips_uuid() {
        let d = DeviceId::from_override("  FEEDFACE-0000  ");
        assert_eq!(d.uuid, "FEEDFACE-0000");
        assert_eq!(d.source, "override");
    }

    #[test]
    fn truncated_blob_does_not_panic() {
        // Header claims a long table but bytes are short — must not panic.
        let mut blob = vec![0u8, 3, 4, 0];
        blob.extend_from_slice(&(9999u32).to_le_bytes());
        blob.extend_from_slice(&[1u8, 0x1B, 0, 0, 1, 2]); // truncated type 1
        let _ = parse_smbios(&blob); // just don't panic
    }
}
