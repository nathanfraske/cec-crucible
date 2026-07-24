// SPDX-License-Identifier: MIT
//! Physical-drive discovery for multi-device cross-load.
//!
//! Testing several SSDs at once exposes slowdowns that a single-drive test
//! never sees: shared PCIe/DMI lanes, a saturated chipset uplink, or a
//! controller that can't sustain all its drives at once. To do that we need to
//! find the *distinct physical disks* (not partitions) and their bus type.
//!
//! On Windows: enumerate fixed logical volumes (`GetLogicalDriveStringsW` +
//! `GetDriveTypeW`), map each to its physical disk number
//! (`IOCTL_STORAGE_GET_DEVICE_NUMBER`) so two partitions on one SSD collapse to
//! one drive, and read the bus type (`IOCTL_STORAGE_QUERY_PROPERTY`) to label
//! NVMe vs SATA. All handles are opened query-only (zero access) — nothing is
//! ever written to the raw volume. Non-Windows targets return an empty list.

/// Storage bus type, as reported by `STORAGE_DEVICE_DESCRIPTOR.BusType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusType {
    Nvme,
    Sata,
    Sas,
    Ata,
    Usb,
    Raid,
    Scsi,
    Unknown,
    Other(u32),
}

impl BusType {
    fn from_raw(v: u32) -> BusType {
        match v {
            0x01 => BusType::Scsi,
            0x03 => BusType::Ata,
            0x07 => BusType::Usb,
            0x08 => BusType::Raid,
            0x0A => BusType::Sas,
            0x0B => BusType::Sata,
            0x11 => BusType::Nvme,
            0x00 => BusType::Unknown,
            other => BusType::Other(other),
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            BusType::Nvme => "NVMe".into(),
            BusType::Sata => "SATA".into(),
            BusType::Sas => "SAS".into(),
            BusType::Ata => "ATA".into(),
            BusType::Usb => "USB".into(),
            BusType::Raid => "RAID".into(),
            BusType::Scsi => "SCSI".into(),
            BusType::Unknown => "unknown".into(),
            BusType::Other(v) => format!("bus-0x{v:02x}"),
        }
    }
}

/// A distinct physical disk and the volume roots that live on it.
#[derive(Debug, Clone)]
pub struct PhysicalDrive {
    /// Windows physical device number (`\\.\PhysicalDriveN`).
    pub number: u32,
    pub bus: BusType,
    /// The volume root used as the scratch target (e.g. `D:\`).
    pub primary_root: String,
    /// All fixed volume roots backed by this physical disk.
    pub roots: Vec<String>,
}

/// Discover distinct fixed physical drives, deduplicated by device number and
/// sorted by it. Empty on non-Windows or if enumeration fails.
pub fn discover() -> Vec<PhysicalDrive> {
    #[cfg(windows)]
    {
        win::discover()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
mod win {
    use super::{BusType, PhysicalDrive};
    use core::ffi::c_void;
    use std::collections::BTreeMap;

    type Handle = *mut c_void;
    const INVALID_HANDLE_VALUE: isize = -1;
    const DRIVE_FIXED: u32 = 3;
    const FILE_SHARE_READ: u32 = 0x1;
    const FILE_SHARE_WRITE: u32 = 0x2;
    const OPEN_EXISTING: u32 = 3;
    const IOCTL_STORAGE_GET_DEVICE_NUMBER: u32 = 0x002D_1080;
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalDriveStringsW(n: u32, buf: *mut u16) -> u32;
        fn GetDriveTypeW(root: *const u16) -> u32;
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            sec: *mut c_void,
            disposition: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn DeviceIoControl(
            device: Handle,
            code: u32,
            in_buf: *mut c_void,
            in_size: u32,
            out_buf: *mut c_void,
            out_size: u32,
            returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        fn CloseHandle(h: Handle) -> i32;
    }

    #[repr(C)]
    struct StorageDeviceNumber {
        device_type: u32,
        device_number: u32,
        partition_number: i32,
    }

    #[repr(C)]
    struct StoragePropertyQuery {
        property_id: u32,
        query_type: u32,
        additional: [u8; 1],
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn logical_roots() -> Vec<String> {
        let mut buf = vec![0u16; 256];
        // SAFETY: buf has capacity; the API writes at most `buf.len()` u16s.
        let len = unsafe { GetLogicalDriveStringsW(buf.len() as u32, buf.as_mut_ptr()) };
        if len == 0 {
            return Vec::new();
        }
        if len as usize > buf.len() {
            buf = vec![0u16; len as usize + 1];
            // SAFETY: as above, with the larger buffer.
            let _ = unsafe { GetLogicalDriveStringsW(buf.len() as u32, buf.as_mut_ptr()) };
        }
        // The result is a double-NUL-terminated set of NUL-separated roots.
        let mut roots = Vec::new();
        let mut start = 0usize;
        for (i, &c) in buf.iter().enumerate() {
            if c == 0 {
                if i > start {
                    roots.push(String::from_utf16_lossy(&buf[start..i]));
                }
                if i + 1 >= buf.len() || buf[i + 1] == 0 {
                    break;
                }
                start = i + 1;
            }
        }
        roots
    }

    fn is_fixed(root: &str) -> bool {
        let w = wide(root);
        // SAFETY: `w` is a valid NUL-terminated wide string.
        unsafe { GetDriveTypeW(w.as_ptr()) == DRIVE_FIXED }
    }

    fn open_volume(root: &str) -> Option<Handle> {
        // "C:\" -> device path "\\.\C:", opened query-only (zero access).
        let letter = root.chars().next()?;
        let path = format!("\\\\.\\{letter}:");
        let w = wide(&path);
        // SAFETY: valid wide path; null security attrs / template are allowed.
        let h = unsafe {
            CreateFileW(
                w.as_ptr(),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null_mut(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if h as isize == INVALID_HANDLE_VALUE {
            None
        } else {
            Some(h)
        }
    }

    /// Returns `(device_number, bus_type_raw)` for the volume, if queryable.
    fn query(root: &str) -> Option<(u32, u32)> {
        let h = open_volume(root)?;

        let mut num = StorageDeviceNumber {
            device_type: 0,
            device_number: 0,
            partition_number: 0,
        };
        let mut returned = 0u32;
        // SAFETY: out buffer is a valid StorageDeviceNumber of the given size.
        let ok = unsafe {
            DeviceIoControl(
                h,
                IOCTL_STORAGE_GET_DEVICE_NUMBER,
                std::ptr::null_mut(),
                0,
                &mut num as *mut _ as *mut c_void,
                std::mem::size_of::<StorageDeviceNumber>() as u32,
                &mut returned,
                std::ptr::null_mut(),
            )
        };
        let devnum = if ok != 0 {
            Some(num.device_number)
        } else {
            None
        };

        // Bus type via the device-property query; BusType is a u32 at offset 28
        // of STORAGE_DEVICE_DESCRIPTOR.
        let mut q = StoragePropertyQuery {
            property_id: 0, // StorageDeviceProperty
            query_type: 0,  // PropertyStandardQuery
            additional: [0],
        };
        let mut out = [0u8; 1024];
        let mut ret2 = 0u32;
        // SAFETY: in/out buffers are valid for the given sizes.
        let ok2 = unsafe {
            DeviceIoControl(
                h,
                IOCTL_STORAGE_QUERY_PROPERTY,
                &mut q as *mut _ as *mut c_void,
                std::mem::size_of::<StoragePropertyQuery>() as u32,
                out.as_mut_ptr() as *mut c_void,
                out.len() as u32,
                &mut ret2,
                std::ptr::null_mut(),
            )
        };
        let bus = if ok2 != 0 && ret2 >= 32 {
            u32::from_le_bytes([out[28], out[29], out[30], out[31]])
        } else {
            0
        };

        // SAFETY: `h` came from CreateFileW and is closed exactly once.
        unsafe {
            CloseHandle(h);
        }
        devnum.map(|d| (d, bus))
    }

    pub(super) fn discover() -> Vec<PhysicalDrive> {
        // device_number -> (bus_raw, roots)
        let mut map: BTreeMap<u32, (u32, Vec<String>)> = BTreeMap::new();
        // Roots we could enumerate but not map to a device number are kept as
        // distinct entries above the real-device-number range.
        let mut unmapped = 1_000_000u32;

        for root in logical_roots() {
            if !is_fixed(&root) {
                continue;
            }
            match query(&root) {
                Some((num, bus)) => {
                    let e = map.entry(num).or_insert((bus, Vec::new()));
                    if e.0 == 0 {
                        e.0 = bus;
                    }
                    e.1.push(root);
                }
                None => {
                    map.insert(unmapped, (0, vec![root]));
                    unmapped += 1;
                }
            }
        }

        map.into_iter()
            .map(|(number, (bus, roots))| PhysicalDrive {
                number,
                bus: BusType::from_raw(bus),
                primary_root: roots.first().cloned().unwrap_or_default(),
                roots,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_does_not_panic_and_is_deduped() {
        let drives = discover();
        // On Windows this box has >=1 fixed drive; elsewhere it's empty. Either
        // way it must not panic and device numbers must be unique.
        let mut nums: Vec<u32> = drives.iter().map(|d| d.number).collect();
        let before = nums.len();
        nums.sort_unstable();
        nums.dedup();
        assert_eq!(before, nums.len(), "device numbers must be unique");
    }

    #[test]
    fn bus_type_labels() {
        assert_eq!(BusType::from_raw(0x11).as_str(), "NVMe");
        assert_eq!(BusType::from_raw(0x0B).as_str(), "SATA");
        assert_eq!(BusType::from_raw(0x99).as_str(), "bus-0x99");
    }
}
