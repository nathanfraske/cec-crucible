// SPDX-License-Identifier: MIT
//! Minimal system facts needed to size loads and populate `info`.
//!
//! Physical-memory totals come from `GlobalMemoryStatusEx` on Windows (zero
//! deps); other targets return `None` and callers fall back to an explicit
//! size argument.

/// Physical memory snapshot, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    pub total_bytes: u64,
    pub avail_bytes: u64,
}

/// Number of logical processors the runtime will schedule threads across.
pub fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Physical memory totals, or `None` if unavailable on this platform.
pub fn memory() -> Option<MemInfo> {
    #[cfg(windows)]
    {
        win::memory()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
mod win {
    use super::MemInfo;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub(super) fn memory() -> Option<MemInfo> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status.length` is set to the struct size as the API requires;
        // the kernel writes only within the struct.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 {
            Some(MemInfo {
                total_bytes: status.total_phys,
                avail_bytes: status.avail_phys,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_least_one_cpu() {
        assert!(logical_cpus() >= 1);
    }

    #[test]
    fn memory_is_sane_when_present() {
        if let Some(m) = memory() {
            assert!(m.total_bytes > 0);
            assert!(m.avail_bytes <= m.total_bytes);
        }
    }
}
