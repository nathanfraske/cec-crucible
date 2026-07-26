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

// ---------------------------------------------------------------------------
// Process / thread scheduling priority
// ---------------------------------------------------------------------------

/// Raise this process above the normal scheduling class.
///
/// A stress run deliberately saturates every core, and at `NORMAL` the tool's
/// own coordination work — the live dashboard, the telemetry sampler, the shape
/// drivers that decide when a burst turns on — competes on equal terms with the
/// 32 worker threads it just launched. The result is a UI that stutters and,
/// worse, load edges that land late because the thread that was supposed to
/// flip them did not get scheduled. Both were observed in the field.
///
/// `ABOVE_NORMAL` is deliberate: `HIGH` (and certainly `REALTIME`) can starve
/// the desktop and the very drivers we are measuring, which would distort the
/// measurement and could hang the box. Returns whether the class was raised.
pub fn raise_process_priority(high: bool) -> bool {
    #[cfg(windows)]
    {
        const ABOVE_NORMAL_PRIORITY_CLASS: u32 = 0x0000_8000;
        const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;
        // SAFETY: both are documented, argument-free/pseudo-handle Win32 calls.
        unsafe {
            let class = if high {
                HIGH_PRIORITY_CLASS
            } else {
                ABOVE_NORMAL_PRIORITY_CLASS
            };
            SetPriorityClass(GetCurrentProcess(), class) != 0
        }
    }
    #[cfg(not(windows))]
    {
        let _ = high;
        false
    }
}

/// Raise the CALLING thread's priority — used for the live-UI and telemetry
/// threads so they stay responsive while every core is pinned.
pub fn raise_current_thread_priority() -> bool {
    #[cfg(windows)]
    {
        const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
        // SAFETY: pseudo-handle + documented constant.
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) != 0 }
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn SetPriorityClass(handle: isize, class: u32) -> i32;
    fn GetCurrentThread() -> isize;
    fn SetThreadPriority(handle: isize, priority: i32) -> i32;
}
