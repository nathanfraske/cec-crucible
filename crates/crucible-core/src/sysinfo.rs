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

/// System memory a test must leave alone, in bytes, given the machine's total.
///
/// A flat "90% of available" is what pushes a box into paging. Available memory
/// is not spare memory: Windows is holding a file cache it will give back under
/// pressure, the GPU driver has pinned allocations, and the operator's session
/// needs a working set. Commit past that and the machine starts swapping — at
/// which point a memory test has become a disk test, it reports throughput that
/// says nothing about the DIMMs, and the technician's desktop stops responding.
///
/// So the reserve scales with the machine and has a floor: **2 GiB or an eighth
/// of total RAM, whichever is larger.** On an 8 GiB box that is 2 GiB; on 32 GiB
/// it is 4 GiB; on 128 GiB it is 16 GiB. Sizing off *total* rather than
/// *available* is deliberate — the reserve should not shrink just because
/// something else is already using memory.
pub fn working_set_reserve_bytes(total_bytes: u64) -> u64 {
    const FLOOR: u64 = 2 * 1024 * 1024 * 1024;
    (total_bytes / 8).max(FLOOR)
}

/// The largest buffer a test may take right now without pushing the machine
/// into paging: available memory minus [`working_set_reserve_bytes`].
///
/// Returns `None` when memory cannot be read at all, and `Some(0)` when the
/// machine is already inside its reserve — the caller must treat that as "do
/// not allocate", not as "allocate nothing and call it a pass".
pub fn safe_test_budget_bytes() -> Option<u64> {
    let m = memory()?;
    Some(
        m.avail_bytes
            .saturating_sub(working_set_reserve_bytes(m.total_bytes)),
    )
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

/// Is this process running elevated (an Administrator token)?
///
/// Matters because it decides whether a child ETW capture is *ours*. PresentMon
/// needs elevation for its trace session; when we are not elevated it relaunches
/// itself with `--restart_as_admin`, and the process that ends up writing the CSV
/// is no longer our child — so we cannot stop it, and have to wait out its own
/// timer. Running elevated turns a wait of up to the whole run duration into an
/// immediate stop.
pub fn is_elevated() -> bool {
    #[cfg(windows)]
    {
        win_elevation::is_elevated()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(windows)]
mod win_elevation {
    use core::ffi::c_void;

    /// `TokenElevation` from the TOKEN_INFORMATION_CLASS enum.
    const TOKEN_ELEVATION: u32 = 20;
    const TOKEN_QUERY: u32 = 0x0008;

    #[link(name = "advapi32")]
    extern "system" {
        // Handles are `isize` here to match the declaration this crate already
        // carries for GetCurrentProcess. Two extern blocks in one crate declaring
        // the same symbol with different signatures is a real hazard, not a
        // style point: the linker keeps one and the other becomes a silent
        // type-confusion at the call site.
        fn OpenProcessToken(process: isize, access: u32, token: *mut isize) -> i32;
        fn GetTokenInformation(
            token: isize,
            class: u32,
            info: *mut c_void,
            len: u32,
            out_len: *mut u32,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn CloseHandle(h: isize) -> i32;
    }

    pub(super) fn is_elevated() -> bool {
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no close;
        // the token handle is closed on every path out.
        unsafe {
            let mut token: isize = 0;
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return false;
            }
            let mut elevated: u32 = 0;
            let mut len: u32 = 0;
            let ok = GetTokenInformation(
                token,
                TOKEN_ELEVATION,
                &mut elevated as *mut u32 as *mut c_void,
                std::mem::size_of::<u32>() as u32,
                &mut len,
            );
            CloseHandle(token);
            ok != 0 && elevated != 0
        }
    }
}
