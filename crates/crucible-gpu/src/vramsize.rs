// SPDX-License-Identifier: MIT
//! How much dedicated video memory the adapter actually has (Windows / DXGI).
//!
//! Needed because **you cannot find the VRAM ceiling by probing for it.** WDDM
//! happily over-commits: an allocation past the end of dedicated VRAM succeeds
//! and silently spills into shared system memory, so `create_buffer` never
//! refuses. The failure surfaces later, when the fill actually writes — and it
//! surfaces as a *device loss*, which takes the run down with it. Measured on an
//! RTX 3070 (8 GiB): 7168 MiB fills clean, 8192 MiB loses the device.
//!
//! So `--vram-mb max` has to know the real number up front. `DXGI_ADAPTER_DESC1`
//! carries it, and reading it needs only `CreateDXGIFactory1` plus two vtable
//! calls — no `windows` crate, matching the hand-rolled FFI used elsewhere in
//! this workspace.

/// Dedicated VRAM in bytes for the adapter at `index`, or `None` when it cannot
/// be determined (non-Windows, no DXGI, or an adapter that reports zero — which
/// is what a software or fully-shared-memory adapter does).
pub fn dedicated_vram_bytes(index: u32) -> Option<u64> {
    #[cfg(windows)]
    {
        win::dedicated(index)
    }
    #[cfg(not(windows))]
    {
        let _ = index;
        None
    }
}

/// The span `--vram-mb max` should ask for, in MiB.
///
/// Deliberately a fraction, not the whole card: the desktop, the compositor and
/// our own command buffers are already resident, so requesting 100% is what
/// triggers the over-commit that kills the device. 85% fills the card hard
/// enough to be a real test while leaving the driver room to breathe.
pub fn max_testable_vram_mb(index: u32) -> Option<usize> {
    let bytes = dedicated_vram_bytes(index)?;
    if bytes == 0 {
        return None;
    }
    let mb = (bytes / (1024 * 1024)) as f64 * VRAM_MAX_FRACTION;
    Some((mb as usize).max(64))
}

/// Fraction of dedicated VRAM used by `max`. See [`max_testable_vram_mb`].
pub const VRAM_MAX_FRACTION: f64 = 0.85;

#[cfg(windows)]
mod win {
    use core::ffi::c_void;

    #[repr(C)]
    struct Guid {
        a: u32,
        b: u16,
        c: u16,
        d: [u8; 8],
    }

    /// IID_IDXGIFactory1 — {770aae78-f26f-4dba-a829-253c83d1b387}
    const IID_FACTORY1: Guid = Guid {
        a: 0x770a_ae78,
        b: 0xf26f,
        c: 0x4dba,
        d: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    };

    #[repr(C)]
    struct AdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        adapter_luid: i64,
        flags: u32,
    }

    #[link(name = "dxgi")]
    extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, out: *mut *mut c_void) -> i32;
    }

    /// COM vtable slots. IUnknown occupies 0-2 and IDXGIObject 3-6 on both
    /// interfaces, so the derived methods start at 7.
    const RELEASE: usize = 2;
    /// `IDXGIFactory1::EnumAdapters1` — after IDXGIFactory's 5 methods (7..=11).
    const ENUM_ADAPTERS1: usize = 12;
    /// `IDXGIAdapter1::GetDesc1` — after IDXGIAdapter's 3 methods (7..=9).
    const GET_DESC1: usize = 10;

    /// Read slot `n` of the vtable `obj` points at and call it.
    ///
    /// # Safety
    /// `obj` must be a live COM interface pointer whose vtable slot `n` has the
    /// signature the caller transmutes to.
    unsafe fn vtbl(obj: *mut c_void, n: usize) -> *const c_void {
        // SAFETY: a COM object's first field is a pointer to its vtable.
        unsafe {
            let vt = *(obj as *const *const *const c_void);
            *vt.add(n)
        }
    }

    unsafe fn release(obj: *mut c_void) {
        if obj.is_null() {
            return;
        }
        // SAFETY: caller guarantees a live interface pointer.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void) -> u32 =
                std::mem::transmute(vtbl(obj, RELEASE));
            f(obj);
        }
    }

    pub(super) fn dedicated(index: u32) -> Option<u64> {
        // SAFETY: every pointer below is checked before use, and each interface
        // obtained is released on every path out.
        unsafe {
            let mut factory: *mut c_void = std::ptr::null_mut();
            if CreateDXGIFactory1(&IID_FACTORY1, &mut factory) < 0 || factory.is_null() {
                return None;
            }

            let enum_adapters: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(vtbl(factory, ENUM_ADAPTERS1));

            let mut adapter: *mut c_void = std::ptr::null_mut();
            let hr = enum_adapters(factory, index, &mut adapter);
            if hr < 0 || adapter.is_null() {
                release(factory);
                return None;
            }

            let get_desc: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32 =
                std::mem::transmute(vtbl(adapter, GET_DESC1));

            let mut desc: AdapterDesc1 = std::mem::zeroed();
            let ok = get_desc(adapter, &mut desc) >= 0;
            let bytes = desc.dedicated_video_memory as u64;

            release(adapter);
            release(factory);

            if ok {
                Some(bytes)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_something_sane_or_nothing_at_all() {
        // Adapter 0 exists on any machine that can run the GPU tests. We assert
        // on plausibility rather than a value, since this runs on whatever
        // hardware the developer has.
        match dedicated_vram_bytes(0) {
            Some(b) if b > 0 => {
                let gib = b as f64 / (1024.0 * 1024.0 * 1024.0);
                assert!(
                    (0.1..=256.0).contains(&gib),
                    "implausible dedicated VRAM: {gib} GiB"
                );
                let mb = max_testable_vram_mb(0).expect("a span follows from a size");
                let total_mb = b / (1024 * 1024);
                assert!(
                    (mb as u64) < total_mb,
                    "the max span ({mb} MiB) must stay UNDER the card's {total_mb} MiB — \
                     asking for all of it is what loses the device"
                );
            }
            // A shared-memory adapter, or no DXGI: both are legitimate, and the
            // caller must fall back rather than guess.
            _ => {}
        }
    }

    #[test]
    fn a_missing_adapter_is_none_not_a_panic() {
        assert!(dedicated_vram_bytes(999).is_none());
    }
}
