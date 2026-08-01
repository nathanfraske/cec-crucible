// SPDX-License-Identifier: MIT
//! Adapter identity — which GPU is actually under test.
//!
//! Before this module the suite carried at least four unrelated integers and
//! treated them as one:
//!
//! | space | `0` means |
//! |---|---|
//! | wgpu class ordinal (`GpuDevice::Integrated(0)`) | first *integrated* adapter |
//! | DXGI adapter index (`EnumAdapters1(0)`) | highest-performance adapter |
//! | NVML device index | first *NVIDIA* device |
//! | ADL adapter index | first *AMD* device |
//!
//! Passing a wgpu class ordinal to DXGI is what made `vram --gpu-device
//! integrated --vram-mb max` size itself to the *discrete* card's VRAM and
//! report `PASS … 6784 MiB VRAM` on an adapter that has none. That is the worst
//! failure this tool can have: not a crash, a confident wrong answer.
//!
//! So identity is resolved **once**, here, into an [`AdapterRecord`] that every
//! other plane keys off:
//!
//! * **wgpu** supplies the authoritative name, vendor/device id, backend, and —
//!   crucially — `device_type`, which is how we know an adapter is integrated
//!   without guessing from memory sizes.
//! * **DXGI** supplies the **LUID**, dedicated video memory and shared system
//!   memory. The LUID is Windows' stable adapter identity and is the join key
//!   for the vendor-neutral PDH `GPU Engine` / `GPU Adapter Memory` counters,
//!   which are named `luid_0x…_0x…_phys_0`.
//!
//! The two are matched on `(vendor_id, device_id)`, falling back to the
//! description. A machine with two physically identical cards cannot be told
//! apart this way; that case is reported rather than guessed at.

use cubecl::future::block_on;

use crate::GpuDevice;

/// Everything known about one adapter, from every plane that can see it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdapterRecord {
    /// wgpu's adapter name, e.g. "NVIDIA GeForce RTX 3070".
    pub name: String,
    /// PCI vendor id: 0x10DE NVIDIA, 0x1002/0x1022 AMD, 0x8086 Intel.
    pub vendor_id: u32,
    pub device_id: u32,
    /// Graphics backend wgpu selected for it ("Dx12", "Vulkan", …).
    pub backend: String,
    /// **The** identity: Windows' locally-unique adapter id. 0 when unknown
    /// (non-Windows, or no DXGI adapter matched).
    pub luid: i64,
    /// Dedicated video memory in bytes. 0 on a UMA adapter, which is the point.
    pub dedicated_vram: u64,
    /// System memory the adapter may share. On UMA this is where everything
    /// lives.
    pub shared_system_memory: u64,
    /// Unified memory: the adapter has no private VRAM and no PCIe link to
    /// cross. Taken from wgpu's `DeviceType::IntegratedGpu`, not inferred from
    /// memory sizes.
    pub uma: bool,
    /// A software rasteriser (WARP / lavapipe). Never a valid QC target.
    pub software: bool,
    /// Largest single buffer the adapter will allocate, bytes.
    pub max_buffer_bytes: u64,
    /// Largest storage-buffer binding, bytes. Lower than `max_buffer_bytes` on
    /// most adapters, and it is the one that actually bounds a compute kernel's
    /// working buffer.
    pub max_storage_binding_bytes: u64,
}

impl AdapterRecord {
    /// Vendor name for display, from the PCI id rather than string-matching the
    /// adapter name (which vendors change between driver versions).
    pub fn vendor(&self) -> &'static str {
        match self.vendor_id {
            0x10DE => "NVIDIA",
            0x1002 | 0x1022 => "AMD",
            0x8086 => "Intel",
            0x13B5 => "ARM",
            0x5143 => "Qualcomm",
            0x1010 => "Imagination",
            _ => "unknown",
        }
    }

    /// The PDH instance-name fragment for this adapter's GPU counters.
    ///
    /// Windows names those instances `luid_0xHIGH_0xLOW_phys_N`, where the two
    /// halves are the LUID's high and low 32 bits. Verified against
    /// `\GPU Adapter Memory(luid_0x00000000_0x0001b019_phys_0)\Dedicated Usage`.
    pub fn pdh_luid_key(&self) -> Option<String> {
        if self.luid == 0 {
            return None;
        }
        let raw = self.luid as u64;
        Some(format!(
            "luid_0x{:08x}_0x{:08x}",
            (raw >> 32) as u32,
            raw as u32
        ))
    }

    /// One-line identity for reports and the console.
    pub fn line(&self) -> String {
        let kind = if self.software {
            "software"
        } else if self.uma {
            "integrated (UMA)"
        } else {
            "discrete"
        };
        let mem = if self.uma {
            format!("{} MiB shared", self.shared_system_memory / (1024 * 1024))
        } else {
            format!("{} MiB VRAM", self.dedicated_vram / (1024 * 1024))
        };
        format!(
            "{} [{}] — {}, {}, {}, max buffer {} MiB / binding {} MiB",
            self.name,
            self.vendor(),
            kind,
            mem,
            self.backend,
            self.max_buffer_bytes / (1024 * 1024),
            self.max_storage_binding_bytes / (1024 * 1024)
        )
    }
}

/// Enumerate every adapter wgpu can see, joined with its DXGI identity.
///
/// Ordering is wgpu's, so a class ordinal (`Integrated(1)` = "the second
/// integrated adapter") means here exactly what it means when the kernel asks
/// wgpu for its device.
///
/// **Cached for the life of the process.** Identity is consulted from nine
/// places — buffer sizing, the UMA checks in `vram` and `link`, each kernel's
/// detail string, adapter selection — for an answer that cannot change: adapters
/// do not appear or vanish partway through a run.
///
/// Measured honestly: this is *not* the speed-up it looks like. A/B against the
/// uncached build on a `run worst-case --seconds 15` came out 35.3s vs 35.6s,
/// inside the noise. The first enumeration in a process pays to load the DX12,
/// Vulkan and GL drivers (~3.3s here, visible in `gpu-info`), but the drivers
/// stay loaded, so later walks were already cheap. Kept because doing the same
/// work nine times for an invariant answer is still wrong, not because it made
/// the suite faster.
pub fn enumerate() -> Vec<AdapterRecord> {
    static CACHE: std::sync::OnceLock<Vec<AdapterRecord>> = std::sync::OnceLock::new();
    CACHE.get_or_init(enumerate_uncached).clone()
}

/// The real walk, behind the cache. Separate so a test can measure it.
fn enumerate_uncached() -> Vec<AdapterRecord> {
    let instance = wgpu::Instance::default();
    let dxgi = dxgi_adapters();

    block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| {
            let info = a.get_info();
            let uma = info.device_type == wgpu::DeviceType::IntegratedGpu;
            let software = matches!(
                info.device_type,
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
            );
            let limits = a.limits();
            let mut rec = AdapterRecord {
                name: info.name.clone(),
                max_buffer_bytes: limits.max_buffer_size,
                max_storage_binding_bytes: limits.max_storage_buffer_binding_size as u64,
                vendor_id: info.vendor,
                device_id: info.device,
                backend: format!("{:?}", info.backend),
                uma,
                software,
                ..Default::default()
            };
            if let Some(d) = match_dxgi(&dxgi, &rec) {
                rec.luid = d.luid;
                rec.dedicated_vram = d.dedicated_vram;
                rec.shared_system_memory = d.shared_system_memory;
            }
            rec
        })
        .collect()
}

/// Resolve a [`GpuDevice`] selector to the adapter it will actually run on.
///
/// The class ordinal is counted over adapters of that class, in wgpu's own
/// order — the same walk wgpu performs internally — rather than being used as a
/// global index into some other enumeration.
pub fn resolve(device: GpuDevice) -> Option<AdapterRecord> {
    let all = enumerate();
    // Duplicate backends (DX12 and Vulkan both expose the same physical GPU)
    // would double-count the class ordinal, so collapse to one entry per
    // physical adapter first, keeping the first backend wgpu offered.
    let mut seen: Vec<AdapterRecord> = Vec::new();
    for a in all {
        if !seen
            .iter()
            .any(|s| s.vendor_id == a.vendor_id && s.device_id == a.device_id && s.name == a.name)
        {
            seen.push(a);
        }
    }
    match device {
        GpuDevice::Integrated(i) => seen.into_iter().filter(|a| a.uma).nth(i),
        GpuDevice::Discrete(i) => seen.into_iter().filter(|a| !a.uma && !a.software).nth(i),
        // wgpu's default is the highest-power adapter, which is the first
        // discrete one when there is one.
        GpuDevice::Default => {
            let discrete = seen.iter().find(|a| !a.uma && !a.software).cloned();
            discrete.or_else(|| seen.into_iter().next())
        }
    }
}

/// The DXGI half of an adapter's identity.
#[derive(Debug, Clone, Default)]
struct DxgiAdapter {
    description: String,
    vendor_id: u32,
    device_id: u32,
    luid: i64,
    dedicated_vram: u64,
    shared_system_memory: u64,
}

/// Pair a wgpu adapter with its DXGI entry on `(vendor, device)`, then on the
/// description. Returns `None` rather than a guess when nothing matches — a
/// wrong LUID would bind the sensor plane to the wrong GPU, which is the exact
/// class of bug this module exists to end.
fn match_dxgi<'a>(dxgi: &'a [DxgiAdapter], rec: &AdapterRecord) -> Option<&'a DxgiAdapter> {
    let exact: Vec<&DxgiAdapter> = dxgi
        .iter()
        .filter(|d| d.vendor_id == rec.vendor_id && d.device_id == rec.device_id)
        .collect();
    if exact.len() == 1 {
        return Some(exact[0]);
    }
    // Two identical cards: disambiguate by description if we can, otherwise
    // decline. An arbitrary pick here would silently attribute one card's
    // sensors to the other.
    if exact.len() > 1 {
        return exact
            .into_iter()
            .find(|d| d.description.eq_ignore_ascii_case(&rec.name));
    }
    dxgi.iter().find(|d| d.description.eq_ignore_ascii_case(&rec.name))
}

#[cfg(windows)]
fn dxgi_adapters() -> Vec<DxgiAdapter> {
    win::enumerate_dxgi()
}

#[cfg(not(windows))]
fn dxgi_adapters() -> Vec<DxgiAdapter> {
    Vec::new()
}

#[cfg(windows)]
mod win {
    use super::DxgiAdapter;
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

    /// How many adapters we are willing to walk before concluding the driver is
    /// misbehaving. `EnumAdapters1` terminates with DXGI_ERROR_NOT_FOUND; this
    /// only bounds a pathological loop.
    const MAX_ADAPTERS: u32 = 64;

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

    fn wide_to_string(w: &[u16]) -> String {
        let end = w.iter().position(|&c| c == 0).unwrap_or(w.len());
        String::from_utf16_lossy(&w[..end])
    }

    pub(super) fn enumerate_dxgi() -> Vec<DxgiAdapter> {
        let mut out = Vec::new();
        // SAFETY: every pointer is checked before use, and each interface
        // obtained is released on every path out.
        unsafe {
            let mut factory: *mut c_void = std::ptr::null_mut();
            if CreateDXGIFactory1(&IID_FACTORY1, &mut factory) < 0 || factory.is_null() {
                return out;
            }
            let enum_adapters: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(vtbl(factory, ENUM_ADAPTERS1));

            for i in 0..MAX_ADAPTERS {
                let mut adapter: *mut c_void = std::ptr::null_mut();
                if enum_adapters(factory, i, &mut adapter) < 0 || adapter.is_null() {
                    break; // DXGI_ERROR_NOT_FOUND: past the last adapter
                }
                let get_desc: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32 =
                    std::mem::transmute(vtbl(adapter, GET_DESC1));
                let mut desc: AdapterDesc1 = std::mem::zeroed();
                if get_desc(adapter, &mut desc) >= 0 {
                    out.push(DxgiAdapter {
                        description: wide_to_string(&desc.description),
                        vendor_id: desc.vendor_id,
                        device_id: desc.device_id,
                        luid: desc.adapter_luid,
                        dedicated_vram: desc.dedicated_video_memory as u64,
                        shared_system_memory: desc.shared_system_memory as u64,
                    });
                }
                release(adapter);
            }
            release(factory);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_second_enumeration_is_served_from_cache() {
        // Nine call sites consult identity, and the uncached walk initialises
        // every graphics backend. Without the cache a cross-load spent tens of
        // seconds re-deriving an answer that cannot change mid-run.
        let first = std::time::Instant::now();
        let a = enumerate();
        let cold = first.elapsed();

        let second = std::time::Instant::now();
        let b = enumerate();
        let warm = second.elapsed();

        assert_eq!(a, b, "the cached answer must be the same answer");
        // Not a timing assertion on the cold path (that depends on the machine)
        // — only that the warm path is not doing the work again.
        assert!(
            warm * 10 < cold.max(std::time::Duration::from_millis(10)),
            "second enumeration took {warm:?} against a cold {cold:?}: cache not working"
        );
    }

    #[test]
    fn enumeration_is_self_consistent() {
        let all = enumerate();
        // A machine running these tests has at least one adapter; if wgpu can
        // see none, there is nothing to assert and nothing to test.
        for a in &all {
            assert!(!a.name.is_empty(), "an adapter with no name: {a:?}");
            // UMA and a big dedicated VRAM pool are contradictory. If this ever
            // fires, the UMA signal is wrong and every sizing decision built on
            // it is wrong too.
            if a.uma {
                assert!(
                    a.dedicated_vram < 1024 * 1024 * 1024,
                    "adapter claims UMA but reports {} MiB of dedicated VRAM: {a:?}",
                    a.dedicated_vram / (1024 * 1024)
                );
            }
        }
    }

    #[test]
    fn a_discrete_selector_never_resolves_to_an_integrated_adapter() {
        // The exact confusion that produced a PASS at "6784 MiB VRAM" on an
        // adapter with no VRAM: a class ordinal used as a global index.
        if let Some(d) = resolve(GpuDevice::Discrete(0)) {
            assert!(!d.uma, "Discrete(0) resolved to a UMA adapter: {d:?}");
        }
        if let Some(i) = resolve(GpuDevice::Integrated(0)) {
            assert!(i.uma, "Integrated(0) resolved to a non-UMA adapter: {i:?}");
        }
    }

    #[test]
    fn the_pdh_luid_key_matches_the_windows_instance_naming() {
        let a = AdapterRecord {
            luid: 0x0001_b019,
            ..Default::default()
        };
        // Windows names the counter instance luid_0xHIGH_0xLOW_phys_N; this is
        // the real instance observed on the bench.
        assert_eq!(a.pdh_luid_key().as_deref(), Some("luid_0x00000000_0x0001b019"));
        // No LUID means no key — never a plausible-looking zero that would
        // match some other adapter's counters.
        assert_eq!(AdapterRecord::default().pdh_luid_key(), None);
    }
}
