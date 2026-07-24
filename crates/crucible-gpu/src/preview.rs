// SPDX-License-Identifier: MIT
//! Optional live-preview window for the `render` kernel (`--preview`).
//!
//! Pops a real OS window and mirrors the stress render into it so you can watch
//! the GPU work in real time — an instant "is it drawing sane frames or garbage?"
//! gut-check on top of the framebuffer checksum.
//!
//! ## Design constraints this satisfies
//!
//! * **Verification is untouched.** The kernel still renders to its offscreen
//!   colour target and checksums *that*; the preview only *copies* that finished
//!   texture to the swapchain. The bytes that get verified are identical to a
//!   headless run — the window is a mirror, never the source of truth.
//! * **The stress test still runs flat-out.** `render` dispatches frames as fast
//!   as the GPU allows; the preview presents at most ~60 Hz (a human can't see
//!   more), so mirroring never throttles the load.
//! * **No new dependency tree.** The suite already hand-rolls Win32 FFI
//!   (SMBIOS, WHEA, QPC, thread affinity); a window is the same pattern —
//!   `RegisterClassExW` / `CreateWindowExW` / a `PeekMessageW` pump. The only
//!   external crate is `raw-window-handle`, which is already in the wgpu tree; we
//!   just hand wgpu the raw `HWND` so it can make a surface. Windows-only, which
//!   is what the whole suite targets.
//!
//! Closing the window stops the test (same effect as `q` in the TUI or Ctrl-C).

#![cfg(all(windows, feature = "preview"))]
// The Win32 struct/type names below (MSG, RECT, WNDCLASSEXW, …) deliberately
// match the Windows API spelling.
#![allow(clippy::upper_case_acronyms)]

use std::cell::Cell;
use std::mem::MaybeUninit;
use std::num::NonZeroIsize;
use std::time::{Duration, Instant};

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
};

// The offscreen colour target is Rgba8Unorm (see render::COLOR_FORMAT). We
// configure the swapchain to the same format so presenting is a straight
// texture-to-texture copy — no blit shader, no colour conversion.
const PREVIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

// ---------------------------------------------------------------------------
// Minimal Win32 windowing FFI
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type WNDPROC = Option<unsafe extern "system" fn(isize, u32, usize, isize) -> isize>;

#[repr(C)]
struct WNDCLASSEXW {
    cb_size: u32,
    style: u32,
    lpfn_wnd_proc: WNDPROC,
    cb_cls_extra: i32,
    cb_wnd_extra: i32,
    h_instance: isize,
    h_icon: isize,
    h_cursor: isize,
    hbr_background: isize,
    lpsz_menu_name: *const u16,
    lpsz_class_name: *const u16,
    h_icon_sm: isize,
}

#[repr(C)]
struct POINT {
    x: i32,
    y: i32,
}

#[repr(C)]
struct MSG {
    hwnd: isize,
    message: u32,
    w_param: usize,
    l_param: isize,
    time: u32,
    pt: POINT,
}

#[repr(C)]
struct RECT {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

// Window styles: caption + system menu + minimize box, but no thick frame and
// no maximize box → a fixed-size, non-resizable window (so the swapchain size
// stays put and we never have to handle live resize).
const WS_CAPTION: u32 = 0x00C0_0000;
const WS_SYSMENU: u32 = 0x0008_0000;
const WS_MINIMIZEBOX: u32 = 0x0002_0000;
const WS_VISIBLE: u32 = 0x1000_0000;
const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
const SW_SHOW: i32 = 5;
const CS_HREDRAW: u32 = 0x0002;
const CS_VREDRAW: u32 = 0x0001;
const CS_OWNDC: u32 = 0x0020;
const IDC_ARROW: u16 = 32512;
const PM_REMOVE: u32 = 0x0001;
const WM_DESTROY: u32 = 0x0002;
const WM_CLOSE: u32 = 0x0010;
const WM_QUIT: u32 = 0x0012;

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> isize;
}

#[link(name = "user32")]
extern "system" {
    fn RegisterClassExW(class: *const WNDCLASSEXW) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: isize,
        menu: isize,
        instance: isize,
        param: *mut core::ffi::c_void,
    ) -> isize;
    fn DefWindowProcW(hwnd: isize, msg: u32, w_param: usize, l_param: isize) -> isize;
    fn DestroyWindow(hwnd: isize) -> i32;
    fn ShowWindow(hwnd: isize, cmd_show: i32) -> i32;
    fn PeekMessageW(
        msg: *mut MSG,
        hwnd: isize,
        filter_min: u32,
        filter_max: u32,
        remove: u32,
    ) -> i32;
    fn TranslateMessage(msg: *const MSG) -> i32;
    fn DispatchMessageW(msg: *const MSG) -> isize;
    fn LoadCursorW(instance: isize, cursor_name: *const u16) -> isize;
    fn PostQuitMessage(exit_code: i32);
    fn AdjustWindowRect(rect: *mut RECT, style: u32, menu: i32) -> i32;
}

/// The window procedure. Closing the window (WM_CLOSE) tears it down; WM_DESTROY
/// posts WM_QUIT, which the pump loop below detects. No per-window state is
/// needed, so this stays a plain function.
unsafe extern "system" fn wnd_proc(hwnd: isize, msg: u32, wp: usize, lp: isize) -> isize {
    match msg {
        WM_CLOSE => {
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A hand-rolled Win32 window, thread-affine to whichever thread opened it
/// (create + pump + destroy all happen on the render worker thread).
pub struct PreviewWindow {
    hwnd: isize,
    hinstance: isize,
    closed: Cell<bool>,
}

impl PreviewWindow {
    /// Open a fixed-size window whose client area is `w`x`h` pixels.
    pub fn open(title: &str, w: u32, h: u32) -> Result<Self, String> {
        let style = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE;
        let class_name = wide("CecCruciblePreview");
        let title_w = wide(title);

        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSEXW {
                cb_size: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
                lpfn_wnd_proc: Some(wnd_proc),
                cb_cls_extra: 0,
                cb_wnd_extra: 0,
                h_instance: hinstance,
                h_icon: 0,
                h_cursor: LoadCursorW(0, IDC_ARROW as usize as *const u16),
                hbr_background: 0,
                lpsz_menu_name: std::ptr::null(),
                lpsz_class_name: class_name.as_ptr(),
                h_icon_sm: 0,
            };
            // Registering an already-registered class fails harmlessly; the
            // existing class is reused. Only a null HWND below is fatal.
            RegisterClassExW(&wc);

            // Grow the window rect so the *client* area is exactly w x h.
            let mut rect = RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            AdjustWindowRect(&mut rect, style, 0);
            let win_w = rect.right - rect.left;
            let win_h = rect.bottom - rect.top;

            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                title_w.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                win_w,
                win_h,
                0,
                0,
                hinstance,
                std::ptr::null_mut(),
            );
            if hwnd == 0 {
                return Err("CreateWindowExW failed".to_string());
            }
            ShowWindow(hwnd, SW_SHOW);
            Ok(PreviewWindow {
                hwnd,
                hinstance,
                closed: Cell::new(false),
            })
        }
    }

    /// Drain pending window messages. Returns `false` once the window has been
    /// closed (WM_QUIT seen) — the caller should then stop the test.
    pub fn pump(&self) -> bool {
        if self.closed.get() {
            return false;
        }
        unsafe {
            let mut msg = MaybeUninit::<MSG>::zeroed();
            while PeekMessageW(msg.as_mut_ptr(), 0, 0, 0, PM_REMOVE) != 0 {
                if (*msg.as_ptr()).message == WM_QUIT {
                    self.closed.set(true);
                    return false;
                }
                TranslateMessage(msg.as_ptr());
                DispatchMessageW(msg.as_ptr());
            }
        }
        true
    }

    pub fn raw_window_handle(&self) -> RawWindowHandle {
        // hwnd is non-null (checked at open()).
        let mut h = Win32WindowHandle::new(NonZeroIsize::new(self.hwnd).unwrap());
        h.hinstance = NonZeroIsize::new(self.hinstance);
        RawWindowHandle::Win32(h)
    }

    pub fn raw_display_handle(&self) -> RawDisplayHandle {
        RawDisplayHandle::Windows(WindowsDisplayHandle::new())
    }
}

impl Drop for PreviewWindow {
    fn drop(&mut self) {
        if !self.closed.get() {
            unsafe { DestroyWindow(self.hwnd) };
        }
    }
}

// ---------------------------------------------------------------------------
// wgpu present path
// ---------------------------------------------------------------------------

/// Owns the swapchain surface and copies the render's offscreen colour target to
/// it, rate-limited so the preview never throttles the stress load.
pub struct Presenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    last: Cell<Instant>,
    interval: Duration,
    src_w: u32,
    src_h: u32,
}

impl Presenter {
    /// Configure the swapchain. `surface` must have been created from `window`'s
    /// raw handles; the caller guarantees the window outlives this Presenter.
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let caps = surface.get_capabilities(adapter);
        if !caps.formats.contains(&PREVIEW_FORMAT) {
            return Err(format!(
                "surface does not support {PREVIEW_FORMAT:?} (preview needs a straight copy)"
            ));
        }
        // Prefer a non-vsync mode so presenting never blocks the loop; our own
        // ~60 Hz timer is what limits the present rate.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: PREVIEW_FORMAT,
            width,
            height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(device, &config);
        Ok(Presenter {
            surface,
            config,
            // Start "due" so the first frame presents immediately.
            last: Cell::new(Instant::now() - Duration::from_secs(1)),
            interval: Duration::from_millis(16),
            src_w: width,
            src_h: height,
        })
    }

    /// If ~16 ms have elapsed, copy the finished colour target to the swapchain
    /// and present. Cheap no-op otherwise, so it is safe to call every frame.
    pub fn maybe_present(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_tex: &wgpu::Texture,
    ) {
        if self.last.get().elapsed() < self.interval {
            return;
        }
        self.last.set(Instant::now());

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                f
            }
            // Minimized / resized / lost: reconfigure and skip this frame.
            _ => {
                self.surface.configure(device, &self.config);
                return;
            }
        };

        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.src_w.min(self.config.width),
                height: self.src_h.min(self.config.height),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        frame.present();
    }
}

// ---------------------------------------------------------------------------
// Pixel present path (for kernels that produce a CPU-side image, e.g. `rt`,
// whose compute runs on raw Vulkan and hands us finished RGBA8 pixels)
// ---------------------------------------------------------------------------

/// A fullscreen-triangle blit that samples an uploaded image into the swapchain.
/// Linear filtering upscales a small traced image to a comfortable window size.
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var o: VsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.uv = vec2<f32>(x, y);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}

@fragment
fn fs(v: VsOut) -> @location(0) vec4<f32> {
    return textureSample(img, samp, v.uv);
}
"#;

/// Presents CPU-supplied RGBA8 images by uploading them to a texture and blitting
/// (with linear upscale) to the swapchain. Used by kernels whose rendering does
/// not happen on this wgpu device.
pub struct PixelPresenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    upload: wgpu::Texture,
    img_w: u32,
    img_h: u32,
    last: Cell<Instant>,
    interval: Duration,
}

impl PixelPresenter {
    /// `win_w`/`win_h` size the window (swapchain); `img_w`/`img_h` are the size
    /// of the CPU images that will be uploaded and upscaled to fill it.
    pub fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        surface: wgpu::Surface<'static>,
        win_w: u32,
        win_h: u32,
        img_w: u32,
        img_h: u32,
    ) -> Result<Self, String> {
        let caps = surface.get_capabilities(adapter);
        // Prefer a non-sRGB format: the shader already gamma-encodes the image, so
        // an sRGB swapchain would double-encode and wash it out.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| "surface reports no formats".to_string())?;
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: win_w,
            height: win_h,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(device, &config);

        let upload = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview-upload"),
            size: wgpu::Extent3d {
                width: img_w,
                height: img_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let upload_view = upload.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preview-samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview-blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&upload_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview-pl"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview-pipe"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: Default::default(),
            cache: None,
        });

        Ok(PixelPresenter {
            surface,
            config,
            pipeline,
            bind_group,
            upload,
            img_w,
            img_h,
            last: Cell::new(Instant::now() - Duration::from_secs(1)),
            interval: Duration::from_millis(16),
        })
    }

    /// Whether enough time has elapsed to present again (~60 Hz). Callers use
    /// this to skip the (comparatively expensive) image read-back when not due.
    pub fn due(&self) -> bool {
        self.last.get().elapsed() >= self.interval
    }

    /// Upload one RGBA8 image (`img_w * img_h * 4` bytes) and blit it to the
    /// window, rate-limited to ~60 Hz. Cheap no-op if called too soon.
    pub fn present_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue, pixels: &[u8]) {
        if self.last.get().elapsed() < self.interval {
            return;
        }
        self.last.set(Instant::now());

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.upload,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.img_w * 4),
                rows_per_image: Some(self.img_h),
            },
            wgpu::Extent3d {
                width: self.img_w,
                height: self.img_h,
                depth_or_array_layers: 1,
            },
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                f
            }
            _ => {
                self.surface.configure(device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview-blit-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(Some(enc.finish()));
        frame.present();
    }
}
