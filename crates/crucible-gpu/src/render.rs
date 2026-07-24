// SPDX-License-Identifier: MIT
//! Graphics-pipeline load — the fixed-function silicon the compute thrasher
//! never touches.
//!
//! `thrash`/`vram`/`link` all run compute or transfers, so they exercise the
//! shader ALU, memory and PCIe but **never** the rasterizer, the texture-mapping
//! units, or the ROP/blend/depth back-end. A card with a dead ROP, a bad TMU, or
//! a broken rasterizer passes every other test and fails first in the customer's
//! game. This kernel drives a real (headless) render pipeline — vertex → raster →
//! textured, lit, blended, depth-tested fragments → offscreen framebuffer — so
//! those units are actually used.
//!
//! ## Why raw wgpu (like `link.rs`, not CubeCL)
//!
//! CubeCL is compute-only: it cannot issue a draw call, sample a texture through
//! the filtering hardware, or run the raster back-end. The graphics pipeline is
//! reachable only through wgpu's render path, so this kernel is built directly on
//! the raw `wgpu` dependency `link.rs` already uses. No window/swapchain: it
//! renders to an offscreen texture, which exercises everything except the
//! display/present path (low QC value on a bench).
//!
//! ## Verification (not optional)
//!
//! The scene is **static and deterministic**, so on one device every frame's
//! framebuffer is bit-identical. Verification reuses the thrasher's model:
//! **liveness** (the framebuffer must not be the clear colour — catches "drew
//! nothing") and **self-consistency** (every checked frame must reproduce the
//! first frame's checksum; a mismatch is a soft error in raster/TMU/ROP or VRAM).
//! Cross-vendor rasterisation is *not* bit-identical (fill rules, filter LSBs
//! differ), so this is same-device self-consistency only — no cross-vendor
//! golden, like the fp32 thrasher.
//!
//! Safety: dimensions and instance count are clamped; adapter/device init and the
//! first frame are probed under `catch_unwind`; a lost device (poll/map error) is
//! caught and reported; a run that verified nothing reports FAIL.

use std::time::Instant;

use crucible_core::kernel::{Budget, Kind, LoadKernel, LoadResult, ShapeDriver, StopFlag, Tick};
use crucible_core::markers::MarkerLog;

use cubecl::future::block_on;

use crate::GpuDevice;

/// A vertex: position, uv, normal (8 f32 = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
}

/// Graphics-pipeline load kernel.
#[derive(Debug, Clone)]
pub struct RenderKernel {
    pub device: GpuDevice,
    /// Framebuffer size. Bigger = more raster/ROP/fill work.
    pub width: u32,
    pub height: u32,
    /// Instances of the base mesh drawn per frame — the geometry/overdraw knob.
    pub instances: u32,
    /// Verify (read back + checksum) every this many frames.
    pub verify_every: u64,
}

impl Default for RenderKernel {
    fn default() -> Self {
        RenderKernel {
            device: GpuDevice::Discrete(0),
            width: 1280,
            height: 720,
            instances: 48,
            verify_every: 32,
        }
    }
}

impl RenderKernel {
    pub fn new(device: GpuDevice) -> Self {
        RenderKernel {
            device,
            ..Default::default()
        }
    }

    fn power_preference(&self) -> wgpu::PowerPreference {
        match self.device {
            GpuDevice::Integrated(_) => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::HighPerformance,
        }
    }
}

const TEX_SIZE: u32 = 256;
const GRID: u32 = 96; // GRID x GRID vertices per instance mesh
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A fixed rotation matrix (column-major) so the mesh isn't screen-aligned —
/// gives the rasteriser varied triangle orientations. Constant → deterministic.
fn view_proj() -> [f32; 16] {
    // Rotate ~35° about a tilted axis, then squash slightly so rotated geometry
    // stays in the [-1,1] clip cube. Hand-built, constant.
    let (ax, ay, az) = {
        let (x, y, z) = (1.0f32, 0.5, 0.25);
        let n = (x * x + y * y + z * z).sqrt();
        (x / n, y / n, z / n)
    };
    let a = 0.6f32; // radians
    let (s, c) = (a.sin(), a.cos());
    let t = 1.0 - c;
    let k = 0.7f32; // uniform scale to keep it in view
                    // Rodrigues rotation, then * k. Column-major [col*4 + row].
    [
        (t * ax * ax + c) * k,
        (t * ax * ay + s * az) * k,
        (t * ax * az - s * ay) * k,
        0.0,
        (t * ax * ay - s * az) * k,
        (t * ay * ay + c) * k,
        (t * ay * az + s * ax) * k,
        0.0,
        (t * ax * az + s * ay) * k,
        (t * ay * az - s * ax) * k,
        (t * az * az + c) * k,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// Build the tessellated grid mesh (positions in [-1,1]^2, wavy z, uv, normal).
fn build_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let n = GRID;
    let mut verts = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let fx = i as f32 / (n - 1) as f32; // 0..1
            let fy = j as f32 / (n - 1) as f32;
            let x = fx * 2.0 - 1.0;
            let y = fy * 2.0 - 1.0;
            // Procedural height + normal (a couple of sine ridges).
            let z = 0.12 * ((x * 4.0).sin() + (y * 4.0).cos());
            let dzdx = 0.12 * 4.0 * (x * 4.0).cos();
            let dzdy = -0.12 * 4.0 * (y * 4.0).sin();
            let nrm = {
                let (nx, ny, nz) = (-dzdx, -dzdy, 1.0);
                let l = (nx * nx + ny * ny + nz * nz).sqrt();
                [nx / l, ny / l, nz / l]
            };
            verts.push(Vertex {
                pos: [x, y, z],
                uv: [fx, fy],
                normal: nrm,
            });
        }
    }
    let mut idx = Vec::with_capacity(((n - 1) * (n - 1) * 6) as usize);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = j * n + i;
            let b = j * n + i + 1;
            let c = (j + 1) * n + i;
            let d = (j + 1) * n + i + 1;
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (verts, idx)
}

/// A procedural RGBA8 texture (checker × value-noise) for the TMU to sample.
fn build_texture() -> Vec<u8> {
    let mut px = vec![0u8; (TEX_SIZE * TEX_SIZE * 4) as usize];
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let checker = ((x / 16) + (y / 16)) & 1;
            // cheap hash noise
            let mut h = x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263);
            h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
            let n = (h & 0xff) as u8;
            let i = ((y * TEX_SIZE + x) * 4) as usize;
            let base: u8 = if checker == 0 { 200 } else { 60 };
            px[i] = base.saturating_add(n / 4);
            px[i + 1] = (n).wrapping_add(40);
            px[i + 2] = 255u8.saturating_sub(base);
            px[i + 3] = 255;
        }
    }
    px
}

const SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>,
           @location(1) uv: vec2<f32>,
           @location(2) normal: vec3<f32>,
           @builtin(instance_index) inst: u32) -> VsOut {
    let fi = f32(inst);
    let n = max(u.params.x, 1.0);
    // Spread instances through depth (overdraw) and a golden-angle xy swirl.
    let ang = fi * 0.61803398875 * 6.2831853;
    let depth = (fi / n) * 1.4 - 0.7;
    let off = vec3<f32>(cos(ang) * 0.18, sin(ang) * 0.18, depth);
    let world = pos * 0.9 + off;
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv + vec2<f32>(fi * 0.017, fi * 0.011);
    out.normal = normal;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Several texture fetches (TMU) + lighting ALU (shader cores).
    let t0 = textureSample(tex, samp, in.uv);
    let t1 = textureSample(tex, samp, in.uv * 3.0 + vec2<f32>(0.3, 0.7));
    let t2 = textureSample(tex, samp, in.uv * 0.5 - vec2<f32>(0.1, 0.2));
    let base = t0 * 0.5 + t1 * 0.3 + t2 * 0.2;
    let ldir = normalize(vec3<f32>(0.4, 0.7, 0.6));
    let ndl = max(dot(normalize(in.normal), ldir), 0.0);
    let spec = pow(ndl, 24.0);
    let rgb = base.rgb * (0.2 + 0.8 * ndl) + vec3<f32>(spec, spec, spec);
    return vec4<f32>(rgb, 0.55); // alpha < 1 → the ROP blends (overdraw)
}
"#;

/// The wgpu device + pipeline + reusable buffers for the render load.
struct RenderGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    index_count: u32,
    color_view: wgpu::TextureView,
    color_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bpr: u32,
    instances: u32,
}

impl RenderGpu {
    fn init(k: &RenderKernel) -> Result<RenderGpu, String> {
        let width = k.width.clamp(64, 7680);
        let height = k.height.clamp(64, 4320);
        let instances = k.instances.clamp(1, 4096);

        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: k.power_preference(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("no usable GPU adapter: {e}"))?;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("crucible-render"),
            ..Default::default()
        }))
        .map_err(|e| format!("could not create device: {e}"))?;

        // Geometry.
        let (verts, idx) = build_mesh();
        let index_count = idx.len() as u32;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-vbuf"),
            size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut v = vbuf.slice(..).get_mapped_range_mut();
            v.copy_from_slice(vertex_bytes(&verts));
        }
        vbuf.unmap();
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-ibuf"),
            size: (idx.len() * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut v = ibuf.slice(..).get_mapped_range_mut();
            v.copy_from_slice(u32_bytes(&idx));
        }
        ibuf.unmap();

        // Texture + sampler (TMU).
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render-tex"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &build_texture(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_SIZE * 4),
                rows_per_image: Some(TEX_SIZE),
            },
            wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("render-samp"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Uniforms.
        let mut ubytes = [0u8; 80];
        ubytes[..64].copy_from_slice(f32_bytes(&view_proj()));
        ubytes[64..].copy_from_slice(f32_bytes(&[instances as f32, 0.0, 0.0, 0.0]));
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-ubuf"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut v = ubuf.slice(..).get_mapped_range_mut();
            v.copy_from_slice(&ubytes);
        }
        ubuf.unmap();

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render-layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 20,
                    shader_location: 2,
                },
            ],
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vbl],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: Default::default(),
            cache: None,
        });

        // Render targets.
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let padded_bpr = ((width * 4).div_ceil(256)) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-readback"),
            size: (padded_bpr * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(RenderGpu {
            device,
            queue,
            pipeline,
            bind_group,
            vbuf,
            ibuf,
            index_count,
            color_view,
            color_tex,
            depth_view,
            readback,
            width,
            height,
            padded_bpr,
            instances,
        })
    }

    /// Render one frame into the offscreen colour target. Returns false on device
    /// loss.
    fn frame(&self) -> bool {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.vbuf.slice(..));
            rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.index_count, 0, 0..self.instances);
        }
        self.queue.submit(Some(enc.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_ok()
    }

    /// Copy the colour target back and checksum it (over real pixels, skipping
    /// row padding). Returns `(hash, uniform)` where `uniform` means every pixel
    /// equals pixel 0 — a blank frame, i.e. nothing was drawn. `None` on device
    /// loss / map failure.
    fn checksum(&self) -> Option<(u64, bool)> {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bpr),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r.is_ok());
            });
        if self
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return None;
        }
        if !rx.recv().unwrap_or(false) {
            return None;
        }
        let row_bytes = (self.width * 4) as usize;
        let (hash, uniform) = {
            let view = self.readback.slice(..).get_mapped_range();
            let first = [view[0], view[1], view[2], view[3]];
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut uniform = true;
            for row in 0..self.height as usize {
                let start = row * self.padded_bpr as usize;
                for (k, &b) in view[start..start + row_bytes].iter().enumerate() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    if b != first[k & 3] {
                        uniform = false;
                    }
                }
            }
            (h, uniform)
        };
        self.readback.unmap();
        Some((hash, uniform))
    }
}

fn vertex_bytes(v: &[Vertex]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl LoadKernel for RenderKernel {
    fn name(&self) -> &str {
        "render"
    }

    fn kind(&self) -> Kind {
        Kind::Gpu
    }

    fn run(&self, budget: &Budget, stop: &StopFlag, markers: &MarkerLog) -> LoadResult {
        // Init + first frame under catch_unwind — a driver/shader-compile panic
        // becomes a clean setup failure, never takes down the run.
        let gpu = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RenderGpu::init(self)
        })) {
            Ok(Ok(g)) => g,
            Ok(Err(e)) => return LoadResult::setup_failure(e),
            Err(_) => {
                return LoadResult::setup_failure(format!(
                    "render init panicked for {}",
                    self.device.label()
                ))
            }
        };

        let label = self.device.label();
        let mut driver = ShapeDriver::start(budget, stop, markers, "render", "render");
        let start = Instant::now();

        let mut frames: u64 = 0;
        let mut verifications: u64 = 0;
        let mut errors: u64 = 0;
        let mut reference: Option<u64> = None;

        loop {
            match driver.tick() {
                Tick::Work => {
                    if !gpu.frame() {
                        errors += 1;
                        return LoadResult::new(
                            true,
                            frames,
                            reference.unwrap_or(0),
                            errors,
                            format!("{label} render: device lost after {frames} frame(s)"),
                        );
                    }
                    frames += 1;
                    if frames.is_multiple_of(self.verify_every) {
                        match gpu.checksum() {
                            Some((h, uniform)) => {
                                verifications += 1;
                                match reference {
                                    None => {
                                        // Liveness: a uniform frame means nothing was drawn.
                                        if uniform {
                                            errors += 1;
                                            return LoadResult::new(
                                                true,
                                                frames,
                                                h,
                                                errors,
                                                format!(
                                                    "{label} render: framebuffer is uniform — \
                                                     nothing was drawn"
                                                ),
                                            );
                                        }
                                        reference = Some(h);
                                    }
                                    Some(r) if r != h => {
                                        errors += 1;
                                        return LoadResult::new(
                                            true,
                                            frames,
                                            h,
                                            errors,
                                            format!(
                                                "{label} render: framebuffer miscompare at frame \
                                                 {frames} (got {h:#018x}, expected {r:#018x})"
                                            ),
                                        );
                                    }
                                    _ => {}
                                }
                            }
                            None => {
                                errors += 1;
                                return LoadResult::new(
                                    true,
                                    frames,
                                    reference.unwrap_or(0),
                                    errors,
                                    format!("{label} render: read-back failed (device lost)"),
                                );
                            }
                        }
                    }
                }
                Tick::Idle => {}
                Tick::Stop => break,
            }
        }

        let secs = start.elapsed().as_secs_f64();
        if verifications == 0 {
            // Never read anything back — refuse to report a confident pass.
            return LoadResult::new(
                false,
                frames,
                0,
                0,
                format!("{label} render: no frame verified (ran {frames} frame(s) in {secs:.1}s)"),
            );
        }
        let fps = if secs > 0.0 {
            frames as f64 / secs
        } else {
            0.0
        };
        let detail = format!(
            "{label} {}x{} x{} inst, {frames} frame(s) ~{fps:.0} fps, {verifications} verified",
            gpu.width, gpu.height, gpu.instances
        );
        LoadResult::new(true, frames, reference.unwrap_or(0), errors, detail)
    }
}
