// SPDX-License-Identifier: MIT
//
// Ray-query traversal load for the RT-core test. One invocation per screen ray;
// each fires `iters` slightly-perturbed rays at the bound top-level acceleration
// structure and folds every committed hit (primitive index + hit distance) into
// a per-ray checksum word. The perturbation keeps each trace distinct work (so
// the RT cores stay fed and nothing is optimized away) while the whole thing
// stays a pure function of the geometry — the same `iters` always produces the
// same output, which is what the host self-consistency check relies on to catch
// an RT core returning a wrong intersection.
//
// The checksum always uses a FIXED camera, so it is deterministic and the
// verification is unaffected by anything below.
//
// When `params.shade == 1` (a frame the preview will present) each pixel is ALSO
// shaded into `colorbuf` from an ORBITING camera (driven by `params.time`), with
// 2x2 supersampled anti-aliasing, so the object appears to rotate smoothly with
// clean edges. Shading reconstructs the smooth surface normal from the bound
// index + normal buffers, does Lambertian lighting and a traced hard shadow ray
// (the object self-shadows — visibly the RT cores doing primary + secondary
// rays). None of this touches `outbuf`, so the animated display never affects
// what is verified.
//
// Compiled to SPIR-V at runtime by naga (no external shader compiler), then run
// on raw Vulkan (ash) with VK_KHR_ray_query enabled.
enable wgpu_ray_query;

struct Params {
    iters: u32,
    width: u32,
    height: u32,
    shade: u32,
    time: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> colorbuf: array<u32>;
// Mesh index + per-vertex normal buffers, read only for display shading.
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> vnormals: array<f32>;

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.03, 0.04, 0.07), vec3<f32>(0.13, 0.19, 0.33), t);
}

fn fetch_normal(i: u32) -> vec3<f32> {
    return vec3<f32>(vnormals[3u * i], vnormals[3u * i + 1u], vnormals[3u * i + 2u]);
}

// Trace one display ray and return its (linear) colour. Reconstructs the smooth
// surface normal at the hit from the mesh, lights it, and traces a hard shadow
// ray so the object self-shadows.
fn shade_ray(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var crq: ray_query;
    rayQueryInitialize(&crq, tlas, RayDesc(0u, 0xFFu, 0.01, 100.0, ro, rd));
    while (rayQueryProceed(&crq)) {}
    let ch = rayQueryGetCommittedIntersection(&crq);

    if (ch.kind == RAY_QUERY_INTERSECTION_NONE) {
        return sky(rd);
    }

    let p = ro + rd * ch.t;
    let prim = ch.primitive_index;
    let i0 = indices[3u * prim + 0u];
    let i1 = indices[3u * prim + 1u];
    let i2 = indices[3u * prim + 2u];
    let bc = ch.barycentrics;
    let w0 = 1.0 - bc.x - bc.y;
    var n = normalize(w0 * fetch_normal(i0) + bc.x * fetch_normal(i1) + bc.y * fetch_normal(i2));
    let viewdir = normalize(ro - p);
    if (dot(n, viewdir) < 0.0) {
        n = -n;
    }

    let lightdir = normalize(vec3<f32>(0.35, 0.85, 0.35));
    // Traced hard shadow ray, lifted along the normal to avoid self-hit.
    var srq: ray_query;
    rayQueryInitialize(&srq, tlas, RayDesc(0u, 0xFFu, 0.02, 50.0, p + n * 0.01, lightdir));
    while (rayQueryProceed(&srq)) {}
    let shadowed = rayQueryGetCommittedIntersection(&srq).kind != RAY_QUERY_INTERSECTION_NONE;

    let ndl = max(dot(n, lightdir), 0.0);
    let ambient = 0.2;
    var diffuse = ndl;
    var spec = 0.0;
    if (shadowed) {
        diffuse = 0.0;
    } else {
        let halfv = normalize(lightdir + viewdir);
        spec = pow(max(dot(n, halfv), 0.0), 40.0);
    }
    let lit = ambient + (1.0 - ambient) * diffuse;

    let orient = 0.5 + 0.5 * n.y;
    let tint = mix(vec3<f32>(0.90, 0.48, 0.28), vec3<f32>(0.30, 0.62, 0.95), orient);
    return tint * lit + vec3<f32>(spec) * 0.6;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }

    let gx = idx % params.width;
    let gy = idx / params.width;

    // ---- Checksum: FIXED camera, deterministic (this is what's verified) ----
    let u = (f32(gx) + 0.5) / f32(params.width) * 2.0 - 1.0;
    let v = (f32(gy) + 0.5) / f32(params.height) * 2.0 - 1.0;
    let cs_origin = vec3<f32>(0.0, 0.0, -4.0);
    let cs_dir = normalize(vec3<f32>(u * 2.2, v * 2.2, 4.0));
    var acc: u32 = idx * 2654435761u;
    for (var k: u32 = 0u; k < params.iters; k = k + 1u) {
        let a = f32(k) * 0.0001;
        let jitter = vec3<f32>(sin(a) * 0.03, cos(a) * 0.03, 0.0);
        let dir = normalize(cs_dir + jitter);
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.01, 100.0, cs_origin, dir));
        while (rayQueryProceed(&rq)) {}
        let hit = rayQueryGetCommittedIntersection(&rq);
        var contrib: u32 = 0x9E3779B9u; // miss sentinel
        if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
            contrib = (hit.primitive_index + 1u) ^ bitcast<u32>(hit.t);
        }
        acc = (acc ^ contrib) * 16777619u;
    }
    outbuf[idx] = acc;

    // ---- Display shading: ORBITING camera, 2x2 AA (animated, NOT verified) ----
    if (params.shade != 1u) {
        return;
    }

    // Camera orbits the object at the origin; look-at basis.
    let ang = params.time * 0.5;
    let radius = 4.2;
    let cam_pos = vec3<f32>(sin(ang) * radius, 1.6, -cos(ang) * radius);
    let fwd = normalize(-cam_pos);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let fov = 0.55;
    let inv_w = 1.0 / f32(params.width);
    let inv_h = 1.0 / f32(params.height);

    // Four sub-samples per pixel (jittered on a 2x2 grid), averaged.
    var col = vec3<f32>(0.0);
    for (var sy = 0u; sy < 2u; sy = sy + 1u) {
        for (var sx = 0u; sx < 2u; sx = sx + 1u) {
            let ox = (f32(sx) - 0.5) * 0.5; // -0.25 / +0.25
            let oy = (f32(sy) - 0.5) * 0.5;
            let su = (f32(gx) + 0.5 + ox) * inv_w * 2.0 - 1.0;
            let sv = (f32(gy) + 0.5 + oy) * inv_h * 2.0 - 1.0;
            let rd = normalize(fwd + right * (su * fov) + up * (-sv * fov));
            col = col + shade_ray(cam_pos, rd);
        }
    }
    col = col * 0.25;

    col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    colorbuf[idx] = pack4x8unorm(vec4<f32>(col, 1.0));
}
