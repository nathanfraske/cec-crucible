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
// When `params.shade == 1` (the `--preview` window is open) each pixel is *also*
// shaded into `colorbuf` for display: a clean camera ray, Lambertian lighting
// from the grid's analytic normal, and a traced hard shadow ray so the surface
// self-shadows — genuinely showing the RT cores doing primary + secondary rays.
// The shading never touches `outbuf`, so verification is identical with or
// without a preview, and it is skipped entirely when `shade == 0`.
//
// This is compiled to SPIR-V at runtime by naga (no external shader compiler),
// then handed to raw Vulkan (ash) with VK_KHR_ray_query enabled.
enable wgpu_ray_query;

struct Params {
    iters: u32,
    width: u32,
    height: u32,
    shade: u32,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> colorbuf: array<u32>;

// Sky gradient for rays that miss the geometry.
fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.04, 0.05, 0.09), vec3<f32>(0.16, 0.24, 0.42), t);
}

// Analytic normal of the displaced grid z = 0.3*sin(3x)*cos(3y) (matches
// build_grid on the host), so shading is smooth rather than faceted.
fn grid_normal(p: vec3<f32>) -> vec3<f32> {
    let nx = -0.9 * cos(3.0 * p.x) * cos(3.0 * p.y);
    let ny = 0.9 * sin(3.0 * p.x) * sin(3.0 * p.y);
    return normalize(vec3<f32>(nx, ny, 1.0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }

    // Fan a ray per pixel out of a point behind the geometry. Scale 2.2 spreads
    // the fan just wide enough to cover the [-2,2] displaced grid, so most rays
    // hit and the BVH is actually traversed.
    let gx = idx % params.width;
    let gy = idx / params.width;
    let u = (f32(gx) + 0.5) / f32(params.width) * 2.0 - 1.0;
    let v = (f32(gy) + 0.5) / f32(params.height) * 2.0 - 1.0;
    let origin = vec3<f32>(0.0, 0.0, -4.0);
    let base_dir = normalize(vec3<f32>(u * 2.2, v * 2.2, 4.0));

    // Seed the accumulator per-ray so miss patterns still vary across the image.
    var acc: u32 = idx * 2654435761u;

    for (var k: u32 = 0u; k < params.iters; k = k + 1u) {
        // Tiny deterministic wobble: distinct traversal each iteration, no RNG.
        let a = f32(k) * 0.0001;
        let jitter = vec3<f32>(sin(a) * 0.03, cos(a) * 0.03, 0.0);
        let dir = normalize(base_dir + jitter);

        var rq: ray_query;
        let desc = RayDesc(0u, 0xFFu, 0.01, 100.0, origin, dir);
        rayQueryInitialize(&rq, tlas, desc);
        while (rayQueryProceed(&rq)) {}

        let hit = rayQueryGetCommittedIntersection(&rq);
        var contrib: u32 = 0x9E3779B9u; // miss sentinel
        if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
            // primitive_index proves the BVH found the right triangle; the hit
            // distance `t` proves the intersection math — both fold in, so any
            // wrong result changes the checksum.
            contrib = (hit.primitive_index + 1u) ^ bitcast<u32>(hit.t);
        }
        acc = (acc ^ contrib) * 16777619u; // FNV-style mix
    }

    outbuf[idx] = acc;

    // ---- Display shading (only when a preview window is open) ----
    // When shading is off, do NOT touch colorbuf at all: it is host-visible, so a
    // write here would burn PCIe bandwidth every dispatch and pollute the pure
    // traversal load. Leaving it unwritten is fine — nothing reads it.
    if (params.shade != 1u) {
        return;
    }

    var col: vec3<f32>;
    var camrq: ray_query;
    rayQueryInitialize(&camrq, tlas, RayDesc(0u, 0xFFu, 0.01, 100.0, origin, base_dir));
    while (rayQueryProceed(&camrq)) {}
    let cam = rayQueryGetCommittedIntersection(&camrq);

    if (cam.kind == RAY_QUERY_INTERSECTION_NONE) {
        col = sky(base_dir);
    } else {
        let p = origin + base_dir * cam.t;
        var n = grid_normal(p);
        let viewdir = normalize(origin - p);
        // Face the normal toward the camera (the grid is drawn two-sided).
        if (dot(n, viewdir) < 0.0) {
            n = -n;
        }
        let lightdir = normalize(vec3<f32>(0.45, 0.55, -0.70));

        // Traced hard shadow ray, lifted along the normal to avoid self-hit.
        var srq: ray_query;
        rayQueryInitialize(&srq, tlas, RayDesc(0u, 0xFFu, 0.02, 50.0, p + n * 0.012, lightdir));
        while (rayQueryProceed(&srq)) {}
        let shadowed = rayQueryGetCommittedIntersection(&srq).kind != RAY_QUERY_INTERSECTION_NONE;

        let ndl = max(dot(n, lightdir), 0.0);
        let ambient = 0.18;
        var diffuse = ndl;
        var spec = 0.0;
        if (shadowed) {
            diffuse = 0.0;
        } else {
            let halfv = normalize(lightdir + viewdir);
            spec = pow(max(dot(n, halfv), 0.0), 32.0) * ndl;
        }
        let lit = ambient + (1.0 - ambient) * diffuse;

        // Base colour ramps with height for depth cueing.
        let hgt = clamp((p.z + 0.3) / 0.6, 0.0, 1.0);
        let base = mix(vec3<f32>(0.85, 0.45, 0.28), vec3<f32>(0.26, 0.55, 0.95), hgt);
        col = base * lit + vec3<f32>(spec);
    }

    // Gamma-correct for display.
    col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    colorbuf[idx] = pack4x8unorm(vec4<f32>(col, 1.0));
}
