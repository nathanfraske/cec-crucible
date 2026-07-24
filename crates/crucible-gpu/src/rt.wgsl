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
// This is compiled to SPIR-V at runtime by naga (no external shader compiler),
// then handed to raw Vulkan (ash) with VK_KHR_ray_query enabled.
enable wgpu_ray_query;

struct Params {
    iters: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

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
}
