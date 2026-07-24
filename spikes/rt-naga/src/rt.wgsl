// SPDX-License-Identifier: MIT
// Minimal ray-query compute shader for the naga->SPIR-V de-risk spike.
// Traces one ray per invocation against a bound top-level acceleration
// structure and folds the committed hit into an output word. This exercises
// every ray-query construct the real rt kernel will need:
//   - the `acceleration_structure` binding
//   - `ray_query` local + `RayDesc` literal
//   - rayQueryInitialize / rayQueryProceed / rayQueryGetCommittedIntersection
//   - RayIntersection field reads (kind, t, primitive_index)
enable wgpu_ray_query;

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Fan a small grid of rays out of a point behind the geometry.
    let px = f32(idx % 64u) - 32.0;
    let py = f32((idx / 64u) % 64u) - 32.0;
    let dir = normalize(vec3<f32>(px, py, 64.0));

    var rq: ray_query;
    let desc = RayDesc(0u, 0xFFu, 0.001, 1000.0, vec3<f32>(0.0, 0.0, -4.0), dir);
    rayQueryInitialize(&rq, tlas, desc);
    while (rayQueryProceed(&rq)) {}

    let hit = rayQueryGetCommittedIntersection(&rq);
    var v: u32 = 0u;
    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        // Combine primitive index and hit distance so the checksum depends on
        // real traversal results, not just "did we hit something".
        v = (hit.primitive_index + 1u) ^ bitcast<u32>(hit.t);
    }
    outbuf[idx] = v;
}
