// SPDX-License-Identifier: MIT
//
// Multi-bounce Monte-Carlo PATH TRACER — the deep/divergent RT-core + SM stress.
// Where the `rt` test fires a coherent fan of primary (+shadow) rays, this
// bounces light around the scene: each pixel casts `samples` paths, each up to
// `bounces` deep, diffusely scattering off the geometry and gathering light from
// an environment (sky + sun). That makes the secondary rays INCOHERENT — they
// scatter across the hemisphere and touch far-apart BVH nodes — which is the
// divergent, cache-thrashing traversal + per-bounce shading load that a real
// path-traced game imposes and that `rt` cannot reach.
//
// ## Determinism (this is what makes it a valid QC test)
//
// A path tracer is stochastic, but the suite verifies by same-device
// self-consistency: every dispatch must reproduce the first's checksum bit-for-
// bit. So this is a MEGAKERNEL — one invocation per pixel loops all samples and
// bounces in REGISTERS, in a fixed order, and writes once. The RNG is a PCG
// stream seeded PURELY from the pixel index (+ a fixed base seed), so the exact
// same pseudo-random sequence — hence the exact same accumulated radiance — is
// produced every dispatch. There is NO float-atomic accumulation (whose
// reduction order is scheduler-dependent and would break bit-exactness) and no
// time-varying input in the verified path. The accumulated radiance is bitcast
// into the FNV checksum, exactly as `rt.wgsl` bitcasts the hit distance.
//
// Compiled to SPIR-V at runtime by naga; run on raw Vulkan (ash) with ray-query.
enable wgpu_ray_query;

const PI: f32 = 3.14159265;

struct Params {
    iters: u32,   // unused in path-trace mode
    width: u32,
    height: u32,
    shade: u32,   // 1 => also write the tonemapped image to colorbuf (preview)
    time: f32,    // unused in path-trace mode
    samples: u32, // paths per pixel
    bounces: u32, // max path depth
    seed: u32,    // fixed base seed (keeps it deterministic)
}

@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> outbuf: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> colorbuf: array<u32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> vnormals: array<f32>;

// PCG (RXS-M-XS) — a good, cheap, fully deterministic per-pixel RNG stream.
fn pcg(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}
fn randf(state: ptr<function, u32>) -> f32 {
    return f32(pcg(state)) * (1.0 / 4294967296.0);
}

fn fetch_normal(i: u32) -> vec3<f32> {
    return vec3<f32>(vnormals[3u * i], vnormals[3u * i + 1u], vnormals[3u * i + 2u]);
}

// Environment radiance: a sky gradient plus a bright, sharp sun — the only light
// in the scene, so the geometry is lit entirely by what the paths gather.
fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let base = mix(vec3<f32>(0.35, 0.42, 0.55), vec3<f32>(0.70, 0.82, 1.00), t);
    let sun = normalize(vec3<f32>(0.45, 0.75, -0.30));
    let s = pow(max(dot(dir, sun), 0.0), 220.0) * 9.0;
    return base * 1.15 + vec3<f32>(s);
}

// Cosine-weighted hemisphere sample around n (for a Lambertian surface the cos
// term and the pdf cancel, so throughput just multiplies by albedo).
fn cosine_hemisphere(n: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let t = normalize(cross(a, n));
    let b = cross(n, t);
    let phi = 2.0 * PI * r1;
    let r = sqrt(r2);
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - r2));
    return normalize(t * x + b * y + n * z);
}

// Trace one full path and return the radiance it gathers.
fn trace_path(ro: vec3<f32>, rd: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var origin = ro;
    var dir = rd;

    for (var b: u32 = 0u; b < params.bounces; b = b + 1u) {
        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, dir));
        while (rayQueryProceed(&rq)) {}
        let hit = rayQueryGetCommittedIntersection(&rq);

        if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
            // Escaped to the environment — gather its light and stop.
            radiance += throughput * sky(dir);
            break;
        }

        let p = origin + dir * hit.t;
        let prim = hit.primitive_index;
        let i0 = indices[3u * prim + 0u];
        let i1 = indices[3u * prim + 1u];
        let i2 = indices[3u * prim + 2u];
        let bc = hit.barycentrics;
        let w0 = 1.0 - bc.x - bc.y;
        var n = normalize(w0 * fetch_normal(i0) + bc.x * fetch_normal(i1) + bc.y * fetch_normal(i2));
        if (dot(n, dir) > 0.0) {
            n = -n; // face the incoming ray
        }

        // Diffuse (Lambertian) bounce. A gently tinted albedo so multi-bounce
        // colour-bleeding is visible.
        let albedo = vec3<f32>(0.72, 0.62, 0.55);
        let r1 = randf(rng);
        let r2 = randf(rng);
        dir = cosine_hemisphere(n, r1, r2);
        throughput *= albedo;
        origin = p + n * 0.001;
    }

    return radiance;
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

    // Per-pixel RNG stream, seeded purely from the pixel index + fixed base seed
    // → identical sequence every dispatch → deterministic radiance.
    var rng: u32 = (idx * 0x9e3779b9u) ^ (params.seed * 0x85ebca6bu) ^ 0xdeadbeefu;

    // Fixed look-at camera (deterministic); a 3/4 view that frames the object.
    let cam_pos = vec3<f32>(2.6, 1.9, -3.4);
    let fwd = normalize(-cam_pos);
    let right = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, fwd);
    let fov = 0.52;
    let inv_w = 1.0 / f32(params.width);
    let inv_h = 1.0 / f32(params.height);

    var radiance = vec3<f32>(0.0);
    let samples = max(params.samples, 1u);
    for (var s: u32 = 0u; s < samples; s = s + 1u) {
        // Jittered primary ray (stochastic AA), jitter from the same RNG.
        let jx = randf(&rng);
        let jy = randf(&rng);
        let u = (f32(gx) + jx) * inv_w * 2.0 - 1.0;
        let v = (f32(gy) + jy) * inv_h * 2.0 - 1.0;
        let dir = normalize(fwd + right * (u * fov) + up * (-v * fov));
        radiance += trace_path(cam_pos, dir, &rng);
    }
    radiance = radiance / f32(samples);

    // Checksum: bitcast the accumulated radiance into the FNV fold (deterministic
    // → self-consistent). Seeded by idx so an all-black pixel still varies.
    var h: u32 = idx * 2654435761u;
    h = (h ^ bitcast<u32>(radiance.x)) * 16777619u;
    h = (h ^ bitcast<u32>(radiance.y)) * 16777619u;
    h = (h ^ bitcast<u32>(radiance.z)) * 16777619u;
    outbuf[idx] = h;

    // Display: Reinhard tonemap + gamma (only when a preview window is open).
    if (params.shade == 1u) {
        var c = radiance / (radiance + vec3<f32>(1.0));
        c = pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
        colorbuf[idx] = pack4x8unorm(vec4<f32>(c, 1.0));
    }
}
