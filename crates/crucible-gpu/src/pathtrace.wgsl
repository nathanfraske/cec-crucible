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
    time: f32,    // preview orbit angle (display only; never affects the checksum)
    samples: u32,  // paths per pixel
    bounces: u32,  // max path depth
    seed: u32,     // fixed base seed (keeps it deterministic)
    material: u32, // surface material id (see MAT_* below)
    // WGSL rounds this uniform struct up to 48 bytes; the host `Params` pads to
    // match. (No explicit pad fields needed here — naga adds the tail padding.)
}

// Material ids (kept in sync with the CLI `--material` map and the host RtKernel).
const MAT_METAL: u32 = 0u;
const MAT_MATTE: u32 = 1u;
const MAT_PLASTIC: u32 = 2u;
const MAT_MIRROR: u32 = 3u;
const MAT_GLASS: u32 = 4u;
const MAT_VELVET: u32 = 5u;
const MAT_MARBLE: u32 = 6u;
const MAT_FUR: u32 = 7u;

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

// Environment radiance: a layered sky (warm horizon, blue zenith, dark ground)
// plus a bright sun. It is the only light, so the geometry is lit entirely by
// what the paths gather — and a glossy surface *reflects* this environment, so a
// structured sky (not a flat gradient) is what makes the metal read as reflective.
fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.85, 0.80, 0.72);
    let zenith = vec3<f32>(0.26, 0.42, 0.72);
    let ground = vec3<f32>(0.26, 0.25, 0.23);
    var base = mix(horizon, zenith, smoothstep(0.0, 0.55, t));
    base = mix(ground, base, smoothstep(-0.12, 0.06, dir.y));
    let sun = normalize(vec3<f32>(0.45, 0.75, -0.30));
    let s = pow(max(dot(dir, sun), 0.0), 180.0) * 8.0;
    return base + vec3<f32>(1.0, 0.92, 0.78) * s;
}

// Cosine-weighted hemisphere sample around n (for the diffuse lobe: the cos term
// and the pdf cancel, so throughput just multiplies by albedo).
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

// ---- Procedural surface: texture + bump (all analytic, no texture assets) ----

// Cheap 3-D value noise (one octave): a hashed lattice, trilinearly smoothed.
fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(mix(mix(c000, c100, u.x), mix(c010, c110, u.x), u.y),
               mix(mix(c001, c101, u.x), mix(c011, c111, u.x), u.y), u.z);
}

// Per-surface material: a copper/steel colour that varies over the surface, plus
// a roughness that varies too (so the reflections sharpen and blur across it).
struct Material {
    albedo: vec3<f32>,
    rough: f32,
    metallic: f32,
}
fn surface_material(p: vec3<f32>) -> Material {
    let tint = vnoise(p * 2.2);
    let grain = vnoise(p * 6.5 + vec3<f32>(5.1, 2.7, 8.3));
    var m: Material;
    m.albedo = mix(
        vec3<f32>(0.95, 0.56, 0.32), // warm copper
        vec3<f32>(0.79, 0.80, 0.83), // cool steel
        smoothstep(0.35, 0.65, tint),
    );
    m.rough = clamp(0.04 + 0.22 * grain, 0.03, 0.34);
    m.metallic = 0.90;
    return m;
}

// Analytic "hammered metal" bump: perturb the shading normal by the gradient of a
// cheap sinusoidal height field (no finite differences, no extra noise taps).
fn bump_normal(p: vec3<f32>, n: vec3<f32>, strength: f32) -> vec3<f32> {
    let a = p * 13.0; // coarse hammer dents
    let c = p * 33.0; // fine grain on top
    let grad = vec3<f32>(
        cos(a.x) + 0.5 * cos(a.y * 1.7 + 1.0),
        cos(a.y) + 0.5 * cos(a.z * 1.7 + 2.0),
        cos(a.z) + 0.5 * cos(a.x * 1.7 + 3.0),
    ) + 0.22 * vec3<f32>(
        cos(c.x + c.y),
        cos(c.y + c.z),
        cos(c.z + c.x),
    );
    let tgrad = grad - n * dot(grad, n); // keep only the tangential part
    return normalize(n - tgrad * strength);
}

// ---- Microfacet BRDF (GGX) ----

fn fresnel_schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}
// Importance-sample the GGX normal-distribution: returns a microfacet normal
// around n. Reflecting the ray about it gives a glossy bounce.
fn ggx_sample_h(n: vec3<f32>, rough: f32, r1: f32, r2: f32) -> vec3<f32> {
    let a = max(rough * rough, 1.0e-3);
    let phi = 2.0 * PI * r1;
    let cos_t = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let hx = sin_t * cos(phi);
    let hy = sin_t * sin(phi);
    let hz = cos_t;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * hx + b * hy + n * hz);
}
fn smith_g1(ndotx: f32, a2: f32) -> f32 {
    let k = a2 * 0.5;
    return ndotx / (ndotx * (1.0 - k) + k);
}

// Scalar Schlick Fresnel — for the dielectric coat (plastic/marble) and glass.
fn fresnel_scalar(cos_t: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
}

// Turbulence (summed |noise|) — the veining field for the marble material.
fn turbulence(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 3; i = i + 1) {
        s = s + amp * abs(vnoise(q) * 2.0 - 1.0);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return s;
}

// The outcome of scattering at a surface hit: the next ray, a throughput
// multiplier, the spawn origin (offset off the surface), and whether the path
// should stop here.
struct Scatter {
    dir: vec3<f32>,
    weight: vec3<f32>,
    origin: vec3<f32>,
    terminate: bool,
}

// Evaluate + importance-sample the chosen material's BSDF at a hit. `ng` is the
// smooth geometric normal (un-flipped); `d` is the incoming ray direction. All
// branches consume the deterministic RNG, so any single material stays a
// reproducible (checksummable) workload — swapping material just swaps the
// verified load. Prioritises a good *look*; the firefly clamp in `trace_path`
// bounds any energy the stochastic-lobe estimators overshoot.
fn scatter(mat: u32, p: vec3<f32>, ng: vec3<f32>, d: vec3<f32>, rng: ptr<function, u32>) -> Scatter {
    var s: Scatter;
    s.terminate = false;
    let front = dot(ng, d) < 0.0;
    let nf = select(-ng, ng, front); // shading normal facing against the ray
    let wo = -d;
    let r1 = randf(rng);
    let r2 = randf(rng);
    let pick = randf(rng);

    switch (mat) {
        // Matte clay/plaster — pure Lambertian diffuse.
        case 1u: {
            s.dir = cosine_hemisphere(nf, r1, r2);
            s.weight = vec3<f32>(0.72, 0.64, 0.56);
            s.origin = p + nf * 0.001;
        }
        // Coloured plastic — diffuse body under a Fresnel clearcoat highlight.
        case 2u: {
            let ndotv = max(dot(nf, wo), 1.0e-3);
            let coat = clamp(fresnel_scalar(ndotv, 0.04) * 6.0, 0.04, 0.7);
            if (pick < coat) {
                let h = ggx_sample_h(nf, 0.06, r1, r2);
                let wi = reflect(d, h);
                s.terminate = dot(wi, nf) <= 0.0;
                s.dir = wi;
                s.weight = vec3<f32>(1.0); // white specular
                s.origin = p + nf * 0.001;
            } else {
                s.dir = cosine_hemisphere(nf, r1, r2);
                s.weight = vec3<f32>(0.85, 0.17, 0.19); // red body
                s.origin = p + nf * 0.001;
            }
        }
        // Near-perfect chrome mirror (a faint bump keeps it from being sterile).
        case 3u: {
            let nb = bump_normal(p, nf, 0.05);
            let h = ggx_sample_h(nb, 0.02, r1, r2);
            let wi = reflect(d, h);
            s.terminate = dot(wi, nb) <= 0.0;
            s.dir = wi;
            s.weight = vec3<f32>(0.96, 0.97, 0.98);
            s.origin = p + nb * 0.001;
        }
        // Glass — dielectric: reflect or refract by Fresnel, with TIR fallback.
        case 4u: {
            let ior = 1.5;
            let eta = select(ior, 1.0 / ior, front); // entering vs exiting
            let cos_i = clamp(dot(wo, nf), 0.0, 1.0);
            let fr = fresnel_scalar(cos_i, 0.04);
            let refr = refract(d, nf, eta);
            if (pick < fr || dot(refr, refr) < 1.0e-6) {
                s.dir = reflect(d, nf);
                s.origin = p + nf * 0.001;
            } else {
                s.dir = normalize(refr);
                s.origin = p - nf * 0.001; // spawn on the far side
            }
            s.weight = vec3<f32>(0.90, 0.95, 0.98); // faint cool tint
        }
        // Velvet / fuzz — the fur-ish one: diffuse with fibre-jittered normals and
        // a grazing-angle sheen.
        case 5u: {
            let jitter = vec3<f32>(randf(rng), randf(rng), randf(rng)) * 2.0 - vec3<f32>(1.0);
            let nz = normalize(nf + jitter * 0.25);
            let ndotv = max(dot(nf, wo), 0.0);
            let sheen = pow(1.0 - ndotv, 3.0);
            s.dir = cosine_hemisphere(nz, r1, r2);
            s.weight = vec3<f32>(0.52, 0.38, 0.33) + vec3<f32>(0.85, 0.72, 0.66) * sheen;
            s.origin = p + nf * 0.001;
        }
        // Polished marble — veined diffuse stone with a light specular sheen.
        case 6u: {
            let vein = smoothstep(0.32, 0.62, turbulence(p * 1.5));
            let body = mix(vec3<f32>(0.92, 0.90, 0.86), vec3<f32>(0.20, 0.23, 0.30), vein);
            let ndotv = max(dot(nf, wo), 1.0e-3);
            let coat = clamp(fresnel_scalar(ndotv, 0.05) * 3.0, 0.03, 0.4);
            if (pick < coat) {
                let h = ggx_sample_h(nf, 0.12, r1, r2);
                let wi = reflect(d, h);
                s.terminate = dot(wi, nf) <= 0.0;
                s.dir = wi;
                s.weight = vec3<f32>(1.0);
                s.origin = p + nf * 0.001;
            } else {
                s.dir = cosine_hemisphere(nf, r1, r2);
                s.weight = body;
                s.origin = p + nf * 0.001;
            }
        }
        // Fibre-fur — a Kajiya-Kay-style anisotropic fibre BSDF over a procedural
        // tangent field: a tangent-aligned specular streak, a forward-scatter
        // transmittance lobe (the soft rim glow light picks up passing through the
        // coat), and a fuzzy diffuse undercoat. The extra RNG draws + the branch
        // make it the most divergent material here — a deliberate SM stressor —
        // while staying deterministic (a fixed three draws per hit, either branch).
        case 7u: {
            let jitter = vec3<f32>(randf(rng), randf(rng), randf(rng)) * 2.0 - vec3<f32>(1.0);
            // A procedural fibre tangent flowing across the surface, jittered per
            // shading point so neighbouring fibres diverge (the fuzz).
            let t0 = normalize(cross(nf, vec3<f32>(0.13, 1.0, 0.07)));
            let bz = cross(nf, t0);
            let flow = t0 * cos(p.y * 9.0) + bz * sin(p.x * 9.0 + p.z * 7.0);
            let tv = flow + jitter * 0.35;
            let tangent = tv / max(length(tv), 1.0e-3);
            let base = vec3<f32>(0.46, 0.32, 0.20); // warm animal-fur brown
            // Kajiya-Kay anisotropic streak: brightest where the view grazes
            // across the fibre length (sin of the tangent-view angle).
            let tdotv = dot(tangent, wo);
            let streak = pow(sqrt(clamp(1.0 - tdotv * tdotv, 0.0, 1.0)), 24.0);
            let fwd = 0.35;
            if (pick < fwd) {
                // TT lobe — scatter roughly forward through the coat with a warm,
                // brighter tint: fur's translucent, back-lit rim glow.
                s.dir = normalize(d + nf * 0.2 + jitter * 0.25);
                s.origin = p - nf * 0.001;
                s.weight = (base + vec3<f32>(0.32, 0.24, 0.14)) * (0.5 / fwd);
            } else {
                // Fuzzy diffuse undercoat with the specular streak layered on top.
                let nz = normalize(nf + jitter * 0.5);
                s.dir = cosine_hemisphere(nz, r1, r2);
                s.origin = p + nf * 0.001;
                s.weight = (base + vec3<f32>(0.70, 0.64, 0.50) * streak) * (1.0 / (1.0 - fwd));
            }
        }
        // Metal (default) — glossy GGX microfacet over a copper/steel texture and
        // hammered bump; the stochastic lobe is Fresnel-weighted.
        default: {
            let n = bump_normal(p, nf, 0.26);
            let m = surface_material(p);
            let ndotv = max(dot(n, wo), 1.0e-3);
            let f0 = mix(vec3<f32>(0.04), m.albedo, m.metallic);
            let fres = fresnel_schlick(ndotv, f0);
            let sp = clamp(max(fres.x, max(fres.y, fres.z)), 0.15, 0.9);
            if (pick < sp) {
                let h = ggx_sample_h(n, m.rough, r1, r2);
                let wi = reflect(d, h);
                s.terminate = dot(wi, n) <= 0.0;
                let ndotl = max(dot(n, wi), 1.0e-3);
                let ndoth = max(dot(n, h), 1.0e-3);
                let vdoth = max(dot(wo, h), 1.0e-3);
                let a2 = max(m.rough * m.rough, 1.0e-3);
                let g = smith_g1(ndotv, a2) * smith_g1(ndotl, a2);
                // NDF-sampled microfacet estimator: F * G * VdotH / (NdotV * NdotH).
                s.dir = wi;
                s.weight = fres * (g * vdoth / (ndotv * ndoth)) / sp;
                s.origin = p + n * 0.001;
            } else {
                s.dir = cosine_hemisphere(n, r1, r2);
                s.weight = (m.albedo * (1.0 - m.metallic)) / (1.0 - sp);
                s.origin = p + n * 0.001;
            }
        }
    }
    return s;
}

// Trace one full path and return the radiance it gathers, scattering per the
// selected material at each hit until it escapes to the environment (sky) or
// reaches the bounce limit.
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
        let ng = normalize(w0 * fetch_normal(i0) + bc.x * fetch_normal(i1) + bc.y * fetch_normal(i2));

        let sc = scatter(params.material, p, ng, dir, rng);
        if (sc.terminate) {
            break;
        }
        throughput *= sc.weight;
        origin = sc.origin;
        dir = sc.dir;

        // Firefly guard: bound one bounce's throughput so a rare grazing estimator
        // spike can't blow up the deterministic checksum (also keeps preview clean).
        throughput = min(throughput, vec3<f32>(4.0));
    }

    return min(radiance, vec3<f32>(24.0));
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

    // ---- Display shading: ORBITING camera (animated, NOT verified) ----
    // The checksum above traced the FIXED camera with the deterministic per-pixel
    // RNG and has already been written; everything below runs ONLY on the ~60 Hz
    // frames the preview presents (shade==1) and never touches `outbuf`, so the
    // on-screen animation cannot affect verification. It re-traces the scene from
    // a camera that orbits the object (driven by `params.time`) so you can see the
    // whole knot turn — a second full path-trace, i.e. the preview makes the GPU
    // work *harder*, not less.
    if (params.shade != 1u) {
        return;
    }

    let ang = params.time * 0.35;
    let radius = 4.2;
    let disp_pos = vec3<f32>(sin(ang) * radius, 1.7, -cos(ang) * radius);
    let dfwd = normalize(-disp_pos);
    let dright = normalize(cross(dfwd, vec3<f32>(0.0, 1.0, 0.0)));
    let dup = cross(dright, dfwd);
    let dfov = 0.52;

    // Display quality is decoupled from the checksum `--pt-samples` (the stress
    // knob): always render the preview at >=96 spp so the glossy image stays clean,
    // capped so a huge `--pt-samples` can't make the orbit crawl.
    let disp_n = clamp(samples, 96u, 160u);
    var dcol = vec3<f32>(0.0);
    for (var s: u32 = 0u; s < disp_n; s = s + 1u) {
        let jx = randf(&rng);
        let jy = randf(&rng);
        let du = (f32(gx) + jx) * inv_w * 2.0 - 1.0;
        let dv = (f32(gy) + jy) * inv_h * 2.0 - 1.0;
        let ddir = normalize(dfwd + dright * (du * dfov) + dup * (-dv * dfov));
        dcol += trace_path(disp_pos, ddir, &rng);
    }
    dcol = dcol / f32(disp_n);
    var c = dcol / (dcol + vec3<f32>(1.0));
    c = pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    colorbuf[idx] = pack4x8unorm(vec4<f32>(c, 1.0));
}
