// SPDX-License-Identifier: MIT
//
// OptiX device code for the NVIDIA-native path-tracing test (Phase 2). This is
// the CUDA/OptiX counterpart of `pathtrace.wgsl`: a deterministic multi-bounce
// Monte-Carlo megakernel. The raygen program runs the whole path loop (iterative,
// not recursive — OptiX only guarantees trace-depth 1); closest-hit records the
// surface into the per-ray-data and samples the next bounce; miss folds in the
// environment and ends the path.
//
// ## Determinism (same rule as every cec-crucible GPU test)
//
// One thread per pixel loops all samples x bounces, RNG seeded PURELY from the
// pixel index (+ a fixed base seed), radiance accumulated in registers and
// written once. No atomics, no time-varying input. The accumulated radiance is
// folded (bit-exact) into a per-pixel FNV checksum for same-device
// self-consistency — identical to the Vulkan `pathtrace`.
//
// Compiled to PTX by nvcc at build (against the public NVIDIA/optix-dev headers);
// the committed .ptx is JIT-linked by the driver at runtime, so the target needs
// no toolkit.

#include <optix.h>

// ---- minimal float3 vector math (nvcc's <vector_types.h> has no operators) ----
__forceinline__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__forceinline__ __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__forceinline__ __device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__forceinline__ __device__ float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__forceinline__ __device__ float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
__forceinline__ __device__ float3& operator*=(float3& a, float3 b) { a = a * b; return a; }
__forceinline__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__forceinline__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__forceinline__ __device__ float3 normalize(float3 a) {
    float inv = rsqrtf(fmaxf(dot(a, a), 1e-20f));
    return a * inv;
}
__forceinline__ __device__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__forceinline__ __device__ float3 mix3(float3 a, float3 b, float t) { return a * (1.0f - t) + b * t; }

// ---- launch parameters (set by the host per launch) ----
struct Params {
    unsigned int* checksum;   // per-pixel FNV over the accumulated radiance (verified)
    unsigned int* image;      // per-pixel packed RGBA8 for the preview (written if shade)
    float3* normals;          // per-vertex smooth normals
    uint3* indices;           // per-triangle vertex indices
    OptixTraversableHandle handle; // the TLAS
    unsigned int width, height;
    unsigned int samples, bounces;
    unsigned int seed, shade;
    float3 cam_pos, cam_fwd, cam_right, cam_up;
    float fov;
};

extern "C" __constant__ Params params;

// Per-ray data carried across each bounce.
struct PRD {
    float3 throughput;
    float3 next_origin;
    float3 next_dir;
    float3 radiance;
    unsigned int seed;
    int done;
};

// ---- deterministic RNG (PCG-RXS-M-XS), same construction as pathtrace.wgsl ----
__forceinline__ __device__ unsigned int pcg(unsigned int& s) {
    unsigned int old = s;
    s = old * 747796405u + 2891336453u;
    unsigned int word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}
__forceinline__ __device__ float randf(unsigned int& s) { return (float)pcg(s) * (1.0f / 4294967296.0f); }

// Environment: sky gradient + a bright sharp sun (the only light).
__forceinline__ __device__ float3 sky(float3 dir) {
    float t = clampf(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 base = mix3(make_float3(0.35f, 0.42f, 0.55f), make_float3(0.70f, 0.82f, 1.00f), t);
    float3 sun = normalize(make_float3(0.45f, 0.75f, -0.30f));
    float s = powf(fmaxf(dot(dir, sun), 0.0f), 220.0f) * 9.0f;
    return base * 1.15f + make_float3(s, s, s);
}

// Cosine-weighted hemisphere sample about n.
__forceinline__ __device__ float3 cosine_hemisphere(float3 n, float r1, float r2) {
    float3 a = (fabsf(n.y) < 0.99f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 t = normalize(cross(a, n));
    float3 b = cross(n, t);
    float phi = 6.2831853f * r1;
    float r = sqrtf(r2);
    float z = sqrtf(fmaxf(0.0f, 1.0f - r2));
    return normalize(t * (r * cosf(phi)) + b * (r * sinf(phi)) + n * z);
}

// ---- payload <-> PRD pointer packing (2 x 32-bit payload registers) ----
__forceinline__ __device__ PRD* getPRD() {
    unsigned long long u = ((unsigned long long)optixGetPayload_0() << 32) | optixGetPayload_1();
    return reinterpret_cast<PRD*>(u);
}

extern "C" __global__ void __raygen__pt() {
    uint3 li = optixGetLaunchIndex();
    unsigned int pixel = li.y * params.width + li.x;
    unsigned int seed = (pixel * 0x9e3779b9u) ^ (params.seed * 0x85ebca6bu) ^ 0xdeadbeefu;

    float inv_w = 1.0f / (float)params.width;
    float inv_h = 1.0f / (float)params.height;

    float3 radiance = make_float3(0, 0, 0);
    unsigned int samples = params.samples < 1u ? 1u : params.samples;
    for (unsigned int s = 0; s < samples; ++s) {
        float jx = randf(seed);
        float jy = randf(seed);
        float u = ((float)li.x + jx) * inv_w * 2.0f - 1.0f;
        float v = ((float)li.y + jy) * inv_h * 2.0f - 1.0f;

        PRD prd;
        prd.throughput = make_float3(1, 1, 1);
        prd.radiance = make_float3(0, 0, 0);
        prd.next_origin = params.cam_pos;
        prd.next_dir = normalize(params.cam_fwd + params.cam_right * (u * params.fov) + params.cam_up * (-v * params.fov));
        prd.seed = seed;
        prd.done = 0;

        unsigned long long p = (unsigned long long)&prd;
        unsigned int u0 = (unsigned int)(p >> 32), u1 = (unsigned int)(p & 0xffffffff);
        for (unsigned int b = 0; b < params.bounces; ++b) {
            optixTrace(params.handle, prd.next_origin, prd.next_dir,
                       0.001f, 100.0f, 0.0f, OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);
            if (prd.done) break;
        }
        radiance += prd.radiance;
        seed = prd.seed;
    }
    radiance = radiance * (1.0f / (float)samples);

    // Checksum: FNV over the bitcast radiance (deterministic -> self-consistent).
    unsigned int h = pixel * 2654435761u;
    h = (h ^ __float_as_uint(radiance.x)) * 16777619u;
    h = (h ^ __float_as_uint(radiance.y)) * 16777619u;
    h = (h ^ __float_as_uint(radiance.z)) * 16777619u;
    params.checksum[pixel] = h;

    if (params.shade) {
        // Reinhard tonemap + gamma, pack RGBA8 (R low byte).
        float3 c = radiance * make_float3(1, 1, 1);
        c = make_float3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
        float rr = powf(clampf(c.x, 0.0f, 1.0f), 1.0f / 2.2f);
        float gg = powf(clampf(c.y, 0.0f, 1.0f), 1.0f / 2.2f);
        float bb = powf(clampf(c.z, 0.0f, 1.0f), 1.0f / 2.2f);
        unsigned int ri = (unsigned int)(rr * 255.0f + 0.5f);
        unsigned int gi = (unsigned int)(gg * 255.0f + 0.5f);
        unsigned int bi = (unsigned int)(bb * 255.0f + 0.5f);
        params.image[pixel] = ri | (gi << 8) | (bi << 16) | (0xffu << 24);
    }
}

extern "C" __global__ void __miss__pt() {
    PRD* prd = getPRD();
    prd->radiance += prd->throughput * sky(optixGetWorldRayDirection());
    prd->done = 1;
}

extern "C" __global__ void __closesthit__pt() {
    PRD* prd = getPRD();
    unsigned int prim = optixGetPrimitiveIndex();
    float2 bc = optixGetTriangleBarycentrics();
    float t = optixGetRayTmax();
    float3 ro = optixGetWorldRayOrigin();
    float3 rd = optixGetWorldRayDirection();
    float3 p = ro + rd * t;

    uint3 tri = params.indices[prim];
    float w0 = 1.0f - bc.x - bc.y;
    float3 n = normalize(params.normals[tri.x] * w0 + params.normals[tri.y] * bc.x + params.normals[tri.z] * bc.y);
    if (dot(n, rd) > 0.0f) n = n * -1.0f;

    // Diffuse (Lambertian) bounce; tinted albedo so colour-bleeding shows.
    float3 albedo = make_float3(0.72f, 0.62f, 0.55f);
    float r1 = randf(prd->seed);
    float r2 = randf(prd->seed);
    prd->next_dir = cosine_hemisphere(n, r1, r2);
    prd->next_origin = p + n * 0.001f;
    prd->throughput *= albedo;
}
