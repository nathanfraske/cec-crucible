// SPDX-License-Identifier: MIT
//! Shared procedural geometry for the ray-tracing tests (`rt`, `pathtrace`,
//! `optix`). Kept feature-independent so every RT backend traces the same mesh.

/// A (2,3) torus-knot tube mesh — a striking, self-occluding object with plenty
/// of triangles + depth complexity for BVH stress. Returns (positions, indices,
/// smooth per-vertex normals). Deterministic.
///
/// Built by sweeping a ring of `SIDES` points around the knot curve at `SEGMENTS`
/// stations; the outward tube direction at each point is exactly the smooth
/// surface normal, so no normal averaging is needed.
pub(crate) fn build_torus_knot() -> (Vec<[f32; 3]>, Vec<u32>, Vec<[f32; 3]>) {
    const SEGMENTS: usize = 256;
    const SIDES: usize = 24;
    const P: f32 = 2.0;
    const Q: f32 = 3.0;
    const TUBE: f32 = 0.5;
    const SCALE: f32 = 0.6;
    let tau = std::f32::consts::TAU;

    let curve = |u: f32| -> [f32; 3] {
        let r = 2.0 + (Q * u).cos();
        [r * (P * u).cos(), r * (P * u).sin(), (Q * u).sin()]
    };
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let normz = |a: [f32; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-8);
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    let mut verts = Vec::with_capacity(SEGMENTS * SIDES);
    let mut normals = Vec::with_capacity(SEGMENTS * SIDES);
    let e = 0.002_f32;
    for i in 0..SEGMENTS {
        let u = i as f32 / SEGMENTS as f32 * tau;
        let c = curve(u);
        let cp = curve(u + e);
        let cm = curve(u - e);
        let t = normz(sub(cp, cm));
        let mut nraw = [
            cp[0] - 2.0 * c[0] + cm[0],
            cp[1] - 2.0 * c[1] + cm[1],
            cp[2] - 2.0 * c[2] + cm[2],
        ];
        let d = dot(nraw, t);
        nraw = [nraw[0] - d * t[0], nraw[1] - d * t[1], nraw[2] - d * t[2]];
        let n = normz(nraw);
        let b = cross(t, n);
        for j in 0..SIDES {
            let v = j as f32 / SIDES as f32 * tau;
            let (cv, sv) = (v.cos(), v.sin());
            let dir = [
                cv * n[0] + sv * b[0],
                cv * n[1] + sv * b[1],
                cv * n[2] + sv * b[2],
            ];
            verts.push([
                (c[0] + TUBE * dir[0]) * SCALE,
                (c[1] + TUBE * dir[1]) * SCALE,
                (c[2] + TUBE * dir[2]) * SCALE,
            ]);
            normals.push(dir);
        }
    }

    let mut idx = Vec::with_capacity(SEGMENTS * SIDES * 6);
    for i in 0..SEGMENTS {
        let inext = (i + 1) % SEGMENTS;
        for j in 0..SIDES {
            let jnext = (j + 1) % SIDES;
            let a = (i * SIDES + j) as u32;
            let b = (i * SIDES + jnext) as u32;
            let c = (inext * SIDES + j) as u32;
            let d = (inext * SIDES + jnext) as u32;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (verts, idx, normals)
}

#[cfg(test)]
mod tests {
    #[test]
    fn torus_knot_mesh_is_consistent() {
        let (verts, idx, normals) = super::build_torus_knot();
        assert_eq!(verts.len(), 256 * 24);
        assert_eq!(normals.len(), verts.len());
        assert_eq!(idx.len(), 256 * 24 * 6);
        assert!(idx.iter().all(|&i| (i as usize) < verts.len()));
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal not unit: {len}");
        }
    }
}
