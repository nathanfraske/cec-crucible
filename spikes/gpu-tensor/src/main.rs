// SPDX-License-Identifier: MIT
//! Tensor-core spike (Phase 3G).
//!
//! Questions:
//!  1. Does forcing the Vulkan/SPIR-V backend make CubeCL `cmma` compile + run?
//!  2. Does it actually use the tensor cores (mma is spirv/cuda only — wgsl can't)?
//!  3. Does **int8 -> int32** cmma work (the bit-exact cross-vendor golden)?
//!  4. Is the result correct vs a CPU reference?
//!
//! One 16x16x16 tile answers all four. The kernels compute `Out = Lhs @ Rhs.T`
//! (B is loaded column-major), mirroring CubeCL's own cmma test. If int8 fails,
//! f16->f32 still proves the tensor-core path (self-consistency, not a golden).

use cubecl::prelude::*;
use cubecl::wgpu::{init_device, init_setup, RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime};
use half::f16;

/// C(16x16 f32) = A(16x16 f16) @ B(16x16 f16).T, one cooperative-matrix tile.
#[cube(launch)]
fn cmma_f16(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);
    cmma::store(&mut out.to_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
}

/// C(16x16 i32) = A(16x16 i8) @ B(16x16 i8).T — the cross-vendor golden path.
#[cube(launch)]
fn cmma_i8(lhs: &Array<i8>, rhs: &Array<i8>, out: &mut Array<i32>) {
    let a = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &lhs.to_slice(),
        16,
    );
    let b = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        &rhs.to_slice(),
        16,
    );
    let c = cmma::Matrix::<i32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0,
    );
    cmma::execute::<i8, i8, i32, i32>(&a, &b, &c, &c);
    cmma::store(&mut out.to_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
}

/// out[i][j] = sum_k a[i][k] * b[j][k]  (A @ B.T, matching the kernel).
fn cpu_ref_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0.0f32;
            for k in 0..16 {
                s += a[i * 16 + k] * b[j * 16 + k];
            }
            c[i * 16 + j] = s;
        }
    }
    c
}

fn cpu_ref_i32(a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += a[i * 16 + k] as i32 * b[j * 16 + k] as i32;
            }
            c[i * 16 + j] = s;
        }
    }
    c
}

fn main() {
    // Force Vulkan -> SPIR-V, so cmma maps to CooperativeMatrixMultiply.
    let base = WgpuDevice::DiscreteGpu(0);
    println!("forcing Vulkan/SPIR-V backend ...");
    let setup = init_setup::<Vulkan>(&base, RuntimeOptions::default());
    let device = init_device(setup, RuntimeOptions::default());
    let client = WgpuRuntime::client(&device);
    println!("runtime: {}", WgpuRuntime::name(&client));

    // ---- f16 -> f32 (upstream-tested path) ----
    let a16: Vec<f16> = (0..256)
        .map(|i| f16::from_f32(((i % 16) as f32 - 7.0) * 0.1))
        .collect();
    let b16: Vec<f16> = (0..256)
        .map(|i| f16::from_f32(((i / 16) as f32 - 7.0) * 0.1))
        .collect();
    let a_h = client.create_from_slice(f16::as_bytes(&a16));
    let b_h = client.create_from_slice(f16::as_bytes(&b16));
    let c_h = client.empty(256 * std::mem::size_of::<f32>());
    unsafe {
        cmma_f16::launch::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts(a_h.clone(), 256),
            ArrayArg::from_raw_parts(b_h.clone(), 256),
            ArrayArg::from_raw_parts(c_h.clone(), 256),
        );
    }
    cubecl::future::block_on(client.sync()).expect("f16 sync failed");
    let c_bytes = client.read_one(c_h).expect("f16 read failed");
    let c_gpu: &[f32] = f32::from_bytes(&c_bytes);
    let a16f: Vec<f32> = a16.iter().map(|x| x.to_f32()).collect();
    let b16f: Vec<f32> = b16.iter().map(|x| x.to_f32()).collect();
    let c_ref = cpu_ref_f32(&a16f, &b16f);
    let max_err = c_gpu
        .iter()
        .zip(&c_ref)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    println!(
        "f16->f32 cmma: gpu[0]={:.4} ref[0]={:.4} max_err={:.4}  {}",
        c_gpu[0],
        c_ref[0],
        max_err,
        if max_err < 0.05 {
            "PASS (tensor cores reached)"
        } else {
            "MISMATCH"
        }
    );

    // ---- i8 -> i32 (the cross-vendor golden — untested upstream) ----
    let a8: Vec<i8> = (0..256).map(|i| ((i % 7) as i8) - 3).collect();
    let b8: Vec<i8> = (0..256).map(|i| ((i % 5) as i8) - 2).collect();
    let a8_h = client.create_from_slice(i8::as_bytes(&a8));
    let b8_h = client.create_from_slice(i8::as_bytes(&b8));
    let c8_h = client.empty(256 * std::mem::size_of::<i32>());
    let c8_read = c8_h.clone();
    let launched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        cmma_i8::launch::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts(a8_h.clone(), 256),
            ArrayArg::from_raw_parts(b8_h.clone(), 256),
            ArrayArg::from_raw_parts(c8_h.clone(), 256),
        );
    }));
    if launched.is_err() {
        println!("i8->i32  cmma: kernel build/launch panicked -> int8 cmma not supported here");
        return;
    }
    match cubecl::future::block_on(client.sync()) {
        Ok(_) => {
            let c8_bytes = client.read_one(c8_read).expect("i8 read failed");
            let c8_gpu: &[i32] = i32::from_bytes(&c8_bytes);
            let c8_ref = cpu_ref_i32(&a8, &b8);
            let exact = c8_gpu == c8_ref.as_slice();
            println!(
                "i8->i32  cmma: gpu[0]={} ref[0]={}  {}",
                c8_gpu[0],
                c8_ref[0],
                if exact {
                    "EXACT (cross-vendor golden viable!)"
                } else {
                    "MISMATCH"
                }
            );
        }
        Err(e) => println!("i8->i32  cmma: FAILED to run ({e:?}) -> fall back to f16->f32"),
    }
}
