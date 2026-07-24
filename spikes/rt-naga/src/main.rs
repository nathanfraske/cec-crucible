// SPDX-License-Identifier: MIT
//! De-risk spike for cec-crucible Build 3 (RT).
//!
//! Question: can naga 29.0.4 (already in the wgpu 29 dependency tree) compile a
//! WGSL ray-query compute shader to Vulkan SPIR-V with NO external shader
//! compiler installed? If yes, the `rt` kernel can ship a `.wgsl` string and
//! compile it at runtime — no glslang/glslc/dxc, no committed `.spv`.
//!
//! Run: cargo run --manifest-path spikes/rt-naga/Cargo.toml

fn main() {
    let src = include_str!("rt.wgsl");

    // 1. Parse WGSL -> naga IR.
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAIL: WGSL parse error:\n{}", e.emit_to_string(src));
            std::process::exit(1);
        }
    };
    println!("parse OK: {} types, {} functions, {} entry points",
        module.types.len(), module.functions.len(), module.entry_points.len());

    // 2. Validate — must allow the RAY_QUERY capability.
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::RAY_QUERY,
    );
    let info = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("FAIL: validation error: {e:?}");
            std::process::exit(1);
        }
    };
    println!("validate OK");

    // 3. Emit SPIR-V. Ray-query needs SPIR-V 1.4 (VK_KHR_ray_query env).
    let mut opts = naga::back::spv::Options::default();
    opts.lang_version = (1, 4);
    let words = match naga::back::spv::write_vec(&module, &info, &opts, None) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("FAIL: SPIR-V emit error: {e:?}");
            std::process::exit(1);
        }
    };

    // 4. Sanity-check the blob: SPIR-V magic + a hunt for the ray-query opcodes
    //    we require (OpTypeRayQueryKHR=4472, OpRayQueryInitializeKHR=4473,
    //    OpRayQueryProceedKHR=4477). The opcode is the low 16 bits of the word.
    const MAGIC: u32 = 0x0723_0203;
    if words.first().copied() != Some(MAGIC) {
        eprintln!("FAIL: bad SPIR-V magic {:#x}", words.first().copied().unwrap_or(0));
        std::process::exit(1);
    }
    let has = |op: u32| words.iter().any(|w| (w & 0xffff) == op);
    let ty = has(4472);
    let init = has(4473);
    let proceed = has(4477);

    println!("SPIR-V OK: {} words ({} bytes), magic {:#x}", words.len(), words.len() * 4, MAGIC);
    println!("  OpTypeRayQueryKHR       present: {ty}");
    println!("  OpRayQueryInitializeKHR present: {init}");
    println!("  OpRayQueryProceedKHR    present: {proceed}");

    // 5. Write the blob out so we can eyeball it / feed a validator later.
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in &words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    if let Err(e) = std::fs::write("rt_naga_spike.spv", &bytes) {
        eprintln!("note: could not write rt_naga_spike.spv: {e}");
    } else {
        println!("wrote rt_naga_spike.spv");
    }

    if ty && init && proceed {
        println!("\nPASS: naga compiled WGSL ray-query -> SPIR-V with ray-query opcodes present.");
    } else {
        eprintln!("\nFAIL: ray-query opcodes missing from emitted SPIR-V.");
        std::process::exit(1);
    }
}
