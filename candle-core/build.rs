use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn detect_hip_arch() -> Option<String> {
    if let Ok(arch) = env::var("HIP_ARCH") {
        let trimmed = arch.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }

    let output = Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .find(|token| token.starts_with("gfx"))
        .map(ToOwned::to_owned)
}

fn run(cmd: &mut Command, context: &str) {
    let status = cmd.status().unwrap_or_else(|err| {
        panic!("{context}: failed to start command {:?}: {err}", cmd);
    });
    assert!(
        status.success(),
        "{context}: command {:?} failed with {status}",
        cmd
    );
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/src/qwen35_delta.hip");
    println!("cargo:rerun-if-changed=../candle-kernels/src/qwen35_delta_hip_bridge.cpp");

    if env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let bridge_src = Path::new("../candle-kernels/src/qwen35_delta_hip_bridge.cpp");
    let bridge_obj = out_dir.join("qwen35_delta_hip_bridge.o");
    let bridge_lib = out_dir.join("libqwen35_hip.a");

    let mut hipcc = Command::new("hipcc");
    hipcc
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg("../candle-kernels/src")
        .arg("-x")
        .arg("hip")
        .arg("-c")
        .arg(bridge_src)
        .arg("-o")
        .arg(&bridge_obj);
    if let Some(arch) = detect_hip_arch() {
        hipcc.arg(format!("--offload-arch={arch}"));
    }
    run(&mut hipcc, "building Qwen3.5 HIP bridge");

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&bridge_lib).arg(&bridge_obj);
    run(&mut ar, "archiving Qwen3.5 HIP bridge");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qwen35_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
