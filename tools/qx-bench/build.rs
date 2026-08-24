use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/llama_draft_shim.c");
    println!("cargo:rerun-if-env-changed=QX_LLAMA_ROOT");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let llama_root = env::var_os("QX_LLAMA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../external/llama.cpp"));
    let llama_root = llama_root
        .canonicalize()
        .expect("llama.cpp root (set QX_LLAMA_ROOT for Git worktrees)");
    println!(
        "cargo:rerun-if-changed={}",
        llama_root.join("include/llama.h").display()
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let source = manifest_dir.join("src/llama_draft_shim.c");
    let object = out.join("llama_draft_shim.o");
    let archive = out.join("libqx_llama_draft.a");
    let include = llama_root.join("include");
    let ggml_include = llama_root.join("ggml/include");

    let status = Command::new("cc")
        .args(["-c", "-O2", "-fPIC"])
        .arg("-I")
        .arg(&include)
        .arg("-I")
        .arg(&ggml_include)
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("compile llama draft shim");
    assert!(status.success(), "failed to compile llama draft shim");

    let status = Command::new("ar")
        .args(["crus"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("archive llama draft shim");
    assert!(status.success(), "failed to archive llama draft shim");

    let llama_bin = llama_root
        .join("build-metal/bin")
        .canonicalize()
        .expect("llama.cpp build directory");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=qx_llama_draft");
    println!("cargo:rustc-link-search=native={}", llama_bin.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_bin.display());
}
