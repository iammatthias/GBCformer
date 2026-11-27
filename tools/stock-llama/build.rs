//! Compiles `src/shim.c` against a llama.cpp checkout and links its shared libraries.
//! `LLAMA_CPP_DIR` names the checkout (tag b12070) and `LLAMA_CPP_BUILD` its build
//! directory (default `$LLAMA_CPP_DIR/build-cpu`, a Mac CPU-only build).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/shim.c");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_BUILD");
    let dir = PathBuf::from(std::env::var("LLAMA_CPP_DIR").expect("set LLAMA_CPP_DIR to a llama.cpp checkout at tag b12070"));
    let build = std::env::var("LLAMA_CPP_BUILD").map(PathBuf::from).unwrap_or_else(|_| dir.join("build-cpu"));
    let lib = build.join("bin");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("shim.o");
    let status = Command::new("cc")
        .args(["-O2", "-c", "src/shim.c", "-o"])
        .arg(&obj)
        .arg("-I")
        .arg(dir.join("include"))
        .arg("-I")
        .arg(dir.join("ggml/include"))
        .status()
        .expect("run cc");
    assert!(status.success(), "compiling shim.c failed");
    let status = Command::new("ar").arg("crs").arg(out.join("libshim.a")).arg(&obj).status().expect("run ar");
    assert!(status.success(), "archiving shim.o failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
