use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_PATH");

    let include_dirs = locate_llama();

    let header = include_dirs
        .iter()
        .map(|d| d.join("llama.h"))
        .find(|p| p.exists())
        .expect("llama.h not found in any include directory");

    println!("cargo:rerun-if-changed={}", header.display());

    let mut builder = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .prepend_enum_name(false)
        .derive_default(true);

    for dir in &include_dirs {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }

    let bindings = builder.generate().expect("bindgen failed on llama.h");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("could not write bindings.rs");
}

/// Locate llama include directories and emit link directives.
///
/// Uses pkg-config when available. Falls back to LLAMA_PATH for custom builds.
fn locate_llama() -> Vec<PathBuf> {
    if let Ok(lib) = pkg_config::probe_library("llama") {
        // probe_library already emitted cargo:rustc-link-* directives.
        return lib.include_paths;
    }

    // pkg-config not available or llama not registered — fall back to a
    // manually specified build directory.
    let root = PathBuf::from(
        std::env::var("LLAMA_PATH").expect(
            "pkg-config could not find llama and LLAMA_PATH is not set.\n\
             Install the llama.cpp dev package or set LLAMA_PATH to your build directory.",
        ),
    );

    println!(
        "cargo:rustc-link-search=native={}",
        root.join("build/bin").display()
    );
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    vec![root.join("include"), root.join("ggml/include")]
}
