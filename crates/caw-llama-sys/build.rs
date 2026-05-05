use std::path::PathBuf;

fn main() {
    let (header, extra_include) = find_llama();

    // Emit link directives. For system installs the library lives in a
    // standard linker search path so no rustc-link-search is needed.
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    println!("cargo:rerun-if-env-changed=LLAMA_PATH");
    println!("cargo:rerun-if-changed={}", header.display());

    let mut builder = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .prepend_enum_name(false)
        .derive_default(true);

    for dir in &extra_include {
        builder = builder.clang_arg(format!("-I{}", dir.display()));
    }

    let bindings = builder.generate().expect("bindgen failed on llama.h");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("could not write bindings.rs");
}

/// Locate llama.h and any extra include directories needed to parse it.
///
/// Search order:
///   1. `/usr/include/llama.h`  — system .deb / package install (no extra
///      link-search or include paths needed since they are standard).
///   2. `$LLAMA_PATH`           — custom build directory. Emits
///      `rustc-link-search` so the non-standard .so location is found.
fn find_llama() -> (PathBuf, Vec<PathBuf>) {
    // 1. System install
    let sys_header = PathBuf::from("/usr/include/llama.h");
    if sys_header.exists() {
        // /usr/include is a standard clang search path; no -I needed.
        return (sys_header, vec![]);
    }

    // 2. Custom build via LLAMA_PATH
    let llama_root = PathBuf::from(
        std::env::var("LLAMA_PATH").expect(
            "llama.h not found in /usr/include and LLAMA_PATH is not set.\n\
             Install the llama.cpp dev package or point LLAMA_PATH at your build directory.",
        ),
    );

    let lib_dir = llama_root.join("build/bin");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let header = llama_root.join("include/llama.h");
    let extra = vec![
        llama_root.join("include"),
        llama_root.join("ggml/include"),
    ];
    (header, extra)
}
