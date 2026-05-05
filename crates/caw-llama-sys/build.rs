use std::path::PathBuf;

fn main() {
    let llama_root = PathBuf::from(
        std::env::var("LLAMA_PATH")
            .unwrap_or_else(|_| "/home/dennis/src/llama-cpp-turboquant".to_string()),
    );

    let lib_dir = llama_root.join("build/bin");
    let include_dir = llama_root.join("include");
    let ggml_include_dir = llama_root.join("ggml/include");
    let header = include_dir.join("llama.h");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    println!("cargo:rerun-if-env-changed=LLAMA_PATH");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", ggml_include_dir.display()))
        // Only generate items explicitly named llama_* or ggml_*; bindgen will
        // pull in referenced ggml types automatically via allowlist_recursively.
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .prepend_enum_name(false)
        .derive_default(true)
        .generate()
        .expect("bindgen failed to generate llama.h bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("could not write bindings.rs");
}
