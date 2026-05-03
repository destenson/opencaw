//! Runtime discovery of the ONNX Runtime shared library and its cuDNN
//! sidecars. Shared between the direct `ort`-using provider (`onnx_provider`)
//! and `fastembed`, which transitively loads the same `libonnxruntime.so`
//! via its own `ort` dep compiled with `load-dynamic`.
//!
//! Any embedder backed by ort must call [`ensure_ort_dylib_path`] before
//! constructing its session — otherwise ort panics with "cannot open shared
//! object file: libonnxruntime.so" on systems without a system-wide install.

use std::path::{Path, PathBuf};

/// If `ORT_DYLIB_PATH` isn't already set, search the uv / pip wheel cache
/// for a CUDA-capable onnxruntime and point ort at it. Falls back silently
/// (ort itself will surface the resulting "library not found" error with
/// a more actionable message than we could).
///
/// Priority order:
///   1. User-set `ORT_DYLIB_PATH` — always wins.
///   2. Newest CUDA-capable onnxruntime.so in `~/.cache/uv/archive-v0`
///      (what `uv pip install onnxruntime-gpu` produces).
///   3. Newest CPU-only onnxruntime.so in the same cache (last resort —
///      loses GPU but at least the program runs).
///
/// Idempotent: subsequent calls are no-ops once `ORT_DYLIB_PATH` is set,
/// so it's safe for every ort-using provider to call at construction time.
pub fn ensure_ort_dylib_path() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }

    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return,
    };
    let caches = [
        format!("{}/.cache/uv/archive-v0", home),
        format!("{}/.local/share/uv/archive-v0", home),
    ];

    let mut cuda_candidates: Vec<(PathBuf, Version)> = Vec::new();
    let mut cpu_candidates: Vec<(PathBuf, Version)> = Vec::new();

    for root in &caches {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            continue;
        }
        for archive in walk_one_level(root_path) {
            let capi = archive.join("onnxruntime").join("capi");
            if !capi.is_dir() {
                continue;
            }
            let has_cuda = capi.join("libonnxruntime_providers_cuda.so").exists();
            for entry in std::fs::read_dir(&capi).into_iter().flatten().flatten() {
                let path = entry.path();
                let fname = match path.file_name().and_then(|s| s.to_str()) {
                    Some(f) => f,
                    None => continue,
                };
                if let Some(v) = fname
                    .strip_prefix("libonnxruntime.so.")
                    .or_else(|| fname.strip_prefix("libonnxruntime.dylib."))
                    .and_then(|rest| Version::parse(rest))
                {
                    if has_cuda {
                        cuda_candidates.push((path, v));
                    } else {
                        cpu_candidates.push((path, v));
                    }
                }
            }
        }
    }

    let cuda_pick = cuda_candidates.into_iter().max_by(|a, b| a.1.cmp(&b.1));
    let cpu_pick = cpu_candidates.into_iter().max_by(|a, b| a.1.cmp(&b.1));

    // Prefer CUDA builds, but only if we can also find the sidecar libs
    // they need (cuDNN, NCCL, etc.) — otherwise dlopen of the CUDA
    // provider will fail and ort silently falls back to CPU. Fall back
    // to the CPU build in that case rather than pretending we have GPU.
    let pick = if let Some((cuda_path, cuda_version)) = cuda_pick {
        if let Some(extra_lib_dirs) = locate_cuda_sidecars(&home) {
            // Set LD_LIBRARY_PATH too — it won't help dlopen in THIS process
            // (glibc caches the search path at startup), but it propagates
            // to any child processes we exec and is harmless here.
            prepend_ld_library_path(&extra_lib_dirs);
            // Preload cuDNN with absolute paths. This is what actually makes
            // ort find cuDNN: once `libcudnn*.so.9` are in the global
            // namespace, ort's later dlopen of the CUDA provider resolves
            // their DT_NEEDED entries against the already-loaded libs
            // instead of going through ld.so's broken search path.
            let n = preload_cuda_sidecars(&extra_lib_dirs);
            eprintln!(
                "onnx: preloaded {} cuDNN libs from {}",
                n,
                extra_lib_dirs[0].display()
            );
            Some((cuda_path, cuda_version))
        } else {
            eprintln!(
                "onnx: CUDA onnxruntime {} found but cuDNN 9 not located; \
                 falling back to CPU build. Install with: \
                 uv pip install nvidia-cudnn-cu12",
                cuda_version
            );
            cpu_pick
        }
    } else {
        cpu_pick
    };

    if let Some((path, version)) = pick {
        eprintln!(
            "onnx: auto-detected onnxruntime {} at {}",
            version,
            path.display()
        );
        // SAFETY: single-threaded at init time, before any ort call.
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &path);
        }
    } else {
        eprintln!(
            "onnx: no onnxruntime shared library found in uv/pip cache.\n\
             Install one with: uv pip install --system onnxruntime-gpu\n\
             (or set ORT_DYLIB_PATH to your onnxruntime .so explicitly)."
        );
    }
}

/// Find one consistent cuDNN install directory. Python's nvidia-cudnn-cu12
/// wheel lays `libcudnn.so.9` plus its siblings (libcudnn_ops.so.9, etc.)
/// in the same directory, so one dir is enough — we don't want to merge
/// cuDNN files from multiple cache entries because different pip installs
/// can have subtly different cuDNN patch versions whose co-loading
/// segfaults.
///
/// Picks the directory with the largest set of cuDNN siblings (better
/// chance of being a complete install). Returns the chosen dir wrapped in
/// a vec so callers can still merge in additional dirs (nvrtc etc.)
/// without a signature change.
fn locate_cuda_sidecars(home: &str) -> Option<Vec<PathBuf>> {
    let cache_roots = [
        format!("{}/.cache/uv/archive-v0", home),
        format!("{}/.local/share/uv/archive-v0", home),
    ];

    let mut best: Option<(PathBuf, usize)> = None;

    for root in &cache_roots {
        let root_path = Path::new(root);
        if !root_path.is_dir() {
            continue;
        }
        for archive in walk_one_level(root_path) {
            let cudnn_lib = archive.join("nvidia").join("cudnn").join("lib");
            if !cudnn_lib.is_dir() {
                continue;
            }
            if !cudnn_lib.join("libcudnn.so.9").exists() {
                continue;
            }
            // Count cuDNN siblings as a completeness heuristic.
            let siblings = std::fs::read_dir(&cudnn_lib)
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("libcudnn"))
                .count();
            if best.as_ref().map(|(_, n)| siblings > *n).unwrap_or(true) {
                best = Some((cudnn_lib, siblings));
            }
        }
    }
    best.map(|(d, _)| vec![d])
}

/// Explicitly dlopen the cuDNN libraries so their symbols are in the global
/// namespace before ort loads `libonnxruntime_providers_cuda.so`. This is
/// the ONLY reliable way to make ort find cuDNN when cuDNN lives in a
/// non-default path: setting `LD_LIBRARY_PATH` from within the process is
/// a no-op for subsequent `dlopen`, because glibc's dynamic linker caches
/// its search path at process startup and ignores later env-var changes.
///
/// Why preload every `libcudnn*.so.9` in the chosen dir rather than just
/// the top-level one: the sibling libs (`libcudnn_ops.so.9`, `_cnn.so.9`,
/// `_graph.so.9`, ...) are pulled in via DT_NEEDED by either cuDNN itself
/// or by the CUDA EP, and their lookup would also hit the broken
/// LD_LIBRARY_PATH path. Loading each sibling by absolute path bypasses
/// ld.so's search entirely, so it doesn't matter that LD_LIBRARY_PATH is
/// ignored.
///
/// Handles are `mem::forget`ted on purpose; they must stay alive for the
/// whole process. Returns the number of libraries loaded.
fn preload_cuda_sidecars(dirs: &[PathBuf]) -> usize {
    let mut loaded = 0usize;
    for dir in dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let fname = match path.file_name().and_then(|s| s.to_str()) {
                Some(f) => f,
                None => continue,
            };
            if !fname.starts_with("libcudnn") || !fname.ends_with(".so.9") {
                continue;
            }
            unsafe {
                match libloading::Library::new(&path) {
                    Ok(lib) => {
                        std::mem::forget(lib);
                        loaded += 1;
                    }
                    Err(e) => {
                        eprintln!("onnx: preload {} failed: {}", path.display(), e);
                    }
                }
            }
        }
    }
    loaded
}

fn prepend_ld_library_path(dirs: &[PathBuf]) {
    let extra = dirs
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(":");
    let new_val = match std::env::var_os("LD_LIBRARY_PATH") {
        Some(existing) => format!("{}:{}", extra, existing.to_string_lossy()),
        None => extra,
    };
    // SAFETY: single-threaded at init time, before ort dlopens anything.
    unsafe {
        std::env::set_var("LD_LIBRARY_PATH", new_val);
    }
}

fn walk_one_level(root: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect()
}

/// Tiny lexicographic version comparator for dotted numeric versions.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Version(Vec<u32>);

impl Version {
    fn parse(s: &str) -> Option<Self> {
        let parts: Result<Vec<u32>, _> = s.split('.').map(|p| p.parse::<u32>()).collect();
        parts.ok().map(Version)
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: Vec<String> = self.0.iter().map(|p| p.to_string()).collect();
        f.write_str(&s.join("."))
    }
}
