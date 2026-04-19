//! ONNX Runtime embedding provider, with optional CUDA execution provider.
//!
//! Why this exists alongside `CandleEmbeddingProvider`: candle's eager
//! evaluation means each tensor op on CUDA incurs a kernel-launch and
//! synchronization cost. For BGE-small the pooling/normalize pipeline
//! involves ~10 ops, which serializes at ~600ms per sub-batch even though
//! the underlying forward pass is ~4ms. ONNX Runtime compiles the whole
//! model graph and executes it as a single kernel stream — typically an
//! order of magnitude faster on the same hardware.
//!
//! With `ort/cuda` enabled and a CUDA-capable onnxruntime available, the
//! provider tries the CUDA execution provider first and falls back to CPU
//! if registration fails. The fallback is visible in a log line so you
//! know which path is actually running.

use caw_core::{CawError, CawResult, EmbeddingProvider};
use hf_hub::api::sync::Api;
use ndarray::Ix2;
use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::TensorRef};
use std::path::Path;
use tokenizers::Tokenizer;

/// BGE-style position-embedding cap. Sequences longer than this are
/// truncated at tokenize time; the model would OOB the positional table
/// otherwise.
const MAX_SEQ_LEN: usize = 512;

/// Pre-tokenization character clip. BPE is superlinear on pathological
/// inputs, and only the first ~2 KB of a text contributes to a 512-token
/// BERT window anyway. Matches the candle provider's clip.
const MAX_TEXT_CHARS: usize = 2_500;

/// BGE query prefix for asymmetric retrieval. Matches the model card.
const BGE_QUERY_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

pub struct OnnxEmbeddingProvider {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
    /// True for BGE / E5 / any encoder trained with distinct query/doc
    /// prompts. False for symmetric models (sentence-t5 etc.).
    asymmetric: bool,
    /// Human-readable backend name ("onnx-cuda" or "onnx-cpu"), logged
    /// once at init and reported by `provider_name()`.
    backend_name: &'static str,
}

impl OnnxEmbeddingProvider {
    /// Load an ONNX embedding model from a local file path.
    /// Expects a `tokenizer.json` in the same directory as the model file.
    /// Tries the CUDA execution provider first and falls back to CPU.
    pub fn from_path(model_path: &str, dimension: usize) -> CawResult<Self> {
        let model_dir = Path::new(model_path)
            .parent()
            .ok_or_else(|| CawError::Embedding("Cannot determine model directory".to_string()))?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(CawError::Embedding(format!(
                "tokenizer.json not found in {}",
                model_dir.display()
            )));
        }
        Self::from_paths(model_path, tokenizer_path.to_str().unwrap(), dimension, true)
    }

    /// Load with explicit paths and a flag for asymmetric prefix handling.
    pub fn from_paths(
        model_path: &str,
        tokenizer_path: &str,
        dimension: usize,
        asymmetric: bool,
    ) -> CawResult<Self> {
        let (session, backend_name) = build_session(model_path)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self {
            session,
            tokenizer,
            dimension,
            asymmetric,
            backend_name,
        })
    }

    /// Convenience constructor for BAAI/bge-small-en-v1.5. Downloads the
    /// ONNX model and tokenizer from HuggingFace Hub on first use; cached
    /// thereafter in `~/.cache/huggingface`.
    pub fn bge_small() -> CawResult<Self> {
        let api =
            Api::new().map_err(|e| CawError::Embedding(format!("HF Hub init failed: {e}")))?;
        let repo = api.model("BAAI/bge-small-en-v1.5".to_string());
        // BGE publishes the ONNX export under `onnx/model.onnx` in the
        // same repo as the PyTorch weights. `tokenizer.json` sits at the
        // root alongside `config.json`.
        let model_path = repo
            .get("onnx/model.onnx")
            .map_err(|e| CawError::Embedding(format!("Failed to download onnx/model.onnx: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| CawError::Embedding(format!("Failed to download tokenizer.json: {e}")))?;
        let model_str = model_path.to_string_lossy().into_owned();
        let tok_str = tokenizer_path.to_string_lossy().into_owned();
        Self::from_paths(&model_str, &tok_str, 384, true)
    }

    fn prefixed_texts(&self, texts: &[&str], is_query: bool) -> Vec<String> {
        if !self.asymmetric {
            return texts.iter().map(|s| clip(s)).collect();
        }
        if is_query {
            texts
                .iter()
                .map(|s| format!("{}{}", BGE_QUERY_PREFIX, clip(s)))
                .collect()
        } else {
            // BGE v1.5 uses no prefix on the passage side — only the query
            // gets the instruction prompt.
            texts.iter().map(|s| clip(s)).collect()
        }
    }

    fn run_embed(&mut self, texts: Vec<String>) -> CawResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| CawError::Embedding(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);
        if padded_len == 0 {
            return Ok(vec![vec![0.0; self.dimension]; batch_size]);
        }

        let mut all_ids: Vec<i64> = Vec::with_capacity(batch_size * padded_len);
        let mut all_mask: Vec<i64> = Vec::with_capacity(batch_size * padded_len);
        let mut all_type_ids: Vec<i64> = Vec::with_capacity(batch_size * padded_len);

        for enc in &encodings {
            let full_ids = enc.get_ids();
            let full_mask = enc.get_attention_mask();
            let full_type_ids = enc.get_type_ids();
            let take = full_ids.len().min(MAX_SEQ_LEN);
            all_ids.extend(full_ids.iter().take(take).map(|&x| x as i64));
            all_mask.extend(full_mask.iter().take(take).map(|&x| x as i64));
            all_type_ids.extend(full_type_ids.iter().take(take).map(|&x| x as i64));
            for _ in take..padded_len {
                all_ids.push(0);
                all_mask.push(0);
                all_type_ids.push(0);
            }
        }

        let shape = [batch_size, padded_len];
        let ids_tensor = TensorRef::from_array_view((shape, &*all_ids))
            .map_err(|e| CawError::Embedding(format!("create input_ids tensor: {e}")))?;
        let mask_tensor = TensorRef::from_array_view((shape, &*all_mask))
            .map_err(|e| CawError::Embedding(format!("create attention_mask tensor: {e}")))?;
        let type_tensor = TensorRef::from_array_view((shape, &*all_type_ids))
            .map_err(|e| CawError::Embedding(format!("create token_type_ids tensor: {e}")))?;

        // BGE's ONNX export expects (input_ids, attention_mask, token_type_ids).
        // Named inputs are more robust than positional across model versions.
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor,
            ])
            .map_err(|e| CawError::Embedding(format!("ONNX inference failed: {e}")))?;

        // Prefer a named `sentence_embedding` output if the export has
        // pooled embeddings baked in; otherwise fall back to the first
        // output and CLS-pool + L2 normalize ourselves.
        if let Some(pooled) = outputs.get("sentence_embedding") {
            let arr = pooled
                .try_extract_array::<f32>()
                .map_err(|e| CawError::Embedding(format!("extract pooled: {e}")))?
                .into_dimensionality::<Ix2>()
                .map_err(|e| CawError::Embedding(format!("pooled shape: {e}")))?;
            let mut result: Vec<Vec<f32>> = arr.rows().into_iter().map(|r| r.to_vec()).collect();
            for row in &mut result {
                l2_normalize_in_place(row);
            }
            return Ok(result);
        }

        // Fall back to last_hidden_state: [batch, seq, hidden]. Take the
        // first token (CLS) and normalize — matches BGE's training.
        let first = outputs
            .iter()
            .next()
            .ok_or_else(|| CawError::Embedding("model produced no outputs".into()))?
            .1;
        let hidden = first
            .try_extract_array::<f32>()
            .map_err(|e| CawError::Embedding(format!("extract hidden: {e}")))?;
        let shape = hidden.shape();
        if shape.len() != 3 {
            return Err(CawError::Embedding(format!(
                "expected [batch, seq, hidden] output, got shape {shape:?}"
            )));
        }
        let (b, _seq, h) = (shape[0], shape[1], shape[2]);
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(b);
        for i in 0..b {
            let mut row = Vec::with_capacity(h);
            for j in 0..h {
                row.push(hidden[[i, 0, j]]);
            }
            l2_normalize_in_place(&mut row);
            result.push(row);
        }
        Ok(result)
    }
}

fn build_session(model_path: &str) -> CawResult<(Session, &'static str)> {
    // ort is compiled with `load-dynamic`, so it looks up the actual
    // onnxruntime shared library at runtime via ORT_DYLIB_PATH. If the
    // caller hasn't set it, auto-discover a CUDA-capable build from the
    // uv / pip cache — most users running this already have onnxruntime-gpu
    // installed via `uv pip install onnxruntime-gpu` or similar.
    ensure_ort_dylib_path();

    // Try CUDA first. If the CUDA EP fails to register (missing lib,
    // incompatible CUDA version, no device), fall back to CPU. Emit a
    // log line either way so the active backend is visible.
    let cuda_ep = CUDAExecutionProvider::default();
    match Session::builder()
        .and_then(|b| b.with_execution_providers([cuda_ep.build()]))
        .and_then(|b| b.commit_from_file(model_path))
    {
        Ok(session) => {
            eprintln!("onnx: using CUDA execution provider");
            Ok((session, "onnx-cuda"))
        }
        Err(err) => {
            eprintln!(
                "onnx: CUDA EP unavailable ({err}); falling back to CPU execution provider"
            );
            let session = Session::builder()
                .map_err(|e| CawError::Embedding(format!("session builder: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| CawError::Embedding(format!("load model CPU: {e}")))?;
            Ok((session, "onnx-cpu"))
        }
    }
}

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
fn ensure_ort_dylib_path() {
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

    let mut cuda_candidates: Vec<(std::path::PathBuf, Version)> = Vec::new();
    let mut cpu_candidates: Vec<(std::path::PathBuf, Version)> = Vec::new();

    for root in &caches {
        let root_path = std::path::Path::new(root);
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
                if let Some(rest) = fname
                    .strip_prefix("libonnxruntime.so.")
                    .or_else(|| fname.strip_prefix("libonnxruntime.dylib."))
                {
                    if let Some(v) = Version::parse(rest) {
                        if has_cuda {
                            cuda_candidates.push((path, v));
                        } else {
                            cpu_candidates.push((path, v));
                        }
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
            prepend_ld_library_path(&extra_lib_dirs);
            eprintln!(
                "onnx: LD_LIBRARY_PATH += {}",
                extra_lib_dirs[0].display()
            );
            // No preload: loading cuDNN manually races against ort's own
            // dlopen of libonnxruntime_providers_cuda.so and the cuDNN
            // version baked into that provider's build. Letting ld.so
            // resolve cuDNN on-demand via LD_LIBRARY_PATH matches how
            // onnxruntime is normally loaded in a Python env and avoids
            // segfaults we've seen from co-loading conflicting patch
            // versions.
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
fn locate_cuda_sidecars(home: &str) -> Option<Vec<std::path::PathBuf>> {
    let cache_roots = [
        format!("{}/.cache/uv/archive-v0", home),
        format!("{}/.local/share/uv/archive-v0", home),
    ];

    let mut best: Option<(std::path::PathBuf, usize)> = None;

    for root in &cache_roots {
        let root_path = std::path::Path::new(root);
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
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("libcudnn")
                })
                .count();
            if best.as_ref().map(|(_, n)| siblings > *n).unwrap_or(true) {
                best = Some((cudnn_lib, siblings));
            }
        }
    }
    best.map(|(d, _)| vec![d])
}

/// Explicitly dlopen the CUDA sidecar libraries so their symbols are in
/// the process namespace before ort loads the CUDA provider. Uses
/// RTLD_LAZY | RTLD_GLOBAL so ort's subsequent dlopen of
/// libonnxruntime_providers_cuda.so resolves against these.
///
/// Leaks the handles intentionally — they must stay loaded for the
/// lifetime of the process. Returns the count loaded (for logging).
fn preload_cuda_sidecars(dirs: &[std::path::PathBuf]) -> usize {
    // Just preload the top-level cuDNN lib. Its DT_NEEDED siblings
    // (libcudnn_ops.so.9, libcudnn_cnn.so.9, etc.) are resolved by ld.so
    // via the LD_LIBRARY_PATH we've already set to the same directory.
    // Attempting to preload all siblings manually caused segfaults when
    // the caches held two different patch versions.
    for dir in dirs {
        let full = dir.join("libcudnn.so.9");
        if full.exists() {
            unsafe {
                if let Ok(lib) = libloading::Library::new(&full) {
                    std::mem::forget(lib);
                    return 1;
                }
            }
        }
    }
    0
}

fn prepend_ld_library_path(dirs: &[std::path::PathBuf]) {
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

fn walk_one_level(root: &std::path::Path) -> Vec<std::path::PathBuf> {
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

fn clip(s: &str) -> String {
    if s.len() <= MAX_TEXT_CHARS {
        s.to_string()
    } else {
        s.chars().take(MAX_TEXT_CHARS).collect()
    }
}

fn l2_normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        // Unprefixed path: treat everything as documents.
        self.run_embed(self.prefixed_texts(&texts, false))
    }

    fn embed_query(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.run_embed(self.prefixed_texts(&texts, true))
    }

    fn embed_document(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.run_embed(self.prefixed_texts(&texts, false))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        self.backend_name
    }
}
