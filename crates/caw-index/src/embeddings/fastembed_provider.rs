use super::ort_setup::ensure_ort_dylib_path;
use caw_core::{CawError, CawResult, EmbeddingProvider};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ort::execution_providers::CUDAExecutionProvider;
use std::sync::Once;

/// ort's global environment (and its logger) must be committed before any
/// session is built. fastembed never does this, so registering the CUDA EP
/// under `error_on_failure()` would surface its error through onnxruntime's
/// DefaultLogger before one exists ("Attempt to use DefaultLogger but none
/// has been registered"), masking the real bind result. Commit it once.
static ORT_ENV_INIT: Once = Once::new();

fn ensure_ort_env_initialized() {
    ORT_ENV_INIT.call_once(|| {
        if let Err(e) = ort::init().with_name("caw-fastembed").commit() {
            eprintln!("fastembed: ort environment init failed: {e}");
        }
    });
}

/// Whether the current process holds any GPU memory, per `nvidia-smi`. An
/// active onnxruntime CUDA EP creates a CUDA context (allocating device
/// memory) for this PID; a silent CPU fallback leaves the process with none.
/// Returns `None` when `nvidia-smi` can't be run (no driver / not installed),
/// so the caller can report "unverified" rather than a false negative.
fn process_holds_gpu_memory() -> Option<bool> {
    let pid = std::process::id();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Rows are "pid, used_mib". Our PID present with >0 MiB means a live
    // CUDA context on some device.
    let holds = text.lines().any(|line| {
        let mut cols = line.split(',').map(str::trim);
        let row_pid = cols.next().and_then(|s| s.parse::<u32>().ok());
        let used = cols.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        row_pid == Some(pid) && used > 0
    });
    Some(holds)
}

pub struct FastEmbedProvider {
    model: TextEmbedding,
    dimension: usize,
    /// BGE models use "query: " prefix for query-side encoding
    asymmetric: bool,
}

impl FastEmbedProvider {
    pub fn new(model_type: FastEmbedModel) -> CawResult<Self> {
        // fastembed transitively pulls `ort` compiled with `load-dynamic`,
        // so it needs ORT_DYLIB_PATH pointing at a real `libonnxruntime.so`
        // at construction time. Same helper the direct onnx provider uses.
        ensure_ort_dylib_path();
        ensure_ort_env_initialized();

        let (model_enum, dimension, asymmetric) = match model_type {
            FastEmbedModel::BGESmallENV15 => (EmbeddingModel::BGESmallENV15, 384, true),
            FastEmbedModel::BGEBaseENV15 => (EmbeddingModel::BGEBaseENV15, 768, true),
            FastEmbedModel::AllMiniLML6V2 => (EmbeddingModel::AllMiniLML6V2, 384, false),
        };

        // Without an explicit execution provider, fastembed registers only the
        // CPU EP — pointing ORT_DYLIB_PATH at a CUDA build does nothing on its
        // own. Request CUDA with `error_on_failure()` so a silent CPU fallback
        // becomes a hard, observable error we catch and report, rather than
        // running every embedding on CPU while looking healthy (the exact trap
        // documented in onnx_provider.rs).
        let cuda_ep = CUDAExecutionProvider::default().build().error_on_failure();
        let model = match TextEmbedding::try_new(
            InitOptions::new(model_enum.clone())
                .with_show_download_progress(false)
                .with_execution_providers(vec![cuda_ep]),
        ) {
            Ok(mut m) => {
                // try_new succeeding with the CUDA EP does NOT prove the model
                // runs on GPU. onnxruntime registers the EP, then can silently
                // fall back to CPU at inference if the provider's runtime deps
                // can't load (e.g. the selected onnxruntime build wants a cuDNN
                // major that doesn't match the preloaded one). error_on_failure
                // guards registration, not kernel placement. So don't claim
                // success on construction alone — warm up and verify this
                // process actually holds a live CUDA context (GPU memory).
                let _ = m.embed(vec!["warmup"], None);
                // A CUDA *context* (GPU memory) is necessary but NOT sufficient
                // for GPU compute: onnxruntime allocates a device arena when the
                // EP initializes, then can still place every kernel on CPU if a
                // runtime dep is wrong (here onnxruntime 1.25 wants cuDNN 9 under
                // CUDA 13). We cannot tell context from kernel placement in
                // process, so report only what's verifiable and never claim
                // "running on GPU" outright.
                match process_holds_gpu_memory() {
                    Some(false) => eprintln!(
                        "fastembed: CUDA EP registered but this process holds NO GPU memory — \
                         embeddings are running on CPU (onnxruntime fell back, likely a \
                         cuDNN/CUDA version mismatch for the auto-selected onnxruntime build)."
                    ),
                    Some(true) => eprintln!(
                        "fastembed: CUDA EP registered and a CUDA context exists for this process. \
                         This does NOT confirm GPU compute — onnxruntime may still run kernels on \
                         CPU. Confirm with nvtop/nvidia-smi GPU utilization during embedding."
                    ),
                    None => eprintln!(
                        "fastembed: CUDA EP registered; GPU engagement unverified (nvidia-smi unavailable)."
                    ),
                }
                m
            }
            Err(cuda_err) => {
                eprintln!(
                    "fastembed: CUDA EP unavailable ({cuda_err}); falling back to CPU embeddings"
                );
                TextEmbedding::try_new(
                    InitOptions::new(model_enum).with_show_download_progress(false),
                )
                .map_err(|e| CawError::Embedding(format!("Failed to initialize fastembed: {}", e)))?
            }
        };

        Ok(Self {
            model,
            dimension,
            asymmetric,
        })
    }

    pub fn bge_small() -> CawResult<Self> {
        Self::new(FastEmbedModel::BGESmallENV15)
    }

    pub fn bge_base() -> CawResult<Self> {
        Self::new(FastEmbedModel::BGEBaseENV15)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FastEmbedModel {
    BGESmallENV15,
    BGEBaseENV15,
    AllMiniLML6V2,
}

impl EmbeddingProvider for FastEmbedProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        let texts_vec: Vec<String> = texts.into_iter().map(|s| s.to_string()).collect();
        self.model
            .embed(texts_vec, None)
            .map_err(|e| CawError::Embedding(format!("Embedding generation failed: {}", e)))
    }

    fn embed_query(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if !self.asymmetric {
            return self.embed(texts);
        }
        let prefixed: Vec<String> = texts.iter().map(|s| format!("query: {}", s)).collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed(refs)
    }

    fn embed_document(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if !self.asymmetric {
            return self.embed(texts);
        }
        let prefixed: Vec<String> = texts.iter().map(|s| format!("passage: {}", s)).collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed(refs)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "fastembed"
    }
}
