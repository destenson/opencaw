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

use super::ort_setup::ensure_ort_dylib_path;
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
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// Precision variant for `bge_small_variant`. Maps to a specific ONNX file
/// in `Xenova/bge-small-en-v1.5`. Naming follows the transformers.js export
/// convention used across Xenova's repos.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxVariant {
    Fp32,
    Fp16,
    Int8,
    Quantized,
}

impl OnnxVariant {
    fn filename(self) -> &'static str {
        match self {
            OnnxVariant::Fp32 => "onnx/model.onnx",
            OnnxVariant::Fp16 => "onnx/model_fp16.onnx",
            OnnxVariant::Int8 => "onnx/model_int8.onnx",
            OnnxVariant::Quantized => "onnx/model_quantized.onnx",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            OnnxVariant::Fp32 => "fp32",
            OnnxVariant::Fp16 => "fp16",
            OnnxVariant::Int8 => "int8",
            OnnxVariant::Quantized => "quantized",
        }
    }
}

/// Bucket boundaries for per-batch padded-seq-length histogram. A batch is
/// counted in the smallest bucket whose value is >= its padded_len. Chosen
/// to roughly match the powers-of-two granularity that TensorRT optimization
/// profiles care about — finer at the short end since BGE inputs concentrate
/// there, coarse above 256 since anything that hits 512 is already maxed out.
const SEQ_HIST_BUCKETS: [usize; 7] = [16, 32, 64, 128, 256, 384, 512];

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
    /// Per-batch padded_len histogram. Shows what TRT would actually see
    /// since TRT runs the whole batch at `max(len)`. Counted by bucket index
    /// in `SEQ_HIST_BUCKETS`. Not atomic — provider is single-threaded in
    /// the benchmark consumer path.
    seq_hist: [u64; SEQ_HIST_BUCKETS.len()],
    /// Per-*item* token-length histogram. Shows actual content distribution
    /// independent of batching — answers "if we had perfect bucketing, how
    /// much compute would we save?" vs. what the batch-max forces us into.
    item_seq_hist: [u64; SEQ_HIST_BUCKETS.len()],
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
        Self::from_paths(
            model_path,
            tokenizer_path.to_str().unwrap(),
            dimension,
            true,
        )
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
            seq_hist: [0; SEQ_HIST_BUCKETS.len()],
            item_seq_hist: [0; SEQ_HIST_BUCKETS.len()],
        })
    }

    /// Convenience constructor for BAAI/bge-small-en-v1.5 (fp32). Downloads
    /// the ONNX model and tokenizer from HuggingFace Hub on first use; cached
    /// thereafter in `~/.cache/huggingface`. Equivalent to
    /// `bge_small_variant(OnnxVariant::Fp32)` against BAAI's repo.
    pub fn bge_small() -> CawResult<Self> {
        let api =
            Api::new().map_err(|e| CawError::Embedding(format!("HF Hub init failed: {e}")))?;
        let repo = api.model("BAAI/bge-small-en-v1.5".to_string());
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

    /// Convenience constructor for bge-small-en-v1.5 variants published by
    /// Xenova/bge-small-en-v1.5. Lets callers pick precision (fp32/fp16/int8/
    /// quantized) without juggling HF paths. Xenova's repo is the transformers.js
    /// export which ships a full set of quantized variants side-by-side.
    pub fn bge_small_variant(variant: OnnxVariant) -> CawResult<Self> {
        let api =
            Api::new().map_err(|e| CawError::Embedding(format!("HF Hub init failed: {e}")))?;
        let repo = api.model("Xenova/bge-small-en-v1.5".to_string());
        let filename = variant.filename();
        let model_path = repo.get(filename).map_err(|e| {
            CawError::Embedding(format!("Failed to download {filename} from Xenova: {e}"))
        })?;
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
        // Histogram the shape TRT EP would see. Smallest bucket whose max
        // covers this batch; anything above the last bucket bumps the last.
        let hist_idx = SEQ_HIST_BUCKETS
            .iter()
            .position(|&b| padded_len <= b)
            .unwrap_or(SEQ_HIST_BUCKETS.len() - 1);
        self.seq_hist[hist_idx] += 1;
        // Per-item length distribution, independent of batching. Shows how
        // much the batch-max padding is wasting: if item distribution is
        // skewed short but batch-max is always 512, the batching policy
        // leaves throughput on the table.
        for enc in &encodings {
            let item_len = enc.get_ids().len().min(MAX_SEQ_LEN);
            let idx = SEQ_HIST_BUCKETS
                .iter()
                .position(|&b| item_len <= b)
                .unwrap_or(SEQ_HIST_BUCKETS.len() - 1);
            self.item_seq_hist[idx] += 1;
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

    // Try CUDA first. `error_on_failure()` is critical: without it,
    // `with_execution_providers` only registers the EP as a *preference*,
    // and if CUDA silently fails to bind at inference time (cuDNN ABI
    // mismatch, CUDA driver/runtime skew, whatever) ort runs the whole
    // model on CPU while our code happily logs "using CUDA". That misled
    // a multi-hour sweep into thinking ONNX was 6x slower than candle when
    // it was really running on 14 CPU cores at 0% GPU utilization. Strict
    // registration makes CUDA-or-CPU a binary, observable outcome.
    let cuda_ep = CUDAExecutionProvider::default().build().error_on_failure();
    match Session::builder()
        .and_then(|b| b.with_execution_providers([cuda_ep]))
        .and_then(|b| b.commit_from_file(model_path))
    {
        Ok(session) => {
            eprintln!("onnx: CUDA execution provider bound");
            Ok((session, "onnx-cuda"))
        }
        Err(err) => {
            eprintln!("onnx: CUDA EP unavailable ({err}); falling back to CPU execution provider");
            let session = Session::builder()
                .map_err(|e| CawError::Embedding(format!("session builder: {e}")))?
                .commit_from_file(model_path)
                .map_err(|e| CawError::Embedding(format!("load model CPU: {e}")))?;
            Ok((session, "onnx-cpu"))
        }
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

    fn seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        Some(
            SEQ_HIST_BUCKETS
                .iter()
                .zip(self.seq_hist.iter())
                .map(|(b, c)| (*b, *c))
                .collect(),
        )
    }

    fn item_seq_len_histogram(&self) -> Option<Vec<(usize, u64)>> {
        Some(
            SEQ_HIST_BUCKETS
                .iter()
                .zip(self.item_seq_hist.iter())
                .map(|(b, c)| (*b, *c))
                .collect(),
        )
    }
}
