use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use caw_core::{CawError, CawResult, EmbeddingProvider};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

/// BGE / most BERT-family models have 512 position embeddings. Sequences
/// longer than this must be truncated before forward; candle's BertModel
/// will otherwise index past the position embedding and either panic or
/// produce garbage. The fastembed backend truncates automatically; we do
/// it explicitly here.
const MAX_SEQ_LEN: usize = 512;

/// Character-level pre-clip before tokenization. BPE tokenization is O(n²)
/// in the worst case on certain inputs (long runs of rare bytes), so a
/// text that's 10MB of noise can hang the tokenizer for minutes. BGE only
/// uses the first 512 tokens anyway (~2KB of normal prose), so clipping at
/// 32KB is a pure safety net that never throws away useful signal.
const MAX_TEXT_CHARS: usize = 32_768;

pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
    /// BGE models use asymmetric prefixes for query vs document encoding
    asymmetric: bool,
}

impl CandleEmbeddingProvider {
    /// Load a BERT-family model from HuggingFace Hub by model ID.
    /// Downloads and caches model weights, config, and tokenizer automatically.
    ///
    /// Tries CUDA:0 first and falls back to CPU if CUDA isn't available. The
    /// fallback keeps non-GPU hosts working; on a CUDA host a successful
    /// `new_cuda(0)` is the whole point of this provider.
    pub fn from_pretrained(model_id: &str) -> CawResult<Self> {
        let device = match Device::new_cuda(0) {
            Ok(d) => {
                eprintln!("candle: using CUDA:0");
                d
            }
            Err(e) => {
                eprintln!("candle: CUDA unavailable ({e}); falling back to CPU");
                Device::Cpu
            }
        };
        let api =
            Api::new().map_err(|e| CawError::Embedding(format!("HF Hub init failed: {e}")))?;
        let repo = api.model(model_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| CawError::Embedding(format!("Failed to download config.json: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| CawError::Embedding(format!("Failed to download tokenizer.json: {e}")))?;
        let weights_path = repo.get("model.safetensors").map_err(|e| {
            CawError::Embedding(format!("Failed to download model.safetensors: {e}"))
        })?;

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| CawError::Embedding(format!("Failed to parse config.json: {e}")))?;

        let dimension = config.hidden_size;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load tokenizer: {e}")))?;

        // Safety: mmap is safe here because we own the file and won't modify it
        // while the model holds a reference.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device) }
            .map_err(|e| CawError::Embedding(format!("Failed to load weights: {e}")))?;

        let model = BertModel::load(vb, &config)
            .map_err(|e| CawError::Embedding(format!("Failed to load BERT model: {e}")))?;

        let asymmetric = model_id.contains("bge-");

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
            asymmetric,
        })
    }

    /// Convenience constructor for BAAI/bge-small-en-v1.5 (384-dim, asymmetric).
    pub fn bge_small() -> CawResult<Self> {
        Self::from_pretrained("BAAI/bge-small-en-v1.5")
    }

    fn embed_inner(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Phase timings make it obvious when a single call hangs which
        // phase is at fault (tokenize/tensor-copy/forward/pool). Cheap
        // relative to the work they bracket.
        let t_start = std::time::Instant::now();
        let raw_max_chars = texts.iter().map(|s| s.len()).max().unwrap_or(0);
        let raw_total_chars: usize = texts.iter().map(|s| s.len()).sum();

        // Clip at the char level before tokenization so a pathological input
        // can't stall the whole run. Done char-wise (not byte) to preserve
        // UTF-8 boundaries.
        let texts_owned: Vec<String> = texts
            .iter()
            .map(|s| {
                if s.len() <= MAX_TEXT_CHARS {
                    (*s).to_string()
                } else {
                    s.chars().take(MAX_TEXT_CHARS).collect()
                }
            })
            .collect();
        let clipped = texts_owned.iter().filter(|s| s.len() < raw_max_chars).count();

        eprintln!(
            "    candle: about to tokenize batch={} raw_max_chars={} raw_total_chars={} clipped={}",
            texts.len(),
            raw_max_chars,
            raw_total_chars,
            clipped
        );

        let t_tok = std::time::Instant::now();
        let encodings = self
            .tokenizer
            .encode_batch(texts_owned, true)
            .map_err(|e| CawError::Embedding(format!("Tokenization failed: {e}")))?;
        let tok_ms = t_tok.elapsed().as_millis();
        let max_tokens = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let total_tokens: usize = encodings.iter().map(|e| e.get_ids().len()).sum();
        eprintln!(
            "    candle: tokenize done max_tok={} total_tok={} in {}ms",
            max_tokens, total_tokens, tok_ms
        );

        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);
        let batch_size = encodings.len();

        let mut all_ids = Vec::with_capacity(batch_size * padded_len);
        let mut all_type_ids = Vec::with_capacity(batch_size * padded_len);
        let mut all_mask = Vec::with_capacity(batch_size * padded_len);

        for enc in &encodings {
            let full_ids = enc.get_ids();
            let full_type_ids = enc.get_type_ids();
            let full_mask = enc.get_attention_mask();
            let take = full_ids.len().min(MAX_SEQ_LEN);

            all_ids.extend(full_ids.iter().take(take).map(|&x| x as i64));
            all_type_ids.extend(full_type_ids.iter().take(take).map(|&x| x as i64));
            all_mask.extend(full_mask.iter().take(take).map(|&x| x as f32));

            // Pad to uniform length (padded_len already capped at MAX_SEQ_LEN).
            for _ in take..padded_len {
                all_ids.push(0);
                all_type_ids.push(0);
                all_mask.push(0.0);
            }
        }

        let shape = &[batch_size, padded_len];

        let token_ids = Tensor::from_vec(all_ids, shape, &self.device)
            .map_err(|e| CawError::Embedding(format!("Failed to create token_ids tensor: {e}")))?;
        let token_type_ids = Tensor::from_vec(all_type_ids, shape, &self.device).map_err(|e| {
            CawError::Embedding(format!("Failed to create token_type_ids tensor: {e}"))
        })?;
        let attention_mask_f32 = Tensor::from_vec(all_mask, shape, &self.device).map_err(|e| {
            CawError::Embedding(format!("Failed to create attention_mask tensor: {e}"))
        })?;

        // BertModel.forward expects i64 attention mask for the causal mask,
        // but we need f32 for mean-pooling below
        let attention_mask_i64 = attention_mask_f32
            .to_dtype(DType::I64)
            .map_err(|e| CawError::Embedding(format!("Attention mask dtype cast failed: {e}")))?;

        let t_fwd = std::time::Instant::now();
        let output = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask_i64))
            .map_err(|e| CawError::Embedding(format!("Forward pass failed: {e}")))?;
        eprintln!(
            "    candle: forward shape=[{}, {}] in {}ms",
            batch_size,
            padded_len,
            t_fwd.elapsed().as_millis()
        );

        // Mean pooling: average token embeddings weighted by attention mask.
        // output shape: [batch, seq_len, hidden_size]
        // mask shape:   [batch, seq_len] -> unsqueeze to [batch, seq_len, 1]
        let mask_expanded = attention_mask_f32
            .unsqueeze(2)
            .map_err(|e| CawError::Embedding(format!("Mask unsqueeze failed: {e}")))?;

        let masked = output
            .broadcast_mul(&mask_expanded)
            .map_err(|e| CawError::Embedding(format!("Broadcast mul failed: {e}")))?;

        let summed = masked
            .sum(1)
            .map_err(|e| CawError::Embedding(format!("Sum failed: {e}")))?;

        let mask_sum = mask_expanded
            .sum(1)
            .map_err(|e| CawError::Embedding(format!("Mask sum failed: {e}")))?;

        // Clamp to avoid division by zero for fully-padded sequences
        let mask_sum_clamped = mask_sum
            .clamp(1e-9, f64::MAX)
            .map_err(|e| CawError::Embedding(format!("Clamp failed: {e}")))?;

        let pooled = summed
            .broadcast_div(&mask_sum_clamped)
            .map_err(|e| CawError::Embedding(format!("Broadcast div failed: {e}")))?;

        // L2 normalize each embedding
        let norms = pooled
            .sqr()
            .and_then(|s| s.sum(1))
            .and_then(|s| s.sqrt())
            .and_then(|s| s.unsqueeze(1))
            .and_then(|s| s.clamp(1e-12, f64::MAX))
            .map_err(|e| CawError::Embedding(format!("Norm computation failed: {e}")))?;

        let normalized = pooled
            .broadcast_div(&norms)
            .map_err(|e| CawError::Embedding(format!("Normalization failed: {e}")))?;

        // Extract to Vec<Vec<f32>>
        let flat: Vec<f32> = normalized
            .to_vec2()
            .map_err(|e| CawError::Embedding(format!("Tensor extraction failed: {e}")))?
            .into_iter()
            .flatten()
            .collect();

        let out = flat
            .chunks(self.dimension)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        eprintln!(
            "    candle: total embed {}ms (tok {}ms)",
            t_start.elapsed().as_millis(),
            tok_ms,
        );
        Ok(out)
    }
}

impl EmbeddingProvider for CandleEmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.embed_inner(texts)
    }

    fn embed_query(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if !self.asymmetric {
            return self.embed(texts);
        }
        let prefixed: Vec<String> = texts.iter().map(|s| format!("query: {s}")).collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed(refs)
    }

    fn embed_document(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if !self.asymmetric {
            return self.embed(texts);
        }
        let prefixed: Vec<String> = texts.iter().map(|s| format!("passage: {s}")).collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        self.embed(refs)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "candle"
    }
}
