use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use caw_core::{CawError, CawResult, EmbeddingProvider};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use tracing::{debug, info};

const DTYPE: DType = DType::F32;

/// BGE / most BERT-family models have 512 position embeddings. Sequences
/// longer than this must be truncated before forward; candle's BertModel
/// will otherwise index past the position embedding and either panic or
/// produce garbage. The fastembed backend truncates automatically; we do
/// it explicitly here.
const MAX_SEQ_LEN: usize = 512;

/// Character-level pre-clip before tokenization. BGE only uses the first
/// 512 tokens (`MAX_SEQ_LEN`), which is roughly 2 KB of normal prose at
/// ~3-4 chars/token. Clipping *before* tokenization means the BPE pass
/// does ~20x less work on large inputs — previously we tokenized 10-16K
/// tokens per text and threw away 95% of that in the truncation step,
/// which dominated batch latency when embedding chunk content.
///
/// 2500 leaves modest headroom over 512×4 (2048) for token-dense inputs
/// (code, heavily-punctuated text). If a text tokenizes denser than the
/// headroom allows, the 512-token seq-len cap still protects the model.
const MAX_TEXT_CHARS: usize = 2_500;

/// Environment variable a caller may set to choose the embedding device
/// explicitly: `cpu`, `cuda`, or `cuda:N` (N = ordinal). This is declared
/// configuration, not runtime inference — the library never probes which
/// GPU is freest; that choice belongs to the caller (e.g. `pick-gpu.sh`).
const EMBED_DEVICE_ENV: &str = "CAW_EMBED_DEVICE";

/// Which compute device the candle embedder should load onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedDevice {
    /// Honor `CAW_EMBED_DEVICE` if set; otherwise CUDA ordinal 0, falling
    /// back to CPU only when CUDA is entirely absent. This is the historical
    /// default and what every existing caller gets.
    #[default]
    Auto,
    /// Force CPU regardless of CUDA availability.
    Cpu,
    /// A specific CUDA ordinal. Construction fails if that device can't be
    /// opened — no silent CPU fallback, so a misconfigured ordinal is loud.
    Cuda(usize),
}

impl EmbedDevice {
    /// Parse the `cpu` / `cuda` / `cuda:N` forms of `CAW_EMBED_DEVICE`.
    fn parse_env(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        match raw.to_ascii_lowercase().as_str() {
            "cpu" => Ok(EmbedDevice::Cpu),
            "cuda" => Ok(EmbedDevice::Cuda(0)),
            other => match other.strip_prefix("cuda:") {
                Some(n) => n
                    .parse::<usize>()
                    .map(EmbedDevice::Cuda)
                    .map_err(|_| format!("invalid CUDA ordinal in {EMBED_DEVICE_ENV}={raw:?}")),
                None => Err(format!(
                    "unrecognized {EMBED_DEVICE_ENV}={raw:?} (expected cpu, cuda, or cuda:N)"
                )),
            },
        }
    }

    /// Resolve to a concrete candle `Device`. `Auto` consults the env var,
    /// then tries CUDA:0, then CPU. Explicit `Cuda(n)` never falls back —
    /// failing to open the requested ordinal is an error the caller must see.
    fn resolve(self) -> CawResult<Device> {
        let selected = match self {
            EmbedDevice::Auto => match std::env::var(EMBED_DEVICE_ENV) {
                Ok(raw) if !raw.trim().is_empty() => {
                    EmbedDevice::parse_env(&raw).map_err(CawError::Embedding)?
                }
                _ => {
                    // Historical Auto behavior: CUDA:0 or CPU if CUDA absent.
                    return Ok(match Device::new_cuda(0) {
                        Ok(d) => {
                            info!("candle: using CUDA:0 (Auto)");
                            d
                        }
                        Err(e) => {
                            info!("candle: CUDA unavailable ({e}); falling back to CPU");
                            Device::Cpu
                        }
                    });
                }
            },
            other => other,
        };
        match selected {
            EmbedDevice::Cpu => {
                info!("candle: using CPU");
                Ok(Device::Cpu)
            }
            EmbedDevice::Cuda(n) => {
                let d = Device::new_cuda(n).map_err(|e| {
                    CawError::Embedding(format!(
                        "open CUDA:{n} failed: {e} (set {EMBED_DEVICE_ENV}=cpu or a free ordinal)"
                    ))
                })?;
                info!("candle: using CUDA:{n}");
                Ok(d)
            }
            EmbedDevice::Auto => unreachable!("Auto resolved above"),
        }
    }
}

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
    /// Uses [`EmbedDevice::Auto`]: honors `CAW_EMBED_DEVICE` if set, else
    /// CUDA:0, else CPU when CUDA is absent. To pin a specific GPU or force
    /// CPU programmatically, use [`Self::from_pretrained_on`].
    pub fn from_pretrained(model_id: &str) -> CawResult<Self> {
        Self::from_pretrained_on(model_id, EmbedDevice::default())
    }

    /// Load a model onto an explicitly chosen device. An explicit
    /// [`EmbedDevice::Cuda`] ordinal that can't be opened is a hard error
    /// (no silent CPU fallback), so a busy/missing GPU surfaces immediately.
    pub fn from_pretrained_on(model_id: &str, device: EmbedDevice) -> CawResult<Self> {
        let device = device.resolve()?;
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

    /// [`Self::bge_small`] on an explicitly chosen device.
    pub fn bge_small_on(device: EmbedDevice) -> CawResult<Self> {
        Self::from_pretrained_on("BAAI/bge-small-en-v1.5", device)
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
        let clipped = texts_owned
            .iter()
            .filter(|s| s.len() < raw_max_chars)
            .count();

        debug!(
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
        let max_tokens = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
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

        // BertModel.forward expects i64 attention mask for the causal mask.
        let attention_mask_i64 = attention_mask_f32
            .to_dtype(DType::I64)
            .map_err(|e| CawError::Embedding(format!("Attention mask dtype cast failed: {e}")))?;
        // attention_mask_f32 is no longer needed after the dtype cast —
        // CLS pooling doesn't touch the mask.
        drop(attention_mask_f32);

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

        // CLS pooling: BGE-v1.5 (and most BERT-style encoders trained with
        // MLM + next-sentence or sentence-pair objectives) use the [CLS]
        // token's hidden state as the sentence embedding. Take the 0th
        // token along the sequence axis.
        //
        // Mean-pooling, which the previous implementation used, is both
        // incorrect for BGE and ~10x slower: each of the 11 tensor ops
        // involved incurs a kernel-launch/sync cost, and candle evaluates
        // eagerly with implicit syncs. CLS is 2 ops (narrow + squeeze)
        // plus the normalize ops that follow.
        let pooled = output
            .narrow(1, 0, 1)
            .and_then(|t| t.squeeze(1))
            .map_err(|e| CawError::Embedding(format!("CLS pooling failed: {e}")))?;

        // L2 normalize each embedding. BGE was trained with normalized
        // CLS vectors, so matching semantics at inference matters.
        let norms = pooled
            .sqr()
            .and_then(|s| s.sum_keepdim(1))
            .and_then(|s| s.sqrt())
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
