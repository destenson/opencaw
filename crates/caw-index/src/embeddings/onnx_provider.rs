use caw_core::{CawError, CawResult, EmbeddingProvider};
use ndarray::Ix2;
use ort::{session::Session, value::TensorRef};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct OnnxEmbeddingProvider {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl OnnxEmbeddingProvider {
    /// Load an ONNX embedding model from a local file path.
    /// Expects a `tokenizer.json` in the same directory as the model file.
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

        let session = Session::builder()
            .map_err(|e| CawError::Embedding(format!("Failed to create session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            session,
            tokenizer,
            dimension,
        })
    }

    /// Load with an explicit tokenizer path rather than expecting it adjacent to the model.
    pub fn from_paths(
        model_path: &str,
        tokenizer_path: &str,
        dimension: usize,
    ) -> CawResult<Self> {
        let session = Session::builder()
            .map_err(|e| CawError::Embedding(format!("Failed to create session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load ONNX model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CawError::Embedding(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            session,
            tokenizer,
            dimension,
        })
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        let encodings = self
            .tokenizer
            .encode_batch(texts_owned, true)
            .map_err(|e| CawError::Embedding(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let padded_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut all_ids: Vec<i64> = Vec::with_capacity(batch_size * padded_len);
        let mut all_mask: Vec<i64> = Vec::with_capacity(batch_size * padded_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len();

            all_ids.extend(ids.iter().map(|&x| x as i64));
            all_mask.extend(mask.iter().map(|&x| x as i64));

            for _ in len..padded_len {
                all_ids.push(0);
                all_mask.push(0);
            }
        }

        let shape = [batch_size, padded_len];

        let ids_tensor = TensorRef::from_array_view((shape, &*all_ids))
            .map_err(|e| CawError::Embedding(format!("Failed to create input_ids tensor: {e}")))?;
        let mask_tensor = TensorRef::from_array_view((shape, &*all_mask)).map_err(|e| {
            CawError::Embedding(format!("Failed to create attention_mask tensor: {e}"))
        })?;

        let outputs = self
            .session
            .run(ort::inputs![ids_tensor, mask_tensor])
            .map_err(|e| CawError::Embedding(format!("ONNX inference failed: {e}")))?;

        // Most sentence-transformer ONNX exports put the pooled embeddings at
        // output index 1 (index 0 is the raw token-level hidden states).
        // Fall back to index 0 if only one output exists.
        let output_idx = if outputs.len() > 1 { 1 } else { 0 };

        let embeddings = outputs[output_idx]
            .try_extract_array::<f32>()
            .map_err(|e| CawError::Embedding(format!("Failed to extract output tensor: {e}")))?
            .into_dimensionality::<Ix2>()
            .map_err(|e| {
                CawError::Embedding(format!("Output tensor has unexpected shape: {e}"))
            })?;

        let result: Vec<Vec<f32>> = embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        Ok(result)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "onnx"
    }
}
