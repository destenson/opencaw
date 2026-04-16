// Placeholder for ONNX-based embedding provider
// Will use ort crate to load custom ONNX models

use caw_core::{CawError, CawResult, EmbeddingProvider};

pub struct OnnxEmbeddingProvider {
    // session: ort::Session,
    dimension: usize,
}

impl OnnxEmbeddingProvider {
    pub fn new(_model_path: &str, dimension: usize) -> CawResult<Self> {
        // TODO: Load ONNX model using ort crate
        Err(CawError::Embedding(
            "ONNX provider not yet implemented".to_string(),
        ))
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn embed(&mut self, _texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        Err(CawError::Embedding(
            "ONNX provider not yet implemented".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "onnx"
    }
}
