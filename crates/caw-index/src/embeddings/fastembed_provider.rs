use caw_core::{CawError, CawResult, EmbeddingProvider};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct FastEmbedProvider {
    model: TextEmbedding,
    dimension: usize,
}

impl FastEmbedProvider {
    pub fn new(model_type: FastEmbedModel) -> CawResult<Self> {
        let (model_enum, dimension) = match model_type {
            FastEmbedModel::BGESmallENV15 => (EmbeddingModel::BGESmallENV15, 384),
            FastEmbedModel::BGEBaseENV15 => (EmbeddingModel::BGEBaseENV15, 768),
            FastEmbedModel::AllMiniLML6V2 => (EmbeddingModel::AllMiniLML6V2, 384),
        };

        let model = TextEmbedding::try_new(
            InitOptions::new(model_enum)
                .with_show_download_progress(false)
        )
        .map_err(|e| CawError::Embedding(format!("Failed to initialize fastembed: {}", e)))?;

        Ok(Self { model, dimension })
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

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        "fastembed"
    }
}
