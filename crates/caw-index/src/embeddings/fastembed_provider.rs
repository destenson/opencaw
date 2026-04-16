use caw_core::{CawError, CawResult, EmbeddingProvider};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct FastEmbedProvider {
    model: TextEmbedding,
    dimension: usize,
    /// BGE models use "query: " prefix for query-side encoding
    asymmetric: bool,
}

impl FastEmbedProvider {
    pub fn new(model_type: FastEmbedModel) -> CawResult<Self> {
        let (model_enum, dimension, asymmetric) = match model_type {
            FastEmbedModel::BGESmallENV15 => (EmbeddingModel::BGESmallENV15, 384, true),
            FastEmbedModel::BGEBaseENV15 => (EmbeddingModel::BGEBaseENV15, 768, true),
            FastEmbedModel::AllMiniLML6V2 => (EmbeddingModel::AllMiniLML6V2, 384, false),
        };

        let model =
            TextEmbedding::try_new(InitOptions::new(model_enum).with_show_download_progress(false))
                .map_err(|e| {
                    CawError::Embedding(format!("Failed to initialize fastembed: {}", e))
                })?;

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
