use caw_core::{CawError, CawResult, EmbeddingProvider};
use serde::{Deserialize, Serialize};

/// API-based embedding provider (OpenAI, Cohere, Voyage, etc.)
pub struct ApiEmbeddingProvider {
    client: reqwest::blocking::Client,
    api_key: String,
    endpoint: String,
    model: String,
    dimension: usize,
    provider_type: ApiProviderType,
}

#[derive(Debug, Clone, Copy)]
pub enum ApiProviderType {
    OpenAI,
    Cohere,
    Voyage,
}

impl ApiEmbeddingProvider {
    pub fn new(
        provider_type: ApiProviderType,
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Self {
        let endpoint = match provider_type {
            ApiProviderType::OpenAI => "https://api.openai.com/v1/embeddings",
            ApiProviderType::Cohere => "https://api.cohere.ai/v1/embed",
            ApiProviderType::Voyage => "https://api.voyageai.com/v1/embeddings",
        }
        .to_string();

        Self {
            client: reqwest::blocking::Client::new(),
            api_key: api_key.into(),
            endpoint,
            model: model.into(),
            dimension,
            provider_type,
        }
    }

    pub fn openai_small() -> CawResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| CawError::Embedding("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(
            ApiProviderType::OpenAI,
            api_key,
            "text-embedding-3-small",
            1536,
        ))
    }

    pub fn openai_large() -> CawResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| CawError::Embedding("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(
            ApiProviderType::OpenAI,
            api_key,
            "text-embedding-3-large",
            3072,
        ))
    }
}

#[derive(Serialize)]
struct OpenAIRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

impl EmbeddingProvider for ApiEmbeddingProvider {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        match self.provider_type {
            ApiProviderType::OpenAI => {
                let request = OpenAIRequest {
                    input: texts.iter().map(|s| s.to_string()).collect(),
                    model: self.model.clone(),
                };

                let response = self
                    .client
                    .post(&self.endpoint)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .map_err(|e| CawError::Embedding(format!("API request failed: {}", e)))?
                    .json::<OpenAIResponse>()
                    .map_err(|e| CawError::Embedding(format!("Failed to parse response: {}", e)))?;

                Ok(response.data.into_iter().map(|e| e.embedding).collect())
            }
            _ => Err(CawError::Embedding(format!(
                "{:?} not yet implemented",
                self.provider_type
            ))),
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &str {
        match self.provider_type {
            ApiProviderType::OpenAI => "openai-api",
            ApiProviderType::Cohere => "cohere-api",
            ApiProviderType::Voyage => "voyage-api",
        }
    }
}
