use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct OllamaAdapter {
    base_url: String,
    model: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl std::fmt::Debug for OllamaAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaAdapter")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OllamaAdapter {
    pub fn new_with(
        base_url: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: Client::new(),
            runtime,
        }
    }

    pub fn local(model: impl Into<String>, runtime: Arc<Runtime>) -> Self {
        Self::new_with("http://localhost:11434", model, runtime)
    }

    pub fn llama3_2(runtime: Arc<Runtime>) -> Self {
        Self::local("llama3.2", runtime)
    }

    pub fn qwen2_5(runtime: Arc<Runtime>) -> Self {
        Self::local("qwen2.5", runtime)
    }

    pub fn deepseek_r1(runtime: Arc<Runtime>) -> Self {
        Self::local("deepseek-r1", runtime)
    }
}

/// Uses Ollama's /api/chat endpoint with proper message roles
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessage,
}

impl ModelAdapter for OllamaAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: false,
            supports_hidden_reasoning: false,
            supports_visible_reasoning: self.model.contains("deepseek-r1")
                || self.model.contains("qwen-qwq"),
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
        let full_user_message = format!("{}{}", req.user, workspace_context);

        let ollama_req = OllamaChatRequest {
            model: self.model.clone(),
            messages: vec![
                OllamaChatMessage {
                    role: "system".to_string(),
                    content: req.system,
                },
                OllamaChatMessage {
                    role: "user".to_string(),
                    content: full_user_message,
                },
            ],
            stream: false,
            options: OllamaOptions {
                temperature: 0.7,
                num_predict: 4096,
            },
        };

        let response = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            self.client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?
                .json::<OllamaChatResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        Ok(CompletionResponse {
            answer: response.message.content,
        })
    }
}
