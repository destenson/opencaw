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

    pub fn qwen3_5_9b(runtime: Arc<Runtime>) -> Self {
        Self::local("qwen3.5:9b", runtime)
    }

    pub fn deepseek_v3_1(runtime: Arc<Runtime>) -> Self {
        Self::local("deepseek-v3.1:671b-cloud", runtime)
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
        // TODO: query Ollama's /api/models endpoint to get actual capabilities per model. For now we hardcode based on known behavior of popular models:
        // hidden_reasoning is overloaded here to mean "follows
        // marker-emission instructions in its final answer" — the gate the
        // orchestrator uses to decide whether to inject probe/note prompts.
        // Modern instruct models (qwen2.5, llama3.2, mistral-instruct) all
        // qualify; without this flag the orchestrator silently degrades to
        // single-shot retrieval. Visible reasoning still gates the
        // <think>-block parsing path and is detected by model name.
        ModelCapabilities {
            supports_tool_calls: false,
            supports_hidden_reasoning: true,
            supports_visible_reasoning: self.model.contains("deepseek")
                || self.model.contains("qwen"),
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

        let body = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            let resp = self
                .client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| CawError::Adapter(format!("read response body: {}", e)))?;
            if !status.is_success() {
                return Err(CawError::Adapter(format!(
                    "Ollama returned HTTP {}: {}",
                    status,
                    text.chars().take(500).collect::<String>()
                )));
            }
            Ok(text)
        })?;

        // Read the body as text first, then parse — when Ollama returns an
        // error envelope (e.g. unknown model) the chat-response shape fails
        // to deserialize and the original error text gets lost. Surfacing
        // both candidate parses gives the caller something to act on.
        match serde_json::from_str::<OllamaChatResponse>(&body) {
            Ok(parsed) => Ok(CompletionResponse {
                answer: parsed.message.content,
            }),
            Err(parse_err) => {
                #[derive(Deserialize)]
                struct OllamaError {
                    error: String,
                }
                if let Ok(err) = serde_json::from_str::<OllamaError>(&body) {
                    return Err(CawError::Adapter(format!("Ollama error: {}", err.error)));
                }
                Err(CawError::Adapter(format!(
                    "Ollama response did not match chat or error schema ({}): {}",
                    parse_err,
                    body.chars().take(500).collect::<String>()
                )))
            }
        }
    }
}
