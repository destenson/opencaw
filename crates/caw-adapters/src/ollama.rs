use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct OllamaAdapter {
    base_url: String,
    model: String,
    client: Client,
}

impl OllamaAdapter {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: Client::new(),
        }
    }

    pub fn local(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434", model)
    }

    pub fn llama3_2() -> Self {
        Self::local("llama3.2")
    }

    pub fn qwen2_5() -> Self {
        Self::local("qwen2.5")
    }

    pub fn deepseek_r1() -> Self {
        Self::local("deepseek-r1")
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
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
        let workspace_context = if req.workspace_fragments.is_empty() {
            String::new()
        } else {
            let fragments = req
                .workspace_fragments
                .iter()
                .map(|f| {
                    format!(
                        "\n[recalled from {source}:{locator}]\n{content}\n[end recall]",
                        source = f.locator.source,
                        locator = f.locator.locator,
                        content = f.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            format!("\n\nRecalled workspace context:\n{}", fragments)
        };

        let full_prompt = format!(
            "System: {}\n\nUser: {}{}\n\nAssistant:",
            req.system, req.user, workspace_context
        );

        let ollama_req = OllamaRequest {
            model: self.model.clone(),
            prompt: full_prompt,
            stream: false,
            options: OllamaOptions {
                temperature: 0.7,
                num_predict: 4096,
            },
        };

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| CawError::Adapter(format!("Failed to create runtime: {}", e)))?;

        let response = runtime.block_on(async {
            let url = format!("{}/api/generate", self.base_url);
            self.client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?
                .json::<OllamaResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        Ok(CompletionResponse {
            answer: response.response,
        })
    }
}
