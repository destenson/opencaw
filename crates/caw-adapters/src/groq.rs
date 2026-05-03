use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat, TokenUsage, is_looping,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::trace;

pub struct GroqAdapter {
    api_key: String,
    model: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl std::fmt::Debug for GroqAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroqAdapter")
            .field("model", &self.model)
            .finish()
    }
}

impl GroqAdapter {
    pub fn new_with(
        api_key: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: Client::new(),
            runtime,
        }
    }

    pub fn groq_model(model: impl Into<String>, runtime: Arc<Runtime>) -> CawResult<Self> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| CawError::Adapter("GROQ_API_KEY not set".into()))?;
        Ok(Self::new_with(api_key, model, runtime))
    }

    pub fn llama_70b(runtime: Arc<Runtime>) -> CawResult<Self> {
        Self::groq_model("llama-3.3-70b-versatile", runtime)
    }

    pub fn llama_8b(runtime: Arc<Runtime>) -> CawResult<Self> {
        Self::groq_model("llama-3.1-8b-instant", runtime)
    }

    pub fn mixtral(runtime: Arc<Runtime>) -> CawResult<Self> {
        Self::groq_model("mixtral-8x7b-32768", runtime)
    }
}

#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl ModelAdapter for GroqAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: true,
            supports_hidden_reasoning: false,
            supports_visible_reasoning: false,
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
        let full_user_message = format!("{}{}", req.user, workspace_context);
        trace!(model = %self.model, system = %req.system, prompt = %full_user_message, "→ llm");

        let groq_req = GroqRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages: vec![
                GroqMessage {
                    role: "system".to_string(),
                    content: req.system,
                },
                GroqMessage {
                    role: "user".to_string(),
                    content: full_user_message,
                },
            ],
        };

        let response = self.runtime.block_on(async {
            let http_resp = self.client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&groq_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?;

            if !http_resp.status().is_success() {
                let status = http_resp.status();
                let body = http_resp.text().await.unwrap_or_default();
                return Err(CawError::Adapter(format!("Groq API error {status}: {body}")));
            }

            http_resp
                .json::<GroqResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        let answer = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        if is_looping(&answer) {
            return Err(CawError::DegenerateOutput {
                model: self.model.clone(),
                sample: answer.chars().take(120).collect(),
            });
        }
        let usage = response.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        });
        trace!(model = %self.model, answer = %answer, "← llm");
        Ok(CompletionResponse {
            answer,
            thinking: None,
            usage,
        })
    }
}
