use caw_core::{
    detect_loop, CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter,
    ModelCapabilities, ProvenanceFormat, TokenUsage,
};
use tracing::trace;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct AnthropicAdapter {
    api_key: String,
    model: String,
    client: Client,
    runtime: Arc<Runtime>,
}

impl std::fmt::Debug for AnthropicAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicAdapter")
            .field("model", &self.model)
            .finish()
    }
}

impl AnthropicAdapter {
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

    pub fn claude_sonnet(runtime: Arc<Runtime>) -> CawResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| CawError::Adapter("ANTHROPIC_API_KEY not set".into()))?;
        Ok(Self::new_with(api_key, "claude-sonnet-4-20250514", runtime))
    }

    pub fn claude_opus(runtime: Arc<Runtime>) -> CawResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| CawError::Adapter("ANTHROPIC_API_KEY not set".into()))?;
        Ok(Self::new_with(api_key, "claude-opus-4-20250514", runtime))
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<AnthropicMessage>,
}

#[derive(Serialize, Deserialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl ModelAdapter for AnthropicAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: true,
            supports_hidden_reasoning: self.model.contains("sonnet-4")
                || self.model.contains("opus-4"),
            supports_visible_reasoning: false,
            ..Default::default()
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Xml);
        let full_system = format!("{}{}", req.system, workspace_context);
        trace!(model = %self.model, system = %full_system, prompt = %req.user, "→ llm");

        let anthropic_req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            system: full_system,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: req.user,
            }],
        };

        let response = self.runtime.block_on(async {
            let http_resp = self.client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&anthropic_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?;

            if !http_resp.status().is_success() {
                let status = http_resp.status();
                let body = http_resp.text().await.unwrap_or_default();
                return Err(CawError::Adapter(format!("Anthropic API error {status}: {body}")));
            }

            http_resp
                .json::<AnthropicResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        let answer = response
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        if let Some(reason) = detect_loop(&answer) {
            tracing::warn!(model = %self.model, reason = %reason, "degenerate output: loop detected");
            return Err(CawError::DegenerateOutput {
                model: self.model.clone(),
                sample: answer.chars().take(120).collect(),
            });
        }
        let usage = response.usage.map(|u| TokenUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        });
        trace!(model = %self.model, answer = %answer, "← llm");
        Ok(CompletionResponse {
            answer,
            thinking: None,
            usage,
        })
    }
}
