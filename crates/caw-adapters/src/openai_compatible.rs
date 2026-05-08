use caw_core::{
    detect_loop, strip_fake_recall_blocks, truncate_at_chat_boundary, CawError, CawResult,
    CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities, ProvenanceFormat,
    TokenUsage,
};
use tracing::trace;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Arbitrary headers to attach to every request.
/// Supports any combination of auth schemes and provider-specific headers.
#[derive(Debug, Clone, Default)]
pub struct RequestHeaders {
    headers: Vec<(String, String)>,
}

impl RequestHeaders {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bearer(token: impl Into<String>) -> Self {
        Self::new().with("Authorization", format!("Bearer {}", token.into()))
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn add(&mut self, name: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

/// Adapter for any provider that speaks the OpenAI chat completions protocol.
/// Covers vLLM, Perplexity, ollama.com, HuggingFace Inference Endpoints,
/// LiteLLM, and anything else that implements /v1/chat/completions.
pub struct OpenAiCompatibleAdapter {
    base_url: String,
    model: String,
    headers: RequestHeaders,
    capabilities: ModelCapabilities,
    client: Client,
    runtime: Arc<Runtime>,
    max_tokens: u32,
    /// Sampling temperature. `None` lets the server pick its default.
    /// Set to 0.0 for deterministic greedy decoding — required when
    /// comparing two orchestrator modes against the same input.
    temperature: Option<f32>,
}

impl std::fmt::Debug for OpenAiCompatibleAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleAdapter")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

impl OpenAiCompatibleAdapter {
    pub fn new_with(
        base_url: impl Into<String>,
        model: impl Into<String>,
        headers: RequestHeaders,
        capabilities: ModelCapabilities,
        runtime: Arc<Runtime>,
    ) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        Self {
            base_url,
            model: model.into(),
            headers,
            capabilities,
            client: Client::new(),
            runtime,
            max_tokens: 4096,
            temperature: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the default sampling temperature. Pass 0.0 for deterministic
    /// greedy decoding (vLLM and most providers honor this).
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.add(name, value);
        self
    }

    // -- vLLM --

    pub fn vllm(model: impl Into<String>, runtime: Arc<Runtime>) -> Self {
        Self::vllm_at("http://localhost:8000", model, runtime)
    }

    pub fn vllm_at(
        base_url: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self::new_with(
            base_url,
            model,
            RequestHeaders::new(),
            ModelCapabilities {
                supports_tool_calls: true,
                supports_hidden_reasoning: false,
                supports_visible_reasoning: false,
                ..Default::default()
            },
            runtime,
        )
    }

    // -- Perplexity --

    pub fn perplexity(runtime: Arc<Runtime>) -> CawResult<Self> {
        Self::perplexity_model("sonar-pro", runtime)
    }

    pub fn perplexity_model(model: impl Into<String>, runtime: Arc<Runtime>) -> CawResult<Self> {
        let api_key = std::env::var("PERPLEXITY_API_KEY")
            .map_err(|_| CawError::Adapter("PERPLEXITY_API_KEY not set".into()))?;
        Ok(Self::new_with(
            "https://api.perplexity.ai",
            model,
            RequestHeaders::bearer(api_key),
            ModelCapabilities {
                supports_tool_calls: false,
                supports_hidden_reasoning: false,
                supports_visible_reasoning: false,
                ..Default::default()
            },
            runtime,
        ))
    }

    // -- HuggingFace Inference Endpoints --

    pub fn huggingface(
        endpoint_url: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<Runtime>,
    ) -> CawResult<Self> {
        let api_key =
            std::env::var("HF_TOKEN").map_err(|_| CawError::Adapter("HF_TOKEN not set".into()))?;
        Ok(Self::new_with(
            endpoint_url,
            model,
            RequestHeaders::bearer(api_key),
            ModelCapabilities {
                supports_tool_calls: false,
                supports_hidden_reasoning: false,
                supports_visible_reasoning: false,
                ..Default::default()
            },
            runtime,
        ))
    }

    // -- ollama.com (cloud) --

    pub fn ollama_cloud(model: impl Into<String>, runtime: Arc<Runtime>) -> CawResult<Self> {
        let api_key = std::env::var("OLLAMA_API_KEY")
            .map_err(|_| CawError::Adapter("OLLAMA_API_KEY not set".into()))?;
        Ok(Self::new_with(
            "https://api.ollama.com",
            model,
            RequestHeaders::bearer(api_key),
            ModelCapabilities {
                supports_tool_calls: true,
                supports_hidden_reasoning: false,
                supports_visible_reasoning: false,
                ..Default::default()
            },
            runtime,
        ))
    }
}

// -- OpenAI-compatible request/response types --

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct ApiErrorResponse {
    error: Option<ApiErrorDetail>,
}

#[derive(Deserialize)]
struct ApiErrorDetail {
    message: Option<String>,
}

impl ModelAdapter for OpenAiCompatibleAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
        let full_system = format!("{}{}", req.system, workspace_context);
        trace!(model = %self.model, system = %full_system, prompt = %req.user, "→ llm");

        let chat_req = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: full_system,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: req.user,
                },
            ],
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self.runtime.block_on(async {
            let mut request = self
                .client
                .post(&url)
                .header("Content-Type", "application/json");

            for (name, value) in &self.headers.headers {
                request = request.header(name.as_str(), value.as_str());
            }

            let http_response = request
                .json(&chat_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?;

            let status = http_response.status();
            if !status.is_success() {
                let body = http_response.text().await.unwrap_or_default();
                // Try to extract a structured error message
                let detail = serde_json::from_str::<ApiErrorResponse>(&body)
                    .ok()
                    .and_then(|r| r.error)
                    .and_then(|e| e.message)
                    .unwrap_or(body);
                return Err(CawError::Adapter(format!(
                    "{} {} — {}",
                    status.as_u16(),
                    status.canonical_reason().unwrap_or(""),
                    detail
                )));
            }

            http_response
                .json::<ChatCompletionResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        let raw = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        let answer = truncate_at_chat_boundary(&raw).to_string();

        let answer_for_loop_check = strip_fake_recall_blocks(&answer);
        if let Some(reason) = detect_loop(&answer_for_loop_check) {
            tracing::warn!(model = %self.model, reason = %reason, "degenerate output: loop detected");
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
