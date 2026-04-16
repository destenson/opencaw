use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct AnthropicAdapter {
    api_key: String,
    model: String,
    client: Client,
}

impl AnthropicAdapter {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: Client::new(),
        }
    }

    pub fn claude_sonnet() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .unwrap_or_else(|_| panic!("ANTHROPIC_API_KEY not set"));
        Self::new(api_key, "claude-sonnet-4-20250514")
    }

    pub fn claude_opus() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .unwrap_or_else(|_| panic!("ANTHROPIC_API_KEY not set"));
        Self::new(api_key, "claude-opus-4-20250514")
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
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

impl ModelAdapter for AnthropicAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: true,
            supports_hidden_reasoning: self.model.contains("sonnet-4") || self.model.contains("opus-4"),
            supports_visible_reasoning: false,
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
                        "\n<recalled from=\"{source}\" locator=\"{locator}\">\n{content}\n</recalled>",
                        source = f.locator.source,
                        locator = f.locator.locator,
                        content = f.content
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            format!("\n\nRecalled workspace context:\n{}", fragments)
        };

        let full_user_message = format!("{}{}", req.user, workspace_context);

        let anthropic_req = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            system: req.system,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: full_user_message,
            }],
        };

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| CawError::Adapter(format!("Failed to create runtime: {}", e)))?;

        let response = runtime.block_on(async {
            self.client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&anthropic_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?
                .json::<AnthropicResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        let answer = response
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        Ok(CompletionResponse { answer })
    }
}
