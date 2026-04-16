use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct GroqAdapter {
    api_key: String,
    model: String,
    client: Client,
}

impl GroqAdapter {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: Client::new(),
        }
    }

    pub fn llama_70b() -> Self {
        let api_key =
            std::env::var("GROQ_API_KEY").unwrap_or_else(|_| panic!("GROQ_API_KEY not set"));
        Self::new(api_key, "llama-3.3-70b-versatile")
    }

    pub fn llama_8b() -> Self {
        let api_key =
            std::env::var("GROQ_API_KEY").unwrap_or_else(|_| panic!("GROQ_API_KEY not set"));
        Self::new(api_key, "llama-3.1-8b-instant")
    }

    pub fn mixtral() -> Self {
        let api_key =
            std::env::var("GROQ_API_KEY").unwrap_or_else(|_| panic!("GROQ_API_KEY not set"));
        Self::new(api_key, "mixtral-8x7b-32768")
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
}

#[derive(Deserialize)]
struct GroqChoice {
    message: GroqMessage,
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

        let full_user_message = format!("{}{}", req.user, workspace_context);

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

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| CawError::Adapter(format!("Failed to create runtime: {}", e)))?;

        let response = runtime.block_on(async {
            self.client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&groq_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?
                .json::<GroqResponse>()
                .await
                .map_err(|e| CawError::Adapter(format!("Failed to parse response: {}", e)))
        })?;

        let answer = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(CompletionResponse { answer })
    }
}
