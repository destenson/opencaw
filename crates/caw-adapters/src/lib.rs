use caw_core::{CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities};

mod anthropic;
mod groq;
mod ollama;

pub use anthropic::AnthropicAdapter;
pub use groq::GroqAdapter;
pub use ollama::OllamaAdapter;

/// Create a shared tokio runtime for all adapters.
/// Call this once at startup and pass the Arc to each adapter.
pub fn create_runtime() -> CawResult<std::sync::Arc<tokio::runtime::Runtime>> {
    tokio::runtime::Runtime::new()
        .map(std::sync::Arc::new)
        .map_err(|e| caw_core::CawError::Adapter(format!("Failed to create runtime: {}", e)))
}

#[derive(Debug, Clone)]
pub struct MockAdapter {
    name: String,
    caps: ModelCapabilities,
}

impl MockAdapter {
    pub fn new(name: impl Into<String>, supports_visible_reasoning: bool) -> Self {
        Self {
            name: name.into(),
            caps: ModelCapabilities {
                supports_tool_calls: true,
                supports_hidden_reasoning: false,
                supports_visible_reasoning,
            },
        }
    }
}

impl ModelAdapter for MockAdapter {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.caps
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let sources = req
            .workspace_fragments
            .iter()
            .map(|f| format!("{}:{}", f.locator.source, f.locator.locator))
            .collect::<Vec<_>>();

        Ok(CompletionResponse {
            answer: format!(
                "[{}] synthesized answer for: {}\n\nGrounded sources:\n- {}",
                self.name,
                req.user,
                sources.join("\n- ")
            ),
        })
    }
}
