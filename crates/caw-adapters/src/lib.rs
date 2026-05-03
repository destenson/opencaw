use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};

pub mod adapter_factory;
mod anthropic;
mod claude_code;
mod groq;
mod ollama;
mod openai_compatible;
mod tracing;

pub use anthropic::AnthropicAdapter;
pub use claude_code::ClaudeCodeAdapter;
pub use groq::GroqAdapter;
pub use ollama::OllamaAdapter;
pub use openai_compatible::{OpenAiCompatibleAdapter, RequestHeaders};
use ::tracing::trace;
pub use tracing::{TraceSink, TracingAdapter};

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
        let name = name.into();
        trace!("New mock adapter: {}", name);
        Self {
            name,
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
        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);

        Ok(CompletionResponse {
            answer: format!(
                "[{}] synthesized answer for: {}{}",
                self.name, req.user, workspace_context,
            ),
            thinking: None,
            usage: None,
        })
    }
}
