use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};
use std::fmt::Write as _;

pub mod adapter_factory;
mod anthropic;
mod claude_code;
mod groq;
#[cfg(feature = "llama")]
mod llama_cpp;
mod ollama;
mod openai_compatible;
mod tracing;

pub use anthropic::AnthropicAdapter;
pub use claude_code::ClaudeCodeAdapter;
pub use groq::GroqAdapter;
#[cfg(feature = "llama")]
pub use llama_cpp::{LlamaCppAdapter, LlamaCppConfig};
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
                ..Default::default()
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

/// Wraps any adapter and prints the contents of each CompletionRequest to
/// stderr before delegating. Lets you verify exactly what context the model
/// receives — system prompt, user message, and every loaded workspace fragment.
pub struct ShowPromptAdapter {
    inner: Box<dyn ModelAdapter>,
}

impl ShowPromptAdapter {
    pub fn new(inner: Box<dyn ModelAdapter>) -> Self {
        Self { inner }
    }
}

fn dump_request(req: &CompletionRequest) {
    let fragment_tokens: usize = req.workspace_fragments.iter().map(|f| f.tokens).sum();
    let mut out = String::new();
    let _ = writeln!(out, "\n─── PROMPT ({} workspace tokens across {} fragments) ───", fragment_tokens, req.workspace_fragments.len());
    let _ = writeln!(out, "[system]\n{}", req.system);
    let _ = writeln!(out, "[user]\n{}", req.user);
    for frag in &req.workspace_fragments {
        let _ = writeln!(out, "[fragment: {} | {} tokens]\n{}", frag.locator.source, frag.tokens, frag.content);
    }
    let _ = writeln!(out, "─────────────────────────");
    eprint!("{out}");
}

impl ModelAdapter for ShowPromptAdapter {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        dump_request(&req);
        self.inner.complete(req)
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        dump_request(&req);
        self.inner.generate_passive(req, check_interval, window_size, on_window)
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<()> {
        dump_request(&req);
        self.inner.thinking_with_steps(req, on_step)
    }
}
