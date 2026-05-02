//! Build a `Box<dyn ModelAdapter>` from a (kind, model, endpoint) triple.
//! Centralizes the adapter-selection logic so the runner and judge paths
//! share the same flag semantics.

use anyhow::Result;
use std::sync::Arc;

use caw_adapters::{
    ClaudeCodeAdapter, OllamaAdapter, OpenAiCompatibleAdapter, RequestHeaders, TraceSink,
    TracingAdapter,
};
use caw_core::{ModelAdapter, ModelCapabilities};

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum AdapterKind {
    /// Local Ollama at `--ollama-url` (default http://localhost:11434).
    Ollama,
    /// vLLM or any OpenAI chat-completions-compatible server at the URL
    /// passed via `--openai-url`. Reads `OPENAI_COMPATIBLE_API_KEY` if set.
    Vllm,
    /// Local `claude` CLI; model name is a Claude alias ("sonnet", "haiku").
    ClaudeCode,
}

pub struct AdapterSpec<'a> {
    pub kind: AdapterKind,
    pub model: &'a str,
    pub ollama_url: &'a str,
    pub openai_url: &'a str,
    /// Sampling temperature for ollama / vllm. `None` lets the server pick
    /// its default; `Some(0.0)` forces deterministic greedy decoding so a
    /// repeated invocation with the same prompt yields the same answer.
    /// ClaudeCodeAdapter ignores this — claude CLI doesn't expose a
    /// temperature flag in `--print` mode.
    pub temperature: Option<f32>,
}

pub fn build(
    spec: AdapterSpec<'_>,
    runtime: &Arc<tokio::runtime::Runtime>,
) -> Result<Box<dyn ModelAdapter>> {
    let adapter = build_inner(spec, runtime)?;

    // If CAW_TRACE_FILE is set, wrap the adapter so every complete() call
    // appends llm_request / llm_response events. The sweep driver sets
    // this per cell automatically; callers can set it themselves for
    // one-off debugging.
    match TraceSink::from_env()? {
        Some(sink) => Ok(Box::new(TracingAdapter::new(adapter, sink))),
        None => Ok(adapter),
    }
}

fn build_inner(
    spec: AdapterSpec<'_>,
    runtime: &Arc<tokio::runtime::Runtime>,
) -> Result<Box<dyn ModelAdapter>> {
    let adapter: Box<dyn ModelAdapter> = match spec.kind {
        AdapterKind::Ollama => {
            let mut a = OllamaAdapter::new_with(spec.ollama_url, spec.model, runtime.clone());
            if let Some(t) = spec.temperature {
                a = a.with_temperature(t);
            }
            Box::new(a)
        }
        AdapterKind::Vllm => {
            let headers = match std::env::var("OPENAI_COMPATIBLE_API_KEY") {
                Ok(key) => RequestHeaders::bearer(key),
                Err(_) => RequestHeaders::new(),
            };
            // vLLM and other OpenAI-protocol servers serving instruct models
            // qualify for marker-emission instructions per the same logic as
            // OllamaAdapter — flag hidden_reasoning so the orchestrator
            // injects probe/note prompts.
            let mut a = OpenAiCompatibleAdapter::new_with(
                spec.openai_url,
                spec.model,
                headers,
                ModelCapabilities {
                    supports_tool_calls: false,
                    supports_hidden_reasoning: true,
                    supports_visible_reasoning: false,
                },
                runtime.clone(),
            );
            if let Some(t) = spec.temperature {
                a = a.with_temperature(t);
            }
            Box::new(a)
        }
        AdapterKind::ClaudeCode => Box::new(ClaudeCodeAdapter::builder().model(spec.model).build()),
    };
    Ok(adapter)
}
