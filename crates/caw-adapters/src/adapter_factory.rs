//! Build a `Box<dyn ModelAdapter>` from a (kind, model, endpoint) triple.
//! Centralizes the adapter-selection logic so any binary in the workspace
//! can share the same flag semantics.
//!
//! The `cli` feature gates `clap::ValueEnum` on `AdapterKind` so CLI tools
//! can use it directly as a parsed argument type without making `clap` a
//! dependency for library consumers.

use anyhow::Result;
use std::sync::Arc;

use crate::{
    AnthropicAdapter, ClaudeCodeAdapter, GroqAdapter, OllamaAdapter, OpenAiCompatibleAdapter,
    RequestHeaders, TraceSink, TracingAdapter,
};
use caw_core::{ModelAdapter, ModelCapabilities};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum AdapterKind {
    /// Local Ollama at `--ollama-url` (default http://localhost:11434).
    Ollama,
    /// vLLM or any OpenAI chat-completions-compatible server at the URL
    /// passed via `--openai-url`. Reads `OPENAI_COMPATIBLE_API_KEY` if set.
    Vllm,
    /// Local `claude` CLI; model name is a Claude alias ("sonnet", "haiku").
    ClaudeCode,
    /// Groq cloud API. Reads `GROQ_API_KEY` from the environment.
    Groq,
    /// Anthropic cloud API. Reads `ANTHROPIC_API_KEY` from the environment.
    Anthropic,
}

/// The intended use of an adapter, used to select an appropriate default model.
/// Callers always know their role, so defaults are role-specific rather than
/// a single arbitrary choice per adapter.
#[derive(Debug, Clone, Copy)]
pub enum ModelRole {
    /// Primary answering / response generation — prefers a capable instruct model.
    Answer,
    /// Scoring and evaluation — prefers a model from a different family than the
    /// answer model to avoid self-agreement bias; mid-tier is usually sufficient.
    Judge,
    /// Query intent classification — small, fast model is sufficient.
    Classify,
}

impl AdapterKind {
    /// Default model name for this adapter in the given role. Each adapter
    /// family has its own naming conventions; defaults vary by role to match
    /// capability requirements (e.g. a small model for classification, a
    /// mid-tier model for judging).
    pub fn default_model(self, role: ModelRole) -> &'static str {
        match (self, role) {
            (AdapterKind::Ollama, ModelRole::Answer) => "hf.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:Q6_K",
            (AdapterKind::Ollama, ModelRole::Judge) => "qwen3.5:9b",
            (AdapterKind::Ollama, ModelRole::Classify) => "granite4:micro",
            (AdapterKind::Vllm, _) => "",
            (AdapterKind::ClaudeCode, ModelRole::Answer) => "sonnet",
            (AdapterKind::ClaudeCode, ModelRole::Judge) => "haiku",
            (AdapterKind::ClaudeCode, ModelRole::Classify) => "haiku",
            (AdapterKind::Groq, ModelRole::Answer) => "llama-3.3-70b-versatile",
            (AdapterKind::Groq, ModelRole::Judge) => "llama-3.3-70b-versatile",
            (AdapterKind::Groq, ModelRole::Classify) => "llama-3.1-8b-instant",
            (AdapterKind::Anthropic, ModelRole::Answer) => "claude-sonnet-4-20250514",
            (AdapterKind::Anthropic, ModelRole::Judge) => "claude-haiku-4-20250514",
            (AdapterKind::Anthropic, ModelRole::Classify) => "claude-haiku-4-20250514",
        }
    }
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
    /// Cap Ollama's context window (`num_ctx`). `None` uses the model default.
    /// Cutting context to 4096 on 32k-default models can reduce loaded VRAM
    /// from 8+ GB to ~2 GB, which matters when running many candidates back
    /// to back on a single GPU.
    pub num_ctx: Option<u32>,
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
            if let Some(ctx) = spec.num_ctx {
                a = a.with_num_ctx(ctx);
            }
            Box::new(a)
        }
        AdapterKind::Vllm => {
            let headers = match std::env::var("OPENAI_COMPATIBLE_API_KEY") {
                Ok(key) => RequestHeaders::bearer(key),
                Err(_) => RequestHeaders::new(),
            };
            // Cooperative probe/annotation injection is off by default — models
            // that can't follow the protocol emit markers as literal text.
            // Enable supports_hidden_reasoning here after verifying with caw-bench-coop.
            let mut a = OpenAiCompatibleAdapter::new_with(
                spec.openai_url,
                spec.model,
                headers,
                ModelCapabilities {
                    supports_tool_calls: false,
                    supports_hidden_reasoning: false,
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
        AdapterKind::Groq => Box::new(GroqAdapter::groq_model(spec.model, runtime.clone())?),
        AdapterKind::Anthropic => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?;
            Box::new(AnthropicAdapter::new_with(api_key, spec.model, runtime.clone()))
        }
    };
    Ok(adapter)
}
