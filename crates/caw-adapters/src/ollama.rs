use caw_core::{
    is_looping, split_thinking, truncate_at_chat_boundary, CawError, CawResult, CompletionRequest,
    CompletionResponse, ModelAdapter, ModelCapabilities, ProvenanceFormat, TokenUsage,
};
use futures_util::StreamExt;
use tracing::{debug, info, trace};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct OllamaAdapter {
    base_url: String,
    model: String,
    client: Client,
    runtime: Arc<Runtime>,
    /// Sampling temperature passed in the request `options`. `None` lets
    /// the model server pick its default (typically 0.8). Set to 0.0 for
    /// deterministic greedy decoding — required when comparing two
    /// orchestrator modes against the same input, since stochastic
    /// sampling otherwise dominates any framework-level signal.
    temperature: Option<f32>,
    /// Context window size passed as `num_ctx` in the Ollama request options.
    /// `None` lets Ollama use the model's default (typically 2048–32768).
    /// Set explicitly to control VRAM usage — KV cache dominates loaded model
    /// size, so capping at 4096 can cut a 32k-default model from 8 GB to ~2 GB.
    num_ctx: Option<u32>,
    /// Whether to signal the orchestrator that this model reliably follows the
    /// cooperative probe/annotation protocol (emitting `<probe>` and `<note>`
    /// markers when instructed). Defaults to false — the orchestrator degrades
    /// to single-shot retrieval rather than injecting instructions that weaker
    /// models will echo as literal text. Enable only for models you've verified
    /// follow the protocol (e.g. via caw-bench-coop).
    cooperative_probes: bool,
    /// Cached result of querying `/api/show` to check whether the model's
    /// chat template handles a `system` role message. Mistral-family models
    /// often omit `{{ .System }}` from their template; sending a system
    /// message to such a model causes degenerate looping output. When false,
    /// the system content is folded into the first user message instead.
    system_supported: std::sync::OnceLock<bool>,
}

impl std::fmt::Debug for OllamaAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaAdapter")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OllamaAdapter {
    pub fn new_with(
        base_url: impl Into<String>,
        model: impl Into<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: Client::new(),
            runtime,
            temperature: None,
            num_ctx: None,
            cooperative_probes: false,
            system_supported: std::sync::OnceLock::new(),
        }
    }

    /// Override the default sampling temperature. Pass 0.0 for deterministic
    /// greedy decoding.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Cap the context window. Controls KV-cache VRAM usage — useful when
    /// running many small models back-to-back in benchmarks.
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    /// Enable cooperative probe/annotation mode for this model. Only set this
    /// after verifying (e.g. via caw-bench-coop) that the model reliably emits
    /// `<probe>` and `<note>` markers when instructed. Without this the
    /// orchestrator uses single-shot retrieval, which avoids confusing literal
    /// marker text in responses from models that can't follow the protocol.
    pub fn with_cooperative_probes(mut self, enabled: bool) -> Self {
        self.cooperative_probes = enabled;
        self
    }

    pub fn local(model: impl Into<String>, runtime: Arc<Runtime>) -> Self {
        Self::new_with("http://localhost:11434", model, runtime)
    }

    pub fn llama3_2(runtime: Arc<Runtime>) -> Self {
        Self::local("llama3.2", runtime)
    }

    pub fn qwen3_5_9b(runtime: Arc<Runtime>) -> Self {
        Self::local("qwen3.5:9b", runtime)
    }

    pub fn deepseek_v3_1(runtime: Arc<Runtime>) -> Self {
        Self::local("deepseek-v3.1:671b-cloud", runtime)
    }

    pub fn phi4_reasoning_3_8b(runtime: Arc<Runtime>) -> Self {
        Self::local("huihui_ai/phi4-reasoning-abliterated:3.8b", runtime)
    }

    pub fn granite4_micro(runtime: Arc<Runtime>) -> Self {
        Self::local("granite4:micro", runtime)
    }

    /// Build the messages array, folding the system content into the first
    /// user message when the model's template doesn't support a system role.
    fn build_messages(&self, system: String, user: String) -> Vec<OllamaChatMessage> {
        let supported = self.system_supported.get_or_init(|| {
            self.runtime
                .block_on(probe_system_support(&self.client, &self.base_url, &self.model))
        });

        if *supported && !system.is_empty() {
            vec![
                OllamaChatMessage { role: "system".to_string(), content: system },
                OllamaChatMessage { role: "user".to_string(), content: user },
            ]
        } else if !system.is_empty() {
            vec![OllamaChatMessage {
                role: "user".to_string(),
                content: format!("{}\n\n{}", system, user),
            }]
        } else {
            vec![OllamaChatMessage { role: "user".to_string(), content: user }]
        }
    }
}

/// Query Ollama's `/api/show` endpoint and return whether the model's chat
/// template includes a system-message placeholder (`{{ .System }}`). Mistral-
/// family models frequently omit it; sending a separate system role message to
/// such a model causes the tokenizer to produce degenerate looping output.
/// Falls back to `true` (assume supported) on any network or parse error so
/// that unexpected failures degrade gracefully rather than silently mangling
/// every request.
async fn probe_system_support(client: &Client, base_url: &str, model: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct ShowResponse {
        template: Option<String>,
    }

    let Ok(resp) = client
        .post(format!("{}/api/show", base_url))
        .json(&serde_json::json!({ "model": model }))
        .send()
        .await
    else {
        return true;
    };

    if !resp.status().is_success() {
        return true;
    }

    let Ok(show) = resp.json::<ShowResponse>().await else {
        return true;
    };

    show.template
        .map(|t| t.contains(".System"))
        .unwrap_or(true)
}

/// Uses Ollama's /api/chat endpoint with proper message roles
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    num_predict: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessage,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaStreamToken {
    message: OllamaChatMessage,
    done: bool,
}

impl ModelAdapter for OllamaAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: false,
            // cooperative_probes gates probe/annotation injection. Default false —
            // models that can't follow the protocol emit markers as literal text.
            // Enable per-model after verifying with caw-bench-coop.
            supports_hidden_reasoning: self.cooperative_probes,
            // Visible reasoning gates <think>-block parsing; detected by model name.
            supports_visible_reasoning: self.model.contains("deepseek")
                || self.model.contains("qwen"),
            ..Default::default()
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
        let full_system = format!("{}{}", req.system, workspace_context);
        trace!(model = %self.model, system = %full_system, prompt = %req.user, "→ llm");

        let ollama_req = OllamaChatRequest {
            model: self.model.clone(),
            messages: self.build_messages(full_system, req.user),
            stream: false,
            options: OllamaOptions {
                temperature: self.temperature,
                num_predict: 4096,
                num_ctx: self.num_ctx,
            },
        };

        let body = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            let resp = self
                .client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {}", e)))?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| CawError::Adapter(format!("read response body: {}", e)))?;
            if !status.is_success() {
                return Err(CawError::Adapter(format!(
                    "Ollama returned HTTP {}: {}",
                    status,
                    text.chars().take(500).collect::<String>()
                )));
            }
            Ok(text)
        })?;

        // Read the body as text first, then parse — when Ollama returns an
        // error envelope (e.g. unknown model) the chat-response shape fails
        // to deserialize and the original error text gets lost. Surfacing
        // both candidate parses gives the caller something to act on.
        match serde_json::from_str::<OllamaChatResponse>(&body) {
            Ok(parsed) => {
                let raw = truncate_at_chat_boundary(&parsed.message.content);
                let (thinking, answer) = split_thinking(raw);
                if is_looping(&answer) {
                    return Err(CawError::DegenerateOutput {
                        model: self.model.clone(),
                        sample: answer.chars().take(120).collect(),
                    });
                }
                let usage = match (parsed.prompt_eval_count, parsed.eval_count) {
                    (Some(i), Some(o)) => Some(TokenUsage { input_tokens: i, output_tokens: o }),
                    _ => None,
                };
                trace!(model = %self.model, answer = %answer, "← llm");
                Ok(CompletionResponse { answer, thinking, usage })
            }
            Err(parse_err) => {
                #[derive(Deserialize)]
                struct OllamaError {
                    error: String,
                }
                if let Ok(err) = serde_json::from_str::<OllamaError>(&body) {
                    return Err(CawError::Adapter(format!("Ollama error: {}", err.error)));
                }
                Err(CawError::Adapter(format!(
                    "Ollama response did not match chat or error schema ({}): {}",
                    parse_err,
                    body.chars().take(500).collect::<String>()
                )))
            }
        }
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<()> {
        if !self.capabilities().supports_visible_reasoning {
            // Non-reasoning model: full completion, split on \n\n, replay.
            let response = self.complete(req)?;
            let thinking = response.thinking.unwrap_or_default();
            for step in thinking
                .split("\n\n")
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                if !on_step(step)? {
                    return Ok(());
                }
            }
            return Ok(());
        }

        debug!(model = %self.model, "streaming thinking trace");

        let workspace_context = req.format_workspace(ProvenanceFormat::Bracketed);
        let full_system = format!("{}{}", req.system, workspace_context);

        let ollama_req = OllamaChatRequest {
            model: self.model.clone(),
            messages: self.build_messages(full_system, req.user),
            stream: true,
            options: OllamaOptions {
                temperature: self.temperature,
                num_predict: 4096,
                num_ctx: self.num_ctx,
            },
        };

        // Stream the thinking trace and collect completed steps. Steps are
        // delimited by \n\n within the <think> block. We stop at </think>
        // without reading the answer tokens — the orchestrator makes a separate
        // complete() call with the enriched workspace for that.
        let steps: Vec<String> = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            let resp = self
                .client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::Adapter(format!("Request failed: {e}")))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(CawError::Adapter(format!("Ollama HTTP {status}: {body}")));
            }

            let mut stream = resp.bytes_stream();
            let mut line_buf = String::new();
            let mut step_buf = String::new();
            let mut steps: Vec<String> = Vec::new();
            let mut in_think = false;

            'outer: while let Some(chunk) = stream.next().await {
                let bytes = chunk.map_err(|e| CawError::Adapter(format!("Stream error: {e}")))?;
                for ch in String::from_utf8_lossy(&bytes).chars() {
                    if ch == '\n' {
                        if let Ok(token) = serde_json::from_str::<OllamaStreamToken>(&line_buf) {
                            let content = &token.message.content;

                            if !in_think {
                                if let Some(after) = content.split_once("<think>").map(|(_, r)| r) {
                                    in_think = true;
                                    step_buf.push_str(after);
                                    debug!("<think> detected — collecting steps");
                                }
                            } else {
                                step_buf.push_str(content);
                            }

                            if in_think {
                                if let Some(end) = step_buf.find("</think>") {
                                    let step = step_buf[..end].trim().to_string();
                                    if !step.is_empty() {
                                        debug!(
                                            step_preview = &step[..step.len().min(80)],
                                            "step at </think>"
                                        );
                                        steps.push(step);
                                    }
                                    debug!("</think> detected — stopping stream");
                                    break 'outer;
                                }
                                // Flush a completed step at \n\n boundary
                                while let Some(boundary) = step_buf.find("\n\n") {
                                    let step = step_buf[..boundary].trim().to_string();
                                    step_buf.drain(..boundary + 2);
                                    if !step.is_empty() {
                                        debug!(
                                            step_preview = &step[..step.len().min(80)],
                                            "step boundary flushed"
                                        );
                                        steps.push(step);
                                    }
                                }
                            }

                            if token.done {
                                break 'outer;
                            }
                        }
                        line_buf.clear();
                    } else {
                        line_buf.push(ch);
                    }
                }
            }

            Ok(steps)
        })?;

        info!(model = %self.model, steps = steps.len(), "thinking trace collected");

        // Replay collected steps through the callback synchronously.
        // on_step returning false means new context was found — the
        // orchestrator will restart with the enriched workspace.
        for (i, step) in steps.iter().enumerate() {
            if !on_step(step)? {
                debug!(step_index = i, "early stop — new context admitted at this step");
                break;
            }
        }
        Ok(())
    }
}
