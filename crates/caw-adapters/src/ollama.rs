use caw_core::{
    detect_loop, split_thinking, strip_fake_recall_blocks, truncate_at_chat_boundary, CawError,
    CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat, TokenUsage,
};
use futures_util::StreamExt;
use tracing::{debug, info, trace};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Default `num_predict` (max generated tokens per completion). High enough to
/// hold a full reasoning trace plus answer for the models used here; override
/// via [`OllamaAdapter::with_num_predict`] when measuring shorter caps.
const DEFAULT_NUM_PREDICT: i32 = 4096;

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
    /// Max tokens the model may generate per completion, sent as `num_predict`.
    /// For a reasoning model this budget covers the thinking trace *and* the
    /// visible answer, so capping it too low truncates the answer before it is
    /// emitted. Defaults to `DEFAULT_NUM_PREDICT`; lower it only with a paired
    /// answer-quality measurement (the trace is the thesis mechanism).
    num_predict: i32,
    /// Whether to signal the orchestrator that this model reliably follows the
    /// cooperative probe/annotation protocol (emitting `<probe>` and `<note>`
    /// markers when instructed). Defaults to false — the orchestrator degrades
    /// to single-shot retrieval rather than injecting instructions that weaker
    /// models will echo as literal text. Enable only for models you've verified
    /// follow the protocol (e.g. via caw-bench-coop).
    cooperative_probes: bool,
    /// When true, the system content is concatenated into the first user
    /// message instead of being sent as a separate `system`-role message.
    /// Default false: send a real system message and let Ollama apply the
    /// model's chat template. Enable this only for a specific model whose
    /// template lacks a system slot (some Mistral-family templates omit
    /// `{{ .System }}`, so a system message is dropped or — worse — triggers
    /// degenerate looping). The correct alternative is to fix that model's
    /// template; this flag is the escape hatch when you can't.
    fold_system: bool,
    /// Cached answer to "does this model emit a reasoning trace?" Resolved
    /// lazily on first use by querying Ollama's `/api/show` capabilities
    /// (the authoritative, non-heuristic source), falling back to a model-name
    /// match only when `/api/show` is unreachable. Gates whether we send
    /// `"think": true` — sending it to a non-reasoning model crashes
    /// llama-server (observed GGML_ASSERT on gemma4) — and whether the
    /// orchestrator routes through the streaming thinking-trace path.
    thinking_supported: std::sync::OnceLock<bool>,
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
            num_predict: DEFAULT_NUM_PREDICT,
            cooperative_probes: false,
            fold_system: false,
            thinking_supported: std::sync::OnceLock::new(),
        }
    }

    /// Whether this model emits a reasoning trace the orchestrator can consume.
    /// Resolved once via `/api/show` capabilities; on any error (Ollama
    /// unreachable, old server without the `capabilities` field) falls back to
    /// the model-name match that predated the capability query. The result is
    /// cached for the adapter's lifetime.
    fn supports_thinking(&self) -> bool {
        *self.thinking_supported.get_or_init(|| match self.query_thinking_capability() {
            Some(v) => v,
            None => self.model.contains("deepseek") || self.model.contains("qwen"),
        })
    }

    /// Query `/api/show` for the model's declared capabilities. Returns
    /// `Some(true)` if "thinking" is advertised, `Some(false)` if the model
    /// responded without it, and `None` if the request failed (caller then
    /// falls back to the name heuristic).
    fn query_thinking_capability(&self) -> Option<bool> {
        #[derive(Deserialize)]
        struct ShowResponse {
            capabilities: Option<Vec<String>>,
        }
        let url = format!("{}/api/show", self.base_url);
        let model = self.model.clone();
        let resp: Option<ShowResponse> = self.runtime.block_on(async {
            self.client
                .post(&url)
                .json(&serde_json::json!({ "model": model }))
                .send()
                .await
                .ok()?
                .json::<ShowResponse>()
                .await
                .ok()
        });
        resp.map(|r| {
            r.capabilities
                .unwrap_or_default()
                .iter()
                .any(|c| c == "thinking")
        })
    }

    /// Concatenate the system content into the first user message instead of
    /// sending a separate `system`-role message. Only needed for a model whose
    /// chat template lacks a `{{ .System }}` slot; see the field docs.
    pub fn with_fold_system(mut self, fold_system: bool) -> Self {
        self.fold_system = fold_system;
        self
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

    /// Cap the per-completion generation budget (`num_predict`). For a
    /// reasoning model this covers the thinking trace plus the visible answer;
    /// setting it too low truncates the answer. Use only with a paired
    /// answer-quality measurement.
    pub fn with_num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = num_predict;
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

    /// Build the messages array. By default a non-empty system goes in a real
    /// `system`-role message; with `fold_system` it is concatenated into the
    /// first user message instead (see the field docs).
    fn build_messages(&self, system: String, user: String) -> Vec<OllamaChatMessage> {
        if system.is_empty() {
            vec![OllamaChatMessage::user(user)]
        } else if self.fold_system {
            vec![OllamaChatMessage::user(format!("{}\n\n{}", system, user))]
        } else {
            vec![OllamaChatMessage::system(system), OllamaChatMessage::user(user)]
        }
    }
}

/// Uses Ollama's /api/chat endpoint with proper message roles
#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    stream: bool,
    options: OllamaOptions,
    /// Request the model's reasoning trace. Sent only for models that advertise
    /// the `thinking` capability — `true` on a non-reasoning model crashes
    /// llama-server. Omitted (None) otherwise so the request shape is unchanged
    /// for non-reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaChatMessage {
    role: String,
    content: String,
    /// Reasoning trace. Modern Ollama (0.30+) returns the model's thinking here
    /// — streamed token-by-token while `content` stays empty — rather than as an
    /// inline `<think>` block in `content`. Never sent on requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

impl OllamaChatMessage {
    fn user(content: String) -> Self {
        Self { role: "user".to_string(), content, thinking: None }
    }

    fn system(content: String) -> Self {
        Self { role: "system".to_string(), content, thinking: None }
    }
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
    /// Nanosecond timings Ollama reports for every non-streaming response.
    /// Logged so the per-completion cost (prompt processing vs token
    /// generation throughput) is visible without external profiling.
    total_duration: Option<u64>,
    /// Time Ollama spent loading the model for this call. Large and recurring
    /// across turns means models are being swapped in/out of VRAM (thrash),
    /// which is a config problem, not generation cost.
    load_duration: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_duration: Option<u64>,
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
            // Gates the streaming thinking-trace path. Sourced from Ollama's
            // `/api/show` capabilities (name match only as offline fallback).
            supports_visible_reasoning: self.supports_thinking(),
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
                num_predict: self.num_predict,
                num_ctx: self.num_ctx,
            },
            think: self.supports_thinking().then_some(true),
        };

        let body = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            let resp = self
                .client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::External(format!("Request failed: {}", e)))?;
            let status = resp.status();
            let text = resp
                .text()
                .await
                .map_err(|e| CawError::External(format!("read response body: {}", e)))?;
            if !status.is_success() {
                return Err(CawError::External(format!(
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
                // Prefer the dedicated `thinking` field (modern Ollama); fall
                // back to splitting an inline <think> block out of content for
                // models/servers that still embed reasoning there.
                let field_thinking = parsed
                    .message
                    .thinking
                    .clone()
                    .filter(|s| !s.trim().is_empty());
                let raw = truncate_at_chat_boundary(&parsed.message.content);
                let (inline_thinking, answer) = split_thinking(raw);
                let thinking = field_thinking.or(inline_thinking);
                if answer.trim().is_empty() {
                    tracing::warn!(model = %self.model, "degenerate output: blank answer after split_thinking");
                    return Err(CawError::DegenerateOutput {
                        model: self.model.clone(),
                        sample: answer.chars().take(120).collect(),
                    });
                }
                let answer_for_loop_check = strip_fake_recall_blocks(&answer);
                if let Some(reason) = detect_loop(&answer_for_loop_check) {
                    tracing::warn!(model = %self.model, reason = %reason, "degenerate output: loop detected");
                    return Err(CawError::DegenerateOutput {
                        model: self.model.clone(),
                        sample: answer.chars().take(120).collect(),
                    });
                }
                let usage = match (parsed.prompt_eval_count, parsed.eval_count) {
                    (Some(i), Some(o)) => Some(TokenUsage { input_tokens: i, output_tokens: o }),
                    _ => None,
                };
                // Report the cost of this completion from Ollama's own timings.
                // gen_tps (eval tokens / eval seconds) is the throughput knob;
                // a large prompt_ms with small eval_ms means context size, not
                // generation, dominates the call.
                let ms = |ns: Option<u64>| ns.map(|n| n as f64 / 1e6).unwrap_or(0.0);
                let eval_s = ms(parsed.eval_duration) / 1e3;
                let gen_tps = if eval_s > 0.0 {
                    parsed.eval_count.unwrap_or(0) as f64 / eval_s
                } else {
                    0.0
                };
                info!(
                    model = %self.model,
                    prompt_tokens = parsed.prompt_eval_count.unwrap_or(0),
                    eval_tokens = parsed.eval_count.unwrap_or(0),
                    load_ms = format_args!("{:.0}", ms(parsed.load_duration)),
                    prompt_ms = format_args!("{:.0}", ms(parsed.prompt_eval_duration)),
                    eval_ms = format_args!("{:.0}", ms(parsed.eval_duration)),
                    total_ms = format_args!("{:.0}", ms(parsed.total_duration)),
                    gen_tps = format_args!("{:.1}", gen_tps),
                    "ollama completion timing"
                );
                trace!(model = %self.model, answer = %answer, "← llm");
                Ok(CompletionResponse { answer, thinking, usage })
            }
            Err(parse_err) => {
                #[derive(Deserialize)]
                struct OllamaError {
                    error: String,
                }
                if let Ok(err) = serde_json::from_str::<OllamaError>(&body) {
                    return Err(CawError::External(format!("Ollama error: {}", err.error)));
                }
                Err(CawError::External(format!(
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
    ) -> CawResult<CompletionResponse> {
        if !self.capabilities().supports_visible_reasoning {
            // Non-reasoning model: full completion, split on \n\n, replay.
            let response = self.complete(req)?;
            let thinking = response.clone().thinking.unwrap_or_default();
            for step in thinking
                .split("\n\n")
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                if !on_step(step)? {
                    return Ok(response);
                }
            }
            return Ok(response);
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
                num_predict: self.num_predict,
                num_ctx: self.num_ctx,
            },
            think: Some(true),
        };

        // Stream the thinking trace and collect completed steps. Steps are
        // delimited by \n\n. Modern Ollama streams reasoning in each chunk's
        // `message.thinking` (with `content` empty until reasoning ends); older
        // servers/models embed it inline as a <think>…</think> block in
        // `content`. We handle both, and stop once the answer text begins (or
        // the inline block closes) — the orchestrator makes a separate
        // complete() call with the enriched workspace for the answer itself.
        let steps: Vec<String> = self.runtime.block_on(async {
            let url = format!("{}/api/chat", self.base_url);
            let resp = self
                .client
                .post(&url)
                .json(&ollama_req)
                .send()
                .await
                .map_err(|e| CawError::External(format!("Request failed: {e}")))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(CawError::External(format!("Ollama HTTP {status}: {body}")));
            }

            let mut stream = resp.bytes_stream();
            let mut line_buf = String::new();
            let mut step_buf = String::new();
            let mut steps: Vec<String> = Vec::new();
            let mut in_think = false;

            // Flush any \n\n-delimited completed steps out of the buffer.
            let flush_steps = |buf: &mut String, steps: &mut Vec<String>| {
                while let Some(boundary) = buf.find("\n\n") {
                    let step = buf[..boundary].trim().to_string();
                    buf.drain(..boundary + 2);
                    if !step.is_empty() {
                        debug!(step_preview = &step[..step.len().min(80)], "step boundary flushed");
                        steps.push(step);
                    }
                }
            };

            'outer: while let Some(chunk) = stream.next().await {
                let bytes = chunk.map_err(|e| CawError::External(format!("Stream error: {e}")))?;
                for ch in String::from_utf8_lossy(&bytes).chars() {
                    if ch != '\n' {
                        line_buf.push(ch);
                        continue;
                    }
                    if let Ok(token) = serde_json::from_str::<OllamaStreamToken>(&line_buf) {
                        let msg = &token.message;

                        // Modern Ollama: reasoning arrives as `thinking` deltas.
                        if let Some(t) = msg.thinking.as_deref().filter(|t| !t.is_empty()) {
                            if !in_think {
                                in_think = true;
                                debug!("thinking field detected — collecting steps");
                            }
                            step_buf.push_str(t);
                            flush_steps(&mut step_buf, &mut steps);
                        } else if in_think && msg.thinking.is_none() {
                            // Inline-block model: keep appending content tokens
                            // until the closing tag.
                            step_buf.push_str(&msg.content);
                        } else if !in_think {
                            // Inline fallback: reasoning embedded in content as
                            // an opening <think> tag.
                            if let Some(after) = msg.content.split_once("<think>").map(|(_, r)| r) {
                                in_think = true;
                                step_buf.push_str(after);
                                debug!("<think> detected — collecting steps");
                            }
                        }

                        if in_think {
                            if let Some(end) = step_buf.find("</think>") {
                                let step = step_buf[..end].trim().to_string();
                                if !step.is_empty() {
                                    debug!(step_preview = &step[..step.len().min(80)], "step at </think>");
                                    steps.push(step);
                                }
                                debug!("</think> detected — stopping stream");
                                break 'outer;
                            }
                            flush_steps(&mut step_buf, &mut steps);
                            // The thinking field carries no closing tag; the
                            // reasoning phase ends when answer content begins.
                            if msg.thinking.is_none() && !msg.content.is_empty() {
                                let tail = step_buf.trim();
                                if !tail.is_empty() {
                                    steps.push(tail.to_string());
                                }
                                debug!("answer content started — stopping thinking stream");
                                break 'outer;
                            }
                        }

                        if token.done {
                            let tail = step_buf.trim();
                            if !tail.is_empty() {
                                steps.push(tail.to_string());
                            }
                            break 'outer;
                        }
                    }
                    line_buf.clear();
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
        Ok(CompletionResponse {
            answer: String::new(),
            thinking: None,
            usage: None,
        })
    }
}
