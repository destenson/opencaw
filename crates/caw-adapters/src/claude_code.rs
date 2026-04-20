use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// How long to wait between streamed NDJSON events before declaring the
/// claude CLI hung. The CLI with `--output-format stream-json` emits an
/// event for session init, per-token generation, thinking blocks, and the
/// final `result`. A silent gap longer than this almost always means the
/// upstream API stopped responding and we're burning wallclock waiting
/// for TCP keepalives to decide. 60s is comfortably above thinking-block
/// pauses on hard prompts but aggressive enough that a single stuck call
/// can't stall a batch run for minutes.
const STREAM_EVENT_TIMEOUT: Duration = Duration::from_secs(60);

/// Adapter that shells out to the Claude Code CLI in single-shot print mode.
/// Intended for cheap auxiliary tasks (summarization, consolidation, outline
/// generation) where the subsidized CLI pricing beats direct API calls.
pub struct ClaudeCodeAdapter {
    model: String,
    /// Tool names to allow (e.g. "Bash(git *)", "Read"). Empty = no tools.
    allowed_tools: Vec<String>,
    /// Per-call budget cap in USD. None = no cap.
    max_budget_usd: Option<f64>,
    /// Additional directories to grant tool access to.
    add_dirs: Vec<String>,
    /// Effort level: "low", "medium", "high", "max". Lower effort = cheaper.
    effort: String,
}

impl std::fmt::Debug for ClaudeCodeAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeCodeAdapter")
            .field("model", &self.model)
            .field("allowed_tools", &self.allowed_tools)
            .finish()
    }
}

/// Builder for ClaudeCodeAdapter — all configuration is optional,
/// defaults produce a minimal no-tools sonnet adapter.
pub struct ClaudeCodeAdapterBuilder {
    model: String,
    allowed_tools: Vec<String>,
    max_budget_usd: Option<f64>,
    add_dirs: Vec<String>,
    effort: String,
}

impl ClaudeCodeAdapterBuilder {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }

    pub fn max_budget_usd(mut self, budget: f64) -> Self {
        self.max_budget_usd = Some(budget);
        self
    }

    pub fn add_dir(mut self, dir: impl Into<String>) -> Self {
        self.add_dirs.push(dir.into());
        self
    }

    pub fn effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = effort.into();
        self
    }

    pub fn build(self) -> ClaudeCodeAdapter {
        ClaudeCodeAdapter {
            model: self.model,
            allowed_tools: self.allowed_tools,
            max_budget_usd: self.max_budget_usd,
            add_dirs: self.add_dirs,
            effort: self.effort,
        }
    }
}

impl ClaudeCodeAdapter {
    pub fn builder() -> ClaudeCodeAdapterBuilder {
        ClaudeCodeAdapterBuilder {
            model: "sonnet".to_string(),
            allowed_tools: Vec::new(),
            max_budget_usd: None,
            add_dirs: Vec::new(),
            effort: "low".to_string(),
        }
    }

    /// Sonnet with no tools — cheapest option for pure text tasks.
    pub fn sonnet() -> Self {
        Self::builder().build()
    }

    /// Haiku for the cheapest/fastest auxiliary tasks.
    pub fn haiku() -> Self {
        Self::builder().model("haiku").build()
    }

    /// Sonnet with read-only file access for tasks that need to inspect files.
    pub fn sonnet_with_read(dirs: Vec<String>) -> Self {
        let mut builder = Self::builder().allowed_tools(vec![
            "Read".to_string(),
            "Glob".to_string(),
            "Grep".to_string(),
        ]);
        for dir in dirs {
            builder = builder.add_dir(dir);
        }
        builder.build()
    }
}

/// Deserialization of the `type: "result"` event emitted at the end of a
/// `--output-format stream-json` session. The CLI emits many event types
/// (system/init, rate_limit_event, assistant, result); we only parse the
/// terminal `result` here, which carries the final string plus cost/latency
/// telemetry. Intermediate events are used only as a liveness signal.
#[derive(Deserialize)]
struct ClaudeStreamResult {
    result: String,
    #[serde(default)]
    is_error: bool,
    #[serde(default)]
    total_cost_usd: Option<f64>,
    #[serde(default)]
    duration_ms: Option<u64>,
}

#[derive(Deserialize)]
struct EventHeader {
    #[serde(rename = "type")]
    ty: String,
}

impl ModelAdapter for ClaudeCodeAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        // Claude models reason internally and will follow marker-emission
        // instructions even though `--print` doesn't surface the reasoning
        // trace. Flagging hidden-reasoning signals to the orchestrator that
        // probe/annotation instructions are worth injecting — visible trace
        // parsing stays off because the CLI doesn't expose it.
        ModelCapabilities {
            supports_tool_calls: !self.allowed_tools.is_empty(),
            supports_hidden_reasoning: true,
            supports_visible_reasoning: false,
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Xml);
        let full_user_message = format!("{}{}", req.user, workspace_context);

        // Use stream-json so a reader-side watchdog can detect upstream
        // hangs. Previously the CLI could sit in ep_poll indefinitely with
        // `--output-format json` because that format only flushes on the
        // terminal `result` event — so an inter-event gap was
        // indistinguishable from slow generation to any outside observer.
        // With stream-json each assistant event becomes a liveness tick.
        let mut cmd = Command::new("claude");
        cmd.arg("--print")
            .arg("--output-format")
            .arg("stream-json")
            // `--output-format stream-json` requires `--verbose` when used
            // with `--print`; claude CLI hard-errors otherwise.
            .arg("--verbose")
            .arg("--model")
            .arg(&self.model)
            .arg("--system-prompt")
            .arg(&req.system)
            .arg("--effort")
            .arg(&self.effort)
            .arg("--no-session-persistence")
            // Skip project/local CLAUDE.md + settings so programmatic
            // callers get predictable context, independent of whatever
            // repo they happen to be invoked from.
            .arg("--setting-sources")
            .arg("user")
            .arg("--disable-slash-commands");

        if let Some(budget) = self.max_budget_usd {
            cmd.arg("--max-budget-usd").arg(budget.to_string());
        }

        if !self.allowed_tools.is_empty() {
            cmd.arg("--allowedTools");
            for tool in &self.allowed_tools {
                cmd.arg(tool);
            }
        } else {
            cmd.arg("--tools").arg("");
        }

        for dir in &self.add_dirs {
            cmd.arg("--add-dir").arg(dir);
        }

        // Pass the prompt via stdin rather than as a positional argument.
        // `--tools <tools...>` is variadic and greedily swallows trailing
        // positional args; stdin delivery sidesteps that argument-parsing
        // trap and also avoids any ARG_MAX ceiling on very large prompts.
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| CawError::Adapter(format!("Failed to spawn claude CLI: {}", e)))?;

        // Write the full prompt, then drop stdin so the CLI sees EOF and
        // starts generating. If we held the handle open the CLI would wait
        // for more input and the whole call would deadlock.
        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| CawError::Adapter("claude CLI stdin unavailable".to_string()))?;
            stdin
                .write_all(full_user_message.as_bytes())
                .map_err(|e| CawError::Adapter(format!("write prompt to claude stdin: {}", e)))?;
            // dropped here
        }

        // A background thread owns the stdout read — sync Rust has no
        // read-with-timeout for pipes, so the channel is the escape hatch:
        // the main loop does `recv_timeout(STREAM_EVENT_TIMEOUT)` and kills
        // the child if no event arrives. Each NDJSON line is one event.
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CawError::Adapter("claude CLI stdout unavailable".to_string()))?;
        let (line_tx, line_rx) = mpsc::channel::<std::io::Result<String>>();
        let reader_thread = std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                // An error here (EOF, broken pipe) is still a valid
                // signal — surface it so the main loop can distinguish
                // "clean EOF with no result event" from "timed out".
                let is_err = line.is_err();
                if line_tx.send(line).is_err() {
                    break;
                }
                if is_err {
                    break;
                }
            }
        });

        let mut final_result: Option<ClaudeStreamResult> = None;
        let timeout_reason;
        loop {
            match line_rx.recv_timeout(STREAM_EVENT_TIMEOUT) {
                Ok(Ok(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    // Peek at the event type; only parse the full body
                    // for the one we care about. Unknown event types are
                    // just liveness ticks.
                    let ty = match serde_json::from_str::<EventHeader>(trimmed) {
                        Ok(h) => h.ty,
                        Err(_) => continue,
                    };
                    if ty == "result" {
                        match serde_json::from_str::<ClaudeStreamResult>(trimmed) {
                            Ok(r) => {
                                final_result = Some(r);
                                break;
                            }
                            Err(e) => {
                                let _ = child.kill();
                                return Err(CawError::Adapter(format!(
                                    "parse stream-json result event: {} (line: {})",
                                    e, trimmed
                                )));
                            }
                        }
                    }
                }
                Ok(Err(io_err)) => {
                    timeout_reason = format!("stdout read error: {}", io_err);
                    return finalize_error(&mut child, timeout_reason);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let _ = child.kill();
                    let _ = reader_thread.join();
                    return Err(CawError::Adapter(format!(
                        "claude CLI stalled: no stream-json event for {}s — child killed",
                        STREAM_EVENT_TIMEOUT.as_secs()
                    )));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Reader finished before we saw a result event. Fall
                    // through to collect child status and stderr.
                    timeout_reason =
                        "stream ended without a result event".to_string();
                    return finalize_error(&mut child, timeout_reason);
                }
            }
        }

        // Reader may still be draining trailing bytes; join so we don't
        // leak the thread and so any pending stderr has time to land.
        let _ = reader_thread.join();
        let _ = child.wait();

        let parsed = final_result.expect("loop exits with final_result set on success path");
        if parsed.is_error {
            return Err(CawError::Adapter(format!(
                "claude CLI returned error: {}",
                parsed.result
            )));
        }

        if let (Some(cost), Some(ms)) = (parsed.total_cost_usd, parsed.duration_ms) {
            eprintln!(
                "[claude-code] model={} cost=${:.4} duration={}ms",
                self.model, cost, ms
            );
        }

        Ok(CompletionResponse {
            answer: parsed.result,
        })
    }
}

/// Shared error-path cleanup: kill the child if it's still alive, drain
/// stderr for diagnostics, and format an adapter error. Used when the
/// stream ends without a result event or stdout dies unexpectedly.
fn finalize_error(
    child: &mut std::process::Child,
    context: String,
) -> CawResult<CompletionResponse> {
    let _ = child.kill();
    let mut stderr_buf = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        use std::io::Read;
        let _ = stderr.read_to_string(&mut stderr_buf);
    }
    let status = child.wait().ok();
    Err(CawError::Adapter(format!(
        "claude CLI {} (status: {:?}, stderr: {})",
        context,
        status,
        stderr_buf.trim()
    )))
}
