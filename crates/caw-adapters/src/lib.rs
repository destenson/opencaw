use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};
use std::fmt::Write as FmtWrite;
use std::path::PathBuf;
use std::sync::Mutex;

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
        .map_err(|e| CawError::Io(format!("Failed to create runtime: {}", e)))
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
        // A cooperative model acknowledges the recalled context without
        // reproducing the `[recalled from …]` injection scaffold in its answer.
        // Reproducing that scaffold is exactly the degenerate behavior the
        // orchestrator detects, so the mock must not emit it — otherwise every
        // mock turn with injected fragments would be flagged degenerate.
        let grounding = if req.workspace_fragments.is_empty() {
            String::new()
        } else {
            format!(
                " (grounded in {} recalled fragment(s))",
                req.workspace_fragments.len()
            )
        };

        Ok(CompletionResponse {
            answer: format!(
                "[{}] synthesized answer for: {}{}",
                self.name, req.user, grounding,
            ),
            thinking: None,
            usage: None,
        })
    }
}

pub struct SavePromptAdapter {
    inner: Box<dyn ModelAdapter>,
    prompt_dir: PathBuf,
    timestamp: String,
    turn: Mutex<usize>,
}

impl SavePromptAdapter {
        pub fn new(inner: Box<dyn ModelAdapter>) -> Self {
        let prompt_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            inner,
            prompt_dir,
            timestamp: prompt_timestamp(),
            turn: Mutex::new(0),
        }
    }

    /// Also save each prompt as a file in `dir` beside the session files.
    pub fn saving_to(mut self, dir: PathBuf) -> Self {
        self.prompt_dir = dir;
        self
    }

    fn log_prompt(&self, req: &CompletionRequest, res: CawResult<CompletionResponse>) -> CawResult<CompletionResponse> {
        let out = render_prompt_dump(req, self.inner.model_name(), self.inner.provenance_format(), &res)?;
        let mut turn = self.turn.lock().unwrap();
        *turn += 1;
        let path = self.prompt_dir.join(format!("prompt-{}-turn-{}.txt", self.timestamp, *turn));
        if let Ok(mut f) = std::fs::File::create(&path) {
            std::io::Write::write_all(&mut f, out.as_bytes())?;
        }
        res
    }

}

/// Render the exact context a model receives for one turn, for prompt-dump
/// tooling. The string mirrors what every adapter builds before its API call:
/// the system text concatenated with the recalled workspace block formatted
/// via the adapter's own `provenance_format` (`format_workspace` is the single
/// renderer all adapters share), followed by the user message. Some chat
/// adapters (e.g. Ollama for models whose template omits a system slot) fold
/// the system text into the first user message — the content is identical, only
/// the message boundary differs.
fn render_prompt_dump(
    req: &CompletionRequest,
    model_name: &str,
    format: ProvenanceFormat,
    res: &CawResult<CompletionResponse>,
) -> CawResult<String> {
    let fragment_tokens: usize = req.workspace_fragments.iter().map(|f| f.tokens).sum();
    let full_system = format!("{}{}", req.system, req.format_workspace(format));
    let mut out = String::new();
    writeln!(
        out,
        "\n─── PROMPT ({} workspace tokens across {} fragments) [MODEL: {} | provenance: {:?}] ───",
        fragment_tokens,
        req.workspace_fragments.len(),
        model_name,
        format,
    )?;
    writeln!(out, "[system message]\n{}", full_system)?;
    writeln!(out, "\n[user message]\n{}", req.user)?;
    writeln!(out, "─────────────────────────")?;
    writeln!(
        out,
        "Response: {}",
        res.as_ref()
            .map(|r| r.answer.clone())
            .unwrap_or_else(|e| format!("Error: {}", e))
    )?;
    writeln!(out, "─────────────────────────")?;
    Ok(out)
}

/// Wraps any adapter and prints the contents of each CompletionRequest to
/// stderr before delegating. Lets you verify exactly what context the model
/// receives — system prompt, user message, and every loaded workspace fragment.
pub struct ShowPromptAdapter {
    inner: Box<dyn ModelAdapter>,
    /// When set, each prompt is also written to this directory as
    /// `prompt-{timestamp}-turn-{N}.txt`. The .txt extension keeps these
    /// files out of the session indexer, which only loads .md files.
    prompt_dir: Option<PathBuf>,
    timestamp: String,
    turn: Mutex<usize>,
}

impl ShowPromptAdapter {
    pub fn new(inner: Box<dyn ModelAdapter>, save_prompt: bool) -> Self {
        let prompt_dir = if save_prompt {
            Some(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        } else {
            None
        };
        Self {
            inner,
            prompt_dir,
            timestamp: prompt_timestamp(),
            turn: Mutex::new(0),
        }
    }

    /// Also save each prompt as a file in `dir` beside the session files.
    pub fn saving_to(mut self, dir: PathBuf) -> Self {
        self.prompt_dir = Some(dir);
        self
    }

    fn log_prompt(&self, req: &CompletionRequest, res: CawResult<CompletionResponse>) -> CawResult<CompletionResponse> {
        let out = render_prompt_dump(req, self.inner.model_name(), self.inner.provenance_format(), &res)?;
        eprint!("{out}");

        if let Some(dir) = &self.prompt_dir {
            let mut turn = self.turn.lock().unwrap();
            *turn += 1;
            let path = dir.join(format!("prompt-{}-turn-{}.txt", self.timestamp, *turn));
            if let Ok(mut f) = std::fs::File::create(&path) {
                std::io::Write::write_all(&mut f, out.as_bytes())?;
            }
        }
        res
    }
}

fn prompt_timestamp() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    // Minimal Gregorian date (good enough for filenames)
    let (y, mo, d) = epoch_days_to_ymd(days);
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, mo, d, hour, min, sec)
}

fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    let mut y = 1970u64;
    let mut rem = days;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dy = if leap { 366 } else { 365 };
        if rem < dy { break; }
        rem -= dy;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months = [31u64, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 1u64;
    for &dm in &months {
        if rem < dm { break; }
        rem -= dm;
        mo += 1;
    }
    (y, mo, rem + 1)
}

impl ModelAdapter for ShowPromptAdapter {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.complete(req.clone()))
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.generate_passive(req.clone(), check_interval, window_size, on_window))
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.thinking_with_steps(req.clone(), on_step))
    }
}

impl ModelAdapter for SavePromptAdapter {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.complete(req.clone()))
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.generate_passive(req.clone(), check_interval, window_size, on_window))
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        self.log_prompt(&req, self.inner.thinking_with_steps(req.clone(), on_step))
    }
}
