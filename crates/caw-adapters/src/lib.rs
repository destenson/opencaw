use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
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
    /// When set, each prompt is also written to this directory as
    /// `prompt-{timestamp}-turn-{N}.txt`. The .txt extension keeps these
    /// files out of the session indexer, which only loads .md files.
    prompt_dir: Option<PathBuf>,
    timestamp: String,
    turn: Mutex<usize>,
}

impl ShowPromptAdapter {
    pub fn new(inner: Box<dyn ModelAdapter>) -> Self {
        Self {
            inner,
            prompt_dir: None,
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
        let fragment_tokens: usize = req.workspace_fragments.iter().map(|f| f.tokens).sum();
        let mut out = String::new();
        writeln!(out, "\n─── PROMPT ({} workspace tokens across {} fragments) [MODEL: {}] ───", fragment_tokens, req.workspace_fragments.len(), self.inner.model_name())?;
        writeln!(out, "[system]\n{}", req.system)?;
        writeln!(out, "[user]\n{}", req.user)?;
        for frag in &req.workspace_fragments {
            writeln!(out, "[fragment: {} | {} tokens]\n{}", frag.locator.source, frag.tokens, frag.content)?;
        }
        writeln!(out, "─────────────────────────")?;
        writeln!(out, "Response: {}", res.as_ref().map(|r| r.answer.clone()).unwrap_or_else(|e| format!("Error: {}", e)))?;
        writeln!(out, "─────────────────────────")?;
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
