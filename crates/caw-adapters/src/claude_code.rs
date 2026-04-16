use caw_core::{
    CawError, CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};
use serde::Deserialize;
use std::process::Command;

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

    pub fn build(self) -> ClaudeCodeAdapter {
        ClaudeCodeAdapter {
            model: self.model,
            allowed_tools: self.allowed_tools,
            max_budget_usd: self.max_budget_usd,
            add_dirs: self.add_dirs,
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
        let mut builder = Self::builder()
            .allowed_tools(vec!["Read".to_string(), "Glob".to_string(), "Grep".to_string()]);
        for dir in dirs {
            builder = builder.add_dir(dir);
        }
        builder.build()
    }
}

#[derive(Deserialize)]
struct ClaudeJsonResponse {
    result: String,
    // other fields exist but we only need the text output
}

impl ModelAdapter for ClaudeCodeAdapter {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tool_calls: !self.allowed_tools.is_empty(),
            supports_hidden_reasoning: false,
            supports_visible_reasoning: false,
        }
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let workspace_context = req.format_workspace(ProvenanceFormat::Xml);
        let full_user_message = format!("{}{}", req.user, workspace_context);

        let mut cmd = Command::new("claude");
        cmd.arg("--print")
            .arg("--bare")
            .arg("--output-format").arg("json")
            .arg("--model").arg(&self.model)
            .arg("--system-prompt").arg(&req.system)
            .arg("--no-session-persistence");

        if let Some(budget) = self.max_budget_usd {
            cmd.arg("--max-budget-usd").arg(budget.to_string());
        }

        if !self.allowed_tools.is_empty() {
            cmd.arg("--allowedTools");
            for tool in &self.allowed_tools {
                cmd.arg(tool);
            }
        } else {
            // Disable all tools when none are configured
            cmd.arg("--tools").arg("");
        }

        for dir in &self.add_dirs {
            cmd.arg("--add-dir").arg(dir);
        }

        // The prompt goes as the positional argument
        cmd.arg(&full_user_message);

        let output = cmd
            .output()
            .map_err(|e| CawError::Adapter(format!("Failed to run claude CLI: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CawError::Adapter(format!(
                "claude CLI exited with {}: {}",
                output.status, stderr
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // JSON output wraps the result in a structured envelope
        match serde_json::from_str::<ClaudeJsonResponse>(&stdout) {
            Ok(parsed) => Ok(CompletionResponse {
                answer: parsed.result,
            }),
            Err(_) => {
                // Fall back to raw text if JSON parsing fails — handles
                // edge cases where --output-format json isn't honored
                Ok(CompletionResponse {
                    answer: stdout.trim().to_string(),
                })
            }
        }
    }
}
