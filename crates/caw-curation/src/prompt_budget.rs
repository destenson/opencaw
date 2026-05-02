#[derive(Debug, Clone)]
pub struct SystemPromptBudget {
    pub max_tokens: usize,
    /// Emit a warning when token usage exceeds this fraction of max_tokens
    pub warn_threshold_pct: f32,
}

impl Default for SystemPromptBudget {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            warn_threshold_pct: 0.80,
        }
    }
}

impl SystemPromptBudget {
    /// Create a budget sized as a fraction of the total context window.
    pub fn from_context_window(total_context_tokens: usize, fraction: f32) -> Self {
        Self {
            max_tokens: (total_context_tokens as f32 * fraction) as usize,
            warn_threshold_pct: 0.80,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PromptBudgetCheck {
    /// Under budget, no issues
    Ok {
        used_tokens: usize,
        max_tokens: usize,
    },
    /// Over the warning threshold but not the hard limit
    Warning {
        used_tokens: usize,
        max_tokens: usize,
        message: String,
    },
    /// Over the hard limit
    Exceeded {
        used_tokens: usize,
        max_tokens: usize,
        message: String,
    },
}

impl PromptBudgetCheck {
    pub fn is_exceeded(&self) -> bool {
        matches!(self, Self::Exceeded { .. })
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warning { .. })
    }
}

/// Check a system prompt against the budget. Token estimation uses the same
/// rough heuristic as tool_output — callers can substitute a real tokenizer
/// count if they have one.
pub fn check_system_prompt(prompt: &str, budget: &SystemPromptBudget) -> PromptBudgetCheck {
    let used = crate::tool_output::estimate_tokens(prompt);
    check_budget(used, budget)
}

/// Check an already-counted token value against the budget.
pub fn check_budget(used_tokens: usize, budget: &SystemPromptBudget) -> PromptBudgetCheck {
    let max = budget.max_tokens;

    if used_tokens > max {
        return PromptBudgetCheck::Exceeded {
            used_tokens,
            max_tokens: max,
            message: format!(
                "System prompt uses {} tokens, exceeding budget of {} by {}",
                used_tokens,
                max,
                used_tokens - max
            ),
        };
    }

    let warn_at = (max as f32 * budget.warn_threshold_pct) as usize;
    if used_tokens > warn_at {
        return PromptBudgetCheck::Warning {
            used_tokens,
            max_tokens: max,
            message: format!(
                "System prompt uses {} tokens ({:.0}% of {} budget)",
                used_tokens,
                (used_tokens as f32 / max as f32) * 100.0,
                max
            ),
        };
    }

    PromptBudgetCheck::Ok {
        used_tokens,
        max_tokens: max,
    }
}

/// Truncate a system prompt to fit within the token budget. Tries to cut
/// at a sentence boundary to avoid mid-thought truncation. Returns the
/// truncated prompt and how many tokens were dropped.
pub fn truncate_to_budget(prompt: &str, budget: &SystemPromptBudget) -> (String, usize) {
    let current = crate::tool_output::estimate_tokens(prompt);
    if current <= budget.max_tokens {
        return (prompt.to_string(), 0);
    }

    // Binary-search-ish: try progressively shorter prefixes at sentence boundaries
    let sentences: Vec<&str> = prompt
        .split_inclusive([ '.', '\n' ])
        .collect();

    let mut result = String::new();
    let mut result_tokens = 0;

    for sentence in &sentences {
        let candidate_tokens = crate::tool_output::estimate_tokens(sentence);
        if result_tokens + candidate_tokens > budget.max_tokens {
            break;
        }
        result.push_str(sentence);
        result_tokens += candidate_tokens;
    }

    let dropped = current.saturating_sub(result_tokens);
    (result, dropped)
}
