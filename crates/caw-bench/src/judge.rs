use anyhow::{Context, Result};

use caw_adapters::ClaudeCodeAdapter;
use caw_core::{CompletionRequest, ModelAdapter};

#[derive(Debug, Clone)]
pub struct JudgeVerdict {
    /// 0.0 - 1.0. 1.0 = fully correct, 0.0 = wrong or missing.
    pub score: f32,
    pub rationale: String,
}

/// Ask a cheap model to compare the answer against a reference and return a
/// numeric score. The judge is intentionally separate from the answering
/// model so the comparison isn't biased by same-model self-agreement.
///
/// Judge runs through the Claude Code CLI; `judge_model` is passed directly
/// to `claude --model` (typical values: "haiku", "sonnet").
pub fn judge_answer(judge_model: &str, answer: &str, reference: &str) -> Result<JudgeVerdict> {
    let adapter = ClaudeCodeAdapter::builder().model(judge_model).build();

    let system = "You are a strict scorer. Compare a candidate answer against a reference \
         answer and output a single line of JSON with fields score (0.0 to 1.0) and rationale \
         (one sentence). 1.0 means the candidate contains all key facts from the reference; \
         0.0 means it's wrong, missing, or contradicts. Be harsh: partial credit only for \
         answers that are substantively correct but incomplete. Output nothing except the JSON."
        .to_string();

    let user = format!(
        "Reference answer:\n{}\n\nCandidate answer:\n{}\n\nScore the candidate.",
        reference, answer
    );

    let response = adapter
        .complete(CompletionRequest {
            system,
            user,
            workspace_fragments: Vec::new(),
        })
        .context("judge model request failed")?;

    parse_verdict(&response.answer)
}

fn parse_verdict(raw: &str) -> Result<JudgeVerdict> {
    // Tolerate preambles, trailing text, or code fences — extract the first
    // {...} block and parse as JSON. Judges occasionally ignore the "nothing
    // except JSON" instruction; robustness is cheaper than retries here.
    let start = raw
        .find('{')
        .context("judge returned no JSON object in output")?;
    let end = raw
        .rfind('}')
        .context("judge returned unterminated JSON object")?;
    if end < start {
        anyhow::bail!("judge JSON braces malformed");
    }
    let slice = &raw[start..=end];

    #[derive(serde::Deserialize)]
    struct RawVerdict {
        score: f32,
        #[serde(default)]
        rationale: String,
    }
    let parsed: RawVerdict = serde_json::from_str(slice)
        .with_context(|| format!("failed to parse judge JSON: {}", slice))?;

    Ok(JudgeVerdict {
        score: parsed.score.clamp(0.0, 1.0),
        rationale: parsed.rationale,
    })
}
