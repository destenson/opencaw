use anyhow::{Context, Result};

use caw_core::{CompletionRequest, ModelAdapter};

#[derive(Debug, Clone)]
pub struct JudgeVerdict {
    /// 0.0 - 1.0. 1.0 = fully correct, 0.0 = wrong or missing.
    pub score: f32,
    pub rationale: String,
}

/// Ask a cheap model to compare the answer against a reference and return a
/// numeric score. The judge is intentionally separate from the answering
/// model so the comparison isn't biased by same-model self-agreement —
/// configure it via `--judge-adapter`/`--judge-model` to point at a model
/// that doesn't share weights with the answer adapter.
pub fn judge_answer(
    judge_adapter: &dyn ModelAdapter,
    answer: &str,
    reference: &str,
) -> Result<JudgeVerdict> {
    let system = "You are a strict scorer. Compare a candidate answer against a reference \
         answer and output a single line of JSON with fields score (0.0 to 1.0) and rationale \
         (one sentence). 1.0 means the candidate asserts all key facts from the reference; \
         0.0 means it's wrong, missing, or contradicts. Be harsh: partial credit only for \
         answers that are substantively correct but incomplete. If the candidate says the \
         information is absent, not provided, or insufficient to answer — even if it mentions \
         the reference string while doing so — score 0.0; mentioning the reference while \
         denying knowledge is a refusal, not a correct answer. Output nothing except the JSON."
        .to_string();

    let user = format!(
        "Reference answer:\n{}\n\nCandidate answer:\n{}\n\nScore the candidate.",
        reference, answer
    );

    let response = judge_adapter
        .complete(CompletionRequest {
            system,
            user,
            ..Default::default()
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
    match serde_json::from_str::<RawVerdict>(slice) {
        Ok(parsed) => Ok(JudgeVerdict {
            score: parsed.score.clamp(0.0, 1.0),
            rationale: parsed.rationale,
        }),
        // Small judge models routinely emit the free-text `rationale` unquoted
        // (e.g. `{"score": 0.0, "rationale": The answer is wrong.}`), which is
        // invalid JSON even though the verdict is sound. The `score` is the only
        // field the metric needs and it is structured (a number), so recover it
        // directly rather than discarding the item. This is a deterministic
        // recovery of a known-shaped field, not a guess at the model's intent;
        // the malformed output is logged so it stays visible.
        Err(strict_err) => match extract_score(slice) {
            Some(score) => {
                eprintln!(
                    "  judge emitted invalid JSON ({strict_err}); recovered score directly from: {slice}"
                );
                Ok(JudgeVerdict {
                    score: score.clamp(0.0, 1.0),
                    rationale: extract_rationale(slice).unwrap_or_default(),
                })
            }
            None => Err(strict_err)
                .with_context(|| format!("failed to parse judge JSON: {}", slice)),
        },
    }
}

/// Pull the numeric value of the `score` key out of a malformed JSON object.
/// Returns `None` if the key is absent or the value isn't a leading number.
fn extract_score(slice: &str) -> Option<f32> {
    let key = slice.find("\"score\"")?;
    let after = &slice[key + "\"score\"".len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .collect();
    num.parse::<f32>().ok()
}

/// Best-effort recovery of the `rationale` text from a malformed JSON object.
/// The rationale is informational only, so an imperfect extraction is fine.
fn extract_rationale(slice: &str) -> Option<String> {
    let key = slice.find("\"rationale\"")?;
    let after = &slice[key + "\"rationale\"".len()..];
    let colon = after.find(':')?;
    let text = after[colon + 1..]
        .trim()
        .trim_end_matches('}')
        .trim()
        .trim_matches('"')
        .trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_json() {
        let v = parse_verdict(r#"{"score": 0.5, "rationale": "partly right"}"#).unwrap();
        assert_eq!(v.score, 0.5);
        assert_eq!(v.rationale, "partly right");
    }

    #[test]
    fn recovers_score_from_unquoted_rationale() {
        // The exact failure shape seen from groq llama-3.3-70b: a valid score
        // followed by a bare, unquoted rationale string.
        let raw = r#"{"score": 0.0, "rationale": The candidate answer is incorrect because it states 500 tokens, but the reference says 2000.}"#;
        let v = parse_verdict(raw).unwrap();
        assert_eq!(v.score, 0.0);
        assert!(v.rationale.starts_with("The candidate answer is incorrect"));
    }

    #[test]
    fn clamps_out_of_range_score() {
        let v = parse_verdict(r#"{"score": 1.4, "rationale": all of it}"#).unwrap();
        assert_eq!(v.score, 1.0);
    }

    #[test]
    fn errors_when_no_score_present() {
        assert!(parse_verdict(r#"{"rationale": "no score here"}"#).is_err());
    }
}
