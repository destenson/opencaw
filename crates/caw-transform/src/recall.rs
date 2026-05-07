use caw_core::{CawResult, ModelAnnotation, ProbeMarker, Range, ThinkingStep};
use regex::Regex;
use std::sync::LazyLock;

static PROBE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<probe>(.*?)</probe>").unwrap());
static THINK_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").unwrap());
static ANNOTATION_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<note id="([^"]+)">(.*?)</note>"#).unwrap());

/// Extract probe markers from model output
pub fn extract_probes(text: &str) -> Vec<ProbeMarker> {
    PROBE_PATTERN
        .captures_iter(text)
        .map(|cap| ProbeMarker {
            content: cap.get(1).unwrap().as_str().to_string(),
            position: cap.get(0).unwrap().start(),
        })
        .collect()
}

/// Extract thinking steps from reasoning model output.
/// Looks for explicit `<think>` tags first, then falls back to
/// heuristic boundary detection.
pub fn extract_thinking_steps(text: &str) -> Vec<ThinkingStep> {
    let mut steps = Vec::new();

    for (idx, cap) in THINK_PATTERN.captures_iter(text).enumerate() {
        steps.push(ThinkingStep {
            content: cap.get(1).unwrap().as_str().to_string(),
            step_number: idx,
        });
    }

    if steps.is_empty() {
        steps = detect_step_boundaries(text);
    }

    steps
}

fn detect_step_boundaries(text: &str) -> Vec<ThinkingStep> {
    let mut steps = Vec::new();
    let mut current_step = String::new();
    let mut step_number = 0;

    for line in text.lines() {
        let is_boundary = line.trim().is_empty()
            || line
                .trim_start()
                .starts_with(|c: char| c.is_numeric() && line.contains('.'))
            || line.trim_start().starts_with("Therefore")
            || line.trim_start().starts_with("Thus")
            || line.trim_start().starts_with("So");

        if is_boundary && !current_step.is_empty() {
            steps.push(ThinkingStep {
                content: current_step.trim().to_string(),
                step_number,
            });
            step_number += 1;
            current_step.clear();
        } else if !line.trim().is_empty() {
            current_step.push_str(line);
            current_step.push('\n');
        }
    }

    if !current_step.is_empty() {
        steps.push(ThinkingStep {
            content: current_step.trim().to_string(),
            step_number,
        });
    }

    steps.retain(|s| s.content.len() > 20);
    steps
}

/// Apply range selection to content — delegates to Range::apply
pub fn apply_range(content: &str, range: &Range) -> CawResult<String> {
    Ok(range.apply(content))
}

/// Strip cooperation-protocol markers from the final answer before returning
/// it to the caller. Note content is kept inline (small models often wrap their
/// entire answer in a note tag). Probe markers, thinking traces, and leaked
/// chat-template tokens are discarded — they are not answer text.
pub fn strip_markers(text: &str) -> String {
    let text = ANNOTATION_PATTERN.replace_all(text, "$2");
    let text = PROBE_PATTERN.replace_all(&text, "");
    // Strip complete <think>...</think> blocks (reasoning traces are internal).
    let text = THINK_PATTERN.replace_all(&text, "");
    // Strip partial <think> block if the model was cut off before </think>.
    let text = match text.find("<think>") {
        Some(pos) => std::borrow::Cow::Owned(text[..pos].to_string()),
        None => text,
    };
    // Chat-template sentinel tokens that leak when stop sequences aren't
    // configured correctly are an unambiguous sign of a degenerate response.
    let text = text.replace("<|im_end|>", "").replace("<|im_start|>", "");
    text.trim().to_string()
}

/// Extract model annotations about specific stubs.
/// Format: `<note id="stub_id">content</note>`
pub fn extract_annotations(text: &str) -> Vec<ModelAnnotation> {
    ANNOTATION_PATTERN
        .captures_iter(text)
        .map(|cap| ModelAnnotation {
            stub_id: cap.get(1).unwrap().as_str().to_string(),
            content: cap.get(2).unwrap().as_str().to_string(),
            position: cap.get(0).unwrap().start(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_probes() {
        let text = "Consider <probe>check config.toml</probe> and <probe>review main.rs</probe>";
        let probes = extract_probes(text);
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].content, "check config.toml");
        assert_eq!(probes[1].content, "review main.rs");
    }

    #[test]
    fn test_range_lines() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let range = Range::Lines { start: 2, end: 4 };
        let result = apply_range(content, &range).unwrap();
        assert_eq!(result, "line2\nline3\nline4");
    }

    #[test]
    fn test_range_parse() {
        assert!(matches!(Range::parse("full"), Range::Full));
        assert!(matches!(
            Range::parse("10-20"),
            Range::Lines { start: 10, end: 20 }
        ));
        assert!(matches!(
            Range::parse("L5-L15"),
            Range::Lines { start: 5, end: 15 }
        ));
        assert!(matches!(
            Range::parse("#Section/Subsection"),
            Range::Heading { .. }
        ));
        assert!(matches!(
            Range::parse("T100:500"),
            Range::Tokens {
                start: 100,
                count: 500
            }
        ));
    }
}
