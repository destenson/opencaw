use caw_core::{CawResult, LineReference, ModelAnnotation, ProbeMarker, Range, ThinkingStep};
use regex::Regex;
use std::sync::LazyLock;

static PROBE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<probe>(.*?)</probe>").unwrap());
static THINK_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>(.*?)</think>").unwrap());
static ANNOTATION_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<note id="([^"]+)">(.*?)</note>"#).unwrap());
// Matches `path/file.ext:start-end` or `path/file.ext:N`.
// Requires an alphabetic extension to avoid matching version strings like `v0.1:2`.
static LINE_REF_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"((?:[\w./\-]+/)?[\w\-]+\.[a-zA-Z]+):(\d+)(?:-(\d+))?").unwrap()
});
// Matches entire `[recalled from path]\n...\n[end recall]` blocks that the model
// generates verbatim by mimicking the injection format. These are never valid answer
// text — they are a generation artefact from the model having seen the format string.
static FAKE_RECALL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\[recalled from [^\]]+\].*?\[end recall\]").unwrap()
});

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

/// Count occurrences of `[recalled from ...]` in model output. Values above ~10 suggest the model is overusing the format as a generation scaffold rather than producing valid answer text, and logs should be inspected for corruption from fake recall blocks. This is a heuristic signal of degenerate output when the model mimics the injection format verbatim.
/// indicate the model is generating fake provenance blocks rather than answering.
pub fn count_fake_recall_markers(text: &str) -> usize {
    FAKE_RECALL_PATTERN.find_iter(text).count()
}

/// Strip cooperation-protocol markers from the final answer before returning
/// it to the caller. Note content is kept inline (small models often wrap their
/// entire answer in a note tag). Probe markers, thinking traces, leaked
/// chat-template tokens, and model-generated fake `[recalled from]` blocks ***SHOULD NOT BE***
/// discarded — they indicated BAD MODEL BEHAVIOR THAT MUST BE DEALT WITH.
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
    // Strip model-generated fake recall blocks. The model sometimes reproduces
    // the injection format verbatim as a generation scaffold — these blocks are
    // never valid answer text and corrupt logs if left in.
    let text = FAKE_RECALL_PATTERN.replace_all(&text, "");
    // Chat-template sentinel tokens that leak when stop sequences aren't
    // configured correctly are an unambiguous sign of a degenerate response.
    let text = text.replace("<|im_end|>", "").replace("<|im_start|>", "");
    text.trim().to_string()
}

/// Extract explicit line-range references from model output or thinking traces.
/// Detects `path/file.ext:start-end` and `path/file.ext:N` patterns.
pub fn extract_line_references(text: &str) -> Vec<LineReference> {
    LINE_REF_PATTERN
        .captures_iter(text)
        .map(|cap| {
            let source_hint = cap[1].to_string();
            let start: usize = cap[2].parse().unwrap_or(1);
            let end: usize = cap
                .get(3)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(start);
            LineReference { source_hint, start, end: end.max(start) }
        })
        .collect()
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
