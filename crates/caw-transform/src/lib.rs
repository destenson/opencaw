use caw_core::{CawResult, Stub};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

pub mod recall;

pub use recall::{apply_range, extract_annotations, extract_line_references, extract_probes, extract_thinking_steps, strip_markers};

static FILE_REF_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap());
static AT_PATH_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@([a-zA-Z0-9_./\-]+)").unwrap());
static STUB_REFERENCE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<file id="([^"]+)""#).unwrap());

/// Prompt transformer that finds file references and replaces them with stubs
pub struct PromptTransformer {
    stubs: HashMap<String, Stub>,
}

impl PromptTransformer {
    pub fn new() -> Self {
        Self {
            stubs: HashMap::new(),
        }
    }

    /// Register a stub for a path
    pub fn register_stub(&mut self, stub: Stub) {
        self.stubs.insert(stub.path.clone(), stub);
    }

    /// Transform a prompt by replacing file references with structured stubs
    pub fn transform(&self, prompt: &str) -> CawResult<TransformedPrompt> {
        let mut result = prompt.to_string();
        let mut references = Vec::new();

        // Replace markdown links
        for cap in FILE_REF_PATTERN.captures_iter(prompt) {
            let full_match = cap.get(0).unwrap().as_str();
            let path = cap.get(2).unwrap().as_str();

            if let Some(stub) = self.stubs.get(path) {
                let stub_text = format_stub(stub);
                result = result.replace(full_match, &stub_text);
                references.push(stub.clone());
            }
        }

        // Replace @path references
        for cap in AT_PATH_PATTERN.captures_iter(prompt) {
            let full_match = cap.get(0).unwrap().as_str();
            let path = cap.get(1).unwrap().as_str();

            if let Some(stub) = self.stubs.get(path) {
                let stub_text = format_stub(stub);
                result = result.replace(full_match, &stub_text);
                references.push(stub.clone());
            }
        }

        Ok(TransformedPrompt {
            original: prompt.to_string(),
            transformed: result,
            referenced_stubs: references,
        })
    }
}

impl Default for PromptTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of prompt transformation
#[derive(Debug, Clone)]
pub struct TransformedPrompt {
    pub original: String,
    pub transformed: String,
    pub referenced_stubs: Vec<Stub>,
}

/// Format a stub as structured text for inclusion in prompt
fn format_stub(stub: &Stub) -> String {
    format!(
        "<file id=\"{}\" path=\"{}\" tokens={} kind=\"{:?}\">\n  summary: {}\n  outline: [{}]\n</file>",
        stub.id.0,
        stub.path,
        stub.token_estimate,
        stub.kind,
        stub.summary,
        stub.outline.join(", ")
    )
}

/// Parse stub tags from model output to detect which stubs it referenced
pub fn extract_stub_references(text: &str) -> Vec<String> {
    STUB_REFERENCE_PATTERN
        .captures_iter(text)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use caw_core::{ContentKind, StubId};

    #[test]
    fn test_transform_markdown_link() {
        let mut transformer = PromptTransformer::new();
        transformer.register_stub(Stub {
            id: StubId("test.md".to_string()),
            path: "docs/test.md".to_string(),
            token_estimate: 100,
            kind: ContentKind::Markdown,
            summary: "Test document".to_string(),
            outline: vec!["Section 1".to_string()],
            content_hash: "abc123".to_string(),
            mtime_unix_secs: 0,
            byte_offset: 0,
            byte_length: 0,
            consolidation_notes: Vec::new(),
        });

        let result = transformer
            .transform("Check [this doc](docs/test.md) for details")
            .unwrap();

        assert!(result.transformed.contains("<file id=\"test.md\""));
        assert!(result.transformed.contains("summary: Test document"));
        assert_eq!(result.referenced_stubs.len(), 1);
    }

    #[test]
    fn test_transform_at_path() {
        let mut transformer = PromptTransformer::new();
        transformer.register_stub(Stub {
            id: StubId("config.toml".to_string()),
            path: "config.toml".to_string(),
            token_estimate: 50,
            kind: ContentKind::PlainText,
            summary: "Configuration file".to_string(),
            outline: vec![],
            content_hash: "def456".to_string(),
            mtime_unix_secs: 0,
            byte_offset: 0,
            byte_length: 0,
            consolidation_notes: Vec::new(),
        });

        let result = transformer
            .transform("See settings in @config.toml")
            .unwrap();

        assert!(result.transformed.contains("<file id=\"config.toml\""));
        assert_eq!(result.referenced_stubs.len(), 1);
    }

    #[test]
    fn test_extract_stub_references() {
        let text = r#"Based on <file id="doc1" path="a.md"> and <file id="doc2" path="b.md">"#;
        let refs = extract_stub_references(text);
        assert_eq!(refs, vec!["doc1", "doc2"]);
    }
}
