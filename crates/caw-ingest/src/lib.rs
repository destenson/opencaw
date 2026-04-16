use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use caw_core::{ContentKind, Stub, StubId};

#[derive(Debug, Clone)]
pub struct SourceDocument {
    pub path: String,
    pub content: String,
    pub kind: ContentKind,
}

#[derive(Debug, Clone)]
pub struct IngestionPipeline;

impl IngestionPipeline {
    pub fn ingest(&self, doc: SourceDocument) -> Stub {
        let outline = extract_outline(doc.kind, &doc.content);
        let summary = summarize(&doc.content);
        let token_estimate = estimate_tokens(&doc.content);

        Stub {
            id: StubId(doc.path.clone()),
            path: doc.path,
            token_estimate,
            kind: doc.kind,
            summary,
            outline,
            content_hash: stable_hash(&doc.content),
            mtime_unix_secs: now_unix_secs(),
        }
    }
}

fn summarize(content: &str) -> String {
    let first_line = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("(empty)");
    let clipped = first_line.chars().take(120).collect::<String>();
    format!("Auto summary: {}", clipped)
}

fn extract_outline(kind: ContentKind, content: &str) -> Vec<String> {
    match kind {
        ContentKind::Markdown => content
            .lines()
            .filter_map(|l| l.strip_prefix('#').map(str::trim))
            .filter(|l| !l.is_empty())
            .map(ToString::to_string)
            .take(12)
            .collect(),
        ContentKind::Code => content
            .lines()
            .filter(|l| {
                l.trim_start().starts_with("fn ")
                    || l.trim_start().starts_with("pub fn ")
                    || l.trim_start().starts_with("class ")
                    || l.trim_start().starts_with("struct ")
            })
            .map(|l| l.trim().to_string())
            .take(20)
            .collect(),
        _ => vec![],
    }
}

fn estimate_tokens(content: &str) -> usize {
    (content.len() / 4).max(1)
}

fn stable_hash(input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}
