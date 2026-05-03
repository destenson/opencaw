use caw_core::{CawResult, CompletionRequest, CompletionResponse, ConsolidationNote, ModelAdapter};

pub trait ConsolidationSynthesizer {
    fn synthesize_eviction_note(
        &self,
        fragment_content: &str,
        query_context: &str,
        decayed_score: f32,
        source: &str,
        annotations: &[ConsolidationNote],
    ) -> CawResult<String>;
}

/// Produces the same format strings the orchestrator used before
/// LLM-driven consolidation was available.
pub struct MechanicalConsolidation;

impl ConsolidationSynthesizer for MechanicalConsolidation {
    fn synthesize_eviction_note(
        &self,
        _fragment_content: &str,
        query_context: &str,
        decayed_score: f32,
        source: &str,
        _annotations: &[ConsolidationNote],
    ) -> CawResult<String> {
        Ok(format!(
            "Evicted (relevance decayed to {:.2}) during query about '{}'. Source: {}",
            decayed_score,
            truncate_str(query_context, 100),
            source,
        ))
    }
}

/// Uses a cheap ModelAdapter to distill what was learned from the
/// fragment during its time in the workspace, producing a richer
/// consolidation note for future recalls.
pub struct LlmConsolidation {
    adapter: Box<dyn ModelAdapter>,
}

impl LlmConsolidation {
    pub fn with_adapter(adapter: Box<dyn ModelAdapter>) -> Self {
        Self { adapter }
    }
}

impl ConsolidationSynthesizer for LlmConsolidation {
    fn synthesize_eviction_note(
        &self,
        fragment_content: &str,
        query_context: &str,
        _decayed_score: f32,
        source: &str,
        annotations: &[ConsolidationNote],
    ) -> CawResult<String> {
        let annotation_text = if annotations.is_empty() {
            "(no annotations recorded)".to_string()
        } else {
            annotations
                .iter()
                .map(|n| format!("- {}", n.content))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let truncated_content = truncate_str(fragment_content, 1500);

        let system = concat!(
            "You write concise consolidation notes for a context management system. ",
            "A fragment is being evicted from the active workspace. Summarize what was ",
            "learned from this content in 1-2 sentences for future reference. Focus on ",
            "conclusions, decisions, and key facts — not the content's structure.",
        )
        .to_string();

        let user = format!(
            "Source: {source}\n\
             Query context: {query_context}\n\
             Model annotations during this session:\n{annotation_text}\n\n\
             Fragment content (truncated):\n{truncated_content}",
        );

        let resp: CompletionResponse = self.adapter.complete(CompletionRequest {
            system,
            user,
            workspace_fragments: Vec::new(),
            workspace_guidance: Vec::new(),
        })?;

        let answer = resp.answer.trim().to_string();
        if answer.is_empty() {
            // Degrade to mechanical if the model returns nothing
            return MechanicalConsolidation.synthesize_eviction_note(
                fragment_content,
                query_context,
                _decayed_score,
                source,
                annotations,
            );
        }
        Ok(answer)
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}
