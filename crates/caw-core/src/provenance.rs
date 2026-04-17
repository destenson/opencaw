use crate::{ConsolidationNote, ProvenanceStore, RecallFragment, StubId, TopicOverlap};
use std::collections::{HashMap, HashSet};

/// Simple provenance store that records recalled fragments.
/// Uses default no-op implementations for ledger capabilities.
#[derive(Debug, Default, Clone)]
pub struct InMemoryProvenanceStore {
    records: Vec<RecallFragment>,
}

impl ProvenanceStore for InMemoryProvenanceStore {
    fn record(&mut self, fragment: RecallFragment) {
        self.records.push(fragment);
    }

    fn all(&self) -> Vec<RecallFragment> {
        self.records.clone()
    }
}

/// Provenance ledger with topic overlap detection and consolidation tracking.
///
/// Extends basic fragment recording with:
/// - Query context for each recall (what triggered it)
/// - Consolidation notes per stub (eviction summaries + model annotations)
/// - Topic overlap detection between fragments from different sources
#[derive(Debug, Default, Clone)]
pub struct ProvenanceLedger {
    records: Vec<RecallFragment>,
    entries: Vec<LedgerEntry>,
    consolidation: HashMap<String, Vec<ConsolidationNote>>,
}

#[derive(Debug, Clone)]
struct LedgerEntry {
    stub_id: StubId,
    source_path: String,
    topic_terms: Vec<String>,
}

impl ProvenanceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract the most frequent non-stopword terms from text
    fn extract_topic_terms(text: &str) -> Vec<String> {
        let mut freq: HashMap<String, u32> = HashMap::new();
        for token in tokenize(text) {
            *freq.entry(token).or_default() += 1;
        }

        let mut terms: Vec<(String, u32)> = freq.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));
        terms.into_iter().take(20).map(|(t, _)| t).collect()
    }
}

impl ProvenanceStore for ProvenanceLedger {
    fn record(&mut self, fragment: RecallFragment) {
        self.records.push(fragment);
    }

    fn all(&self) -> Vec<RecallFragment> {
        self.records.clone()
    }

    fn record_with_context(&mut self, fragment: RecallFragment, _query: &str, _turn: usize) {
        let topic_terms = Self::extract_topic_terms(&fragment.content);
        let entry = LedgerEntry {
            stub_id: fragment.stub_id.clone(),
            source_path: fragment.locator.source.clone(),
            topic_terms,
        };
        self.entries.push(entry);
        self.records.push(fragment);
    }

    fn record_consolidation(&mut self, stub_id: StubId, note: ConsolidationNote) {
        self.consolidation.entry(stub_id.0).or_default().push(note);
    }

    fn consolidation_notes_for(&self, stub_id: &StubId) -> Vec<ConsolidationNote> {
        self.consolidation
            .get(&stub_id.0)
            .cloned()
            .unwrap_or_default()
    }

    fn check_topic_overlaps(&self) -> Vec<TopicOverlap> {
        let mut overlaps = Vec::new();

        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                let a = &self.entries[i];
                let b = &self.entries[j];

                // Only flag overlaps between different source files
                if a.source_path == b.source_path {
                    continue;
                }

                let a_terms: HashSet<&String> = a.topic_terms.iter().collect();
                let b_terms: HashSet<&String> = b.topic_terms.iter().collect();

                let shared: Vec<String> = a_terms
                    .intersection(&b_terms)
                    .map(|s| (*s).clone())
                    .collect();

                if shared.is_empty() {
                    continue;
                }

                let overlap_score =
                    shared.len() as f32 / a_terms.len().min(b_terms.len()).max(1) as f32;

                // Only surface overlaps above 30% — below that is noise
                if overlap_score > 0.3 {
                    overlaps.push(TopicOverlap {
                        stub_a: a.stub_id.clone(),
                        path_a: a.source_path.clone(),
                        stub_b: b.stub_id.clone(),
                        path_b: b.source_path.clone(),
                        shared_terms: shared,
                        overlap_score,
                    });
                }
            }
        }

        overlaps
    }

    fn format_overlap_warnings(&self) -> String {
        let overlaps = self.check_topic_overlaps();
        if overlaps.is_empty() {
            return String::new();
        }

        let mut warnings = String::from(
            "Topic overlaps detected between recalled fragments. Verify consistency:\n",
        );
        for overlap in &overlaps {
            warnings.push_str(&format!(
                "- '{}' and '{}' share terms [{}] ({:.0}% overlap)\n",
                overlap.path_a,
                overlap.path_b,
                overlap.shared_terms.join(", "),
                overlap.overlap_score * 100.0,
            ));
        }
        warnings
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2 && !is_stopword(s))
        .map(String::from)
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "and"
            | "for"
            | "are"
            | "but"
            | "not"
            | "you"
            | "all"
            | "can"
            | "has"
            | "was"
            | "one"
            | "our"
            | "out"
            | "his"
            | "her"
            | "had"
            | "how"
            | "its"
            | "may"
            | "who"
            | "did"
            | "get"
            | "let"
            | "say"
            | "she"
            | "too"
            | "use"
            | "way"
            | "with"
            | "this"
            | "that"
            | "from"
            | "have"
            | "been"
            | "they"
            | "them"
            | "then"
            | "than"
            | "each"
            | "which"
            | "their"
            | "will"
            | "would"
            | "there"
            | "what"
            | "about"
            | "could"
            | "other"
            | "into"
            | "more"
            | "some"
            | "very"
            | "when"
            | "also"
            | "just"
            | "should"
    )
}
