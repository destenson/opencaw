use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ConsolidationNote, ConsolidationSource,
    EmbeddingProvider, ModelAdapter, ProvenanceStore, Range, RecallFragment, RecallThresholds,
    Retriever, ScoredStub, StubId, VectorIndex,
};
use caw_transform::{extract_annotations, extract_probes, extract_thinking_steps};
use std::collections::HashSet;

/// Advanced orchestrator with iterative multi-pass recall.
///
/// Each turn follows: initial retrieval -> complete -> extract probes/traces ->
/// load new fragments -> evict stale fragments -> re-complete with enriched
/// workspace. Repeats until convergence or max iterations.
pub struct DynamicRecallOrchestrator<R, E, V, P, M>
where
    R: Retriever,
    E: EmbeddingProvider,
    V: VectorIndex,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub retriever: R,
    pub embedder: E,
    pub vector_index: V,
    pub provenance: P,
    pub adapter: M,
    pub loaded: Vec<RecallFragment>,
    pub loaded_ids: HashSet<StubId>,
    pub config: DynamicRecallConfig,
}

#[derive(Debug, Clone)]
pub struct DynamicRecallConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub max_workspace_tokens: usize,
    pub max_recall_iterations: usize,
    /// Term-overlap score below which a fragment is eligible for eviction
    pub eviction_relevance_floor: f32,
    pub enable_thinking_trace_recall: bool,
    pub enable_probe_recall: bool,
}

impl Default for DynamicRecallConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            thresholds: RecallThresholds::default_hysteresis(),
            max_workspace_tokens: 12_000,
            max_recall_iterations: 3,
            eviction_relevance_floor: 0.15,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
        }
    }
}

impl<R, E, V, P, M> DynamicRecallOrchestrator<R, E, V, P, M>
where
    R: Retriever,
    E: EmbeddingProvider,
    V: VectorIndex,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub fn new(
        retriever: R,
        embedder: E,
        vector_index: V,
        provenance: P,
        adapter: M,
        config: DynamicRecallConfig,
    ) -> Self {
        Self {
            retriever,
            embedder,
            vector_index,
            provenance,
            adapter,
            loaded: Vec::new(),
            loaded_ids: HashSet::new(),
            config,
        }
    }

    /// Run a single conversational turn with iterative multi-pass recall.
    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        // Phase 1: Initial retrieval on the user query
        let initial_hits = self.retriever.search(user, self.config.top_k)?;
        self.load_fragments(initial_hits)?;

        let mut last_response = self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        // Phase 2: Iterative recall refinement
        for _ in 0..self.config.max_recall_iterations {
            let loaded_before = self.loaded.len();

            if self.config.enable_thinking_trace_recall
                && self.adapter.capabilities().supports_visible_reasoning
            {
                self.process_thinking_trace(&last_response.answer)?;
            }

            if self.config.enable_probe_recall {
                self.process_probes(&last_response.answer)?;
            }

            self.process_annotations(&last_response.answer);

            // Workspace converged — no new fragments loaded
            if self.loaded.len() == loaded_before {
                break;
            }

            self.evict_stale_fragments(user, &last_response.answer);

            // Inject topic overlap warnings if the provenance ledger detected any
            let warnings = self.provenance.format_overlap_warnings();
            let enriched_system = if warnings.is_empty() {
                system.to_string()
            } else {
                format!("{}\n\n{}", system, warnings)
            };

            // Re-complete with enriched workspace
            last_response = self.adapter.complete(CompletionRequest {
                system: enriched_system,
                user: user.to_string(),
                workspace_fragments: self.loaded.clone(),
            })?;
        }

        Ok(last_response)
    }

    fn process_thinking_trace(&mut self, output: &str) -> CawResult<()> {
        let steps = extract_thinking_steps(output);

        for step in steps {
            if step.content.len() < 20 {
                continue;
            }

            let embeddings = self.embedder.embed(vec![&step.content])?;
            if let Some(embedding) = embeddings.first() {
                let hits = self.vector_index.search(embedding, self.config.top_k);
                let scored: Vec<ScoredStub> = hits
                    .into_iter()
                    .filter_map(|(id, score)| {
                        match self.retriever.read_range(&id, "full") {
                            Ok(frag) => Some(ScoredStub {
                                stub: caw_core::Stub {
                                    id,
                                    path: frag.locator.source.clone(),
                                    token_estimate: frag.tokens,
                                    kind: caw_core::ContentKind::Other,
                                    summary: String::new(),
                                    outline: Vec::new(),
                                    content_hash: String::new(),
                                    mtime_unix_secs: 0,
                                    consolidation_notes: Vec::new(),
                                },
                                score,
                            }),
                            Err(_) => None,
                        }
                    })
                    .collect();
                self.load_fragments(scored)?;
            }
        }

        Ok(())
    }

    fn process_probes(&mut self, output: &str) -> CawResult<()> {
        let probes = extract_probes(output);

        for probe in probes {
            let hits = self.retriever.search(&probe.content, self.config.top_k)?;
            self.load_fragments(hits)?;
        }

        Ok(())
    }

    /// Extract model annotations (<note id="...">...</note>) and record them
    /// as mid-session consolidation notes.
    fn process_annotations(&mut self, output: &str) {
        let annotations = extract_annotations(output);
        for ann in annotations {
            let note = ConsolidationNote {
                content: ann.content,
                source: ConsolidationSource::ModelAnnotation,
                created_at_secs: current_timestamp(),
            };
            self.provenance
                .record_consolidation(StubId(ann.stub_id), note);
        }
    }

    /// Evict fragments whose relevance to the current context has decayed.
    /// Uses term overlap as a fast proxy for relevance. Only triggers when
    /// the workspace is near its token budget.
    fn evict_stale_fragments(&mut self, query: &str, response: &str) {
        let current_tokens: usize = self.loaded.iter().map(|f| f.tokens).sum();
        let budget = self.config.max_workspace_tokens;

        // Only evict when workspace is 80%+ full
        if current_tokens < budget * 4 / 5 {
            return;
        }

        let context = format!("{} {}", query, response);

        let mut scored: Vec<(usize, f32)> = self
            .loaded
            .iter()
            .enumerate()
            .map(|(idx, frag)| {
                let score = term_overlap_score(&context, &frag.content);
                (idx, score)
            })
            .collect();

        // Evict lowest-relevance fragments first
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut to_evict = Vec::new();
        let mut tokens_remaining = current_tokens;

        for (idx, score) in &scored {
            if *score >= self.config.eviction_relevance_floor {
                break;
            }
            if tokens_remaining <= budget * 7 / 10 {
                break;
            }
            to_evict.push(*idx);
            tokens_remaining -= self.loaded[*idx].tokens;
        }

        // Remove in reverse index order to preserve indices
        to_evict.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_evict {
            let fragment = self.loaded.remove(idx);
            self.loaded_ids.remove(&fragment.stub_id);

            let note = ConsolidationNote {
                content: format!(
                    "Evicted during query about '{}'. Source: {}",
                    truncate_str(query, 100),
                    fragment.locator.source,
                ),
                source: ConsolidationSource::Eviction,
                created_at_secs: current_timestamp(),
            };

            self.provenance
                .record_consolidation(fragment.stub_id, note);
        }
    }

    fn load_fragments(&mut self, hits: Vec<ScoredStub>) -> CawResult<()> {
        for hit in hits {
            if self.loaded_ids.contains(&hit.stub.id) {
                continue;
            }

            if hit.score < self.config.thresholds.load {
                continue;
            }

            let current_tokens: usize = self.loaded.iter().map(|f| f.tokens).sum();
            if current_tokens >= self.config.max_workspace_tokens {
                break;
            }

            let fragment = self.retriever.read_range(&hit.stub.id, "full")?;

            if current_tokens + fragment.tokens <= self.config.max_workspace_tokens {
                self.loaded_ids.insert(hit.stub.id.clone());
                self.provenance.record_with_context(fragment.clone(), "", 0);
                self.loaded.push(fragment);
            }
        }

        Ok(())
    }

    pub fn read_range(&self, stub_id: &StubId, range: &Range) -> CawResult<RecallFragment> {
        let full_fragment = self.retriever.read_range(stub_id, "full")?;
        let range_content = range.apply(&full_fragment.content);
        let token_estimate = range_content.split_whitespace().count().max(1);

        Ok(RecallFragment {
            stub_id: stub_id.clone(),
            content: range_content,
            locator: caw_core::Locator {
                source: full_fragment.locator.source,
                locator: range.to_locator_string(),
            },
            tokens: token_estimate,
        })
    }
}

fn term_overlap_score(context: &str, content: &str) -> f32 {
    let ctx_terms: HashSet<String> = tokenize_for_scoring(context).into_iter().collect();
    let doc_terms: HashSet<String> = tokenize_for_scoring(content).into_iter().collect();

    if ctx_terms.is_empty() || doc_terms.is_empty() {
        return 0.0;
    }

    let intersection = ctx_terms.intersection(&doc_terms).count() as f32;
    intersection / ctx_terms.len().min(doc_terms.len()) as f32
}

fn tokenize_for_scoring(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 2 && !is_stopword(s))
        .map(String::from)
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all"
            | "can" | "has" | "was" | "one" | "our" | "out" | "his"
            | "her" | "had" | "how" | "its" | "may" | "who" | "did"
            | "get" | "let" | "say" | "she" | "too" | "use" | "way"
            | "with" | "this" | "that" | "from" | "have" | "been"
            | "they" | "them" | "then" | "than" | "each" | "which"
            | "their" | "will" | "would" | "there" | "what" | "about"
            | "could" | "other" | "into" | "more" | "some" | "very"
            | "when" | "also" | "just" | "should"
    )
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
