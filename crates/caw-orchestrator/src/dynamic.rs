use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ConsolidationNote, ConsolidationSource,
    EmbeddingProvider, ModelAdapter, ProvenanceStore, Range, RecallFragment, RecallThresholds,
    Retriever, ScoredStub, StubId, StubStore, VectorIndex,
};
use caw_transform::{extract_annotations, extract_probes, extract_thinking_steps};
use std::collections::{HashMap, HashSet};

/// Advanced orchestrator with iterative multi-pass recall.
///
/// Each turn follows: initial retrieval -> complete -> extract probes/traces ->
/// load new fragments -> evict stale fragments -> re-complete with enriched
/// workspace. Repeats until convergence or max iterations.
pub struct DynamicRecallOrchestrator<R, E, V, P, M, S = ()> {
    pub retriever: R,
    pub embedder: E,
    pub vector_index: V,
    pub provenance: P,
    pub adapter: M,
    pub loaded: Vec<RecallFragment>,
    pub loaded_ids: HashSet<StubId>,
    /// Per-fragment relevance scores. Decay each reasoning step;
    /// refreshed when the model's output re-engages with the fragment.
    relevance_scores: HashMap<StubId, f32>,
    pub config: DynamicRecallConfig,
    /// Optional persistent store for consolidation notes.
    /// When present, notes survive across sessions.
    pub store: Option<S>,
}

#[derive(Debug, Clone)]
pub struct DynamicRecallConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub max_workspace_tokens: usize,
    pub max_recall_iterations: usize,
    /// Multiplicative decay applied to each fragment's relevance score per
    /// reasoning step. 0.8 means a fragment loses 20% of its score each
    /// step it isn't re-engaged with. Lower values = more aggressive eviction.
    pub relevance_decay_rate: f32,
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
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
        }
    }
}

impl<R, E, V, P, M, S> DynamicRecallOrchestrator<R, E, V, P, M, S>
where
    R: Retriever,
    E: EmbeddingProvider,
    V: VectorIndex,
    P: ProvenanceStore,
    M: ModelAdapter,
    S: StubStore,
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
            relevance_scores: HashMap::new(),
            config,
            store: None,
        }
    }

    pub fn with_store(mut self, store: S) -> Self {
        self.store = Some(store);
        self
    }

    /// Build the system prompt, optionally injecting cooperation instructions
    /// for models that support tool calls or visible reasoning.
    fn build_system_prompt(&self, base: &str) -> String {
        let caps = self.adapter.capabilities();
        if !caps.supports_tool_calls && !caps.supports_visible_reasoning {
            return base.to_string();
        }

        let mut prompt = base.to_string();
        prompt.push_str(concat!(
            "\n\nYou have access to recalled workspace content tagged with source locators. ",
            "When you learn something important from a recalled fragment — a key fact, decision, ",
            "or conclusion — emit a brief annotation: <note id=\"source_path\">what you learned</note>. ",
            "These annotations help the system remember what was relevant if the fragment is ",
            "later unloaded. Keep annotations concise (1-2 sentences).",
        ));

        if caps.supports_visible_reasoning {
            prompt.push_str(concat!(
                " You can also emit <probe>topic or question</probe> in your reasoning ",
                "to request additional context on a topic. The system will automatically ",
                "retrieve relevant material.",
            ));
        }

        prompt
    }

    /// Run a single conversational turn with iterative multi-pass recall.
    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        let system_prompt = self.build_system_prompt(system);

        // Phase 1: Initial retrieval on the user query
        let initial_hits = self.retriever.search(user, self.config.top_k)?;
        self.load_fragments(initial_hits)?;

        let mut last_response = self.adapter.complete(CompletionRequest {
            system: system_prompt.clone(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        // Phase 2: Iterative recall refinement
        for _ in 0..self.config.max_recall_iterations {
            let loaded_before = self.loaded.len();

            // Decay all scores before this reasoning step
            self.decay_relevance_scores();

            // Refresh scores for fragments the model engaged with
            self.refresh_relevance_scores(user, &last_response.answer);

            if self.config.enable_thinking_trace_recall
                && self.adapter.capabilities().supports_visible_reasoning
            {
                self.process_thinking_trace(&last_response.answer)?;
            }

            if self.config.enable_probe_recall {
                self.process_probes(&last_response.answer)?;
            }

            self.process_annotations(&last_response.answer);

            // Evict fragments whose relevance has decayed below threshold
            self.evict_stale_fragments(user);

            // Workspace converged — no new fragments loaded, none evicted
            if self.loaded.len() == loaded_before {
                break;
            }

            // Inject topic overlap warnings if the provenance ledger detected any
            let warnings = self.provenance.format_overlap_warnings();
            let enriched_system = if warnings.is_empty() {
                system_prompt.clone()
            } else {
                format!("{}\n\n{}", system_prompt, warnings)
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

            let embeddings = self.embedder.embed_query(vec![&step.content])?;
            if let Some(embedding) = embeddings.first() {
                let hits = self.vector_index.search(embedding, self.config.top_k);
                let scored: Vec<ScoredStub> = hits
                    .into_iter()
                    .filter_map(|(id, score)| match self.retriever.read_range(&id, "full") {
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
            let stub_id = StubId(ann.stub_id);
            self.persist_consolidation(&stub_id, &note);
        }
    }

    /// Record a consolidation note to both the in-memory provenance ledger
    /// and the persistent store (if available).
    fn persist_consolidation(&mut self, stub_id: &StubId, note: &ConsolidationNote) {
        self.provenance
            .record_consolidation(stub_id.clone(), note.clone());
        if let Some(store) = &mut self.store {
            let _ = store.save_consolidation(stub_id, note);
        }
    }

    /// Apply multiplicative decay to all loaded fragment scores.
    /// Called once per reasoning step so fragments the model stops
    /// engaging with gradually become eviction candidates.
    fn decay_relevance_scores(&mut self) {
        let rate = self.config.relevance_decay_rate;
        for score in self.relevance_scores.values_mut() {
            *score *= rate;
        }
    }

    /// Refresh relevance scores for fragments that the model's output
    /// re-engages with. Uses term overlap between the current context
    /// and each loaded fragment as a proxy for engagement.
    fn refresh_relevance_scores(&mut self, query: &str, response: &str) {
        let context = format!("{} {}", query, response);
        for frag in &self.loaded {
            let overlap = term_overlap_score(&context, &frag.content);
            if let Some(current) = self.relevance_scores.get_mut(&frag.stub_id) {
                // Take the higher of: decayed score or fresh overlap.
                // A fragment being discussed should never be penalized by decay.
                if overlap > *current {
                    *current = overlap;
                }
            }
        }
    }

    /// Evict fragments whose decayed relevance has dropped below the
    /// unload threshold (hysteresis). Also enforces the token budget
    /// as a hard ceiling — if the workspace is over budget, evict the
    /// lowest-scoring fragments until it fits.
    fn evict_stale_fragments(&mut self, query: &str) {
        let unload_threshold = self.config.thresholds.unload;
        let budget = self.config.max_workspace_tokens;

        // Collect (index, score) pairs sorted by score ascending
        let mut scored: Vec<(usize, f32)> = self
            .loaded
            .iter()
            .enumerate()
            .map(|(idx, frag)| {
                let score = self.relevance_scores.get(&frag.stub_id).copied().unwrap_or(0.0);
                (idx, score)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut to_evict = Vec::new();
        let mut tokens_after_eviction: usize = self.loaded.iter().map(|f| f.tokens).sum();

        for (idx, score) in &scored {
            let below_threshold = *score < unload_threshold;
            let over_budget = tokens_after_eviction > budget;

            if !below_threshold && !over_budget {
                break;
            }

            to_evict.push(*idx);
            tokens_after_eviction -= self.loaded[*idx].tokens;
        }

        // Remove in reverse index order to preserve indices
        to_evict.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_evict {
            let fragment = self.loaded.remove(idx);
            self.loaded_ids.remove(&fragment.stub_id);

            let decayed_score = self.relevance_scores.remove(&fragment.stub_id).unwrap_or(0.0);
            let note = ConsolidationNote {
                content: format!(
                    "Evicted (relevance decayed to {:.2}) during query about '{}'. Source: {}",
                    decayed_score,
                    truncate_str(query, 100),
                    fragment.locator.source,
                ),
                source: ConsolidationSource::Eviction,
                created_at_secs: current_timestamp(),
            };

            self.persist_consolidation(&fragment.stub_id, &note);
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
                self.relevance_scores
                    .insert(hit.stub.id.clone(), hit.score);
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
