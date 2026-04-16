use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, EmbeddingProvider, ModelAdapter,
    ProvenanceStore, Range, RecallFragment, RecallThresholds, Retriever, ScoredStub,
    StubId, VectorIndex,
};
use caw_transform::{extract_probes, extract_thinking_steps};
use std::collections::HashSet;

/// Advanced orchestrator with thinking-trace and probe-based recall.
/// Holds a VectorIndex directly so that thinking-trace steps can be
/// embedded and matched against stored document embeddings without
/// going through the text-based Retriever.
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
    pub enable_thinking_trace_recall: bool,
    pub enable_probe_recall: bool,
}

impl Default for DynamicRecallConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            thresholds: RecallThresholds::default_hysteresis(),
            max_workspace_tokens: 12_000,
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

    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        let initial_hits = self.retriever.search(user, self.config.top_k)?;
        self.load_fragments(initial_hits)?;

        let response = self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        if self.config.enable_thinking_trace_recall
            && self.adapter.capabilities().supports_visible_reasoning
        {
            self.process_thinking_trace(&response.answer)?;
        }

        if self.config.enable_probe_recall {
            self.process_probes(&response.answer)?;
        }

        Ok(response)
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
                        // We need the stub to build a ScoredStub, but the vector
                        // index only returns (id, score). Look it up via retriever.
                        // If not found (shouldn't happen), skip silently.
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
                self.provenance.record(fragment.clone());
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
