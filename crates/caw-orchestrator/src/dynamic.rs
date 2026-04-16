use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, EmbeddingProvider, ModelAdapter,
    ProvenanceStore, Range, RecallFragment, RecallThresholds, Retriever, ScoredStub,
    StubId, VectorStore,
};
use caw_transform::{extract_probes, extract_thinking_steps};
use std::collections::HashSet;

/// Advanced orchestrator with thinking-trace and probe-based recall.
/// Unlike the simpler orchestrators, this one holds a VectorStore directly
/// so that thinking-trace steps can be embedded and matched against stored
/// document embeddings without going through the text-based Retriever.
pub struct DynamicRecallOrchestrator<R, E, V, P, M>
where
    R: Retriever,
    E: EmbeddingProvider,
    V: VectorStore,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub retriever: R,
    pub embedder: E,
    pub vector_store: V,
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
    V: VectorStore,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub fn new(
        retriever: R,
        embedder: E,
        vector_store: V,
        provenance: P,
        adapter: M,
        config: DynamicRecallConfig,
    ) -> Self {
        Self {
            retriever,
            embedder,
            vector_store,
            provenance,
            adapter,
            loaded: Vec::new(),
            loaded_ids: HashSet::new(),
            config,
        }
    }

    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        // Initial text-based retrieval on the user query
        let initial_hits = self.retriever.search(user, self.config.top_k)?;
        self.load_fragments(initial_hits)?;

        let response = self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        // Process thinking trace for embedding-based recall
        if self.config.enable_thinking_trace_recall
            && self.adapter.capabilities().supports_visible_reasoning
        {
            self.process_thinking_trace(&response.answer)?;
        }

        // Process explicit probe markers for text-based recall
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
                let hits = self.vector_store.search_by_embedding(embedding, self.config.top_k)?;
                self.load_fragments(hits)?;
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

        self.evict_low_score_fragments()?;
        Ok(())
    }

    fn evict_low_score_fragments(&mut self) -> CawResult<()> {
        // Re-score loaded fragments against recent query context would go here.
        // For now, eviction happens through the scheduler in the other orchestrators.
        // The DynamicRecallOrchestrator relies on budget limits in load_fragments
        // to prevent unbounded growth.
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
