use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, EmbeddingProvider, ModelAdapter,
    ProvenanceStore, Range, RecallFragment, Retriever, ScoredStub, Stub, StubId,
};
use caw_transform::{apply_range, extract_probes, extract_thinking_steps};
use std::collections::HashSet;

/// Advanced orchestrator with thinking-trace and probe-based recall
pub struct DynamicRecallOrchestrator<R, E, P, M>
where
    R: Retriever,
    E: EmbeddingProvider,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub retriever: R,
    pub embedder: E,
    pub provenance: P,
    pub adapter: M,
    pub loaded: Vec<RecallFragment>,
    pub loaded_ids: HashSet<StubId>,
    pub config: DynamicRecallConfig,
}

#[derive(Debug, Clone)]
pub struct DynamicRecallConfig {
    pub top_k: usize,
    pub load_threshold: f32,
    pub unload_threshold: f32,
    pub max_workspace_tokens: usize,
    pub enable_thinking_trace_recall: bool,
    pub enable_probe_recall: bool,
}

impl Default for DynamicRecallConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            load_threshold: 0.7,
            unload_threshold: 0.4,
            max_workspace_tokens: 12_000,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
        }
    }
}

impl<R, E, P, M> DynamicRecallOrchestrator<R, E, P, M>
where
    R: Retriever,
    E: EmbeddingProvider,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub fn new(retriever: R, embedder: E, provenance: P, adapter: M, config: DynamicRecallConfig) -> Self {
        Self {
            retriever,
            embedder,
            provenance,
            adapter,
            loaded: Vec::new(),
            loaded_ids: HashSet::new(),
            config,
        }
    }

    /// Execute a turn with dynamic recall
    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        // Initial retrieval based on user query
        let initial_hits = self.retriever.search(user, self.config.top_k)?;
        self.load_fragments(initial_hits)?;

        // Get initial response
        let response = self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        // Process thinking trace or probes for additional recall
        if self.config.enable_thinking_trace_recall && self.adapter.capabilities().supports_visible_reasoning {
            self.process_thinking_trace(&response.answer)?;
        }

        if self.config.enable_probe_recall {
            self.process_probes(&response.answer)?;
        }

        Ok(response)
    }

    /// Process thinking steps and trigger recall
    fn process_thinking_trace(&mut self, output: &str) -> CawResult<()> {
        let steps = extract_thinking_steps(output);
        
        for step in steps {
            if step.content.len() < 20 {
                continue; // Skip very short steps
            }
            
            // Embed the step
            let embeddings = self.embedder.embed(vec![&step.content])?;
            if let Some(embedding) = embeddings.first() {
                // Search for relevant stubs
                let hits = self.search_by_embedding(embedding)?;
                self.load_fragments(hits)?;
            }
        }
        
        Ok(())
    }

    /// Process probe markers and trigger recall
    fn process_probes(&mut self, output: &str) -> CawResult<()> {
        let probes = extract_probes(output);
        
        for probe in probes {
            let hits = self.retriever.search(&probe.content, self.config.top_k)?;
            self.load_fragments(hits)?;
        }
        
        Ok(())
    }

    /// Load fragments from scored stubs with hysteresis
    fn load_fragments(&mut self, hits: Vec<ScoredStub>) -> CawResult<()> {
        for hit in hits {
            // Suppress already-loaded files
            if self.loaded_ids.contains(&hit.stub.id) {
                continue;
            }
            
            // Check threshold
            if hit.score < self.config.load_threshold {
                continue;
            }
            
            // Check budget
            let current_tokens: usize = self.loaded.iter().map(|f| f.tokens).sum();
            if current_tokens >= self.config.max_workspace_tokens {
                break;
            }
            
            // Load fragment
            let fragment = self.retriever.read_range(&hit.stub.id, "full")?;
            
            if current_tokens + fragment.tokens <= self.config.max_workspace_tokens {
                self.loaded_ids.insert(hit.stub.id.clone());
                self.provenance.record(fragment.clone());
                self.loaded.push(fragment);
            }
        }
        
        // Evict fragments below unload threshold
        self.evict_low_score_fragments()?;
        
        Ok(())
    }

    /// Evict fragments that fall below unload threshold
    fn evict_low_score_fragments(&mut self) -> CawResult<()> {
        // For MVP, we don't re-score. In production, track scores and evict based on threshold.
        // This is a placeholder for hysteresis-based eviction.
        Ok(())
    }

    /// Search by embedding (helper for thinking-trace recall)
    fn search_by_embedding(&self, embedding: &[f32]) -> CawResult<Vec<ScoredStub>> {
        // This would need VectorStore trait access
        // For now, fall back to empty results
        // TODO: Add direct VectorStore access to orchestrator
        Ok(Vec::new())
    }

    /// Read specific range from a stub
    pub fn read_range(&self, stub_id: &StubId, range: &Range) -> CawResult<RecallFragment> {
        let content = self.retriever.read_range(stub_id, "full")?.content;
        let range_content = apply_range(&content, range)?;
        
        // Get stub for metadata
        let fragment = self.retriever.read_range(stub_id, "full")?;
        
        Ok(RecallFragment {
            stub_id: stub_id.clone(),
            content: range_content,
            locator: caw_core::Locator {
                source: fragment.locator.source,
                locator: range.to_locator_string(),
            },
            tokens: (range_content.len() / 4).max(1),
        })
    }
}
