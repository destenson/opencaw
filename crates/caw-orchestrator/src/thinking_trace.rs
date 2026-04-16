use caw_core::{
    BudgetScheduler, CawResult, CompletionRequest, CompletionResponse, EmbeddingProvider,
    ModelAdapter, ProvenanceStore, RecallFragment, RecallThresholds, Retriever, SchedulerInput,
    TokenBudget,
};

/// Orchestrator that implements thinking-trace recall for reasoning models
pub struct ThinkingTraceOrchestrator<R, S, P, M, E>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
    E: EmbeddingProvider,
{
    pub retriever: R,
    pub scheduler: S,
    pub provenance: P,
    pub adapter: M,
    pub embedder: E,
    pub loaded: Vec<RecallFragment>,
    pub config: ThinkingTraceConfig,
}

#[derive(Debug, Clone)]
pub struct ThinkingTraceConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub default_range: String,
    pub budget: TokenBudget,
    pub step_boundary_pattern: String,
}

impl Default for ThinkingTraceConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            thresholds: RecallThresholds::default_hysteresis(),
            default_range: "full".to_string(),
            budget: TokenBudget {
                max_total: 16_000,
                reserved_for_prompt: 2_000,
                reserved_for_answer: 2_000,
            },
            step_boundary_pattern: r"\n\n".to_string(),
        }
    }
}

impl<R, S, P, M, E> ThinkingTraceOrchestrator<R, S, P, M, E>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
    E: EmbeddingProvider,
{
    pub fn new(
        retriever: R,
        scheduler: S,
        provenance: P,
        adapter: M,
        embedder: E,
        config: ThinkingTraceConfig,
    ) -> Self {
        Self {
            retriever,
            scheduler,
            provenance,
            adapter,
            embedder,
            loaded: Vec::new(),
            config,
        }
    }

    pub fn run_turn_with_recall(
        &mut self,
        system: &str,
        user: &str,
    ) -> CawResult<CompletionResponse> {
        self.recall_for_query(user)?;

        // Streaming step-boundary recall would go here for models with
        // visible reasoning. Requires a streaming completion API that
        // yields partial responses — not yet available in this crate.
        let response = self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })?;

        Ok(response)
    }

    fn recall_for_query(&mut self, query: &str) -> CawResult<Vec<RecallFragment>> {
        let hits = self.retriever.search(query, self.config.top_k)?;
        let mut candidates = Vec::new();
        let mut candidate_scores = Vec::new();

        for hit in hits {
            if hit.score >= self.config.thresholds.load {
                let fragment = self
                    .retriever
                    .read_range(&hit.stub.id, &self.config.default_range)?;
                candidate_scores.push(hit.score);
                candidates.push(fragment);
            }
        }

        let decision = self.scheduler.schedule(SchedulerInput {
            currently_loaded: self.loaded.clone(),
            candidates,
            candidate_scores,
            budget: self.config.budget,
        });

        self.loaded = decision.keep.clone();

        for frag in &decision.admitted {
            self.provenance.record(frag.clone());
        }

        Ok(decision.admitted)
    }

    pub fn recall_for_step(&mut self, step_text: &str) -> CawResult<Vec<RecallFragment>> {
        self.recall_for_query(step_text)
    }
}

/// Detect step boundaries in thinking trace
pub fn extract_thinking_steps(text: &str, boundary_pattern: &str) -> Vec<String> {
    text.split(boundary_pattern)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_steps() {
        let trace = "Step 1: First thought\n\nStep 2: Second thought\n\nStep 3: Third thought";
        let steps = extract_thinking_steps(trace, "\n\n");
        assert_eq!(steps.len(), 3);
        assert!(steps[0].contains("First thought"));
    }
}
