pub mod consolidation;
pub mod dynamic;
pub mod probe_recall;
pub mod thinking_trace;

use caw_core::{
    BudgetScheduler, CawResult, CompletionRequest, CompletionResponse, ModelAdapter,
    ProvenanceStore, RecallFragment, RecallThresholds, Retriever, SchedulerInput, TokenBudget,
};

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub default_range: String,
    pub budget: TokenBudget,
}

impl Default for OrchestratorConfig {
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
        }
    }
}

pub struct RecallOrchestrator<R, S, P, M>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub retriever: R,
    pub scheduler: S,
    pub provenance: P,
    pub adapter: M,
    pub loaded: Vec<RecallFragment>,
    pub config: OrchestratorConfig,
}

impl<R, S, P, M> RecallOrchestrator<R, S, P, M>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
    M: ModelAdapter,
{
    pub fn run_turn(&mut self, system: &str, user: &str) -> CawResult<CompletionResponse> {
        let hits = self.retriever.search(user, self.config.top_k)?;
        let mut candidates = Vec::new();
        let mut candidate_scores = Vec::new();

        for hit in hits
            .into_iter()
            .filter(|h| h.score >= self.config.thresholds.load)
        {
            let fragment = self
                .retriever
                .read_range(&hit.stub.id, &self.config.default_range)?;
            candidate_scores.push(hit.score);
            candidates.push(fragment);
        }

        let decision = self.scheduler.schedule(SchedulerInput {
            currently_loaded: self.loaded.clone(),
            candidates,
            candidate_scores,
            budget: self.config.budget,
        });

        self.loaded = decision.keep;

        for frag in &decision.admitted {
            self.provenance.record(frag.clone());
        }

        self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })
    }
}
