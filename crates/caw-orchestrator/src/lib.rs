pub mod consolidation;
pub mod degradation;
pub mod dynamic;
pub mod probe_recall;
pub mod thinking_trace;

use caw_core::RecallThresholds;
use caw_core::{
    BudgetScheduler, CawResult, CompletionRequest, CompletionResponse, ModelAdapter,
    ProvenanceStore, RecallFragment, Retriever, SchedulerInput, TokenBudget,
};
use std::collections::HashSet;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub top_k: usize,
    pub thresholds: RecallThresholds,
    pub default_range: String,
    pub budget: TokenBudget,
    /// Maximum number of thinking-phase restarts per turn for reasoning
    /// models. Each restart happens when a thinking step triggers a new
    /// recall hit — the stream is stopped, context is injected, and the
    /// model re-thinks from the start with the richer workspace.
    pub max_recall_iterations: usize,
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
            max_recall_iterations: 3,
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
        let has_reasoning = self.adapter.capabilities().supports_visible_reasoning;
        debug!(
            model = self.adapter.model_name(),
            query = &user[..user.len().min(80)],
            visible_reasoning = has_reasoning,
            "run_turn start"
        );

        // seen_this_turn prevents re-injecting a stub across thinking restarts,
        // even if it gets evicted by the budget scheduler mid-loop.
        let mut seen_this_turn: HashSet<String> = HashSet::new();

        // Initial recall from the user query itself.
        do_recall(
            user,
            &mut self.retriever,
            &self.scheduler,
            &mut self.loaded,
            &mut self.provenance,
            &self.config,
            &mut seen_this_turn,
        )?;

        debug!(loaded = self.loaded.len(), "initial query recall complete");

        if has_reasoning {
            let mut iterations = 0usize;
            for i in 0..self.config.max_recall_iterations {
                debug!(iteration = i + 1, loaded = self.loaded.len(), "thinking phase start");

                let req = CompletionRequest {
                    system: system.to_string(),
                    user: user.to_string(),
                    workspace_fragments: self.loaded.clone(),
                };

                // Split field borrows so the closure can mutate retriever/loaded/
                // provenance while adapter is held as an immutable reference.
                let adapter = &self.adapter;
                let retriever = &mut self.retriever;
                let scheduler = &self.scheduler;
                let loaded = &mut self.loaded;
                let provenance = &mut self.provenance;
                let config = &self.config;

                let mut admitted_any = false;
                adapter.thinking_with_steps(req, &mut |step| {
                    debug!(
                        step = &step[..step.len().min(120)],
                        "thinking step — searching"
                    );
                    let new = do_recall(
                        step,
                        retriever,
                        scheduler,
                        loaded,
                        provenance,
                        config,
                        &mut seen_this_turn,
                    )?;
                    if new {
                        admitted_any = true;
                        debug!("new context admitted — stopping stream for restart");
                    }
                    // Stop stream on first new admission — restart with richer workspace.
                    Ok(!new)
                })?;

                iterations = i + 1;
                if !admitted_any {
                    debug!(iterations, "thinking loop converged — no new admissions");
                    break;
                }
            }

            info!(
                iterations,
                loaded = self.loaded.len(),
                "thinking-trace recall complete"
            );
        }

        debug!(workspace_frags = self.loaded.len(), "sending final complete");
        self.adapter.complete(CompletionRequest {
            system: system.to_string(),
            user: user.to_string(),
            workspace_fragments: self.loaded.clone(),
        })
    }
}

/// Search, filter against `seen`, schedule, and record admitted fragments.
/// Returns true if anything new was admitted.
fn do_recall<R, S, P>(
    query: &str,
    retriever: &mut R,
    scheduler: &S,
    loaded: &mut Vec<RecallFragment>,
    provenance: &mut P,
    config: &OrchestratorConfig,
    seen: &mut HashSet<String>,
) -> CawResult<bool>
where
    R: Retriever,
    S: BudgetScheduler,
    P: ProvenanceStore,
{
    let hits = retriever.search(query, config.top_k)?;
    let total_hits = hits.len();

    let mut candidates = Vec::new();
    let mut candidate_scores = Vec::new();

    for hit in hits
        .into_iter()
        .filter(|h| h.score >= config.thresholds.load && !seen.contains(&h.stub.id.0))
    {
        debug!(
            path = %hit.stub.path,
            score = hit.score,
            "candidate above threshold"
        );
        let fragment = retriever.read_range(&hit.stub.id, &config.default_range)?;
        candidate_scores.push(hit.score);
        candidates.push(fragment);
    }

    debug!(
        query = &query[..query.len().min(80)],
        total_hits,
        candidates = candidates.len(),
        "recall search done"
    );

    if candidates.is_empty() {
        return Ok(false);
    }

    let decision = scheduler.schedule(SchedulerInput {
        currently_loaded: loaded.clone(),
        candidates,
        candidate_scores,
        budget: config.budget,
    });

    for frag in &decision.admitted {
        debug!(
            path = %frag.locator.source,
            tokens = frag.tokens,
            "fragment admitted"
        );
        seen.insert(frag.stub_id.0.clone());
        provenance.record(frag.clone());
    }

    for frag in &decision.evicted {
        debug!(path = %frag.locator.source, "fragment evicted");
    }

    *loaded = decision.keep;
    Ok(!decision.admitted.is_empty())
}
