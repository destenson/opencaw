use crate::consolidation::{ConsolidationSynthesizer, MechanicalConsolidation};
use crate::degradation::DegradationMonitor;
use crate::session::{self, SessionFile};
use caw_core::{
    AugmentationSignals, CawError, CawResult, CompletionRequest, CompletionResponse,
    ConsolidationNote, ConsolidationSource, EmbeddingProvider, LineReference, Locator, ModelAdapter,
    ProvenanceStore, Range, RecallFragment, RecallThresholds, Retriever, ScoredStub, StubId,
    StubStore, VectorIndex, candidate_list_fragment, count_tokens_cl100k, tokenize_terms,
};
use caw_eval::SessionEvaluator;
use caw_ingest::{DocumentIdSet, IngestionPipeline};
use caw_transform::{extract_annotations, extract_line_references, extract_probes, extract_thinking_steps, strip_markers};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, trace, warn};

/// Controls whether the orchestrator injects probe/annotation cooperation
/// instructions into the system prompt.
///
/// The default (`Auto`) preserves the pre-calibration behavior: instructions
/// are injected only for models whose adapter reports `supports_visible_reasoning`.
/// After running `caw-bench-coop` you can override this per-model based on
/// empirical compliance data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CooperationMode {
    /// Use adapter capability flags: inject instructions only for models with
    /// `supports_visible_reasoning`. This is the default.
    #[default]
    Auto,
    /// Always inject probe/annotation instructions regardless of capability flags.
    Cooperative,
    /// Never inject probe/annotation instructions.
    Transparent,
}

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
    /// Source paths of currently loaded fragments. Prevents admitting
    /// multiple chunks from the same file — one per source is enough.
    loaded_sources: HashSet<String>,
    /// Per-fragment relevance scores. Decay each reasoning step;
    /// refreshed when the model's output re-engages with the fragment.
    relevance_scores: HashMap<StubId, f32>,
    pub config: DynamicRecallConfig,
    /// Optional persistent store for consolidation notes.
    /// When present, notes survive across sessions.
    pub store: Option<S>,
    consolidation_synthesizer: Box<dyn ConsolidationSynthesizer>,
    /// When present, enables graceful degradation based on component health.
    /// Without this, the orchestrator runs at full capability unconditionally.
    pub degradation_monitor: Option<DegradationMonitor>,
    /// Append-only session log for crash recovery and cross-session recall.
    session: Option<SessionFile>,
    /// In-memory content for turns recorded this session. Served directly
    /// rather than going through the StubStore file-reading path.
    session_content: HashMap<StubId, String>,
    session_turn: usize,
    /// Optional evaluator for collecting recall, eviction, probe, and annotation
    /// metrics. When present, events are recorded automatically during run_turn.
    pub evaluator: Option<SessionEvaluator>,
}

#[derive(Debug, Clone)]
pub struct DynamicRecallConfig {
    /// Candidate pool size passed to ANN search. The load threshold — not
    /// this value — controls how many candidates are actually admitted.
    pub max_candidates: usize,
    pub thresholds: RecallThresholds,
    pub max_workspace_tokens: usize,
    pub max_recall_iterations: usize,
    /// Multiplicative decay applied to each fragment's relevance score per
    /// reasoning step. 0.8 means a fragment loses 20% of its score each
    /// step it isn't re-engaged with. Lower values = more aggressive eviction.
    pub relevance_decay_rate: f32,
    pub enable_thinking_trace_recall: bool,
    pub enable_probe_recall: bool,
    pub enable_line_reference_recall: bool,
    /// Hard cap on the number of fragments that can be loaded simultaneously.
    /// Prevents unbounded accumulation across long conversations regardless of
    /// the token budget. When hit, new admissions are blocked.
    pub max_loaded_fragments: usize,
    /// Maximum fragments to load in the initial (pre-probe) phase. If more
    /// candidates than this clear the load threshold, the query is too broad
    /// for confident initial augmentation — load nothing and let probes drive.
    pub max_initial_fragments: usize,
    /// Whether to inject probe/annotation cooperation instructions into the
    /// system prompt. Defaults to `Auto` (capability-based). Override with
    /// calibration data from `caw-bench-coop`.
    pub cooperation_mode: CooperationMode,
    /// How many tokens to generate between passive injection checks. Only used
    /// when the adapter reports `supports_passive_injection = true`.
    pub passive_injection_interval: usize,
    /// How many of the most recently generated tokens to include in the
    /// embedding query at each check. Must be >= `passive_injection_interval`
    /// to avoid missing sequences that span an interval boundary.
    pub passive_injection_window_size: usize,
}

impl Default for DynamicRecallConfig {
    fn default() -> Self {
        Self {
            max_candidates: 20,
            thresholds: RecallThresholds::default_hysteresis(),
            max_workspace_tokens: 12_000,
            max_recall_iterations: 3,
            relevance_decay_rate: 0.8,
            enable_thinking_trace_recall: true,
            enable_probe_recall: true,
            enable_line_reference_recall: true,
            max_loaded_fragments: 15,
            max_initial_fragments: 4,
            cooperation_mode: CooperationMode::Auto,
            passive_injection_interval: 32,
            passive_injection_window_size: 192,
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
            loaded_sources: HashSet::new(),
            relevance_scores: HashMap::new(),
            config,
            store: None,
            consolidation_synthesizer: Box::new(MechanicalConsolidation),
            degradation_monitor: None,
            session: None,
            session_content: HashMap::new(),
            session_turn: 0,
            evaluator: None,
        }
    }

    pub fn with_degradation_monitor(mut self, monitor: DegradationMonitor) -> Self {
        self.degradation_monitor = Some(monitor);
        self
    }

    pub fn with_store(mut self, store: S) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_consolidation_synthesizer(
        mut self,
        synthesizer: Box<dyn ConsolidationSynthesizer>,
    ) -> Self {
        self.consolidation_synthesizer = synthesizer;
        self
    }

    pub fn with_evaluator(mut self, evaluator: SessionEvaluator) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// Remove and return the evaluator so its accumulated metrics can be read.
    pub fn take_evaluator(&mut self) -> Option<SessionEvaluator> {
        self.evaluator.take()
    }

    /// Enable session history: write each turn to an append-only file in
    /// `session_dir` and recall them semantically in future turns.
    /// Previous runs' session files in the same directory are ingested
    /// read-only at startup so prior exchanges are immediately retrievable.
    pub fn with_session(mut self, session_dir: &Path) -> CawResult<Self> {
        std::fs::create_dir_all(session_dir).map_err(|e| caw_core::CawError::Io(e.to_string()))?;
        let pipeline = IngestionPipeline::new();
        let file_name = format!("session-{}.md", session::timestamp_str());
        let file_path = session_dir.join(file_name);
        let loaded = SessionFile::load_previous(
            session_dir,
            &file_path,
            &pipeline,
            &mut self.retriever,
            &DocumentIdSet::new(),
        )?;
        debug!(dir = %session_dir.display(), stubs = loaded, "loaded prior session history");
        self.session = Some(SessionFile::create(file_path)?);
        Ok(self)
    }

    /// Build the system prompt, optionally injecting cooperation instructions
    /// for models capable enough to follow them.
    ///
    /// Per the design doc (§ 3.2), probe markers are *for* hidden-reasoning
    /// models — they're the explicit instruction-based equivalent of visible
    /// `<think>` blocks. Any reasoning-capable model (tool-call, hidden, or
    /// visible) can follow the marker instruction. The `<probe>` tag name
    /// itself is a convention; the orchestrator's `extract_probes` parses
    /// whatever tag the system prompt asks the model to emit.
    fn build_system_prompt(&self, base: &str) -> String {
        let inject = match self.config.cooperation_mode {
            CooperationMode::Cooperative => true,
            CooperationMode::Transparent => false,
            // Auto: inject when the adapter reports either visible reasoning
            // (thinking-trace models — probes are mid-trace signals) or hidden
            // reasoning (models verified to follow the cooperative protocol via
            // caw-bench-coop). Both paths are opt-in per adapter.
            CooperationMode::Auto => {
                let caps = self.adapter.capabilities();
                caps.supports_visible_reasoning || caps.supports_hidden_reasoning
            }
        };
        if !inject {
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

        // Probes: explicit tagged markers the model emits when it wants more
        // context. Works for hidden-reasoning models (the design doc's
        // canonical use case) as well as visible-reasoning models.
        prompt.push_str(concat!(
            " You can also emit <probe>topic or question</probe> anywhere in your response ",
            "to request additional context on a topic. The system will automatically retrieve ",
            "relevant material. Use probes when the recalled context is missing something you ",
            "need to answer well — not for every turn.",
        ));

        // Thinking-trace instructions: for models that support visible reasoning, the model's
        prompt.push_str(concat!(
            "\n\nFeel free to think out loud as you work through the problem. The system will ",
            "parse your thinking steps and recall relevant context for each step. This helps ",
            "you get the information you need even if your initial retrieval missed ",
            "something important.",
        ));

        if self.config.enable_line_reference_recall {
            prompt.push_str(concat!(
                " To load specific lines from a recalled file, write the path and range in your ",
                "thinking as `path/to/file.ext:start-end` (e.g. `./crates/caw-core/src/lib.rs:45-67`). ",
                "The system will automatically load that line range into your workspace.",
            ));
        }

        prompt
    }

    fn auto_recall_enabled(&self) -> bool {
        self.degradation_monitor
            .as_ref()
            .is_none_or(|m| m.should_auto_recall())
    }

    fn stub_generation_enabled(&self) -> bool {
        self.degradation_monitor
            .as_ref()
            .is_none_or(|m| m.should_generate_stubs())
    }

    /// Run a single conversational turn with iterative multi-pass recall.
    ///
    /// `guidance` is appended to every completion request as `workspace_guidance`
    /// (used by the intent classifier in the CLI to inject query-specific hints).
    pub fn run_turn(
        &mut self,
        system: &str,
        user: &str,
        guidance: &[String],
        signals: Option<&AugmentationSignals>,
    ) -> CawResult<CompletionResponse> {
        let caps = self.adapter.capabilities();
        debug!(
            model = %self.adapter.model_name(),
            query = %user,
            visible_reasoning = caps.supports_visible_reasoning,
            "run_turn start"
        );

        // Pass-through mode: skip all recall machinery
        if !self.stub_generation_enabled() {
            debug!("pass-through mode active — skipping recall");
            return self.adapter.complete(CompletionRequest {
                system: system.to_string(),
                user: user.to_string(),
                workspace_guidance: guidance.to_vec(),
                ..Default::default()
            });
        }

        // Cross-turn eviction: fragments from the previous turn decay once more
        // before the new query runs initial retrieval, so stale context doesn't
        // crowd out material relevant to the new query.
        if !self.loaded.is_empty() {
            self.decay_relevance_scores();
            self.evict_stale_fragments(user, signals);
        }

        let system_prompt = self.build_system_prompt(system);

        // Phase 1: Initial retrieval on the user query.
        // Many candidates clearing the threshold is a signal that the query is
        // too broad for confident augmentation — load nothing and let probes drive.
        //
        // For inventory queries the user's natural-language phrasing often doesn't
        // share vocabulary with the file names and artifact paths we want to surface,
        // so we append a small set of path-biasing terms before embedding. This is
        // pure string manipulation — no extra model call.
        let retrieval_query: std::borrow::Cow<str> =
            if signals.map_or(false, |s| s.is_inventory_request) {
                format!("{user} files paths artifacts names list").into()
            } else {
                user.into()
            };
        let initial_hits = self
            .retriever
            .search(&retrieval_query, self.config.max_candidates)?;
        let above_threshold = initial_hits
            .iter()
            .filter(|h| h.score >= self.config.thresholds.load)
            .count();
        debug!(
            hits = initial_hits.len(),
            above_threshold,
            max_initial = self.config.max_initial_fragments,
            "initial retrieval complete"
        );
        let distinct_above_unload = initial_hits
            .iter()
            .filter(|h| h.score >= self.config.thresholds.unload)
            .map(|h| h.stub.path.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let show_listing = above_threshold > self.config.max_initial_fragments
            || distinct_above_unload > self.config.max_initial_fragments;
        // Build the candidate list before consuming initial_hits. Returns Some
        // only when the gate fires; the borrow ends before the move below.
        let candidate_fragment: Option<RecallFragment> = show_listing.then(|| {
            let list_threshold = if above_threshold > self.config.max_initial_fragments {
                self.config.thresholds.load
            } else {
                self.config.thresholds.unload
            };
            candidate_list_fragment(&initial_hits, list_threshold)
        });

        // Session turn recall: embed the query against vector_index and load any
        // current-session turns above threshold. These are not in the corpus retriever,
        // so they need a separate pass. This runs regardless of the ambiguity gate
        // because session history is always small and never contributes to the gate count.
        if !self.session_content.is_empty() {
            if let Ok(embeddings) = self.embedder.embed_query(vec![user]) {
                if let Some(embedding) = embeddings.first() {
                    let session_hits: Vec<(StubId, f32)> = self
                        .vector_index
                        .search(embedding, self.config.max_candidates)
                        .into_iter()
                        .filter(|(id, _)| self.session_content.contains_key(id))
                        .collect();
                    if !session_hits.is_empty() {
                        debug!(hits = session_hits.len(), "session turn initial recall");
                        self.load_fragments(session_hits)?;
                    }
                }
            }
        }

        // When the gate fires, retain initial_hits for the post-completion
        // mentioned-files pass. When it doesn't, consume them into load_fragments.
        let candidate_initial_hits: Option<Vec<ScoredStub>>;
        if candidate_fragment.is_none() {
            self.load_fragments(
                initial_hits
                    .into_iter()
                    .map(|hit| (hit.stub.id, hit.score))
                    .collect(),
            )?;
            candidate_initial_hits = None;
        } else {
            debug!(
                above_threshold,
                distinct_above_unload,
                max_initial = self.config.max_initial_fragments,
                "query matches multiple sources — surfacing candidate list"
            );
            candidate_initial_hits = Some(initial_hits);
        }
        debug!(loaded = self.loaded.len(), "initial query recall complete");

        // Candidate list fragment (if any) is injected only into this first
        // completion — it's not tracked in self.loaded and won't be evicted
        // or counted against the workspace budget across turns.
        let mut initial_fragments = self.loaded.clone();
        initial_fragments.extend(candidate_fragment);

        let initial_req = CompletionRequest {
            system: system_prompt.clone(),
            user: user.to_string(),
            workspace_fragments: initial_fragments,
            workspace_guidance: guidance.to_vec(),
        };

        let mut last_response = if caps.supports_passive_injection {
            self.run_with_passive_injection(initial_req)?
        } else {
            self.adapter.complete(initial_req)?
        };

        // When the candidate list was shown, check if the model's response
        // mentions any candidate file paths explicitly. If so, load those files
        // and re-complete so the final answer is grounded in actual content.
        if let Some(ref hits) = candidate_initial_hits {
            let search_text = match &last_response.thinking {
                Some(t) => format!("{}\n{}", t, last_response.answer),
                None => last_response.answer.clone(),
            };
            let mut seen_paths = std::collections::HashSet::new();
            let mentioned: Vec<(StubId, f32)> = hits
                .iter()
                .filter(|h| h.score >= self.config.thresholds.load)
                .filter(|h| {
                    let norm = h.stub.path.trim_start_matches("./");
                    search_text.contains(norm) || search_text.contains(&h.stub.path)
                })
                .filter(|h| seen_paths.insert(h.stub.path.clone()))
                .map(|h| (h.stub.id.clone(), h.score))
                .collect();
            if !mentioned.is_empty() {
                debug!(
                    count = mentioned.len(),
                    "loading files mentioned in response to candidate list"
                );
                self.load_fragments(mentioned)?;
                last_response = self.adapter.complete(CompletionRequest {
                    system: system_prompt.clone(),
                    user: user.to_string(),
                    workspace_fragments: self.loaded.clone(),
                    workspace_guidance: guidance.to_vec(),
                })?;
            }
        }

        // Phase 2: Iterative recall refinement
        for i in 0..self.config.max_recall_iterations {
            let loaded_before = self.loaded.len();
            debug!(
                iteration = i + 1,
                loaded = loaded_before,
                "refinement iteration start"
            );

            self.decay_relevance_scores();
            self.refresh_relevance_scores(user, &last_response.answer);

            // Automatic recall only runs when the monitor permits it
            if self.auto_recall_enabled() {
                if self.config.enable_thinking_trace_recall {
                    self.process_thinking_trace(&last_response.answer)?;
                }

                if self.config.enable_probe_recall {
                    self.process_probes(&last_response.answer)?;
                }

                if self.config.enable_line_reference_recall {
                    let text = match &last_response.thinking {
                        Some(t) => format!("{}\n{}", t, last_response.answer),
                        None => last_response.answer.clone(),
                    };
                    self.process_line_references(&text)?;
                }
            }

            self.process_annotations(&last_response.answer);
            self.evict_stale_fragments(user, signals);

            if self.loaded.len() == loaded_before {
                debug!(
                    iterations = i + 1,
                    "refinement converged — no new admissions"
                );
                break;
            }

            debug!(
                new_fragments = self.loaded.len() - loaded_before,
                total_loaded = self.loaded.len(),
                "new context admitted — re-completing"
            );

            let warnings = self.provenance.format_overlap_warnings();
            let enriched_system = if warnings.is_empty() {
                system_prompt.clone()
            } else {
                format!("{}\n\n{}", system_prompt, warnings)
            };

            last_response = self.adapter.complete(CompletionRequest {
                system: enriched_system,
                user: user.to_string(),
                workspace_fragments: self.loaded.clone(),
                workspace_guidance: guidance.to_vec(),
            })?;
        }

        info!(
            model = %self.adapter.model_name(),
            workspace_frags = self.loaded.len(),
            workspace_tokens = self.loaded.iter().map(|f| f.tokens).sum::<usize>(),
            "run_turn complete"
        );
        last_response.answer = strip_markers(&last_response.answer);

        // Record completed turn to session history. Take the SessionFile out of
        // self so the borrow checker allows simultaneous access to other fields.
        if let Some(mut sf) = self.session.take() {
            self.session_turn += 1;
            match sf.write_turn(self.session_turn, user, &last_response.answer) {
                Ok(text) => {
                    let stub_id = StubId(format!("session-turn-{}", self.session_turn));
                    match self.embedder.embed_document(vec![text.as_str()]) {
                        Ok(embeddings) => {
                            if let Some(emb) = embeddings.into_iter().next() {
                                self.vector_index.add(stub_id.clone(), emb);
                                self.session_content.insert(stub_id, text);
                            }
                        }
                        Err(e) => warn!(error = %e, "session turn embedding failed"),
                    }
                }
                Err(e) => warn!(error = %e, "session turn write failed"),
            }
            self.session = Some(sf);
        }

        if let Some(eval) = &mut self.evaluator {
            eval.record_turn(count_tokens_cl100k(&last_response.answer));
        }

        Ok(last_response)
    }

    fn process_thinking_trace(&mut self, output: &str) -> CawResult<()> {
        let steps = extract_thinking_steps(output);
        debug!(steps = steps.len(), "thinking-trace steps extracted");

        for step in steps {
            if step.content.len() < 20 {
                continue;
            }

            let start = Instant::now();
            let embed_result = self.embedder.embed_query(vec![&step.content]);
            let latency_ms = start.elapsed().as_millis() as u64;

            if let Some(monitor) = &mut self.degradation_monitor {
                monitor.record_embedding_call(latency_ms, embed_result.is_ok());
            }

            let embeddings = embed_result?;
            if let Some(embedding) = embeddings.first() {
                let hits = self
                    .vector_index
                    .search(embedding, self.config.max_candidates);
                self.load_fragments(hits)?;
            }
        }

        Ok(())
    }

    /// Run one generation pass with passive mid-stream recall injection.
    ///
    /// Every `passive_injection_interval` tokens the adapter calls back with
    /// the last `passive_injection_window_size` tokens as a sliding window.
    /// We embed it, search for matches above threshold, and return content to
    /// inject directly into the KV cache — no generation restart.
    ///
    /// The `retriever`, `loaded_ids`, and `adapter` fields are disjoint struct
    /// members, so Rust NLL allows the split borrow across the closure and the
    /// method call simultaneously.
    fn run_with_passive_injection(
        &mut self,
        req: CompletionRequest,
    ) -> CawResult<CompletionResponse> {
        let load_threshold = self.config.thresholds.load;
        let max_candidates = self.config.max_candidates;
        let check_interval = self.config.passive_injection_interval;
        let window_size = self.config.passive_injection_window_size;
        let initial_budget = self
            .config
            .max_workspace_tokens
            .saturating_sub(self.loaded.iter().map(|f| f.tokens).sum::<usize>());

        let mut injected: Vec<(StubId, f32)> = Vec::new();
        let mut remaining_budget = initial_budget;

        let response = {
            let retriever = &mut self.retriever;
            let loaded_ids = &self.loaded_ids;

            let mut on_window = |window: &str| -> CawResult<Option<String>> {
                let hits = retriever.search(window, max_candidates)?;
                for hit in &hits {
                    if hit.score < load_threshold {
                        break;
                    }
                    if loaded_ids.contains(&hit.stub.id) {
                        continue;
                    }
                    if injected.iter().any(|(id, _)| *id == hit.stub.id) {
                        continue;
                    }
                    if hit.stub.token_estimate > remaining_budget {
                        continue;
                    }
                    let fragment = retriever.read_range(&hit.stub.id, "full")?;
                    remaining_budget = remaining_budget.saturating_sub(fragment.tokens);
                    injected.push((hit.stub.id.clone(), hit.score));
                    debug!(
                        stub_id = hit.stub.id.0.as_str(),
                        score = hit.score,
                        tokens = fragment.tokens,
                        "passive injection"
                    );
                    return Ok(Some(format!(
                        "\n[recalled from {}]\n{}\n",
                        fragment.locator.source, fragment.content
                    )));
                }
                Ok(None)
            };

            self.adapter
                .generate_passive(req, check_interval, window_size, &mut on_window)?
        };

        // Update the loaded workspace state to reflect what was injected mid-generation.
        for (stub_id, score) in injected {
            self.load_fragments(vec![(stub_id, score)])?;
        }

        Ok(response)
    }

    fn process_probes(&mut self, output: &str) -> CawResult<()> {
        let probes = extract_probes(output);
        debug!(probes = probes.len(), "probes extracted");

        for probe in probes {
            if let Some(monitor) = &mut self.degradation_monitor {
                monitor.record_probe();
                // Stop processing probes if the rate limiter just tripped
                if !monitor.should_auto_recall() {
                    break;
                }
            }

            let loaded_before = self.loaded.len();
            let hits = self
                .retriever
                .search(&probe.content, self.config.max_candidates)?;
            self.load_fragments(
                hits.into_iter()
                    .map(|hit| (hit.stub.id, hit.score))
                    .collect(),
            )?;
            if let Some(eval) = &mut self.evaluator {
                eval.record_probe(&probe.content, self.loaded.len() > loaded_before, 0.0);
            }
        }

        Ok(())
    }

    /// Load specific line ranges explicitly referenced in model output or thinking.
    /// Detects `path/file.ext:start-end` patterns, resolves the stub by matching
    /// the source hint against loaded fragments first, then falls back to search.
    fn process_line_references(&mut self, text: &str) -> CawResult<()> {
        let refs: Vec<LineReference> = extract_line_references(text);
        if refs.is_empty() {
            return Ok(());
        }
        debug!(refs = refs.len(), "line references extracted");

        for line_ref in refs {
            // Prefer stubs already in the workspace; fall back to a targeted search.
            let stub_id = self
                .loaded
                .iter()
                .find(|f| {
                    let src = f.locator.source.trim_start_matches("./");
                    src.ends_with(&line_ref.source_hint)
                        || line_ref.source_hint.ends_with(src)
                })
                .map(|f| f.stub_id.clone());

            let stub_id = match stub_id {
                Some(id) => id,
                None => {
                    let hits = self.retriever.search(&line_ref.source_hint, 5)?;
                    match hits.into_iter().find(|h| {
                        let p = h.stub.path.trim_start_matches("./");
                        p.ends_with(&line_ref.source_hint) || line_ref.source_hint.ends_with(p)
                    }) {
                        Some(h) => h.stub.id,
                        None => {
                            debug!(source = %line_ref.source_hint, "line reference: no matching stub");
                            continue;
                        }
                    }
                }
            };

            if self.loaded_ids.contains(&stub_id) {
                // Full file already loaded; the specific range is already visible.
                debug!(source = %line_ref.source_hint, "line reference: stub already loaded");
                continue;
            }

            let range = if line_ref.start == line_ref.end {
                line_ref.start.to_string()
            } else {
                format!("{}-{}", line_ref.start, line_ref.end)
            };

            if self.loaded.len() >= self.config.max_loaded_fragments {
                debug!("fragment cap reached — line reference skipped");
                break;
            }

            match self.retriever.read_range(&stub_id, &range) {
                Ok(fragment) => {
                    let source = &fragment.locator.source;
                    let current_tokens: usize = self.loaded.iter().map(|f| f.tokens).sum();
                    if current_tokens + fragment.tokens <= self.config.max_workspace_tokens {
                        debug!(
                            source = %line_ref.source_hint,
                            range = %range,
                            tokens = fragment.tokens,
                            "line reference admitted"
                        );
                        self.loaded_ids.insert(stub_id);
                        if source != "session history" {
                            self.loaded_sources.insert(source.clone());
                        }
                        self.provenance.record(fragment.clone());
                        self.loaded.push(fragment);
                    }
                }
                Err(e) => debug!(error = %e, source = %line_ref.source_hint, "line reference read failed"),
            }
        }

        Ok(())
    }

    /// Extract model annotations (<note id="...">...</note>) and record them
    /// as mid-session consolidation notes.
    fn process_annotations(&mut self, output: &str) {
        let annotations = extract_annotations(output);
        for ann in annotations {
            if let Some(eval) = &mut self.evaluator {
                eval.record_annotation(&ann.stub_id, &ann.content);
            }
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
                // Only restore a decayed score when the fragment is strongly
                // re-engaged — not on generic topic overlap that appears in
                // every fragment of a domain-specific codebase. Without the
                // margin, decay can never push a score below the unload
                // threshold because even tangential overlap tops it up each turn.
                if overlap > *current + 0.2 {
                    *current = overlap;
                }
            }
        }
    }

    /// Evict fragments whose decayed relevance has dropped below the
    /// unload threshold (hysteresis). Also enforces the token budget
    /// as a hard ceiling — if the workspace is over budget, evict the
    /// lowest-scoring fragments until it fits.
    fn evict_stale_fragments(&mut self, query: &str, signals: Option<&AugmentationSignals>) {
        let unload_threshold = self.config.thresholds.unload;
        let budget = self.config.max_workspace_tokens;
        let wants_latest = signals.map_or(false, |s| s.wants_latest_run_only);

        // Collect (index, relevance_score, mtime) sorted eviction-first.
        // Primary sort: relevance ascending. Secondary sort (when wants_latest_run_only):
        // mtime ascending so older fragments are evicted before newer ones at equal relevance.
        let mut scored: Vec<(usize, f32, u64)> = self
            .loaded
            .iter()
            .enumerate()
            .map(|(idx, frag)| {
                let score = self
                    .relevance_scores
                    .get(&frag.stub_id)
                    .copied()
                    .unwrap_or(0.0);
                (idx, score, frag.mtime_unix_secs)
            })
            .collect();
        if wants_latest {
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
        } else {
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        let mut to_evict = Vec::new();
        let mut tokens_after_eviction: usize = self.loaded.iter().map(|f| f.tokens).sum();

        for (idx, score, _mtime) in scored {
            let below_threshold = score < unload_threshold;
            let over_budget = tokens_after_eviction > budget;

            if !below_threshold && !over_budget {
                break;
            }

            to_evict.push(idx);
            tokens_after_eviction -= self.loaded[idx].tokens;
        }

        // Remove in reverse index order to preserve indices
        to_evict.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_evict {
            let fragment = self.loaded.remove(idx);
            self.loaded_ids.remove(&fragment.stub_id);
            if fragment.locator.source != "session history" {
                self.loaded_sources.remove(&fragment.locator.source);
            }
            debug!(path = %fragment.locator.source, "fragment evicted");

            let decayed_score = self
                .relevance_scores
                .remove(&fragment.stub_id)
                .unwrap_or(0.0);

            if let Some(eval) = &mut self.evaluator {
                eval.record_eviction(&fragment.stub_id, self.session_turn);
            }

            let existing_annotations = self.provenance.consolidation_notes_for(&fragment.stub_id);

            let content = self
                .consolidation_synthesizer
                .synthesize_eviction_note(
                    &fragment.content,
                    query,
                    decayed_score,
                    &fragment.locator.source,
                    &existing_annotations,
                )
                .unwrap_or_else(|_| {
                    // Degrade to mechanical on LLM failure
                    format!(
                        "Evicted (relevance decayed to {:.2}) during query about '{}'. Source: {}",
                        decayed_score,
                        truncate_str(query, 100),
                        fragment.locator.source,
                    )
                });

            let note = ConsolidationNote {
                content,
                source: ConsolidationSource::Eviction,
                created_at_secs: current_timestamp(),
            };

            self.persist_consolidation(&fragment.stub_id, &note);
        }
    }

    fn load_fragments(&mut self, hits: Vec<(StubId, f32)>) -> CawResult<()> {
        for (stub_id, score) in hits {
            if self.loaded_ids.contains(&stub_id) {
                continue;
            }

            if self.loaded.len() >= self.config.max_loaded_fragments {
                debug!(
                    count = self.loaded.len(),
                    max = self.config.max_loaded_fragments,
                    "fragment cap reached — stopping"
                );
                break;
            }

            if score < self.config.thresholds.load {
                trace!(
                    score,
                    threshold = self.config.thresholds.load,
                    "candidate below threshold — skipped"
                );
                continue;
            }

            let current_tokens: usize = self.loaded.iter().map(|f| f.tokens).sum();
            if current_tokens >= self.config.max_workspace_tokens {
                debug!(
                    current_tokens,
                    max = self.config.max_workspace_tokens,
                    "workspace budget full — stopping"
                );
                break;
            }

            let fragment = match self.session_content.get(&stub_id) {
                Some(content) => {
                    let tokens = count_tokens_cl100k(content);
                    RecallFragment {
                        stub_id: stub_id.clone(),
                        content: content.clone(),
                        locator: Locator {
                            source: "session history".to_string(),
                            locator: "full".to_string(),
                        },
                        tokens,
                        mtime_unix_secs: 0,
                    }
                }
                None => match self.retriever.read_range(&stub_id, "full") {
                    Ok(mut f) => {
                        // If there are persisted consolidation notes for this stub
                        // (written on prior evictions), prepend them so the model
                        // can see what was previously learned — not just the raw content.
                        if let Some(store) = &self.store {
                            if let Ok(notes) = store.load_consolidation(&stub_id) {
                                if !notes.is_empty() {
                                    let notes_block = notes
                                        .iter()
                                        .map(|n| format!("- {}", n.content))
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    f.content = format!(
                                        "[Prior session notes for this source:\n{}\n]\n\n{}",
                                        notes_block, f.content
                                    );
                                    f.tokens = count_tokens_cl100k(&f.content);
                                }
                            }
                        }
                        f
                    }
                    // The source file changed since the index was built. Skip this
                    // candidate — treating it as a miss is correct and matches the
                    // documented intent of StaleStub.
                    Err(CawError::StaleStub { path }) => {
                        debug!(%path, "stale stub skipped during load_fragments");
                        continue;
                    }
                    Err(e) => return Err(e),
                },
            };

            // Skip a second chunk from a source already in the workspace.
            // "session history" is exempt — multiple turns are distinct content.
            let source = &fragment.locator.source;
            if source != "session history" && self.loaded_sources.contains(source) {
                debug!(source = %source, "source already loaded — skipping duplicate chunk");
                continue;
            }

            if current_tokens + fragment.tokens <= self.config.max_workspace_tokens {
                let fragment_tokens = fragment.tokens;
                debug!(
                    path = %fragment.locator.source,
                    score,
                    tokens = fragment_tokens,
                    "fragment admitted"
                );
                self.relevance_scores.insert(stub_id.clone(), score);
                self.loaded_ids.insert(stub_id.clone());
                if source != "session history" {
                    self.loaded_sources.insert(source.clone());
                }
                self.provenance.record_with_context(fragment.clone(), "", 0);
                self.loaded.push(fragment);
                if let Some(eval) = &mut self.evaluator {
                    eval.record_recall(&stub_id, score, fragment_tokens);
                }
            } else {
                debug!(
                    path = %fragment.locator.source,
                    fragment_tokens = fragment.tokens,
                    current_tokens,
                    max = self.config.max_workspace_tokens,
                    "fragment would exceed budget — skipped"
                );
            }
        }

        Ok(())
    }

    pub fn read_range(&self, stub_id: &StubId, range: &Range) -> CawResult<RecallFragment> {
        let full_fragment = self.retriever.read_range(stub_id, "full")?;
        let range_content = range.apply(&full_fragment.content);
        let token_estimate = count_tokens_cl100k(&range_content);

        Ok(RecallFragment {
            stub_id: stub_id.clone(),
            content: range_content,
            locator: caw_core::Locator {
                source: full_fragment.locator.source,
                locator: range.to_locator_string(),
            },
            tokens: token_estimate,
            mtime_unix_secs: full_fragment.mtime_unix_secs,
        })
    }
}

impl<R, E, V, P: Default, M, S> DynamicRecallOrchestrator<R, E, V, P, M, S> {
    /// Reset per-turn state so the orchestrator can be reused across independent
    /// queries. Clears loaded fragments, relevance scores, and provenance.
    pub fn reset_session(&mut self) {
        self.loaded.clear();
        self.loaded_ids.clear();
        self.loaded_sources.clear();
        self.relevance_scores.clear();
        self.provenance = P::default();
    }
}

fn term_overlap_score(context: &str, content: &str) -> f32 {
    let ctx_terms: HashSet<String> = tokenize_terms(context).into_iter().collect();
    let doc_terms: HashSet<String> = tokenize_terms(content).into_iter().collect();

    if ctx_terms.is_empty() || doc_terms.is_empty() {
        return 0.0;
    }

    let intersection = ctx_terms.intersection(&doc_terms).count() as f32;
    intersection / ctx_terms.len().min(doc_terms.len()) as f32
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
