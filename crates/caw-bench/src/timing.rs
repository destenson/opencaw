//! Phase-timing instrumentation for the bench.
//!
//! `caw-bench` reports a single `latency_ms` per item but never attributed it
//! to a phase, so "the eval loop is slow (~20s/item)" had no breakdown to act
//! on. `TimingAdapter` is a transparent `ModelAdapter` decorator that records
//! how much wall time each model call costs and how many calls happen, into a
//! shared atomic counter. Wrapping the *answer* adapter and the *judge* adapter
//! separately lets the runner split each item into three buckets:
//!
//!   gen_ms   — answer-model generation (the multi-pass completions)
//!   judge_ms — the judge model (one call/item, often a CLI spawn)
//!   other    — latency_ms − gen_ms − judge_ms (embed + retrieve + orchestration)
//!
//! It lives in the bench, not the library, so the orchestrator's hot path is
//! untouched. The decorator only measures `complete`/`thinking_with_steps`/
//! `generate_passive` — the three trait methods that actually invoke a model —
//! and delegates everything else unchanged.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use caw_core::{
    CawResult, CompletionRequest, CompletionResponse, ModelAdapter, ModelCapabilities,
    ProvenanceFormat,
};

/// Shared, thread-safe accumulator for one logical model role (answer or
/// judge). Cloned `Arc` handles let the runner read the totals after the
/// wrapped adapter has been moved into the orchestrator.
#[derive(Debug, Default)]
pub struct PhaseCounters {
    calls: AtomicU64,
    nanos: AtomicU64,
}

impl PhaseCounters {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn record(&self, elapsed_nanos: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(elapsed_nanos, Ordering::Relaxed);
    }

    /// `(calls, total_milliseconds)` accumulated so far. For the judge, which
    /// reuses one counter across all items, snapshot before and after each
    /// item and subtract to get the per-item slice.
    pub fn snapshot(&self) -> (u64, u64) {
        let calls = self.calls.load(Ordering::Relaxed);
        let ms = self.nanos.load(Ordering::Relaxed) / 1_000_000;
        (calls, ms)
    }
}

/// Wraps a `ModelAdapter`, timing each generation call into `counters`.
pub struct TimingAdapter {
    inner: Box<dyn ModelAdapter>,
    counters: Arc<PhaseCounters>,
}

impl TimingAdapter {
    pub fn new(inner: Box<dyn ModelAdapter>, counters: Arc<PhaseCounters>) -> Self {
        Self { inner, counters }
    }
}

impl ModelAdapter for TimingAdapter {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.inner.capabilities()
    }

    fn provenance_format(&self) -> ProvenanceFormat {
        self.inner.provenance_format()
    }

    fn complete(&self, req: CompletionRequest) -> CawResult<CompletionResponse> {
        let start = Instant::now();
        let result = self.inner.complete(req);
        self.counters.record(start.elapsed().as_nanos() as u64);
        result
    }

    fn thinking_with_steps(
        &self,
        req: CompletionRequest,
        on_step: &mut dyn FnMut(&str) -> CawResult<bool>,
    ) -> CawResult<CompletionResponse> {
        let start = Instant::now();
        let result = self.inner.thinking_with_steps(req, on_step);
        self.counters.record(start.elapsed().as_nanos() as u64);
        result
    }

    fn generate_passive(
        &self,
        req: CompletionRequest,
        check_interval: usize,
        window_size: usize,
        on_window: &mut dyn FnMut(&str) -> CawResult<Option<String>>,
    ) -> CawResult<CompletionResponse> {
        let start = Instant::now();
        let result = self
            .inner
            .generate_passive(req, check_interval, window_size, on_window);
        self.counters.record(start.elapsed().as_nanos() as u64);
        result
    }
}
