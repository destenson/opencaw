//! Deterministic, mechanism-level replay harness for the recall orchestrator.
//!
//! Unlike the end-to-end answer benchmark (where both the answer model and the
//! judge are nondeterministic), this harness drives the orchestrator with a
//! scripted adapter (`ReplayAdapter`) and a deterministic hash embedder. The
//! multi-pass recall trajectory is therefore fixed by construction, so the
//! orchestrator's actual load/evict decisions can be asserted exactly — no GPU,
//! no live LLM, no judge.
//!
//! The canonical proof here is trace-driven recall: a corpus stub the initial
//! user query does NOT retrieve must still be admitted when the model's scripted
//! reasoning trace mentions that stub's vocabulary. This is the OpenCAW
//! differentiator (the thinking trace is the retrieval signal), and the part
//! that is otherwise impossible to verify from noisy end-to-end scores.

use std::sync::atomic::{AtomicUsize, Ordering};

use caw_adapters::{ReplayAdapter, ScriptedResponse};
use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::{CawResult, ContentKind, EmbeddingProvider, RecallThresholds};
use caw_index::{HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};

/// Deterministic embedder: hashes tokens into fixed-dimensional buckets and L2
/// normalizes. For disjoint vocabularies it separates documents cleanly, so
/// retrieval is unambiguous and the test is offline. A wide dimension keeps
/// bucket collisions between the disjoint fixture vocabularies near zero.
struct HashEmbedder {
    dim: usize,
    call_count: AtomicUsize,
}

impl HashEmbedder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            call_count: AtomicUsize::new(0),
        }
    }
}

impl EmbeddingProvider for HashEmbedder {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(texts.into_iter().map(|t| hash_to_unit_vector(t, self.dim)).collect())
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn provider_name(&self) -> &str {
        "hash-embedder-test"
    }
}

fn hash_to_unit_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let idx = (fnv1a(word) as usize) % dim;
        v[idx] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::EPSILON);
    v.iter().map(|x| x / norm).collect()
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Two docs with fully disjoint vocabularies. The auth doc is the initial-query
/// target; the telemetry doc shares no words with the auth query and is only
/// reachable via the scripted reasoning trace.
const AUTH_PATH: &str = "docs/auth.md";
const TELEMETRY_PATH: &str = "docs/telemetry.md";

fn fixture_docs() -> Vec<SourceDocument> {
    vec![
        SourceDocument {
            path: AUTH_PATH.to_string(),
            content: "# Authentication Middleware\n\n\
                This module validates session tokens against a Redis backing store. \
                SessionValidator checks expiry and refreshes tokens near their TTL. \
                Failed validations emit structured audit events to the access log."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 1_700_000_000,
        },
        SourceDocument {
            path: TELEMETRY_PATH.to_string(),
            content: "# Telemetry Pipeline\n\n\
                Collects spans from instrumented services and forwards to the aggregator. \
                Sampling strategy is head-based at 1% for healthy traffic, 100% for errors. \
                Retention is 30 days for raw spans and 365 days for aggregates."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 1_700_000_000,
        },
    ]
}

/// Write fixture docs under a temp corpus root (the store reads body text back
/// from disk via recorded byte ranges) and build a retriever over them.
fn build_retriever(
    dim: usize,
) -> (
    tempfile::TempDir,
    SemanticRetriever<HashEmbedder, SqliteStubStore, HnswVectorIndex>,
) {
    let corpus_root = tempfile::tempdir().expect("tempdir");
    let mut docs = fixture_docs();
    for doc in &mut docs {
        let full = corpus_root.path().join(&doc.path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).expect("create fixture parent dir");
        }
        std::fs::write(&full, &doc.content).expect("write fixture doc");
        doc.mtime_unix_secs = std::fs::metadata(&full)
            .expect("fixture metadata")
            .modified()
            .expect("fixture modified time")
            .duration_since(std::time::UNIX_EPOCH)
            .expect("fixture mtime before epoch")
            .as_secs();
    }

    let store = SqliteStubStore::in_memory(dim)
        .expect("sqlite in-memory store")
        .with_corpus_root(corpus_root.path().to_path_buf());
    let mut retriever = SemanticRetriever::new(HashEmbedder::new(dim), store, HnswVectorIndex::new());

    let pipeline = IngestionPipeline::new();
    for doc in docs {
        for (stub, embed_text) in pipeline.ingest(doc) {
            retriever.insert(stub, &embed_text).expect("insert into retriever");
        }
    }
    (corpus_root, retriever)
}

const USER_QUERY: &str =
    "how does session token validation work in the authentication middleware";

/// Run one turn and return the sorted set of source paths left in the loaded
/// workspace. `trace_recall` toggles only the thinking-trace mechanism; the
/// scripted reasoning trace and corpus are otherwise identical, so the
/// difference between the two runs isolates exactly what trace recall admits.
fn loaded_sources_after_turn(trace_recall: bool) -> Vec<String> {
    let dim = 256;
    let (_corpus_root, retriever) = build_retriever(dim);

    let config = DynamicRecallConfig {
        max_candidates: 4,
        // The hash embedder is not perfectly orthogonal across the fixture
        // vocabularies: on the auth query the telemetry doc still scores 0.1456
        // (auth scores 0.1949). The load threshold sits between them so the
        // initial query admits ONLY auth, leaving the telemetry doc to be
        // surfaced solely by the reasoning trace (which scores it 0.6605).
        thresholds: RecallThresholds {
            load: 0.18,
            unload: 0.1,
        },
        max_workspace_tokens: 8_000,
        max_recall_iterations: 1,
        enable_thinking_trace_recall: trace_recall,
        enable_probe_recall: false,
        enable_line_reference_recall: false,
        ..Default::default()
    };

    // The reasoning trace names telemetry concepts (disjoint from the auth
    // query) but no file path — so only trace recall, not file expansion or the
    // mentioned-files path, can surface the telemetry doc.
    let script = vec![ScriptedResponse::new(
        "To answer this I should consider how the telemetry pipeline collects spans \
         from instrumented services and forwards them to the aggregator with head-based sampling.",
        "Session tokens are validated against the Redis backing store.",
    )];
    let adapter = ReplayAdapter::new("replay", script);

    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
        DynamicRecallOrchestrator::new(
            retriever,
            HashEmbedder::new(dim),
            HnswVectorIndex::new(), // session index — intentionally empty of corpus
            InMemoryProvenanceStore::default(),
            adapter,
            config,
        );

    orchestrator.run_turn("", USER_QUERY, &[], None).expect("run_turn");

    let mut sources: Vec<String> = orchestrator
        .loaded
        .iter()
        .map(|f| f.locator.source.clone())
        .collect();
    sources.sort();
    sources.dedup();
    sources
}

/// Trace-driven recall is the core mechanism. The initial query retrieves the
/// auth doc only; the telemetry doc shares no real vocabulary with it. The
/// scripted reasoning trace then mentions telemetry vocabulary — and that alone
/// must pull the telemetry stub into the loaded workspace.
///
/// The orchestrator's own `vector_index` is left EMPTY: the corpus lives in the
/// retriever, exactly as the bench wires it, so this asserts the true contract
/// — trace recall queries the corpus, not a separately pre-populated index.
/// (Before the fix, `process_thinking_trace` searched only the empty
/// `vector_index`, so this stub was never admitted and the differentiator
/// mechanism was inert in the bench.)
///
/// The test asserts the *transition*, not just the final set: with trace recall
/// OFF the telemetry doc must be absent (proving the initial query alone does
/// not admit it), and with it ON the telemetry doc must appear. Asserting the
/// final set alone would silently pass if a future embedding change pushed the
/// telemetry-on-initial score back above threshold — the exact false positive
/// this fixture hit during development.
#[test]
fn thinking_trace_admits_stub_the_initial_query_missed() {
    let without_trace = loaded_sources_after_turn(false);
    let with_trace = loaded_sources_after_turn(true);

    assert!(
        without_trace.iter().any(|s| s == AUTH_PATH),
        "initial query should admit the auth doc regardless of trace recall; loaded = {without_trace:?}",
    );
    assert!(
        !without_trace.iter().any(|s| s == TELEMETRY_PATH),
        "without trace recall the telemetry doc must NOT be admitted — if it is, the \
         initial query already scores it above threshold and the test no longer isolates \
         the mechanism; loaded = {without_trace:?}",
    );
    assert!(
        with_trace.iter().any(|s| s == TELEMETRY_PATH),
        "the scripted reasoning trace mentions telemetry vocabulary, so trace-driven \
         recall must admit the telemetry doc the initial query missed; loaded = {with_trace:?}",
    );
}
