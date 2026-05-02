//! End-to-end integration smoke test for the recall orchestrator.
//!
//! Wires the real ingestion pipeline, real semantic retriever, real vector
//! index, and real provenance store against a deterministic embedder (so the
//! test is fast and offline) and MockAdapter (so no network or API keys are
//! required). Exercises the full admission path: ingest -> embed -> index ->
//! retrieve -> schedule -> complete -> provenance tagging.
//!
//! This is the contract: if this test breaks, the orchestrator API has
//! shifted in a way that affects every downstream consumer.

use std::sync::atomic::{AtomicUsize, Ordering};

use caw_adapters::MockAdapter;
use caw_core::provenance::InMemoryProvenanceStore;
use caw_core::{CawResult, ContentKind, EmbeddingProvider, RecallThresholds};
use caw_index::{HnswVectorIndex, SemanticRetriever, SqliteStubStore};
use caw_ingest::{IngestionPipeline, SourceDocument};
use caw_orchestrator::dynamic::{DynamicRecallConfig, DynamicRecallOrchestrator};

/// Deterministic embedder that hashes tokens into fixed-dimensional buckets.
/// Not semantically meaningful in general, but for the disjoint vocabularies
/// used in this test it separates documents cleanly enough to verify the
/// retrieval -> admission -> provenance path.
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

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl EmbeddingProvider for HashEmbedder {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(texts
            .into_iter()
            .map(|t| hash_to_unit_vector(t, self.dim))
            .collect())
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
        let h = fnv1a(word);
        let idx = (h as usize) % dim;
        v[idx] += 1.0;
    }
    // L2 normalize so HNSW cosine is bounded in [-1, 1]
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
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

/// Build a workspace of three docs with disjoint vocabularies so the hash
/// embedder's bucket collisions can't confuse one for another.
fn fixture_docs() -> Vec<SourceDocument> {
    vec![
        SourceDocument {
            path: "docs/auth.md".to_string(),
            content: "# Authentication Middleware\n\n\
                This module validates session tokens against a Redis backing store. \
                SessionValidator checks expiry and refreshes tokens near their TTL. \
                Failed validations emit structured audit events to the access log."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 1_700_000_000,
        },
        SourceDocument {
            path: "docs/billing.md".to_string(),
            content: "# Billing Reconciliation\n\n\
                Nightly job reconciles provider invoices against customer ledger entries. \
                Discrepancies flag for manual review via InvoiceAuditor. Currency \
                conversion uses the rate captured at invoice issuance, not at reconcile time."
                .to_string(),
            kind: ContentKind::Markdown,
            mtime_unix_secs: 1_700_000_000,
        },
        SourceDocument {
            path: "docs/telemetry.md".to_string(),
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

#[test]
fn recall_loop_admits_fragment_and_tags_provenance() {
    // The store now reads body text back from disk using (path, byte_offset,
    // byte_length) recorded on each stub, so the fixture docs need to exist
    // as real files under a corpus root for the recall path to resolve them.
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

    let dim = 64;
    let embedder = HashEmbedder::new(dim);
    let store = SqliteStubStore::in_memory(dim)
        .expect("sqlite in-memory store")
        .with_corpus_root(corpus_root.path().to_path_buf());
    let index = HnswVectorIndex::new();
    let mut retriever = SemanticRetriever::new(embedder, store, index);

    // Ingest and index the fixture docs.
    // IngestionPipeline::new() now defaults to cl100k tiktoken, so token
    // estimates reflect a real BPE — exercising the wired-up default.
    let pipeline = IngestionPipeline::new();
    for doc in docs {
        for (stub, _embed_text) in pipeline.ingest(doc) {
            retriever.insert(stub).expect("insert into retriever");
        }
    }

    // Permissive thresholds: the hash embedder's cosine similarities between
    // short queries and full docs land in the 0.1-0.4 range. This test is about
    // the admission machinery, not threshold calibration — the default
    // hysteresis (load=0.7) is tuned for real embedding models.
    let config = DynamicRecallConfig {
        max_candidates: 4,
        thresholds: RecallThresholds {
            load: 0.05,
            unload: 0.02,
        },
        max_workspace_tokens: 4_000,
        max_recall_iterations: 1,
        relevance_decay_rate: 0.8,
        enable_thinking_trace_recall: false,
        enable_probe_recall: false,
        ..Default::default()
    };

    // Deterministic second embedder for the orchestrator's recall path
    // (thinking-trace recall, disabled here — but the generic param still
    // needs a concrete embedder and the vector_index still needs to be a
    // real VectorIndex for probe matching).
    let orchestrator_embedder = HashEmbedder::new(dim);
    let orchestrator_index = HnswVectorIndex::new();
    let provenance = InMemoryProvenanceStore::default();
    let adapter = MockAdapter::new("test-model", false);

    let mut orchestrator: DynamicRecallOrchestrator<_, _, _, _, _, SqliteStubStore> =
        DynamicRecallOrchestrator::new(
            retriever,
            orchestrator_embedder,
            orchestrator_index,
            provenance,
            adapter,
            config,
        );

    // A query with vocabulary aligned to the auth doc. With disjoint vocabs
    // and bucket hashing, the cosine match should prefer auth over others.
    let user_query = "how does session token validation work in the authentication middleware";
    let response = orchestrator.run_turn("", user_query).expect("run_turn");

    // MockAdapter echoes the workspace via format_workspace — so if any
    // fragment was admitted, its provenance locator appears in the answer.
    // The bracketed format emits `[recalled from path:locator]`.
    assert!(
        response.answer.contains("recalled from"),
        "answer should contain provenance locator from admitted fragment; got: {}",
        response.answer,
    );
    assert!(
        response.answer.contains("docs/auth.md"),
        "auth doc should be the top match for an auth-vocabulary query; got: {}",
        response.answer,
    );

    // Orchestrator-side embedder only runs when thinking-trace or probe
    // recall fires. Both are disabled here, so the second embedder should
    // not have been called — proves the recall path respects the config flags.
    assert_eq!(
        orchestrator.embedder.calls(),
        0,
        "recall-path embedder must not run with thinking-trace + probe recall disabled",
    );

    // The retriever's own embedder ran at least once: one call for ingestion
    // (batched or per-doc) and one for the user query.
    assert!(
        orchestrator
            .loaded
            .iter()
            .any(|f| f.locator.source == "docs/auth.md"),
        "loaded workspace should contain the auth fragment",
    );
}
