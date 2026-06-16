//! OpenAI-compatible proxy that augments the last user message with
//! retrieved workspace context before forwarding to an upstream model
//! server. This is the thin deployment target from the design doc: no
//! orchestrator, no probes, no multi-pass — just retrieve → inject →
//! forward. The sweep evidence says most of the measured opencaw gain
//! comes from multi-pass + reasoning, but v0 proves the UX: zero tool
//! calls, automatic context.
//!
//! Alongside the `/v1/chat/completions` proxy route there is one
//! non-OpenAI route, `/v1/retrieve`, that runs the identical retrieval
//! and returns the ranked candidates as JSON — scores, paths, token
//! cost, and whether each survived the token-budget clamp — without
//! forwarding to any model. It exists so the operator can inspect what
//! retrieval actually produced for a query (was a relevant stub ranked
//! out, or clamped out?) instead of inferring it from a model answer.
//! This is a deliberate deviation from a strictly OpenAI-only surface,
//! confined to a read-only diagnostic that never mutates request flow.
//!
//! A second proxy route, `/v1/messages`, accepts the Anthropic Messages
//! API shape so the official `anthropic` SDK (and Anthropic-protocol
//! clients generally) can sit in front of the server. It is the same
//! single-shot retrieve → inject → forward path: the only protocol
//! difference is request-side (user content is often a block array, not
//! a plain string) and the upstream path. The response is streamed back
//! verbatim, so this route requires the upstream to itself speak the
//! Anthropic protocol (e.g. `https://api.anthropic.com`). It is *not* a
//! cross-protocol translator: an Anthropic client in front of an
//! OpenAI-only upstream is out of scope (see `docs/scope.md`).

use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use caw_core::{
    count_tokens_cl100k, CompletionRequest, EmbeddingProvider, Locator, ProvenanceFormat,
    RecallFragment, StubId, StubStore, VectorIndex,
};
use caw_index::{BM25Index, CandleEmbeddingProvider, FlatVectorIndex, HnswVectorIndex, SqliteStubStore};

/// Which in-memory vector index backs the server's retrieval path.
/// Chosen once at startup via `--retriever` and fixed for the life of
/// the process. The proxy route (`/v1/chat/completions`) stays a pure
/// OpenAI passthrough; the only non-OpenAI surface is the read-only
/// `/v1/retrieve` diagnostic, which reports the same retrieval as JSON.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum RetrieverKind {
    /// Brute-force cosine over all embeddings. Zero warmup cost, O(n·d)
    /// per query. Correct choice for corpora under ~1M stubs.
    Flat,
    /// HNSW graph via instant-distance. Needs an eager build at startup
    /// to avoid a first-query latency spike; use when the flat scan is
    /// too slow.
    Hnsw,
    /// Flat cosine fused with a BM25 lexical index over `path + summary +
    /// body`. Fixes the failure mode where a definitional chunk's one
    /// relevant sentence is mean-pooled into the noise of a same-domain
    /// corpus: the lexical half catches exact symbol/term matches the
    /// embedding washes out. Pays a startup cost to read every body once
    /// and build the posting lists.
    Hybrid,
}

/// Reciprocal weighting for hybrid fusion, mirroring
/// `caw_index::HybridRetriever::balanced`. Semantic and keyword scores
/// are each min-max normalized within their own candidate set, then
/// summed with these weights. Tuned by that retriever's defaults, not by
/// this corpus.
const HYBRID_SEMANTIC_WEIGHT: f32 = 0.6;
const HYBRID_KEYWORD_WEIGHT: f32 = 0.4;

/// Number of indexed stub bodies to read at startup to verify `--corpus-root`
/// actually resolves to the directory the index was built against. Small
/// because one missing body is enough signal and the Hybrid path re-reads all
/// bodies for BM25 anyway.
const CORPUS_ROOT_PROBE_SAMPLE: usize = 8;
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Everything the request handler needs. Index, store, embedder, and
/// upstream config live here for the life of the process.
///
/// The three core bits each sit behind their own Mutex:
/// - `embedder` because candle's forward pass takes `&mut self` and GPU
///   context isn't Send across threads without serialization anyway;
/// - `store` because `rusqlite::Connection` is `Send` but not `Sync`;
/// - `index` because `VectorIndex::search` takes `&mut self` (instant-
///   distance maintains per-search scratch state).
///
/// Held only for the synchronous duration of a retrieval call, never
/// across `.await`, so there's no contention with request concurrency
/// beyond the natural single-threading these components require.
pub struct AppState {
    pub embedder: Mutex<CandleEmbeddingProvider>,
    pub store: Mutex<SqliteStubStore>,
    /// Boxed as a trait object so the retriever choice is a runtime flag
    /// without leaking a generic parameter all the way through the
    /// handlers and router state.
    pub index: Mutex<Box<dyn VectorIndex + Send>>,
    /// Present only under `RetrieverKind::Hybrid`: a BM25 lexical index
    /// over every stub's `path + summary + body`. When set, retrieval
    /// fuses its scores with the cosine candidates. `None` leaves the
    /// pure-cosine path untouched.
    pub bm25: Option<Mutex<BM25Index>>,
    /// Which retriever was selected at startup. Stored only so the
    /// `/v1/retrieve` diagnostic route can report it back; the retrieval
    /// path itself dispatches on the presence of `bm25` and the concrete
    /// `index` type, not on this field.
    pub retriever: RetrieverKind,
    /// Base URL of the upstream model server. The handler appends a
    /// route-specific suffix, so the convention differs by route:
    /// - `/v1/chat/completions` appends `/chat/completions`, so set this
    ///   to the OpenAI-style root including `/v1`, e.g.
    ///   `http://localhost:11434/v1` for Ollama.
    /// - `/v1/messages` appends `/v1/messages`, so set this to the
    ///   provider root *without* `/v1`, e.g. `https://api.anthropic.com`.
    ///
    /// A given proxy instance fronts one upstream, so only the matching
    /// route is exercised per deployment.
    pub upstream_base: String,
    /// Candidate pool size for ANN search; the load threshold controls actual admissions.
    pub max_candidates: usize,
    /// Hard cap on total tokens of recalled content injected into the
    /// user message (rough token ≈ whitespace-split count).
    pub max_workspace_tokens: usize,
    /// Shared HTTP client with connection pooling.
    pub http: reqwest::Client,
}

/// Build the full application: shared state + the single augmenting
/// route. Kept public so integration tests can mount it against a mock
/// upstream without touching real HTTP.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/retrieve", post(retrieve))
        .with_state(state)
}

/// Load the retrieval stack that `AppState` wraps. Separate function so
/// callers can wire up a process before picking an HTTP port.
pub fn build_state(
    index_path: &str,
    corpus_root: std::path::PathBuf,
    upstream_base: String,
    max_candidates: usize,
    max_workspace_tokens: usize,
    retriever: RetrieverKind,
) -> Result<AppState> {
    let embedder = CandleEmbeddingProvider::bge_small()
        .map_err(|e| anyhow::anyhow!("init bge-small (candle): {e}"))?;
    let dim = embedder.dimension();

    let corpus_root_display = corpus_root.display().to_string();
    // Read-only: the proxy serves a prebuilt index and has no reingest worker,
    // so get_content must not mark rows stale on a missing/misconfigured path —
    // that would persist into the shared index and degrade every later run.
    let store = SqliteStubStore::new(index_path, dim)
        .with_context(|| format!("open prebuilt index {index_path}"))?
        .with_corpus_root(corpus_root)
        .with_read_only(true);

    let all = store
        .all_embeddings()
        .context("read all embeddings from prebuilt index")?;
    anyhow::ensure!(
        !all.is_empty(),
        "prebuilt index {} is empty — rebuild with caw-bench-build-index",
        index_path
    );

    // A wrong --corpus-root joins indexed (relative) stub paths against the
    // wrong directory, so every body read misses and the proxy forwards
    // unaugmented with a normal HTTP 200 — a working-looking server that
    // silently provides no context. Detect that misconfiguration loudly here,
    // before the (expensive) BM25 body pass, rather than letting it surface
    // only as per-request debug noise. We read the stub path before calling
    // get_content because a missing body marks the row stale, after which
    // get_stub itself would fail and we'd lose the example path.
    let sample_n = CORPUS_ROOT_PROBE_SAMPLE.min(all.len());
    let mut readable = 0usize;
    let mut first_missing: Option<String> = None;
    for (stub_id, _emb) in all.iter().take(sample_n) {
        let path = store.get_stub(stub_id).ok().map(|s| s.path);
        match store.get_content(stub_id) {
            Ok(_) => readable += 1,
            Err(_) => {
                if first_missing.is_none() {
                    first_missing = path;
                }
            }
        }
    }
    if sample_n > 0 && readable == 0 {
        error!(
            "corpus-root probe: 0/{sample_n} sampled stub bodies are readable under corpus-root '{}' (e.g. '{}'). \
             The index was almost certainly built against a different directory — every request will forward UNAUGMENTED. \
             Re-run with the same directory passed to build-index.",
            corpus_root_display,
            first_missing.as_deref().unwrap_or("<unknown>"),
        );
    } else if readable < sample_n {
        warn!(
            "corpus-root probe: only {readable}/{sample_n} sampled stub bodies are readable under corpus-root '{}' — some content fetches will miss",
            corpus_root_display,
        );
    }

    // The vector half is Flat for both Flat and Hybrid; Hnsw only for Hnsw.
    let index: Box<dyn VectorIndex + Send> = match retriever {
        RetrieverKind::Flat | RetrieverKind::Hybrid => {
            Box::new(FlatVectorIndex::from_points(all.clone()))
        }
        RetrieverKind::Hnsw => {
            let mut hnsw = HnswVectorIndex::new();
            for (stub_id, embedding) in &all {
                hnsw.add(stub_id.clone(), embedding.clone());
            }
            // Trigger the graph build here rather than paying the cost
            // on the first request. instant-distance defers construction
            // until .search() is called, which pegs all cores for ~2 min
            // on 158k points — unacceptable for the live-query path.
            let _ = hnsw.search(&vec![0.0_f32; dim], 1);
            Box::new(hnsw)
        }
    };

    // Hybrid pays a one-time cost to read every body and build BM25 posting
    // lists, indexed on the same `path + summary + body` text that
    // `HybridRetriever::insert` uses so lexical scoring matches the library
    // retriever. Bodies are read transiently and not retained here.
    let bm25 = match retriever {
        RetrieverKind::Hybrid => {
            let mut bm25 = BM25Index::new();
            let mut missing = 0usize;
            for (stub_id, _emb) in &all {
                let stub = match store.get_stub(stub_id) {
                    Ok(s) => s,
                    Err(_) => {
                        missing += 1;
                        continue;
                    }
                };
                let body = match store.get_content(stub_id) {
                    Ok(c) => c,
                    Err(_) => {
                        missing += 1;
                        continue;
                    }
                };
                let text = format!("{} {} {}", stub.path, stub.summary, body);
                bm25.add(stub_id.clone(), &text);
            }
            if missing > 0 {
                warn!("BM25 build: {missing} stubs had no readable stub/body and were skipped");
            }
            info!("BM25 lexical index built: {} docs", bm25.len());
            Some(Mutex::new(bm25))
        }
        RetrieverKind::Flat | RetrieverKind::Hnsw => None,
    };

    info!(
        "caw-server ready: {} stubs, retriever={:?}, upstream={}, max_candidates={}, max_workspace_tokens={}",
        all.len(),
        retriever,
        upstream_base,
        max_candidates,
        max_workspace_tokens,
    );

    Ok(AppState {
        embedder: Mutex::new(embedder),
        store: Mutex::new(store),
        index: Mutex::new(index),
        bm25,
        retriever,
        upstream_base,
        max_candidates,
        max_workspace_tokens,
        http: reqwest::Client::builder()
            .build()
            .context("build reqwest client")?,
    })
}

/// Upstream path suffixes appended to `AppState::upstream_base`, one per
/// proxy route. They differ in where `/v1` sits — see the `upstream_base`
/// doc for the per-route base convention.
const OPENAI_CHAT_PATH: &str = "/chat/completions";
const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

/// OpenAI `/v1/chat/completions` proxy handler: parse as opaque JSON so
/// we preserve every field the client sent (model, temperature, tools,
/// response_format, …) and only mutate `messages` to splice the recalled
/// fragments into the last user message, then forward verbatim.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::extract::Json<Value>,
) -> Response {
    let mut req_json = body.0;
    augment_if_possible(&state, &mut req_json);
    forward(&state, req_json, &headers, OPENAI_CHAT_PATH).await
}

/// Anthropic `/v1/messages` proxy handler. Identical retrieve → inject →
/// forward flow as `chat_completions`; the protocol differences (block-
/// array user content, upstream path) are absorbed by
/// `extract_last_user_content` / `augment_last_user_message` and the
/// `ANTHROPIC_MESSAGES_PATH` suffix. The response is streamed back
/// verbatim, so the upstream must itself speak the Anthropic protocol.
async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::extract::Json<Value>,
) -> Response {
    let mut req_json = body.0;
    augment_if_possible(&state, &mut req_json);
    forward(&state, req_json, &headers, ANTHROPIC_MESSAGES_PATH).await
}

/// Retrieve for the request's last user turn and splice the fragments
/// into it, mutating `req_json` in place. Shared by both proxy routes so
/// they inject identically. Any failure (no user turn, no fragments,
/// retrieval or augmentation error) leaves `req_json` unmodified and is
/// logged, never fatal — the request still forwards.
fn augment_if_possible(state: &AppState, req_json: &mut Value) {
    let query = match extract_last_user_content(req_json) {
        Some(q) => q,
        None => {
            // No user turn yet — pass through unchanged. This covers
            // tool-result-only turns the client might send mid-session.
            debug!("no user message found; forwarding unmodified");
            return;
        }
    };

    match retrieve_fragments(state, &query) {
        Ok(fragments) if !fragments.is_empty() => {
            if let Err(e) = augment_last_user_message(req_json, &fragments) {
                warn!("augmentation failed ({e}); forwarding unmodified");
            } else {
                debug!(
                    "augmented with {} fragments ({} tokens)",
                    fragments.len(),
                    fragments.iter().map(|f| f.tokens).sum::<usize>()
                );
            }
        }
        Ok(_) => debug!("retrieval returned no fragments; forwarding unmodified"),
        Err(e) => warn!("retrieval error ({e}); forwarding unmodified"),
    }
}

/// Read-only diagnostic: run the identical retrieval the proxy would run
/// for a query and return the ranked candidates as JSON, without touching
/// any upstream model. Accepts either `{"query": "..."}` or an OpenAI-style
/// `{"messages": [...]}` body (the last user message is used), so the same
/// request shape works against both routes.
///
/// The response exposes what the proxy's DEBUG log otherwise buries: every
/// candidate's fused score, path, token cost, and disposition (admitted /
/// clamped / budget_full / content_miss), plus the body of each admitted
/// fragment — i.e. exactly the text that would have been injected.
async fn retrieve(
    State(state): State<Arc<AppState>>,
    body: axum::extract::Json<Value>,
) -> Response {
    let req = body.0;
    let query = req
        .get("query")
        .and_then(|q| q.as_str())
        .map(|s| s.to_string())
        .or_else(|| extract_last_user_content(&req));

    let query = match query {
        Some(q) if !q.trim().is_empty() => q,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "expected a non-empty `query` string or OpenAI `messages` with a user turn",
            )
                .into_response();
        }
    };

    let candidates = match retrieve_scored(&state, &query) {
        Ok(c) => c,
        Err(e) => {
            error!("retrieve diagnostic failed: {e}");
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("retrieval error: {e}"))
                .into_response();
        }
    };

    let admitted_tokens: usize = candidates
        .iter()
        .filter(|c| c.disposition == Disposition::Admitted)
        .filter_map(|c| c.tokens)
        .sum();
    let admitted_count = candidates
        .iter()
        .filter(|c| c.disposition == Disposition::Admitted)
        .count();

    let rows: Vec<Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "rank": c.rank,
                "path": c.path,
                "stub_id": c.stub_id.0,
                "score": c.score,
                "tokens": c.tokens,
                "disposition": c.disposition.as_str(),
                "content": c.content,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "query": query,
        "retriever": format!("{:?}", state.retriever),
        "max_candidates": state.max_candidates,
        "max_workspace_tokens": state.max_workspace_tokens,
        "candidate_count": candidates.len(),
        "admitted_count": admitted_count,
        "admitted_tokens": admitted_tokens,
        "candidates": rows,
    });

    axum::Json(payload).into_response()
}

/// Pull out the last message whose role is "user". OpenAI-style messages
/// are `{"role": "...", "content": "..."}`. Returns None if there's no
/// user message or the content isn't a string (vision/multipart is out
/// of scope for v0).
fn extract_last_user_content(req: &Value) -> Option<String> {
    let messages = req.get("messages")?.as_array()?;
    messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(message_text)
}

/// Pull the plain-text query out of a message `content` field, which the
/// OpenAI and Anthropic shapes represent differently:
/// - a plain string (OpenAI text turns, simple Anthropic turns), or
/// - an array of content blocks (`{"type":"text","text":…}`, plus image /
///   tool_use / tool_result blocks we ignore here). Both protocols use
///   the same `{"type":"text","text":…}` block, so one pass over text
///   blocks covers both.
///
/// Text blocks are joined with newlines. Returns `None` if there is no
/// usable text (e.g. an image-only or tool-result-only turn), which the
/// callers treat as "nothing to retrieve on".
fn message_text(content: &Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    let blocks = content.as_array()?;
    let text = blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// What retrieval decided to do with one ranked candidate. The proxy
/// injects exactly the `Admitted` ones, in rank order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Read and fit within the remaining token budget — injected.
    Admitted,
    /// Read, but its body would overflow `max_workspace_tokens`. The
    /// proxy stops admitting at the first such candidate (greedy prefix),
    /// so this marks where the budget ran out.
    Clamped,
    /// Ranked, but never read because an earlier candidate already hit the
    /// budget. Listed so the operator sees it was a near-miss on rank.
    BudgetFull,
    /// The stub's body could not be read (almost always a corpus-root
    /// mismatch). Skipped without consuming budget.
    ContentMiss,
    /// `MAX_CHUNKS_PER_SOURCE` chunks from this source file were already
    /// admitted. Up to the cap is allowed (different chunks of a multi-chunk
    /// document carry different answers); beyond it, further chunks are skipped
    /// without consuming budget — mirrors the orchestrator's per-source cap.
    DuplicateSource,
}

/// Maximum chunks admitted from a single source file. Mirrors the orchestrator's
/// `DynamicRecallConfig::max_chunks_per_source` default so the proxy and the CLI
/// engine inject the same per-source breadth. A one-per-source cap silently drops
/// the answer-bearing chunk of a multi-chunk document (long changelogs, READMEs)
/// when a higher-scoring chunk of the same file is admitted first; the budget
/// remains the hard ceiling.
const MAX_CHUNKS_PER_SOURCE: usize = 3;

impl Disposition {
    fn as_str(self) -> &'static str {
        match self {
            Disposition::Admitted => "admitted",
            Disposition::Clamped => "clamped",
            Disposition::BudgetFull => "budget_full",
            Disposition::ContentMiss => "content_miss",
            Disposition::DuplicateSource => "duplicate_source",
        }
    }
}

/// One ranked candidate plus retrieval's decision about it. This is the
/// single source of truth for what the proxy injects: `retrieve_fragments`
/// is just the `Admitted` rows projected into `RecallFragment`.
pub struct ScoredCandidate {
    pub rank: usize,
    pub stub_id: StubId,
    pub path: String,
    pub score: f32,
    /// Body token cost, or `None` for `BudgetFull` candidates the proxy
    /// never read.
    pub tokens: Option<usize>,
    pub mtime_unix_secs: u64,
    pub disposition: Disposition,
    /// The exact body that would be injected; `Some` only when `Admitted`.
    pub content: Option<String>,
}

impl ScoredCandidate {
    fn into_fragment(self) -> Option<RecallFragment> {
        match (self.disposition, self.content, self.tokens) {
            (Disposition::Admitted, Some(content), Some(tokens)) => Some(RecallFragment {
                stub_id: self.stub_id,
                content,
                locator: Locator {
                    source: self.path,
                    locator: "full".to_string(),
                },
                tokens,
                mtime_unix_secs: self.mtime_unix_secs,
            }),
            _ => None,
        }
    }
}

/// Embed the query, search the vector index (fused with BM25 under Hybrid),
/// and return the ranked `(stub, score)` candidates — the input to both the
/// proxy's injection clamp and the `/v1/retrieve` diagnostic.
fn rank_candidates(state: &AppState, query: &str) -> Result<Vec<(StubId, f32)>> {
    let embeddings = {
        let mut embedder = state
            .embedder
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder mutex poisoned"))?;
        embedder
            .embed_query(vec![query])
            .map_err(|e| anyhow::anyhow!("embed query: {e}"))?
    };
    let query_embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty embedding batch"))?;

    // Fetch a wider pool than we ultimately keep so the lexical half can
    // pull in stubs the embedding ranks out entirely (the failure mode that
    // motivated hybrid). Mirrors `HybridRetriever`'s `fetch_k = top_k * 3`.
    let fetch_k = state
        .max_candidates
        .saturating_mul(3)
        .max(state.max_candidates);

    let semantic_hits = {
        let mut index = state
            .index
            .lock()
            .map_err(|_| anyhow::anyhow!("index mutex poisoned"))?;
        index.search(&query_embedding, fetch_k)
    };

    let hits: Vec<(StubId, f32)> = match &state.bm25 {
        Some(bm25) => {
            let bm25_hits = {
                let bm = bm25
                    .lock()
                    .map_err(|_| anyhow::anyhow!("bm25 mutex poisoned"))?;
                bm.search(query, fetch_k)
            };
            fuse_hybrid(&semantic_hits, &bm25_hits, state.max_candidates)
        }
        None => {
            let mut h = semantic_hits;
            h.truncate(state.max_candidates);
            h
        }
    };

    // Diagnostic: the full ranked candidate list with (fused) scores, before
    // the token-budget clamp decides which survive into the workspace. Lets
    // you see whether a relevant stub was ranked out vs. clamped out.
    if tracing::enabled!(tracing::Level::DEBUG) {
        for (rank, (id, score)) in hits.iter().enumerate() {
            debug!("candidate #{rank} score={score:.4} {}", id.0);
        }
    }

    Ok(hits)
}

/// Materialize the ranked candidates into per-candidate dispositions,
/// applying the same greedy token-budget clamp the proxy uses to decide
/// what to inject. The `Admitted` rows, in order, are exactly the
/// fragments `retrieve_fragments` returns; the rest are retained with
/// their scores so the diagnostic route can show what was ranked but not
/// injected, and why.
fn retrieve_scored(state: &AppState, query: &str) -> Result<Vec<ScoredCandidate>> {
    let hits = rank_candidates(state, query)?;
    let candidate_count = hits.len();

    let store = state
        .store
        .lock()
        .map_err(|_| anyhow::anyhow!("store mutex poisoned"))?;

    let mut out = Vec::with_capacity(candidate_count);
    let mut used_tokens = 0usize;
    let mut admitted = 0usize;
    let mut content_misses = 0usize;
    // Chunks already injected per source. Up to MAX_CHUNKS_PER_SOURCE are
    // admitted; further chunks of the same file are dropped before charging
    // budget (the renderer now injects every admitted chunk).
    let mut admitted_source_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    // Set once the budget is hit: subsequent candidates are listed as
    // BudgetFull without reading their bodies, mirroring the proxy's
    // greedy-prefix `break`.
    let mut budget_exhausted = false;

    for (rank, (stub_id, score)) in hits.into_iter().enumerate() {
        // Metadata (incl. path/mtime) is cheap and wanted for every row,
        // including ones whose body we won't read.
        let stub = match store.get_stub(&stub_id) {
            Ok(s) => s,
            Err(e) => {
                content_misses += 1;
                warn!("get_stub({}) failed: {e}", stub_id.0);
                out.push(ScoredCandidate {
                    rank,
                    path: stub_id.0.clone(),
                    score,
                    tokens: None,
                    mtime_unix_secs: 0,
                    disposition: Disposition::ContentMiss,
                    content: None,
                    stub_id,
                });
                continue;
            }
        };

        // Allow up to MAX_CHUNKS_PER_SOURCE chunks per file; beyond the cap a
        // further chunk would crowd out distinct sources for diminishing return,
        // so drop it here without charging budget — same per-source cap the
        // orchestrator's load_fragments applies at admission.
        if admitted_source_counts.get(&stub.path).copied().unwrap_or(0) >= MAX_CHUNKS_PER_SOURCE {
            out.push(ScoredCandidate {
                rank,
                path: stub.path,
                score,
                tokens: None,
                mtime_unix_secs: stub.mtime_unix_secs,
                disposition: Disposition::DuplicateSource,
                content: None,
                stub_id,
            });
            continue;
        }

        if budget_exhausted {
            out.push(ScoredCandidate {
                rank,
                path: stub.path,
                score,
                tokens: None,
                mtime_unix_secs: stub.mtime_unix_secs,
                disposition: Disposition::BudgetFull,
                content: None,
                stub_id,
            });
            continue;
        }

        let content = match store.get_content(&stub_id) {
            Ok(c) => c,
            Err(e) => {
                content_misses += 1;
                warn!("get_content({}) failed: {e}", stub_id.0);
                out.push(ScoredCandidate {
                    rank,
                    path: stub.path,
                    score,
                    tokens: None,
                    mtime_unix_secs: stub.mtime_unix_secs,
                    disposition: Disposition::ContentMiss,
                    content: None,
                    stub_id,
                });
                continue;
            }
        };

        let tokens = count_tokens_cl100k(&content);
        if used_tokens + tokens > state.max_workspace_tokens && admitted > 0 {
            // At least one fragment always gets through so a pathologically
            // large top-1 doesn't silently produce a zero-fragment response.
            // The first over-budget candidate is the clamp boundary; from
            // here on we stop reading bodies.
            budget_exhausted = true;
            out.push(ScoredCandidate {
                rank,
                path: stub.path,
                score,
                tokens: Some(tokens),
                mtime_unix_secs: stub.mtime_unix_secs,
                disposition: Disposition::Clamped,
                content: None,
                stub_id,
            });
            continue;
        }

        used_tokens += tokens;
        admitted += 1;
        *admitted_source_counts.entry(stub.path.clone()).or_insert(0) += 1;
        out.push(ScoredCandidate {
            rank,
            path: stub.path,
            score,
            tokens: Some(tokens),
            mtime_unix_secs: stub.mtime_unix_secs,
            disposition: Disposition::Admitted,
            content: Some(content),
            stub_id,
        });
    }

    // If we had candidates but every body read missed, the workspace is empty
    // for a fetch-side reason (almost always a corpus-root mismatch), not
    // because nothing matched the query. Surface that as one aggregate error
    // rather than only per-stub warns the operator has to add up by hand.
    if candidate_count > 0 && content_misses == candidate_count {
        error!(
            "all {candidate_count} retrieved candidates had unreadable bodies — request forwarded UNAUGMENTED. \
             Check that --corpus-root matches the directory the index was built against."
        );
    }

    Ok(out)
}

/// The fragments the proxy injects: the `Admitted` candidates, in rank
/// order, projected into `RecallFragment`.
fn retrieve_fragments(state: &AppState, query: &str) -> Result<Vec<RecallFragment>> {
    Ok(retrieve_scored(state, query)?
        .into_iter()
        .filter_map(ScoredCandidate::into_fragment)
        .collect())
}

/// Fuse cosine and BM25 candidate lists into a single ranking. Each side's
/// scores are min-max normalized within its own list (so the two unrelated
/// score scales become comparable), weighted, and summed per stub. A stub
/// present in only one list still scores on that side alone — this is what
/// lets a strong lexical match enter even when the embedding ranked it out
/// of the cosine pool entirely.
///
/// Identical normalization, weighting, and tie-break to
/// `caw_index::HybridRetriever::search`, kept in sync deliberately: the
/// proxy can't reuse that type directly (it owns its embedder/store/index
/// behind separate mutexes), but the fusion math must match the library
/// retriever so results are the same.
fn fuse_hybrid(
    semantic: &[(StubId, f32)],
    bm25: &[(StubId, f32)],
    top_k: usize,
) -> Vec<(StubId, f32)> {
    use std::collections::HashMap;
    let mut combined: HashMap<StubId, f32> = HashMap::new();

    if !semantic.is_empty() {
        let max = semantic.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let min = semantic.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
        let range = (max - min).max(f32::EPSILON);
        for (id, score) in semantic {
            let normalized = (score - min) / range;
            *combined.entry(id.clone()).or_default() += normalized * HYBRID_SEMANTIC_WEIGHT;
        }
    }

    if !bm25.is_empty() {
        let max = bm25.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let min = bm25.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
        let range = (max - min).max(f32::EPSILON);
        for (id, score) in bm25 {
            let normalized = (score - min) / range;
            *combined.entry(id.clone()).or_default() += normalized * HYBRID_KEYWORD_WEIGHT;
        }
    }

    let total_weight = HYBRID_SEMANTIC_WEIGHT + HYBRID_KEYWORD_WEIGHT;
    let mut fused: Vec<(StubId, f32)> = combined
        .into_iter()
        .map(|(id, score)| (id, score / total_weight))
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused.truncate(top_k);
    fused
}

/// Splice recalled fragments into the final user message's content. We
/// reuse caw-core's bracketed format so the preamble matches what the
/// rest of the stack uses, and because openai-protocol models handle
/// brackets more reliably than XML.
fn augment_last_user_message(req: &mut Value, fragments: &[RecallFragment]) -> Result<()> {
    let messages = req
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
        .ok_or_else(|| anyhow::anyhow!("missing messages array"))?;

    let last_user_idx = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .ok_or_else(|| anyhow::anyhow!("no user message to augment"))?;

    // CompletionRequest carries the fragments; format_workspace does the
    // preamble + bracketed per-fragment wrapping. system/user strings
    // aren't used by that helper, so empty placeholders are fine.
    let cr = CompletionRequest {
        system: String::new(),
        user: String::new(),
        workspace_fragments: fragments.to_vec(),
        workspace_guidance: Vec::new(),
    };
    let workspace = cr.format_workspace(ProvenanceFormat::Bracketed);

    // Content is either a plain string (OpenAI text turns, simple
    // Anthropic turns) or an array of content blocks (Anthropic, and
    // OpenAI multimodal parts). For a string we append the workspace
    // text; for a block array we append a new `{"type":"text"}` block so
    // existing blocks (images, tool_result, …) are left untouched.
    let content = &mut messages[last_user_idx]["content"];
    if let Some(original) = content.as_str() {
        *content = Value::String(format!("{original}{workspace}"));
    } else if let Some(blocks) = content.as_array_mut() {
        blocks.push(serde_json::json!({ "type": "text", "text": workspace }));
    } else {
        return Err(anyhow::anyhow!(
            "last user content is neither a string nor a block array"
        ));
    }
    Ok(())
}

/// Proxy the (possibly augmented) request to the upstream server and
/// stream the response back verbatim. We forward Authorization and
/// relevant content-type headers; everything else we drop (Host, Accept,
/// user agent — reqwest supplies its own).
async fn forward(
    state: &AppState,
    req_json: Value,
    client_headers: &HeaderMap,
    upstream_path: &str,
) -> Response {
    let url = format!(
        "{}{upstream_path}",
        state.upstream_base.trim_end_matches('/')
    );

    let mut builder = state.http.post(&url).json(&req_json);

    if let Some(auth) = client_headers.get(header::AUTHORIZATION) {
        builder = builder.header(header::AUTHORIZATION, auth);
    }
    if let Some(v) = client_headers.get("anthropic-version") {
        builder = builder.header("anthropic-version", v);
    }
    if let Some(v) = client_headers.get("x-api-key") {
        builder = builder.header("x-api-key", v);
    }

    let upstream_resp = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            warn!("upstream request failed: {e}");
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
    let mut response_headers = HeaderMap::new();
    for k in [header::CONTENT_TYPE, header::CACHE_CONTROL] {
        if let Some(v) = upstream_resp.headers().get(k.clone()) {
            response_headers.insert(k, v.clone());
        }
    }

    let body = Body::from_stream(upstream_resp.bytes_stream());

    let mut resp = Response::builder()
        .status(status)
        .body(body)
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build response",
            )
                .into_response()
        });
    *resp.headers_mut() = response_headers;
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frag(content: &str) -> RecallFragment {
        RecallFragment {
            stub_id: StubId("test-stub".to_string()),
            content: content.to_string(),
            locator: Locator::full("test.md"),
            tokens: 0,
            mtime_unix_secs: 0,
        }
    }

    #[test]
    fn extract_handles_string_content() {
        let req = json!({"messages": [{"role": "user", "content": "what is foo"}]});
        assert_eq!(extract_last_user_content(&req).as_deref(), Some("what is foo"));
    }

    #[test]
    fn extract_handles_anthropic_block_array() {
        // Anthropic user turns commonly carry an array of content blocks;
        // we want the text blocks joined, ignoring non-text ones.
        let req = json!({"messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is foo"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "and bar"}
            ]
        }]});
        assert_eq!(
            extract_last_user_content(&req).as_deref(),
            Some("what is foo\nand bar")
        );
    }

    #[test]
    fn extract_none_for_textless_turn() {
        // An image-only or tool-result-only turn has no text to retrieve on.
        let req = json!({"messages": [{
            "role": "user",
            "content": [{"type": "tool_result", "content": "..."}]
        }]});
        assert_eq!(extract_last_user_content(&req), None);
    }

    #[test]
    fn augment_appends_to_string_content() {
        let mut req = json!({"messages": [{"role": "user", "content": "question"}]});
        augment_last_user_message(&mut req, &[frag("BODY")]).unwrap();
        let content = req["messages"][0]["content"].as_str().unwrap();
        assert!(content.starts_with("question"), "original text preserved");
        assert!(content.contains("BODY"), "fragment body injected");
        assert!(content.len() > "question".len(), "workspace appended");
    }

    #[test]
    fn augment_appends_text_block_to_array_content() {
        // The injected workspace must arrive as a new text block, leaving
        // the client's existing blocks (here an image) untouched.
        let mut req = json!({"messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "image", "source": {}}
            ]
        }]});
        augment_last_user_message(&mut req, &[frag("BODY")]).unwrap();
        let blocks = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3, "one block appended, originals intact");
        assert_eq!(blocks[1]["type"], "image", "image block untouched");
        let appended = &blocks[2];
        assert_eq!(appended["type"], "text");
        assert!(appended["text"].as_str().unwrap().contains("BODY"));
    }

    #[test]
    fn augment_targets_last_user_message() {
        // Retrieval splices into the most recent user turn, not an earlier one.
        let mut req = json!({"messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"}
        ]});
        augment_last_user_message(&mut req, &[frag("BODY")]).unwrap();
        assert_eq!(req["messages"][0]["content"], "old");
        assert!(req["messages"][2]["content"].as_str().unwrap().contains("BODY"));
    }
}
