//! OpenAI-compatible proxy that augments the last user message with
//! retrieved workspace context before forwarding to an upstream model
//! server. This is the thin deployment target from the design doc: no
//! orchestrator, no probes, no multi-pass — just retrieve → inject →
//! forward. The sweep evidence says most of the measured opencaw gain
//! comes from multi-pass + reasoning, but v0 proves the UX: zero tool
//! calls, automatic context.

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
/// the process — the public HTTP surface stays pure OpenAI.
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
use serde_json::Value;
use std::sync::Arc;
use tracing::{debug, info, warn};

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
    /// Base URL of the upstream chat-completions server, e.g.
    /// `http://localhost:11434/v1` for Ollama. The `/chat/completions`
    /// suffix is appended by the handler.
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

    let store = SqliteStubStore::new(index_path, dim)
        .with_context(|| format!("open prebuilt index {index_path}"))?
        .with_corpus_root(corpus_root);

    let all = store
        .all_embeddings()
        .context("read all embeddings from prebuilt index")?;
    anyhow::ensure!(
        !all.is_empty(),
        "prebuilt index {} is empty — rebuild with caw-bench-build-index",
        index_path
    );

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
        upstream_base,
        max_candidates,
        max_workspace_tokens,
        http: reqwest::Client::builder()
            .build()
            .context("build reqwest client")?,
    })
}

/// The one handler: parse as opaque JSON so we preserve every field the
/// client sent (model, temperature, tools, response_format, …) and only
/// mutate `messages` to splice the recalled fragments into the last user
/// message.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::extract::Json<Value>,
) -> Response {
    let mut req_json = body.0;

    let query = match extract_last_user_content(&req_json) {
        Some(q) => q,
        None => {
            // No user turn yet — pass through unchanged. This covers
            // tool-result-only turns the client might send mid-session.
            debug!("no user message found; forwarding unmodified");
            return forward(&state, req_json, &headers).await;
        }
    };

    match retrieve_fragments(&state, &query) {
        Ok(fragments) if !fragments.is_empty() => {
            if let Err(e) = augment_last_user_message(&mut req_json, &fragments) {
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

    forward(&state, req_json, &headers).await
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
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

/// Embed the query, hit the HNSW index for the top K, then materialize
/// each stub's byte range into a RecallFragment. Clamps total tokens to
/// `max_workspace_tokens` in order.
fn retrieve_fragments(state: &AppState, query: &str) -> Result<Vec<RecallFragment>> {
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
    // the token-budget clamp below decides which survive into the workspace.
    // Lets you see whether a relevant stub was ranked out vs. clamped out.
    if tracing::enabled!(tracing::Level::DEBUG) {
        for (rank, (id, score)) in hits.iter().enumerate() {
            debug!("candidate #{rank} score={score:.4} {}", id.0);
        }
    }

    let store = state
        .store
        .lock()
        .map_err(|_| anyhow::anyhow!("store mutex poisoned"))?;

    let mut fragments = Vec::with_capacity(hits.len());
    let mut used_tokens = 0usize;
    for (stub_id, _score) in hits {
        let stub = match store.get_stub(&stub_id) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let content = match store.get_content(&stub_id) {
            Ok(c) => c,
            Err(e) => {
                warn!("get_content({}) failed: {e}", stub_id.0);
                continue;
            }
        };
        let tokens = count_tokens_cl100k(&content);
        if used_tokens + tokens > state.max_workspace_tokens && !fragments.is_empty() {
            // At least one fragment always gets through so a pathologically
            // large top-1 doesn't silently produce a zero-fragment response.
            break;
        }
        fragments.push(RecallFragment {
            stub_id,
            content,
            locator: Locator {
                source: stub.path,
                locator: "full".to_string(),
            },
            tokens,
            mtime_unix_secs: stub.mtime_unix_secs,
        });
        used_tokens += tokens;
    }
    Ok(fragments)
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

    let original = messages[last_user_idx]
        .get("content")
        .and_then(|c| c.as_str())
        .ok_or_else(|| anyhow::anyhow!("last user content is not a string"))?
        .to_string();

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

    let augmented = format!("{original}{workspace}");
    messages[last_user_idx]["content"] = Value::String(augmented);
    Ok(())
}

/// Proxy the (possibly augmented) request to the upstream server and
/// stream the response back verbatim. We forward Authorization and
/// relevant content-type headers; everything else we drop (Host, Accept,
/// user agent — reqwest supplies its own).
async fn forward(state: &AppState, req_json: Value, client_headers: &HeaderMap) -> Response {
    let url = format!(
        "{}/chat/completions",
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
