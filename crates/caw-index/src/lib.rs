use caw_core::{
    count_tokens_cl100k, CawError, CawResult, EmbeddingProvider, Locator, Range, RecallFragment,
    Retriever, ScoredStub, Stub, StubId, StubStore, VectorIndex,
};
use std::collections::HashMap;

pub mod bm25;

pub mod embeddings {
    #[cfg(any(feature = "fastembed", feature = "onnx"))]
    pub(crate) mod ort_setup;

    #[cfg(feature = "fastembed")]
    pub mod fastembed_provider;

    #[cfg(feature = "candle")]
    pub mod candle_provider;

    #[cfg(feature = "onnx")]
    pub mod onnx_provider;

    pub mod api_provider;
}

pub mod storage {
    #[cfg(feature = "sqlite")]
    pub mod sqlite_store;

    #[cfg(feature = "qdrant")]
    pub mod qdrant_store;
}

pub mod flat_index;
pub mod hnsw_index;

#[cfg(feature = "sqlite")]
pub mod graph_edges;

#[cfg(feature = "fastembed")]
pub use embeddings::fastembed_provider::FastEmbedProvider;

#[cfg(feature = "candle")]
pub use embeddings::candle_provider::{CandleEmbeddingProvider, EmbedDevice};

#[cfg(feature = "onnx")]
pub use embeddings::onnx_provider::{OnnxEmbeddingProvider, OnnxVariant};

pub use bm25::{tokenize as bm25_tokenize, BM25Index};
pub use embeddings::api_provider::ApiEmbeddingProvider;

#[cfg(feature = "sqlite")]
pub use storage::sqlite_store::SqliteStubStore;

#[cfg(feature = "qdrant")]
pub use storage::qdrant_store::QdrantStubStore;

pub use flat_index::FlatVectorIndex;
pub use hnsw_index::HnswVectorIndex;

/// Semantic retriever combining embedding, storage, and vector index.
pub struct SemanticRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    embedder: E,
    store: S,
    index: I,
}

impl<E, S, I> SemanticRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    pub fn new(embedder: E, store: S, index: I) -> Self {
        Self {
            embedder,
            store,
            index,
        }
    }

    /// Embed the stub using its chunk content as the primary signal, with
    /// path and summary as a prefix. Asymmetric models benefit from
    /// document-side context, and code behavior is only recoverable from the
    /// body — metadata-only embeddings degrade to symbol-name lookup.
    /// Pass an empty string for `content` only when the body is unavailable
    /// (e.g. legacy call sites that construct stubs without chunking).
    pub fn insert(&mut self, stub: Stub, content: &str) -> CawResult<()> {
        let text = if content.is_empty() {
            format!("{} {}", stub.path, stub.summary)
        } else {
            format!("{}\n{}\n\n{}", stub.path, stub.summary, content)
        };
        let embeddings = self.embedder.embed_document(vec![text.as_str()])?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| CawError::Embedding("No embedding generated".to_string()))?;

        let id = stub.id.clone();
        self.store.insert(stub, embedding.clone())?;
        self.index.add(id, embedding);
        Ok(())
    }

    pub fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        self.store.get_stub(id)
    }

    pub fn index_mut(&mut self) -> &mut I {
        &mut self.index
    }
}

impl<E, S, I> Retriever for SemanticRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>> {
        let embeddings = self.embedder.embed_query(vec![query])?;
        let query_embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| CawError::Embedding("No embedding generated for query".to_string()))?;

        let hits = self.index.search(&query_embedding, top_k);

        let mut results = Vec::new();
        for (stub_id, score) in hits {
            match self.store.get_stub(&stub_id) {
                Ok(stub) => results.push(ScoredStub { stub, score }),
                Err(_) => continue,
            }
        }

        Ok(results)
    }

    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment> {
        let stub = self.store.get_stub(id)?;

        // "stub" range returns the summary + outline without loading body content.
        // Used for progressive disclosure: the model sees a compact descriptor
        // and can probe for full content if the source turns out to be relevant.
        if range.eq_ignore_ascii_case("stub") {
            let content = stub_descriptor(&stub.summary, &stub.outline);
            let tokens = count_tokens_cl100k(&content);
            return Ok(RecallFragment {
                stub_id: id.clone(),
                content,
                locator: Locator {
                    source: stub.path,
                    locator: "stub".to_string(),
                },
                tokens,
                mtime_unix_secs: stub.mtime_unix_secs,
            });
        }

        let full_content = self.store.get_content(id)?;
        let parsed = Range::parse(range);
        let content = parsed.apply(&full_content);
        let tokens = count_tokens_cl100k(&content);

        Ok(RecallFragment {
            stub_id: id.clone(),
            content,
            locator: Locator {
                source: stub.path,
                locator: range.to_string(),
            },
            tokens,
            mtime_unix_secs: stub.mtime_unix_secs,
        })
    }

    fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        self.insert(stub, &content)
    }

    fn chunk_ids_for_source(&self, source: &str) -> CawResult<Vec<StubId>> {
        self.store.chunk_ids_for_source(source)
    }
}

/// Hybrid retriever combining semantic (embedding) and keyword (BM25) search.
/// Fuses results using min-max normalized scores with configurable weights.
pub struct HybridRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    semantic: SemanticRetriever<E, S, I>,
    /// Behind an `Arc` so a corpus-wide BM25 index, which is expensive to
    /// build (it reads every body), can be built once and shared read-only
    /// across many per-item retrievers (the eval's shared-index path). The
    /// incremental `insert` path copies-on-write via `Arc::make_mut`, which
    /// is free while the `Arc` is uniquely held (the ingest-time case).
    bm25: std::sync::Arc<BM25Index>,
    semantic_weight: f32,
    keyword_weight: f32,
}

impl<E, S, I> HybridRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    pub fn new(
        semantic: SemanticRetriever<E, S, I>,
        semantic_weight: f32,
        keyword_weight: f32,
    ) -> Self {
        Self {
            semantic,
            bm25: std::sync::Arc::new(BM25Index::new()),
            semantic_weight,
            keyword_weight,
        }
    }

    /// Construct over a BM25 index that was already built elsewhere (e.g. once
    /// at load time, then shared read-only across per-item retrievers). The
    /// caller is responsible for having populated `bm25` with the same
    /// `path + summary + body` text that [`HybridRetriever::insert`] uses, so
    /// lexical scoring matches the ingest-time path.
    pub fn with_shared_bm25(
        semantic: SemanticRetriever<E, S, I>,
        bm25: std::sync::Arc<BM25Index>,
        semantic_weight: f32,
        keyword_weight: f32,
    ) -> Self {
        Self {
            semantic,
            bm25,
            semantic_weight,
            keyword_weight,
        }
    }

    /// Default 0.6 semantic / 0.4 keyword weights
    pub fn balanced(semantic: SemanticRetriever<E, S, I>) -> Self {
        Self::new(semantic, 0.6, 0.4)
    }

    /// Default 0.6 / 0.4 weights over a prebuilt, shared BM25 index.
    pub fn balanced_shared(
        semantic: SemanticRetriever<E, S, I>,
        bm25: std::sync::Arc<BM25Index>,
    ) -> Self {
        Self::with_shared_bm25(semantic, bm25, 0.6, 0.4)
    }

    /// `content` is used only transiently to build the BM25 posting list;
    /// it is NOT persisted anywhere. The downstream store records only
    /// `(path, byte_offset, byte_length)` and re-reads body text from disk.
    pub fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        let bm25_text = format!("{} {} {}", stub.path, stub.summary, content);
        // Copy-on-write: free while this retriever uniquely holds the Arc
        // (the ingest-time path); clones only if a shared index is mutated.
        std::sync::Arc::make_mut(&mut self.bm25).add(stub.id.clone(), &bm25_text);
        self.semantic.insert(stub, &content)
    }
}

impl<E, S, I> Retriever for HybridRetriever<E, S, I>
where
    E: EmbeddingProvider,
    S: StubStore,
    I: VectorIndex,
{
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>> {
        let fetch_k = top_k * 3;

        let semantic_results = self.semantic.search(query, fetch_k)?;
        let bm25_results = self.bm25.search(query, fetch_k);

        let mut combined: HashMap<StubId, f32> = HashMap::new();

        if !semantic_results.is_empty() {
            let max = semantic_results
                .iter()
                .map(|r| r.score)
                .fold(0.0f32, f32::max);
            let min = semantic_results
                .iter()
                .map(|r| r.score)
                .fold(f32::MAX, f32::min);
            let range = (max - min).max(f32::EPSILON);

            for result in &semantic_results {
                let normalized = (result.score - min) / range;
                *combined.entry(result.stub.id.clone()).or_default() +=
                    normalized * self.semantic_weight;
            }
        }

        if !bm25_results.is_empty() {
            let max = bm25_results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
            let min = bm25_results
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::MAX, f32::min);
            let range = (max - min).max(f32::EPSILON);

            for (id, score) in &bm25_results {
                let normalized = (score - min) / range;
                *combined.entry(id.clone()).or_default() += normalized * self.keyword_weight;
            }
        }

        let total_weight = self.semantic_weight + self.keyword_weight;

        let mut results: Vec<ScoredStub> = Vec::new();
        for (id, score) in combined {
            let normalized_score = score / total_weight;
            // Prefer stub from semantic results; fall back to store lookup
            let stub = semantic_results
                .iter()
                .find(|r| r.stub.id == id)
                .map(|r| r.stub.clone())
                .or_else(|| self.semantic.get_stub(&id).ok());

            if let Some(stub) = stub {
                results.push(ScoredStub {
                    stub,
                    score: normalized_score,
                });
            }
        }

        // Tie-break by stub id so a top_k cutoff that falls among equal fused
        // scores is deterministic. `combined` is a HashMap, so its iteration
        // order is randomized per process; without a total ordering here, which
        // tied stubs survive `truncate` varies run-to-run, changing the injected
        // context for identical inputs.
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.stub.id.0.cmp(&b.stub.id.0))
        });
        results.truncate(top_k);
        Ok(results)
    }

    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment> {
        self.semantic.read_range(id, range)
    }

    fn chunk_ids_for_source(&self, source: &str) -> CawResult<Vec<StubId>> {
        self.semantic.chunk_ids_for_source(source)
    }

    fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        self.insert(stub, content)
    }
}

/// Build a [`BM25Index`] over the stubs in `store`, indexing the same
/// `path + summary + body` text that [`HybridRetriever::insert`] uses so a
/// retriever constructed via [`HybridRetriever::with_shared_bm25`] scores
/// lexically the same way the ingest-time path would. Stubs whose stub row
/// or body can't be read are skipped.
///
/// This is the shared corpus-wide BM25 build used by the eval (`caw-bench`),
/// the CLI (`caw-cli`), and mirrored by the `caw-server` proxy — keeping
/// lexical scoring identical across all three retrieval surfaces.
pub fn build_bm25_over_store<S, Ids>(store: &S, stub_ids: Ids) -> BM25Index
where
    S: StubStore,
    Ids: IntoIterator<Item = StubId>,
{
    let mut bm25 = BM25Index::new();
    let mut missing = 0usize;
    for id in stub_ids {
        let stub = match store.get_stub(&id) {
            Ok(s) => s,
            Err(_) => {
                missing += 1;
                continue;
            }
        };
        let body = match store.get_content(&id) {
            Ok(c) => c,
            Err(_) => {
                missing += 1;
                continue;
            }
        };
        let text = format!("{} {} {}", stub.path, stub.summary, body);
        bm25.add(id, &text);
    }
    if missing > 0 {
        tracing::warn!("BM25 build: {missing} stubs had no readable stub/body and were skipped");
    }
    bm25
}

/// Simple keyword-overlap index for fallback/demo use
#[derive(Debug, Default, Clone)]
pub struct InMemoryIndex {
    stubs: Vec<Stub>,
    docs: Vec<(StubId, String)>,
}

impl InMemoryIndex {
    pub fn insert(&mut self, stub: Stub, content: String) {
        self.docs.push((stub.id.clone(), content));
        self.stubs.push(stub);
    }
}

impl Retriever for InMemoryIndex {
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>> {
        let mut scored = self
            .stubs
            .iter()
            .map(|stub| ScoredStub {
                stub: stub.clone(),
                score: score_query_against_stub(query, stub),
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment> {
        let stub = self
            .stubs
            .iter()
            .find(|s| &s.id == id)
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        if range.eq_ignore_ascii_case("stub") {
            let content = stub_descriptor(&stub.summary, &stub.outline);
            let tokens = count_tokens_cl100k(&content);
            return Ok(RecallFragment {
                stub_id: id.clone(),
                content,
                locator: Locator {
                    source: stub.path.clone(),
                    locator: "stub".to_string(),
                },
                tokens,
                mtime_unix_secs: stub.mtime_unix_secs,
            });
        }

        let full_content = self
            .docs
            .iter()
            .find_map(|(stub_id, content)| (stub_id == id).then_some(content))
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        let parsed = Range::parse(range);
        let content = parsed.apply(full_content);
        let tokens = count_tokens_cl100k(&content);

        Ok(RecallFragment {
            stub_id: id.clone(),
            content,
            locator: Locator {
                source: stub.path.clone(),
                locator: range.to_string(),
            },
            tokens,
            mtime_unix_secs: stub.mtime_unix_secs,
        })
    }
}

fn stub_descriptor(summary: &str, outline: &[String]) -> String {
    if outline.is_empty() {
        summary.to_string()
    } else {
        format!("{}\nOutline: {}", summary, outline.join(", "))
    }
}

fn score_query_against_stub(query: &str, stub: &Stub) -> f32 {
    let q = query.to_ascii_lowercase();
    let mut score = 0.0_f32;

    if stub.path.to_ascii_lowercase().contains(&q) {
        score += 0.8;
    }
    if stub.summary.to_ascii_lowercase().contains(&q) {
        score += 0.6;
    }
    if stub
        .outline
        .iter()
        .any(|line| line.to_ascii_lowercase().contains(&q))
    {
        score += 0.4;
    }

    if score == 0.0 {
        let query_terms = q.split_whitespace().collect::<Vec<_>>();
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "))
            .to_ascii_lowercase();
        let matched = query_terms.iter().filter(|t| text.contains(*t)).count();
        score += matched as f32 / (query_terms.len().max(1) as f32);
    }

    score
}
