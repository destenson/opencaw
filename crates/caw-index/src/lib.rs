use caw_core::{
    CawError, CawResult, EmbeddingProvider, Locator, Range, RecallFragment, Retriever, ScoredStub,
    Stub, StubId, StubStore, VectorIndex,
};
use std::collections::HashMap;

pub mod bm25;

pub mod embeddings {
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

pub mod hnsw_index;

#[cfg(feature = "fastembed")]
pub use embeddings::fastembed_provider::FastEmbedProvider;

#[cfg(feature = "candle")]
pub use embeddings::candle_provider::CandleEmbeddingProvider;

#[cfg(feature = "onnx")]
pub use embeddings::onnx_provider::{OnnxEmbeddingProvider, OnnxVariant};

pub use bm25::BM25Index;
pub use embeddings::api_provider::ApiEmbeddingProvider;

#[cfg(feature = "sqlite")]
pub use storage::sqlite_store::SqliteStubStore;

#[cfg(feature = "qdrant")]
pub use storage::qdrant_store::QdrantStubStore;

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

    /// Embed the stub (path + summary + outline only — content is not
    /// included here because at this layer we don't have it, and the
    /// bench-index path has its own content-aware embedding builder).
    /// Persist the stub + embedding to the store and register with the
    /// in-memory vector index.
    pub fn insert(&mut self, stub: Stub) -> CawResult<()> {
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
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
        let full_content = self.store.get_content(id)?;
        let stub = self.store.get_stub(id)?;

        let parsed = Range::parse(range);
        let content = parsed.apply(&full_content);
        let tokens = estimate_tokens(&content);

        Ok(RecallFragment {
            stub_id: id.clone(),
            content,
            locator: Locator {
                source: stub.path,
                locator: range.to_string(),
            },
            tokens,
        })
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
    bm25: BM25Index,
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
            bm25: BM25Index::new(),
            semantic_weight,
            keyword_weight,
        }
    }

    /// Default 0.6 semantic / 0.4 keyword weights
    pub fn balanced(semantic: SemanticRetriever<E, S, I>) -> Self {
        Self::new(semantic, 0.6, 0.4)
    }

    /// `content` is used only transiently to build the BM25 posting list;
    /// it is NOT persisted anywhere. The downstream store records only
    /// `(path, byte_offset, byte_length)` and re-reads body text from disk.
    pub fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        let bm25_text = format!("{} {} {}", stub.path, stub.summary, content);
        self.bm25.add(stub.id.clone(), &bm25_text);
        self.semantic.insert(stub)
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

        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(top_k);
        Ok(results)
    }

    fn read_range(&self, id: &StubId, range: &str) -> CawResult<RecallFragment> {
        self.semantic.read_range(id, range)
    }
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
        let full_content = self
            .docs
            .iter()
            .find_map(|(stub_id, content)| (stub_id == id).then_some(content))
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        let stub = self
            .stubs
            .iter()
            .find(|s| &s.id == id)
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        let parsed = Range::parse(range);
        let content = parsed.apply(full_content);
        let tokens = estimate_tokens(&content);

        Ok(RecallFragment {
            stub_id: id.clone(),
            content,
            locator: Locator {
                source: stub.path.clone(),
                locator: range.to_string(),
            },
            tokens,
        })
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

fn estimate_tokens(content: &str) -> usize {
    content.split_whitespace().count().max(1)
}
