use caw_core::{CawError, CawResult, EmbeddingProvider, Locator, Range, RecallFragment, Retriever, ScoredStub, Stub, StubId, StubStore, VectorIndex};

pub mod embeddings {
    #[cfg(feature = "fastembed")]
    pub mod fastembed_provider;

    #[cfg(feature = "candle")]
    pub mod candle_provider;

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

pub use embeddings::api_provider::ApiEmbeddingProvider;

#[cfg(feature = "sqlite")]
pub use storage::sqlite_store::SqliteStubStore;

pub use hnsw_index::HnswVectorIndex;

/// Semantic retriever that combines embedding, storage, and vector index.
/// The store persists stubs/content/embeddings; the index provides fast
/// approximate nearest neighbor search.
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
        Self { embedder, store, index }
    }

    pub fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
        let embeddings = self.embedder.embed(vec![text.as_str()])?;
        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| CawError::Embedding("No embedding generated".to_string()))?;

        let id = stub.id.clone();
        self.store.insert(stub, embedding.clone(), content)?;
        self.index.add(id, embedding);
        Ok(())
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
        let embeddings = self.embedder.embed(vec![query])?;
        let query_embedding = embeddings.into_iter().next()
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

/// Simple keyword-overlap index. No embeddings, no vector search —
/// just text matching against stub metadata. Useful for probes and
/// as a fallback when embedding infrastructure isn't available.
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
