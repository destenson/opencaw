use caw_core::{CawError, CawResult, EmbeddingProvider, Locator, Range, RecallFragment, Retriever, ScoredStub, Stub, StubId, VectorStore};

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

#[cfg(feature = "fastembed")]
pub use embeddings::fastembed_provider::FastEmbedProvider;

pub use embeddings::api_provider::ApiEmbeddingProvider;

#[cfg(feature = "sqlite")]
pub use storage::sqlite_store::SqliteVectorStore;

#[cfg(feature = "qdrant")]
pub use storage::qdrant_store::QdrantVectorStore;

/// Semantic retriever that combines embedding generation and vector search
pub struct SemanticRetriever<E, S>
where
    E: EmbeddingProvider,
    S: VectorStore,
{
    embedder: E,
    store: S,
}

impl<E, S> SemanticRetriever<E, S>
where
    E: EmbeddingProvider,
    S: VectorStore,
{
    pub fn new(embedder: E, store: S) -> Self {
        Self { embedder, store }
    }

    pub fn insert(&mut self, stub: Stub, content: String) -> CawResult<()> {
        let text = format!("{} {} {}", stub.path, stub.summary, stub.outline.join(" "));
        let embeddings = self.embedder.embed(vec![text.as_str()])?;
        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| CawError::Embedding("No embedding generated".to_string()))?;

        self.store.insert(stub, embedding, content)
    }

    /// Direct access to the underlying vector store for embedding-based search
    pub fn store(&self) -> &S {
        &self.store
    }
}

impl<E, S> Retriever for SemanticRetriever<E, S>
where
    E: EmbeddingProvider,
    S: VectorStore,
{
    fn search(&mut self, query: &str, top_k: usize) -> CawResult<Vec<ScoredStub>> {
        let embeddings = self.embedder.embed(vec![query])?;
        let query_embedding = embeddings.into_iter().next()
            .ok_or_else(|| CawError::Embedding("No embedding generated for query".to_string()))?;

        self.store.search_by_embedding(&query_embedding, top_k)
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
    // Split on whitespace and punctuation boundaries for a rough
    // subword-tokenizer approximation (closer than len/4 for mixed content)
    content.split_whitespace().count().max(1)
}
