//! Arc<Mutex<_>> wrappers that let multiple `SemanticRetriever`s share a
//! single underlying embedder/store/index.
//!
//! The bench runs items sequentially but needs a fresh orchestrator per
//! item (for independent `loaded`, eviction, trace state). The heavyweight
//! retrieval state — embedding model, stub store, HNSW index — is identical
//! across items, so it's built once and shared. These wrappers adapt the
//! shared handles to the `EmbeddingProvider` / `StubStore` / `VectorIndex`
//! trait surfaces the orchestrator expects, cloning cheaply per item.

use std::sync::{Arc, Mutex};

use caw_core::{
    CawError, CawResult, ConsolidationNote, EmbeddingProvider, Stub, StubId, StubStore,
    VectorIndex,
};

fn poisoned<T>(_: std::sync::PoisonError<T>) -> CawError {
    CawError::VectorStore("shared lock poisoned".to_string())
}

pub struct SharedEmbedder<E: EmbeddingProvider> {
    inner: Arc<Mutex<E>>,
    dim: usize,
    name: &'static str,
}

impl<E: EmbeddingProvider> SharedEmbedder<E> {
    pub fn new(inner: E, name: &'static str) -> Self {
        let dim = inner.dimension();
        Self {
            inner: Arc::new(Mutex::new(inner)),
            dim,
            name,
        }
    }
}

impl<E: EmbeddingProvider> Clone for SharedEmbedder<E> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dim: self.dim,
            name: self.name,
        }
    }
}

impl<E: EmbeddingProvider> EmbeddingProvider for SharedEmbedder<E> {
    fn embed(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.inner.lock().map_err(poisoned)?.embed(texts)
    }

    fn embed_query(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.inner.lock().map_err(poisoned)?.embed_query(texts)
    }

    fn embed_document(&mut self, texts: Vec<&str>) -> CawResult<Vec<Vec<f32>>> {
        self.inner.lock().map_err(poisoned)?.embed_document(texts)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn provider_name(&self) -> &str {
        self.name
    }
}

pub struct SharedStore<S: StubStore> {
    inner: Arc<Mutex<S>>,
}

impl<S: StubStore> SharedStore<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn inner(&self) -> Arc<Mutex<S>> {
        self.inner.clone()
    }
}

impl<S: StubStore> Clone for SharedStore<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<S: StubStore> StubStore for SharedStore<S> {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>) -> CawResult<()> {
        self.inner.lock().map_err(poisoned)?.insert(stub, embedding)
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        self.inner.lock().map_err(poisoned)?.get_content(id)
    }

    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        self.inner.lock().map_err(poisoned)?.get_stub(id)
    }

    fn get_by_content_hash(&self, hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        self.inner
            .lock()
            .map_err(poisoned)?
            .get_by_content_hash(hash)
    }

    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        self.inner.lock().map_err(poisoned)?.all_embeddings()
    }

    fn save_consolidation(&mut self, stub_id: &StubId, note: &ConsolidationNote) -> CawResult<()> {
        self.inner
            .lock()
            .map_err(poisoned)?
            .save_consolidation(stub_id, note)
    }

    fn load_consolidation(&self, stub_id: &StubId) -> CawResult<Vec<ConsolidationNote>> {
        self.inner.lock().map_err(poisoned)?.load_consolidation(stub_id)
    }
}

/// Read-only view over a `SharedStore`. Writes are silently dropped so the
/// prebuilt index stays byte-identical across bench items. Used by the
/// shared-index runner path, where ingestion happens ahead of time and
/// eviction-consolidation must not leak state between items.
pub struct ReadOnlyStore<S: StubStore> {
    inner: Arc<Mutex<S>>,
}

impl<S: StubStore> ReadOnlyStore<S> {
    pub fn new(shared: &SharedStore<S>) -> Self {
        Self {
            inner: shared.inner(),
        }
    }
}

impl<S: StubStore> Clone for ReadOnlyStore<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<S: StubStore> StubStore for ReadOnlyStore<S> {
    fn insert(&mut self, _stub: Stub, _embedding: Vec<f32>) -> CawResult<()> {
        Ok(())
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        self.inner.lock().map_err(poisoned)?.get_content(id)
    }

    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        self.inner.lock().map_err(poisoned)?.get_stub(id)
    }

    fn get_by_content_hash(&self, hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        self.inner
            .lock()
            .map_err(poisoned)?
            .get_by_content_hash(hash)
    }

    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        self.inner.lock().map_err(poisoned)?.all_embeddings()
    }

    fn save_consolidation(&mut self, _stub_id: &StubId, _note: &ConsolidationNote) -> CawResult<()> {
        Ok(())
    }

    fn load_consolidation(&self, stub_id: &StubId) -> CawResult<Vec<ConsolidationNote>> {
        self.inner.lock().map_err(poisoned)?.load_consolidation(stub_id)
    }
}

pub struct SharedIndex<V: VectorIndex> {
    inner: Arc<Mutex<V>>,
}

impl<V: VectorIndex> SharedIndex<V> {
    pub fn new(inner: V) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }
}

impl<V: VectorIndex> Clone for SharedIndex<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V: VectorIndex> VectorIndex for SharedIndex<V> {
    fn add(&mut self, id: StubId, embedding: Vec<f32>) {
        if let Ok(mut g) = self.inner.lock() {
            g.add(id, embedding);
        }
    }

    fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(StubId, f32)> {
        match self.inner.lock() {
            Ok(mut g) => g.search(query_embedding, top_k),
            Err(_) => Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }
}
