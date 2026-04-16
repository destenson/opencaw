// Placeholder for Qdrant-based stub store
// Qdrant would serve as both StubStore and VectorIndex in production,
// since it natively supports vector similarity search with metadata.

use caw_core::{CawError, CawResult, Stub, StubId, StubStore};

pub struct QdrantStubStore {
    collection_name: String,
}

impl QdrantStubStore {
    pub fn new(_url: &str, collection_name: impl Into<String>) -> CawResult<Self> {
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }
}

impl StubStore for QdrantStubStore {
    fn insert(&mut self, _stub: Stub, _embedding: Vec<f32>, _content: String) -> CawResult<()> {
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        Err(CawError::NotFound(id.0.clone()))
    }

    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        Err(CawError::NotFound(id.0.clone()))
    }

    fn get_by_content_hash(&self, _hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }

    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }
}
