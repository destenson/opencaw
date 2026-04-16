// Placeholder for Qdrant-based vector store
// Will use qdrant-client for Docker/cloud Qdrant instances

use caw_core::{CawError, CawResult, ScoredStub, Stub, StubId, VectorStore};

pub struct QdrantVectorStore {
    // client: qdrant_client::Qdrant,
    collection_name: String,
}

impl QdrantVectorStore {
    pub fn new(_url: &str, collection_name: impl Into<String>) -> CawResult<Self> {
        // TODO: Connect to Qdrant instance
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }

    pub fn local(collection_name: impl Into<String>) -> CawResult<Self> {
        Self::new("http://localhost:6334", collection_name)
    }
}

impl VectorStore for QdrantVectorStore {
    fn insert(&mut self, _stub: Stub, _embedding: Vec<f32>, _content: String) -> CawResult<()> {
        Err(CawError::VectorStore(
            "Qdrant store not yet implemented".to_string(),
        ))
    }

    fn search_by_embedding(&self, _query_embedding: &[f32], _top_k: usize) -> CawResult<Vec<ScoredStub>> {
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
}
