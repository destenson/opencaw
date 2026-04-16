use std::collections::HashMap;
use std::sync::Arc;

use caw_core::{CawError, CawResult, ConsolidationNote, Stub, StubId, StubStore, VectorIndex};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Distance, FieldType,
    Filter, GetPointsBuilder, PointStruct, PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, Value, VectorParamsBuilder, vector_output,
};
use serde_json::json;
use tokio::runtime::Runtime;

pub struct QdrantStubStore {
    client: Qdrant,
    collection_name: String,
    runtime: Arc<Runtime>,
    /// Tracks count locally to avoid a round-trip for every len() call.
    point_count: usize,
}

/// Deterministic mapping from arbitrary StubId strings to Qdrant's u64 point IDs.
/// Uses FNV-1a for speed — collisions are astronomically unlikely at the scale
/// this system operates (thousands of stubs, not billions).
fn stub_id_to_point_id(id: &StubId) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in id.0.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn qdrant_err(msg: impl std::fmt::Display) -> CawError {
    CawError::VectorStore(msg.to_string())
}

/// Extract a string from a Qdrant payload Value.
fn value_as_string(v: &Value) -> Option<String> {
    v.as_str().map(|s| s.to_string())
}

/// Extract the dense vector floats from a VectorsOutput, if present.
fn extract_dense_vector(
    vectors: &Option<qdrant_client::qdrant::VectorsOutput>,
) -> Option<Vec<f32>> {
    let vout = vectors.as_ref()?;
    let vector = vout.get_vector()?;
    match vector {
        vector_output::Vector::Dense(dense) => Some(dense.data),
        _ => None,
    }
}

impl QdrantStubStore {
    /// Connect to a running Qdrant instance. Creates the collection if it doesn't exist.
    pub fn connect(
        url: &str,
        collection_name: &str,
        dimension: usize,
        runtime: Arc<Runtime>,
    ) -> CawResult<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|e| qdrant_err(format!("failed to connect to Qdrant at {url}: {e}")))?;

        let coll = collection_name.to_string();

        let point_count = runtime.block_on(async {
            let exists = client
                .collection_exists(&coll)
                .await
                .map_err(|e| qdrant_err(format!("failed to check collection existence: {e}")))?;

            if !exists {
                client
                    .create_collection(CreateCollectionBuilder::new(&coll).vectors_config(
                        VectorParamsBuilder::new(dimension as u64, Distance::Cosine),
                    ))
                    .await
                    .map_err(|e| qdrant_err(format!("failed to create collection: {e}")))?;

                // Index content_hash for get_by_content_hash lookups
                client
                    .create_field_index(CreateFieldIndexCollectionBuilder::new(
                        &coll,
                        "content_hash",
                        FieldType::Keyword,
                    ))
                    .await
                    .map_err(|e| qdrant_err(format!("failed to create content_hash index: {e}")))?;

                // Index stub_id for payload-based lookups
                client
                    .create_field_index(CreateFieldIndexCollectionBuilder::new(
                        &coll,
                        "stub_id",
                        FieldType::Keyword,
                    ))
                    .await
                    .map_err(|e| qdrant_err(format!("failed to create stub_id index: {e}")))?;

                Ok::<usize, CawError>(0)
            } else {
                let info = client
                    .collection_info(&coll)
                    .await
                    .map_err(|e| qdrant_err(format!("failed to get collection info: {e}")))?;

                let count = info
                    .result
                    .map_or(0, |r| r.points_count.unwrap_or(0) as usize);
                Ok(count)
            }
        })?;

        Ok(Self {
            client,
            collection_name: coll,
            runtime,
            point_count,
        })
    }

    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        self.runtime.block_on(f)
    }

    fn payload_string(payload: &HashMap<String, Value>, key: &str) -> CawResult<String> {
        payload
            .get(key)
            .and_then(value_as_string)
            .ok_or_else(|| qdrant_err(format!("missing or invalid payload field: {key}")))
    }

    fn stub_from_payload(payload: &HashMap<String, Value>) -> CawResult<Stub> {
        let json_str = Self::payload_string(payload, "stub_json")?;
        serde_json::from_str(&json_str)
            .map_err(|e| qdrant_err(format!("failed to deserialize stub: {e}")))
    }
}

impl StubStore for QdrantStubStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>, content: String) -> CawResult<()> {
        let stub_json = serde_json::to_string(&stub)
            .map_err(|e| qdrant_err(format!("failed to serialize stub: {e}")))?;

        let point_id = stub_id_to_point_id(&stub.id);

        let payload = Payload::try_from(json!({
            "stub_id": stub.id.0,
            "stub_json": stub_json,
            "content": content,
            "content_hash": stub.content_hash,
            "consolidation_notes": "[]",
        }))
        .map_err(|e| qdrant_err(format!("failed to build payload: {e}")))?;

        let point = PointStruct::new(point_id, embedding, payload);

        self.block_on(async {
            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.collection_name, vec![point]).wait(true),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to upsert point: {e}")))
        })?;

        self.point_count += 1;
        Ok(())
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        let point_id = stub_id_to_point_id(id);

        let result = self.block_on(async {
            self.client
                .get_points(
                    GetPointsBuilder::new(&self.collection_name, vec![point_id.into()])
                        .with_payload(true)
                        .with_vectors(false),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to get point: {e}")))
        })?;

        let point = result
            .result
            .first()
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        Self::payload_string(&point.payload, "content")
    }

    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        let point_id = stub_id_to_point_id(id);

        let result = self.block_on(async {
            self.client
                .get_points(
                    GetPointsBuilder::new(&self.collection_name, vec![point_id.into()])
                        .with_payload(true)
                        .with_vectors(false),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to get point: {e}")))
        })?;

        let point = result
            .result
            .first()
            .ok_or_else(|| CawError::NotFound(id.0.clone()))?;

        Self::stub_from_payload(&point.payload)
    }

    fn get_by_content_hash(&self, hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        let result = self.block_on(async {
            self.client
                .scroll(
                    ScrollPointsBuilder::new(&self.collection_name)
                        .filter(Filter::must([Condition::matches(
                            "content_hash",
                            hash.to_string(),
                        )]))
                        .limit(1)
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to scroll for content hash: {e}")))
        })?;

        let Some(point) = result.result.first() else {
            return Ok(None);
        };

        let stub = Self::stub_from_payload(&point.payload)?;

        let embedding = extract_dense_vector(&point.vectors)
            .ok_or_else(|| qdrant_err("point missing vector data"))?;

        Ok(Some((stub, embedding)))
    }

    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        let mut results = Vec::new();
        let mut offset = None;

        loop {
            let mut builder = ScrollPointsBuilder::new(&self.collection_name)
                .limit(100)
                .with_payload(true)
                .with_vectors(true);

            if let Some(off) = offset {
                builder = builder.offset(off);
            }

            let page = self.block_on(async {
                self.client
                    .scroll(builder)
                    .await
                    .map_err(|e| qdrant_err(format!("failed to scroll all embeddings: {e}")))
            })?;

            for point in &page.result {
                let stub_id_str =
                    Self::payload_string(&point.payload, "stub_id").unwrap_or_default();

                if let Some(emb) = extract_dense_vector(&point.vectors) {
                    results.push((StubId(stub_id_str), emb));
                }
            }

            match page.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }

        Ok(results)
    }

    fn save_consolidation(&mut self, stub_id: &StubId, note: &ConsolidationNote) -> CawResult<()> {
        let point_id = stub_id_to_point_id(stub_id);

        let mut notes = self.load_consolidation(stub_id)?;
        notes.push(note.clone());

        let notes_json = serde_json::to_string(&notes)
            .map_err(|e| qdrant_err(format!("failed to serialize consolidation notes: {e}")))?;

        let payload: Payload = Payload::try_from(json!({
            "consolidation_notes": notes_json,
        }))
        .map_err(|e| qdrant_err(format!("failed to build payload: {e}")))?;

        self.block_on(async {
            self.client
                .set_payload(
                    SetPayloadPointsBuilder::new(&self.collection_name, payload)
                        .points_selector(PointsIdsList {
                            ids: vec![point_id.into()],
                        })
                        .wait(true),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to update consolidation notes: {e}")))
        })?;

        Ok(())
    }

    fn load_consolidation(&self, stub_id: &StubId) -> CawResult<Vec<ConsolidationNote>> {
        let point_id = stub_id_to_point_id(stub_id);

        let result = self.block_on(async {
            self.client
                .get_points(
                    GetPointsBuilder::new(&self.collection_name, vec![point_id.into()])
                        .with_payload(true)
                        .with_vectors(false),
                )
                .await
                .map_err(|e| qdrant_err(format!("failed to get point: {e}")))
        })?;

        let Some(point) = result.result.first() else {
            return Ok(Vec::new());
        };

        let notes_str = Self::payload_string(&point.payload, "consolidation_notes")
            .unwrap_or_else(|_| "[]".to_string());

        serde_json::from_str(&notes_str)
            .map_err(|e| qdrant_err(format!("failed to deserialize consolidation notes: {e}")))
    }
}

impl VectorIndex for QdrantStubStore {
    fn add(&mut self, id: StubId, embedding: Vec<f32>) {
        // VectorIndex::add is infallible by trait signature, so errors are swallowed.
        // In practice, insert() is the primary write path; this exists for trait compliance.
        let point_id = stub_id_to_point_id(&id);

        let payload = match Payload::try_from(json!({ "stub_id": id.0 })) {
            Ok(p) => p,
            Err(_) => return,
        };

        let point = PointStruct::new(point_id, embedding, payload);

        let _ = self.block_on(async {
            self.client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.collection_name, vec![point]).wait(true),
                )
                .await
        });

        self.point_count += 1;
    }

    fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(StubId, f32)> {
        let result = self.block_on(async {
            self.client
                .search_points(
                    SearchPointsBuilder::new(
                        &self.collection_name,
                        query_embedding.to_vec(),
                        top_k as u64,
                    )
                    .with_payload(true)
                    .with_vectors(false),
                )
                .await
        });

        match result {
            Ok(response) => response
                .result
                .iter()
                .filter_map(|scored| {
                    let stub_id_str = value_as_string(scored.get("stub_id"))?;
                    Some((StubId(stub_id_str), scored.score))
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.point_count
    }
}
