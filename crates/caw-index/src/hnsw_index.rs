use caw_core::{StubId, VectorIndex};
use instant_distance::{Builder, HnswMap, Search};

#[derive(Clone)]
struct EmbeddingPoint(Vec<f32>);

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // instant-distance expects distance (lower = closer).
        // Cosine distance = 1 - cosine_similarity
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - (dot / (norm_a * norm_b))
    }
}

/// HNSW-based vector index for approximate nearest neighbor search.
///
/// Uses instant-distance which builds an immutable graph structure.
/// Points accumulate in a staging buffer and the graph is (re)built
/// lazily on the next search. For the expected corpus sizes (hundreds
/// to low thousands of documents), rebuild time is negligible.
pub struct HnswVectorIndex {
    /// All points — the source of truth for rebuilds
    points: Vec<(StubId, Vec<f32>)>,
    /// Built HNSW index; None when stale
    index: Option<HnswMap<EmbeddingPoint, StubId>>,
    /// Reusable search scratch space
    search_buf: Search,
    /// Whether the built index is stale
    dirty: bool,
}

impl HnswVectorIndex {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            index: None,
            search_buf: Search::default(),
            dirty: false,
        }
    }

    fn rebuild(&mut self) {
        if self.points.is_empty() {
            self.index = None;
            self.dirty = false;
            return;
        }

        let embeddings: Vec<EmbeddingPoint> = self
            .points
            .iter()
            .map(|(_, emb)| EmbeddingPoint(emb.clone()))
            .collect();

        let ids: Vec<StubId> = self
            .points
            .iter()
            .map(|(id, _)| id.clone())
            .collect();

        let map = Builder::default()
            .seed(42)
            .build(embeddings, ids);

        self.index = Some(map);
        self.dirty = false;
    }
}

impl Default for HnswVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for HnswVectorIndex {
    fn add(&mut self, id: StubId, embedding: Vec<f32>) {
        self.points.push((id, embedding));
        self.dirty = true;
    }

    fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(StubId, f32)> {
        if self.dirty || self.index.is_none() {
            self.rebuild();
        }

        let index = match &self.index {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let query = EmbeddingPoint(query_embedding.to_vec());

        index
            .search(&query, &mut self.search_buf)
            .take(top_k)
            .map(|item| {
                let similarity = 1.0 - item.distance;
                (item.value.clone(), similarity)
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}
