use caw_core::{StubId, VectorIndex};
use instant_distance::{Builder, HnswMap, Search};

#[derive(Clone)]
struct EmbeddingPoint(Vec<f32>);

/// L2-normalize a vector so cosine similarity reduces to a dot product. A
/// zero vector is left as-is (its dot with anything is 0 → max distance).
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Points are stored L2-normalized (see `normalize`), so cosine
        // distance is just `1 - dot`. instant-distance calls this O(n·log n)
        // times per build; recomputing both norms + two sqrt per call here
        // (the previous implementation) made a 750-point rebuild take ~17s.
        let dot: f32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot
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

    /// Build the graph now if stale, so the cost lands here (e.g. right after
    /// ingest) instead of lazily on the first `search` — which otherwise makes
    /// the first query of a turn pay the full build. No-op if already current.
    pub fn ensure_built(&mut self) {
        if self.dirty || self.index.is_none() {
            self.rebuild();
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

        let ids: Vec<StubId> = self.points.iter().map(|(id, _)| id.clone()).collect();

        let t = std::time::Instant::now();
        let n = embeddings.len();
        let map = Builder::default().seed(42).build(embeddings, ids);
        tracing::debug!(
            points = n,
            elapsed_s = t.elapsed().as_secs_f64(),
            "HnswVectorIndex rebuilt"
        );

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
        // Store normalized so the build-time distance fn is a bare dot product.
        self.points.push((id, normalize(&embedding)));
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

        // Query must be normalized to match the stored points so the dot
        // product equals cosine similarity.
        let query = EmbeddingPoint(normalize(query_embedding));

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
