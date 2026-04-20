//! Brute-force cosine-similarity vector index.
//!
//! For corpora under a few million stubs, a flat scan is faster
//! end-to-end than HNSW: the per-query cost is O(n·d) floating-point
//! ops (well under 100 ms for 158k × 384 with a vectorizing compiler),
//! and there's no construction/rebuild cost to amortize. The HNSW path
//! wins only when the corpus is too large for the flat scan to hit
//! whatever latency budget the caller has.
//!
//! Assumes embeddings are L2-normalized at insert time (the default for
//! both `CandleEmbeddingProvider` and the fastembed BGE variants). Under
//! that assumption cosine similarity reduces to a plain dot product, so
//! we skip norms in the hot loop.

use caw_core::{StubId, VectorIndex};

pub struct FlatVectorIndex {
    points: Vec<(StubId, Vec<f32>)>,
}

impl FlatVectorIndex {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Bulk constructor for when the caller has all points up front — saves
    /// N reallocations vs. calling `add()` in a loop.
    pub fn from_points(points: Vec<(StubId, Vec<f32>)>) -> Self {
        Self { points }
    }
}

impl Default for FlatVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for FlatVectorIndex {
    fn add(&mut self, id: StubId, embedding: Vec<f32>) {
        self.points.push((id, embedding));
    }

    fn search(&mut self, query: &[f32], top_k: usize) -> Vec<(StubId, f32)> {
        if self.points.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Compute similarities. Dot product assumes unit-norm vectors; if
        // callers violate that, the returned scores are no longer bounded
        // to [-1, 1] but the ranking is still monotone in cosine for
        // equal-norm vectors. Wrong norms would bias toward longer
        // vectors, which is an indexer bug to fix upstream.
        let mut scored: Vec<(StubId, f32)> = self
            .points
            .iter()
            .map(|(id, emb)| {
                let sim: f32 = emb.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                (id.clone(), sim)
            })
            .collect();

        // Partial sort: we only need top_k descending. For top_k ≪ n the
        // select_nth approach is cheaper than a full sort but std
        // stable-sort here is still < 10 ms at 158k so the extra
        // complexity isn't worth it. Revisit if latency matters.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);
        scored
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}
