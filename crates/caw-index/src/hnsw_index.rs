use caw_core::{StubId, VectorIndex};
use instant_distance::{Builder, HnswMap, Search};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize)]
struct EmbeddingPoint(Vec<f32>);

/// Failure persisting or loading a built HNSW graph.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("hnsw (de)serialization error: {0}")]
    Codec(#[from] bincode::Error),
}

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

    /// Serialize the built graph plus its source points to `path` (bincode).
    /// Builds first if stale so the persisted file is query-ready on load.
    /// At 43k stubs the in-process rebuild is several seconds of GPU-idle CPU
    /// work that a process restart otherwise repeats; this trades it for a
    /// one-time disk read. Both `points` and the built `index` are written so
    /// the loaded index stays internally consistent (`len`, and a later `add`
    /// that triggers a rebuild, both see the full point set).
    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        self.ensure_built();
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|source| PersistError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, &(&self.points, &self.index))?;
        Ok(())
    }

    /// Load a graph previously written by [`save`](Self::save). The returned
    /// index is query-ready without a rebuild; a subsequent `add` marks it
    /// dirty and rebuilds from `points` exactly as a freshly-built index would.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|source| PersistError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut reader = std::io::BufReader::new(file);
        let (points, index): (Vec<(StubId, Vec<f32>)>, Option<HnswMap<EmbeddingPoint, StubId>>) =
            bincode::deserialize_from(&mut reader)?;
        Ok(Self {
            points,
            index,
            search_buf: Search::default(),
            dirty: false,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("caw_hnsw_{tag}_{}_{nanos}.bin", std::process::id()))
    }

    // A saved graph loads back query-ready: the same query returns the same
    // top neighbor without any rebuild, and len is preserved.
    #[test]
    fn save_load_roundtrip_preserves_search() {
        let mut index = HnswVectorIndex::new();
        let vecs = [
            (StubId("a".into()), vec![1.0_f32, 0.0, 0.0]),
            (StubId("b".into()), vec![0.0, 1.0, 0.0]),
            (StubId("c".into()), vec![0.0, 0.0, 1.0]),
            (StubId("d".into()), vec![0.9, 0.1, 0.0]),
        ];
        for (id, v) in &vecs {
            index.add(id.clone(), v.clone());
        }
        let query = vec![1.0_f32, 0.05, 0.0];
        let before = index.search(&query, 2);

        let path = unique_temp_path("roundtrip");
        index.save(&path).expect("save");
        let mut loaded = HnswVectorIndex::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), vecs.len());
        let after = loaded.search(&query, 2);
        let ids_before: Vec<_> = before.iter().map(|(id, _)| id.clone()).collect();
        let ids_after: Vec<_> = after.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids_before, ids_after, "loaded graph returns same ranking");
    }

    // A loaded index still accepts new points: the rebuild path sees the full
    // point set (persisted + added), so the new point is searchable.
    #[test]
    fn add_after_load_rebuilds_full_set() {
        let mut index = HnswVectorIndex::new();
        index.add(StubId("a".into()), vec![1.0_f32, 0.0]);
        index.add(StubId("b".into()), vec![0.0, 1.0]);

        let path = unique_temp_path("addafter");
        index.save(&path).expect("save");
        let mut loaded = HnswVectorIndex::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        loaded.add(StubId("c".into()), vec![0.5, 0.5]);
        assert_eq!(loaded.len(), 3);
        let hits = loaded.search(&vec![0.5_f32, 0.5], 3);
        assert!(hits.iter().any(|(id, _)| id.0 == "c"), "new point searchable");
    }
}
