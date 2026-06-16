//! Sidecar storage for structural edges between stubs, plus graph-neighbor
//! expansion of retrieval results.
//!
//! This module is deliberately ignorant of where the edges come from. An
//! external extractor (e.g. a code-structure graph) is responsible for mapping
//! its own nodes onto OpenCAW `StubId`s and handing this module a flat list of
//! `StubEdge`s. The edges live in a `stub_edge` table next to the stub store in
//! the same SQLite file, keyed by `StubId`, so a prebuilt index and its graph
//! travel together.
//!
//! The retrieval side is a pure function (`plan_expansion`): given the seed
//! hits and the edges touching them, it decides which neighbors to admit and at
//! what score. IO (reading edges, fetching neighbor stubs) is the caller's job,
//! which keeps the scoring logic testable without a database.

use caw_core::{CawError, CawResult, ScoredStub, StubId};
use rusqlite::{params_from_iter, Connection};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A directed structural relationship between two stubs (e.g. the chunk holding
/// a caller and the chunk holding its callee). `relation` is the edge kind as
/// produced by the extractor (`calls`, `imports_from`, ...); it is stored
/// verbatim so expansion can filter and attribute by kind.
#[derive(Debug, Clone)]
pub struct StubEdge {
    pub src: StubId,
    pub dst: StubId,
    pub relation: String,
    pub weight: f32,
}

/// Byte geometry of one stub, used by an extractor to map a source location
/// (resolved to a byte offset) onto the chunk-stub that covers it.
#[derive(Debug, Clone)]
pub struct StubGeometry {
    pub id: StubId,
    pub byte_offset: u64,
    pub byte_length: u64,
}

/// An edge reachable from a seed hit: `neighbor` is the stub that might be
/// admitted, `seed` is the hit it was reached from, carrying the relation and
/// weight so the planner can score and attribute the admission.
#[derive(Debug, Clone)]
pub struct NeighborEdge {
    pub seed: StubId,
    pub neighbor: StubId,
    pub relation: String,
    pub weight: f32,
}

/// A neighbor selected for admission, with the score it was given and the
/// (seed, relation) it came through. The provenance fields exist so a benchmark
/// can attribute any rank improvement to a specific edge kind.
#[derive(Debug, Clone)]
pub struct PlannedNeighbor {
    pub id: StubId,
    pub score: f32,
    pub via_seed: StubId,
    pub via_relation: String,
}

/// How aggressively to walk edges out of the seed set.
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Only traverse edges whose relation is in this set. Empty means "no
    /// expansion" rather than "all relations" — a graph is a strong signal and
    /// the caller should name the kinds it trusts.
    pub relations: Vec<String>,
    /// Multiplier applied to the seed's score when scoring a neighbor. A
    /// neighbor is never more relevant than the hit that pulled it in, so this
    /// is < 1.0. The neighbor's edge `weight` multiplies on top.
    pub discount: f32,
    /// Hard cap on neighbors admitted per query, across all seeds.
    pub max_neighbors: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            relations: Vec::new(),
            discount: 0.5,
            max_neighbors: 8,
        }
    }
}

/// Read/write handle to the `stub_edge` table inside an existing index file.
pub struct GraphEdgeStore {
    conn: Connection,
}

impl GraphEdgeStore {
    /// Open the edge table in the SQLite file at `db_path`, creating it if
    /// absent. The file is expected to already contain a `stubs` table (the
    /// stub store); `stub_geometry` reads from it.
    pub fn open(db_path: &Path) -> CawResult<Self> {
        let conn = Connection::open(db_path).map_err(|e| {
            CawError::VectorStore(format!("graph edges: open {}: {}", db_path.display(), e))
        })?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS stub_edge (
                src_stub_id TEXT NOT NULL,
                dst_stub_id TEXT NOT NULL,
                relation    TEXT NOT NULL,
                weight      REAL NOT NULL,
                PRIMARY KEY (src_stub_id, dst_stub_id, relation)
             );
             CREATE INDEX IF NOT EXISTS stub_edge_src_idx ON stub_edge(src_stub_id);
             CREATE INDEX IF NOT EXISTS stub_edge_dst_idx ON stub_edge(dst_stub_id);",
        )
        .map_err(|e| CawError::VectorStore(format!("graph edges: create table: {}", e)))?;
        Ok(Self { conn })
    }

    /// All chunk-stubs for a source path, with their byte geometry, ordered by
    /// offset. An extractor uses this to find which chunk covers a given source
    /// location.
    pub fn stub_geometry(&self, path: &str) -> CawResult<Vec<StubGeometry>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, byte_offset, byte_length FROM stubs WHERE path = ?1 ORDER BY byte_offset")
            .map_err(|e| CawError::VectorStore(format!("graph edges: prepare geometry: {}", e)))?;
        let rows = stmt
            .query_map([path], |row| {
                Ok(StubGeometry {
                    id: StubId(row.get::<_, String>(0)?),
                    byte_offset: row.get::<_, i64>(1)? as u64,
                    byte_length: row.get::<_, i64>(2)? as u64,
                })
            })
            .map_err(|e| CawError::VectorStore(format!("graph edges: query geometry: {}", e)))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| CawError::VectorStore(format!("graph edges: read geometry: {}", e)))
    }

    /// Replace the entire edge set. Edge extraction is a whole-graph rebuild, so
    /// a partial-update API would invite stale edges from deleted code; clearing
    /// first keeps the table a faithful image of the latest graph.
    pub fn replace_edges(&mut self, edges: &[StubEdge]) -> CawResult<usize> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| CawError::VectorStore(format!("graph edges: begin tx: {}", e)))?;
        tx.execute("DELETE FROM stub_edge", [])
            .map_err(|e| CawError::VectorStore(format!("graph edges: clear: {}", e)))?;
        let mut written = 0usize;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT OR REPLACE INTO stub_edge (src_stub_id, dst_stub_id, relation, weight)
                     VALUES (?1, ?2, ?3, ?4)",
                )
                .map_err(|e| CawError::VectorStore(format!("graph edges: prepare insert: {}", e)))?;
            for e in edges {
                stmt.execute(rusqlite::params![e.src.0, e.dst.0, e.relation, e.weight])
                    .map_err(|err| {
                        CawError::VectorStore(format!("graph edges: insert: {}", err))
                    })?;
                written += 1;
            }
        }
        tx.commit()
            .map_err(|e| CawError::VectorStore(format!("graph edges: commit: {}", e)))?;
        Ok(written)
    }

    /// In-memory edge store for tests: creates only the `stub_edge` table (no
    /// `stubs` table, so `stub_geometry` is unavailable here).
    #[cfg(test)]
    fn in_memory() -> Self {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE stub_edge (
                src_stub_id TEXT NOT NULL,
                dst_stub_id TEXT NOT NULL,
                relation    TEXT NOT NULL,
                weight      REAL NOT NULL,
                PRIMARY KEY (src_stub_id, dst_stub_id, relation)
             );",
        )
        .unwrap();
        Self { conn }
    }

    pub fn edge_count(&self) -> CawResult<usize> {
        self.conn
            .query_row("SELECT COUNT(*) FROM stub_edge", [], |r| r.get::<_, i64>(0))
            .map(|n| n as usize)
            .map_err(|e| CawError::VectorStore(format!("graph edges: count: {}", e)))
    }

    /// Edges with at least one endpoint in `seed_ids` whose relation is in
    /// `relations`. Each returned `NeighborEdge` is oriented so `seed` is the
    /// endpoint that was in the seed set and `neighbor` is the other end. An
    /// edge with both endpoints in the seed set yields two `NeighborEdge`s; the
    /// planner discards neighbors that are themselves seeds.
    pub fn neighbors(
        &self,
        seed_ids: &[StubId],
        relations: &[String],
    ) -> CawResult<Vec<NeighborEdge>> {
        if seed_ids.is_empty() || relations.is_empty() {
            return Ok(Vec::new());
        }
        let seed_set: HashSet<&str> = seed_ids.iter().map(|s| s.0.as_str()).collect();

        // Distinct placeholder ranges for the two IN-clauses: SQLite numbered
        // params are positional, so reusing one range while binding the seed
        // list twice would mismatch bind count against the highest index. Match
        // a seed on either endpoint so expansion is direction-agnostic: a callee
        // is worth pulling in for its caller and vice versa.
        let n = seed_ids.len();
        let src_ph = placeholders(1, n);
        let dst_ph = placeholders(1 + n, n);
        let rel_ph = placeholders(1 + 2 * n, relations.len());
        let sql = format!(
            "SELECT src_stub_id, dst_stub_id, relation, weight FROM stub_edge
             WHERE (src_stub_id IN ({src_ph}) OR dst_stub_id IN ({dst_ph}))
               AND relation IN ({rel_ph})"
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| CawError::VectorStore(format!("graph edges: prepare neighbors: {}", e)))?;
        // Bind order matches the placeholder ranges: seeds for src, seeds again
        // for dst, then relations.
        let mut binds: Vec<&str> = Vec::with_capacity(n * 2 + relations.len());
        for s in seed_ids {
            binds.push(s.0.as_str());
        }
        for s in seed_ids {
            binds.push(s.0.as_str());
        }
        for r in relations {
            binds.push(r.as_str());
        }

        let rows = stmt
            .query_map(params_from_iter(binds), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)? as f32,
                ))
            })
            .map_err(|e| CawError::VectorStore(format!("graph edges: query neighbors: {}", e)))?;

        let mut out = Vec::new();
        for row in rows {
            let (src, dst, relation, weight) =
                row.map_err(|e| CawError::VectorStore(format!("graph edges: read neighbor: {}", e)))?;
            if seed_set.contains(src.as_str()) {
                out.push(NeighborEdge {
                    seed: StubId(src.clone()),
                    neighbor: StubId(dst.clone()),
                    relation: relation.clone(),
                    weight,
                });
            }
            if seed_set.contains(dst.as_str()) {
                out.push(NeighborEdge {
                    seed: StubId(dst),
                    neighbor: StubId(src),
                    relation,
                    weight,
                });
            }
        }
        Ok(out)
    }
}

/// Byte offset of the start of each line in `content`. `out[k]` is the offset
/// of line `k+1` (1-indexed): line 1 starts at byte 0, and each `\n` opens the
/// next line. Used to turn a source location (a line) into a byte offset so it
/// can be matched against a stub's byte range.
pub fn line_start_offsets(content: &[u8]) -> Vec<u64> {
    let mut starts = vec![0u64];
    for (i, b) in content.iter().enumerate() {
        if *b == b'\n' {
            starts.push((i + 1) as u64);
        }
    }
    starts
}

/// Byte offset for a 1-indexed line, clamped to the last known line start if the
/// line number exceeds the file (a graph and an index can drift between
/// rebuilds, and a clamped match is better than dropping the location).
pub fn byte_offset_of_line(line_starts: &[u64], line: usize) -> u64 {
    if line == 0 || line_starts.is_empty() {
        return 0;
    }
    let idx = (line - 1).min(line_starts.len() - 1);
    line_starts[idx]
}

/// Find the chunk-stub whose byte range covers `byte`. Chunks tile a file
/// contiguously, so a byte past the last chunk's end (e.g. trailing whitespace
/// the chunker dropped) maps to the last chunk rather than going unmapped.
pub fn stub_for_byte(geom: &[StubGeometry], byte: u64) -> Option<StubId> {
    for g in geom {
        if byte >= g.byte_offset && byte < g.byte_offset + g.byte_length {
            return Some(g.id.clone());
        }
    }
    geom.iter()
        .filter(|g| g.byte_offset <= byte)
        .max_by_key(|g| g.byte_offset)
        .map(|g| g.id.clone())
}

/// SQL placeholder list `?n, ?n+1, ...` of length `count`, starting at `start`.
fn placeholders(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Decide which neighbors to admit and at what score. Pure: no IO, so the
/// scoring policy can be tested directly.
///
/// A neighbor's score is `seed_score * discount * edge_weight`, taking the best
/// over all seeds that reach it. Neighbors already present in the seed set are
/// dropped (they're loaded anyway). The result is sorted by score descending
/// and truncated to `max_neighbors`.
pub fn plan_expansion(
    seeds: &[ScoredStub],
    neighbors: &[NeighborEdge],
    cfg: &ExpansionConfig,
) -> Vec<PlannedNeighbor> {
    if cfg.max_neighbors == 0 {
        return Vec::new();
    }
    let seed_score: HashMap<&str, f32> =
        seeds.iter().map(|s| (s.stub.id.0.as_str(), s.score)).collect();
    let seed_ids: HashSet<&str> = seed_score.keys().copied().collect();

    let mut best: HashMap<String, PlannedNeighbor> = HashMap::new();
    for ne in neighbors {
        if seed_ids.contains(ne.neighbor.0.as_str()) {
            continue;
        }
        let base = match seed_score.get(ne.seed.0.as_str()) {
            Some(s) => *s,
            None => continue,
        };
        let score = base * cfg.discount * ne.weight;
        let candidate = PlannedNeighbor {
            id: ne.neighbor.clone(),
            score,
            via_seed: ne.seed.clone(),
            via_relation: ne.relation.clone(),
        };
        match best.get(&ne.neighbor.0) {
            Some(existing) if existing.score >= score => {}
            _ => {
                best.insert(ne.neighbor.0.clone(), candidate);
            }
        }
    }

    let mut planned: Vec<PlannedNeighbor> = best.into_values().collect();
    planned.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    planned.truncate(cfg.max_neighbors);
    planned
}

#[cfg(test)]
mod tests {
    use super::*;
    use caw_core::Stub;

    fn scored(id: &str, score: f32) -> ScoredStub {
        ScoredStub {
            stub: Stub {
                id: StubId(id.to_string()),
                path: format!("{id}.rs"),
                token_estimate: 1,
                kind: caw_core::ContentKind::Code,
                summary: String::new(),
                outline: Vec::new(),
                content_hash: String::new(),
                mtime_unix_secs: 0,
                byte_offset: 0,
                byte_length: 0,
                chunk_total: 1,
                consolidation_notes: Vec::new(),
            },
            score,
        }
    }

    fn edge(seed: &str, neighbor: &str, relation: &str, weight: f32) -> NeighborEdge {
        NeighborEdge {
            seed: StubId(seed.to_string()),
            neighbor: StubId(neighbor.to_string()),
            relation: relation.to_string(),
            weight,
        }
    }

    #[test]
    fn admits_neighbor_discounted_below_seed() {
        let seeds = vec![scored("a", 0.8)];
        let neigh = vec![edge("a", "b", "calls", 1.0)];
        let cfg = ExpansionConfig {
            relations: vec!["calls".into()],
            discount: 0.5,
            max_neighbors: 8,
        };
        let plan = plan_expansion(&seeds, &neigh, &cfg);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].id.0, "b");
        assert!((plan[0].score - 0.4).abs() < 1e-6);
        assert_eq!(plan[0].via_relation, "calls");
    }

    #[test]
    fn neighbor_already_a_seed_is_dropped() {
        let seeds = vec![scored("a", 0.8), scored("b", 0.7)];
        let neigh = vec![edge("a", "b", "calls", 1.0)];
        let cfg = ExpansionConfig {
            relations: vec!["calls".into()],
            discount: 0.5,
            max_neighbors: 8,
        };
        assert!(plan_expansion(&seeds, &neigh, &cfg).is_empty());
    }

    #[test]
    fn best_seed_wins_for_shared_neighbor() {
        let seeds = vec![scored("a", 0.2), scored("c", 0.9)];
        let neigh = vec![
            edge("a", "b", "calls", 1.0),
            edge("c", "b", "calls", 1.0),
        ];
        let cfg = ExpansionConfig {
            relations: vec!["calls".into()],
            discount: 0.5,
            max_neighbors: 8,
        };
        let plan = plan_expansion(&seeds, &neigh, &cfg);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].via_seed.0, "c");
        assert!((plan[0].score - 0.45).abs() < 1e-6);
    }

    #[test]
    fn line_offsets_and_lookup() {
        let content = b"aaa\nbbbb\ncc\n";
        let starts = line_start_offsets(content);
        // line 1 @ 0, line 2 @ 4 ("bbbb"), line 3 @ 9 ("cc"), line 4 @ 12 (EOF)
        assert_eq!(starts, vec![0, 4, 9, 12]);
        assert_eq!(byte_offset_of_line(&starts, 1), 0);
        assert_eq!(byte_offset_of_line(&starts, 2), 4);
        assert_eq!(byte_offset_of_line(&starts, 3), 9);
        assert_eq!(byte_offset_of_line(&starts, 99), 12); // beyond EOF clamps
    }

    #[test]
    fn byte_maps_to_covering_chunk() {
        let geom = vec![
            StubGeometry { id: StubId("c0".into()), byte_offset: 0, byte_length: 10 },
            StubGeometry { id: StubId("c1".into()), byte_offset: 10, byte_length: 10 },
        ];
        assert_eq!(stub_for_byte(&geom, 0).unwrap().0, "c0");
        assert_eq!(stub_for_byte(&geom, 9).unwrap().0, "c0");
        assert_eq!(stub_for_byte(&geom, 10).unwrap().0, "c1");
        assert_eq!(stub_for_byte(&geom, 19).unwrap().0, "c1");
        assert_eq!(stub_for_byte(&geom, 50).unwrap().0, "c1"); // past end clamps
    }

    #[test]
    fn neighbors_match_either_endpoint_and_filter_relation() {
        let mut store = GraphEdgeStore::in_memory();
        let edge = |s: &str, d: &str, r: &str| StubEdge {
            src: StubId(s.into()),
            dst: StubId(d.into()),
            relation: r.into(),
            weight: 1.0,
        };
        store
            .replace_edges(&[
                edge("a", "b", "calls"),
                edge("b", "c", "calls"),
                edge("x", "a", "imports_from"),
            ])
            .unwrap();

        // From `a` along `calls`: a->b gives b. The imports_from edge x->a is
        // filtered out by relation.
        let got = store
            .neighbors(&[StubId("a".into())], &["calls".to_string()])
            .unwrap();
        let mut ns: Vec<String> = got.into_iter().map(|n| n.neighbor.0).collect();
        ns.sort();
        assert_eq!(ns, vec!["b"]);

        // From `b` along `calls`: a->b (b is dst -> neighbor a) and b->c
        // (b is src -> neighbor c). Exercises the both-endpoint match and the
        // distinct-placeholder binding for the two IN-clauses.
        let got = store
            .neighbors(&[StubId("b".into())], &["calls".to_string()])
            .unwrap();
        let mut ns: Vec<String> = got.into_iter().map(|n| n.neighbor.0).collect();
        ns.sort();
        assert_eq!(ns, vec!["a", "c"]);

        // Multi-seed query: the binding must scale with seed count.
        let got = store
            .neighbors(
                &[StubId("a".into()), StubId("b".into())],
                &["calls".to_string()],
            )
            .unwrap();
        assert!(!got.is_empty());
    }

    #[test]
    fn cap_truncates_lowest_scoring() {
        let seeds = vec![scored("a", 1.0)];
        let neigh = vec![
            edge("a", "b", "calls", 1.0),
            edge("a", "c", "calls", 0.5),
            edge("a", "d", "calls", 0.1),
        ];
        let cfg = ExpansionConfig {
            relations: vec!["calls".into()],
            discount: 1.0,
            max_neighbors: 2,
        };
        let plan = plan_expansion(&seeds, &neigh, &cfg);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].id.0, "b");
        assert_eq!(plan[1].id.0, "c");
    }
}
