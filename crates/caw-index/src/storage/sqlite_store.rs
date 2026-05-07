use caw_core::{
    CawError, CawResult, ConsolidationNote, ConsolidationSource, ReindexQueue, Stub, StubId,
    StubStore,
};
use caw_ingest::DocumentId;
use rusqlite::{Connection, params};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

pub struct SqliteStubStore {
    conn: Connection,
    dimension: usize,
    /// Retained so workers can open their own connection to the same file
    /// via `open_another` without the caller having to track the path.
    /// `None` for in-memory databases, since `:memory:` connections are
    /// isolated — opening another `:memory:` gives you a different DB.
    db_path: Option<PathBuf>,
    /// Source tree root. Stubs store paths relative to this; `get_content`
    /// resolves `corpus_root.join(stub.path)` and reads
    /// `[byte_offset, byte_offset + byte_length)` from the file. When `None`,
    /// `get_content` returns an error — useful for tests and indexing-only
    /// callers that never need to read body text back.
    corpus_root: Option<PathBuf>,
    /// Optional sink for "this path needs reindexing" signals. When set,
    /// `get_content` pushes the stub's path onto the queue on staleness
    /// detection in addition to marking the row stale. Without a queue
    /// attached, staleness still surfaces as `CawError::StaleStub` — the
    /// row is just left stale until someone else schedules reingestion
    /// (e.g. via a startup sweep).
    reindex_queue: Option<Arc<dyn ReindexQueue>>,
}

impl SqliteStubStore {
    pub fn new(path: &str, dimension: usize) -> CawResult<Self> {
        let db_path = if path == ":memory:" {
            None
        } else {
            Some(PathBuf::from(path))
        };
        let conn = Connection::open(path)
            .map_err(|e| CawError::VectorStore(format!("Failed to open database: {}", e)))?;

        // WAL + synchronous=NORMAL is ~10-20x faster for bulk inserts than the
        // default rollback-journal + synchronous=FULL, with a minor durability
        // trade-off: a crash in the last second can lose recently-committed
        // transactions but the DB itself stays consistent. Acceptable for a
        // cache of embeddings that can always be re-derived from source.
        // In-memory DBs ignore these PRAGMAs harmlessly.
        for pragma in [
            "journal_mode=WAL",
            "synchronous=NORMAL",
            "temp_store=MEMORY",
        ] {
            conn.execute_batch(&format!("PRAGMA {};", pragma))
                .map_err(|e| CawError::VectorStore(format!("PRAGMA {}: {}", pragma, e)))?;
        }

        conn.execute(
            "CREATE TABLE IF NOT EXISTS stubs (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                outline TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                mtime_unix_secs INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                stub_json TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                ignored INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs table: {}", e)))?;

        // Migration for databases created before `stale` existed. SQLite's
        // ADD COLUMN is cheap; DEFAULT 0 backfills every row in O(metadata).
        // Swallow "duplicate column" so the migration is idempotent.
        match conn.execute(
            "ALTER TABLE stubs ADD COLUMN stale INTEGER NOT NULL DEFAULT 0",
            [],
        ) {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("duplicate column name") {
                    return Err(CawError::VectorStore(format!(
                        "Failed to add stale column: {}",
                        e
                    )));
                }
            }
        }

        // Migration for databases created before `ignored` existed.
        match conn.execute(
            "ALTER TABLE stubs ADD COLUMN ignored INTEGER NOT NULL DEFAULT 0",
            [],
        ) {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("duplicate column name") {
                    return Err(CawError::VectorStore(format!(
                        "Failed to add ignored column: {}",
                        e
                    )));
                }
            }
        }

        // Path lookup for startup sweeps and mark-stale-by-path. Index on
        // stale lets the sweep skip the majority of non-stale rows on large
        // corpora without a full scan.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS stubs_path_idx ON stubs(path)",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs_path_idx: {}", e)))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS stubs_stale_idx ON stubs(stale) WHERE stale = 1",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs_stale_idx: {}", e)))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS stubs_ignored_idx ON stubs(ignored) WHERE ignored = 1",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs_ignored_idx: {}", e)))?;

        // NOTE: no `contents` table. Previously this stored the full chunk
        // text per row, which on multi-chunk files duplicated the source
        // text N times (the bench indexer was shipping the whole doc to
        // every chunk stub, producing 160x blowup on real corpora). Body
        // text is re-derived from disk by slicing the source file using
        // (path, byte_offset, byte_length) from the stubs table.

        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                stub_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY(stub_id) REFERENCES stubs(id)
            )",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create embeddings table: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS consolidation_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stub_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at_secs INTEGER NOT NULL,
                FOREIGN KEY(stub_id) REFERENCES stubs(id)
            )",
            [],
        )
        .map_err(|e| {
            CawError::VectorStore(format!("Failed to create consolidation_notes table: {}", e))
        })?;

        Ok(Self {
            conn,
            dimension,
            db_path,
            corpus_root: None,
            reindex_queue: None,
        })
    }

    /// Open a second connection to the same on-disk database. Intended for
    /// reindex workers — each worker holds its own `Connection` (rusqlite's
    /// `Connection` is `Send` but not `Sync`), and SQLite's WAL mode
    /// serializes writes internally without application-level locking.
    /// Errors if this store was opened in-memory: `:memory:` connections
    /// are isolated per-handle, so a second one would be an empty database.
    pub fn open_another(&self) -> CawResult<Self> {
        let path = self.db_path.as_ref().ok_or_else(|| {
            CawError::VectorStore(
                "cannot open_another on an in-memory store — :memory: connections are isolated"
                    .to_string(),
            )
        })?;
        let path_str = path.to_str().ok_or_else(|| {
            CawError::VectorStore(format!("db path is not valid UTF-8: {}", path.display()))
        })?;
        let mut other = Self::new(path_str, self.dimension)?;
        other.corpus_root = self.corpus_root.clone();
        other.reindex_queue = self.reindex_queue.clone();
        Ok(other)
    }

    /// Attach a reindex queue. When a stale stub is detected in
    /// `get_content`, the row is marked stale and the path is enqueued.
    /// Without a queue, staleness still surfaces as `StaleStub` but no
    /// background work is scheduled.
    pub fn with_reindex_queue(mut self, queue: Arc<dyn ReindexQueue>) -> Self {
        self.reindex_queue = Some(queue);
        self
    }

    /// Mark a single stub stale by id. Idempotent — already-stale rows
    /// just stay stale. Called internally when `get_content` detects an
    /// mtime mismatch; exposed publicly so external signalers (file
    /// watchers, cache invalidators) can mark things stale without
    /// going through a recall.
    pub fn mark_stale(&self, id: &StubId) -> CawResult<()> {
        self.conn
            .execute("UPDATE stubs SET stale = 1 WHERE id = ?1", params![id.0])
            .map_err(|e| CawError::VectorStore(format!("mark_stale({}): {}", id.0, e)))?;
        Ok(())
    }

    /// Mark all stubs at a given path stale. A single file typically has
    /// many stubs (one per chunk); when the file changes they all go
    /// stale together.
    pub fn mark_path_stale(&self, path: &str) -> CawResult<usize> {
        let n = self
            .conn
            .execute(
                "UPDATE stubs SET stale = 1 WHERE path = ?1 AND stale = 0",
                params![path],
            )
            .map_err(|e| CawError::VectorStore(format!("mark_path_stale({}): {}", path, e)))?;
        Ok(n)
    }

    /// One-shot recovery for workers that crashed mid-reindex: returns
    /// every path currently flagged stale. Callers enqueue the result on
    /// their `ReindexQueue` at startup so any channel messages lost to a
    /// crash are replayed from the durable flag in the DB.
    /// Remove every stub at `path` along with its embeddings and
    /// consolidation notes. Used by reindex workers when the source file
    /// has been deleted — there's nothing to reingest, so the stubs have
    /// to go. Single transaction; all-or-nothing.
    pub fn delete_path(&mut self, path: &str) -> CawResult<usize> {
        let tx = self
            .conn
            .transaction()
            .map_err(|e| CawError::VectorStore(format!("begin tx delete_path: {}", e)))?;
        let n;
        {
            tx.execute(
                "DELETE FROM embeddings WHERE stub_id IN (SELECT id FROM stubs WHERE path = ?1)",
                params![path],
            )
            .map_err(|e| CawError::VectorStore(format!("delete embeddings for {}: {}", path, e)))?;
            tx.execute(
                "DELETE FROM consolidation_notes WHERE stub_id IN (SELECT id FROM stubs WHERE path = ?1)",
                params![path],
            )
            .map_err(|e| CawError::VectorStore(format!("delete notes for {}: {}", path, e)))?;
            n = tx
                .execute("DELETE FROM stubs WHERE path = ?1", params![path])
                .map_err(|e| CawError::VectorStore(format!("delete stubs for {}: {}", path, e)))?;
        }
        tx.commit()
            .map_err(|e| CawError::VectorStore(format!("commit delete_path: {}", e)))?;
        Ok(n)
    }

    /// Replace every stub at `path` with the given set in a single
    /// transaction: first delete the old rows (so a chunking change can
    /// leave orphans behind — their ids won't collide with new ones), then
    /// insert the new ones. New rows get `stale = 0` from the column
    /// default, so this clears the stale flag in one shot.
    pub fn replace_path(&mut self, path: &str, items: Vec<(Stub, Vec<f32>)>) -> CawResult<()> {
        for (_, embedding) in &items {
            if embedding.len() != self.dimension {
                return Err(CawError::VectorStore(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                )));
            }
        }

        let tx = self
            .conn
            .transaction()
            .map_err(|e| CawError::VectorStore(format!("begin tx replace_path: {}", e)))?;
        {
            tx.execute(
                "DELETE FROM embeddings WHERE stub_id IN (SELECT id FROM stubs WHERE path = ?1)",
                params![path],
            )
            .map_err(|e| CawError::VectorStore(format!("delete embeddings for {}: {}", path, e)))?;
            tx.execute(
                "DELETE FROM consolidation_notes WHERE stub_id IN (SELECT id FROM stubs WHERE path = ?1)",
                params![path],
            )
            .map_err(|e| CawError::VectorStore(format!("delete notes for {}: {}", path, e)))?;
            tx.execute("DELETE FROM stubs WHERE path = ?1", params![path])
                .map_err(|e| CawError::VectorStore(format!("delete stubs for {}: {}", path, e)))?;

            let mut stub_stmt = tx
                .prepare(
                    "INSERT INTO stubs \
                     (id, path, token_estimate, kind, summary, outline, content_hash, mtime_unix_secs, byte_offset, byte_length, stub_json) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                )
                .map_err(|e| CawError::VectorStore(format!("prepare stubs: {}", e)))?;
            let mut embed_stmt = tx
                .prepare("INSERT INTO embeddings (stub_id, embedding) VALUES (?1, ?2)")
                .map_err(|e| CawError::VectorStore(format!("prepare embeddings: {}", e)))?;

            for (stub, embedding) in items {
                let stub_json = serde_json::to_string(&stub)
                    .map_err(|e| CawError::VectorStore(format!("serialize stub: {}", e)))?;
                let kind_str = format!("{:?}", stub.kind);
                let outline_json = serde_json::to_string(&stub.outline)
                    .map_err(|e| CawError::VectorStore(format!("serialize outline: {}", e)))?;

                stub_stmt
                    .execute(params![
                        stub.id.0,
                        stub.path,
                        stub.token_estimate as i64,
                        kind_str,
                        stub.summary,
                        outline_json,
                        stub.content_hash,
                        stub.mtime_unix_secs as i64,
                        stub.byte_offset as i64,
                        stub.byte_length as i64,
                        stub_json,
                    ])
                    .map_err(|e| {
                        CawError::VectorStore(format!("insert stub {}: {}", stub.id.0, e))
                    })?;
                embed_stmt
                    .execute(params![stub.id.0, Self::embedding_to_blob(&embedding)])
                    .map_err(|e| {
                        CawError::VectorStore(format!("insert embedding {}: {}", stub.id.0, e))
                    })?;
            }
        }
        tx.commit()
            .map_err(|e| CawError::VectorStore(format!("commit replace_path: {}", e)))?;
        Ok(())
    }

    pub fn stale_paths(&self) -> CawResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT path FROM stubs WHERE stale = 1")
            .map_err(|e| CawError::VectorStore(format!("prepare stale_paths: {}", e)))?;
        let paths = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| CawError::VectorStore(format!("query stale_paths: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(paths)
    }

    /// Configure the corpus root used by `get_content` to resolve the
    /// relative paths stored on each stub. Call once after construction when
    /// the store will be queried for body text.
    pub fn with_corpus_root(mut self, root: PathBuf) -> Self {
        self.corpus_root = Some(root);
        self
    }

    pub fn in_memory(dimension: usize) -> CawResult<Self> {
        Self::new(":memory:", dimension)
    }

    /// Return the distinct (path, mtime) pairs already indexed. Used by
    /// resumable builders to skip files that have already been ingested
    /// without having to decode full stubs. Cheap — a single SELECT DISTINCT
    /// against the indexed `path` column.
    pub fn indexed_paths(&self) -> CawResult<Vec<DocumentId>> {
        // Exclude stale rows so the resumable builder reingests files whose
        // source changed since last index — otherwise the skip-already-indexed
        // fast path would keep stale stubs alive forever.
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT path, mtime_unix_secs FROM stubs WHERE stale = 0")
            .map_err(|e| CawError::VectorStore(format!("prepare indexed_paths: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let mtime: i64 = row.get(1)?;
                Ok((path, mtime as u64))
            })
            .map_err(|e| CawError::VectorStore(format!("query indexed_paths: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Insert many stubs + embeddings inside a single transaction with
    /// prepared statements reused across rows. Orders of magnitude faster
    /// than calling `insert` in a loop because it collapses fsyncs into one
    /// commit and avoids re-planning each statement.
    ///
    /// The whole batch is atomic: on error, nothing from this call lands.
    /// Resume logic upstream can just re-ingest the affected files.
    pub fn insert_batch(&mut self, items: Vec<(Stub, Vec<f32>)>) -> CawResult<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (_, embedding) in &items {
            if embedding.len() != self.dimension {
                return Err(CawError::VectorStore(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                )));
            }
        }

        let tx = self
            .conn
            .transaction()
            .map_err(|e| CawError::VectorStore(format!("begin transaction: {}", e)))?;
        {
            let mut stub_stmt = tx
                .prepare(
                    "INSERT OR REPLACE INTO stubs \
                     (id, path, token_estimate, kind, summary, outline, content_hash, mtime_unix_secs, byte_offset, byte_length, stub_json) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                )
                .map_err(|e| CawError::VectorStore(format!("prepare stubs: {}", e)))?;
            let mut embed_stmt = tx
                .prepare("INSERT OR REPLACE INTO embeddings (stub_id, embedding) VALUES (?1, ?2)")
                .map_err(|e| CawError::VectorStore(format!("prepare embeddings: {}", e)))?;

            for (stub, embedding) in items {
                let stub_json = serde_json::to_string(&stub)
                    .map_err(|e| CawError::VectorStore(format!("serialize stub: {}", e)))?;
                let kind_str = format!("{:?}", stub.kind);
                let outline_json = serde_json::to_string(&stub.outline)
                    .map_err(|e| CawError::VectorStore(format!("serialize outline: {}", e)))?;

                stub_stmt
                    .execute(params![
                        stub.id.0,
                        stub.path,
                        stub.token_estimate as i64,
                        kind_str,
                        stub.summary,
                        outline_json,
                        stub.content_hash,
                        stub.mtime_unix_secs as i64,
                        stub.byte_offset as i64,
                        stub.byte_length as i64,
                        stub_json,
                    ])
                    .map_err(|e| {
                        CawError::VectorStore(format!("insert stub {}: {}", stub.id.0, e))
                    })?;
                embed_stmt
                    .execute(params![stub.id.0, Self::embedding_to_blob(&embedding)])
                    .map_err(|e| {
                        CawError::VectorStore(format!("insert embedding {}: {}", stub.id.0, e))
                    })?;
            }
        }
        tx.commit()
            .map_err(|e| CawError::VectorStore(format!("commit batch: {}", e)))?;
        Ok(())
    }

    fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
        embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Apply `.cawignore` patterns from `root` to the stub index.
    ///
    /// Stubs whose paths match the patterns are flagged `ignored=1` and
    /// excluded from all subsequent searches. Stubs that no longer match are
    /// cleared to `ignored=0`. If `.cawignore` does not exist all flags are
    /// cleared. Returns `(ignored, cleared)`.
    pub fn apply_cawignore(&self, root: &std::path::Path) -> CawResult<(usize, usize)> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT path FROM stubs")
            .map_err(|e| CawError::VectorStore(format!("apply_cawignore prepare: {}", e)))?;
        let all_paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| CawError::VectorStore(format!("apply_cawignore query: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        let cawignore_path = root.join(".cawignore");
        let to_ignore: std::collections::HashSet<String> = if cawignore_path.exists() {
            let mut builder = ignore::gitignore::GitignoreBuilder::new(root);
            builder.add(&cawignore_path);
            let gitignore = builder
                .build()
                .map_err(|e| CawError::VectorStore(format!("apply_cawignore build: {}", e)))?;
            all_paths
                .iter()
                .filter(|p| {
                    let rel = p.trim_start_matches("./").trim_start_matches('/');
                    let abs = root.join(rel);
                    matches!(gitignore.matched(&abs, false), ignore::Match::Ignore(_))
                })
                .cloned()
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        // Clear all existing flags first, then re-apply the current pattern set.
        let cleared = self
            .conn
            .execute("UPDATE stubs SET ignored = 0 WHERE ignored = 1", [])
            .map_err(|e| CawError::VectorStore(format!("apply_cawignore clear: {}", e)))?;

        let mut ignored = 0usize;
        for path in &to_ignore {
            let n = self
                .conn
                .execute("UPDATE stubs SET ignored = 1 WHERE path = ?1", params![path])
                .map_err(|e| CawError::VectorStore(format!("apply_cawignore mark: {}", e)))?;
            ignored += n;
        }

        Ok((ignored, cleared))
    }
}

impl StubStore for SqliteStubStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>) -> CawResult<()> {
        self.insert_batch(vec![(stub, embedding)])
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        let (path, offset, length, stored_mtime, stale): (String, i64, i64, i64, i64) = self
            .conn
            .query_row(
                "SELECT path, byte_offset, byte_length, mtime_unix_secs, stale \
                 FROM stubs WHERE id = ?1",
                params![id.0],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => CawError::NotFound(id.0.clone()),
                _ => CawError::VectorStore(format!("Failed to get stub location: {}", e)),
            })?;

        // Already-stale rows short-circuit — the worker hasn't caught up yet,
        // and serving the old byte range would hand the caller bytes that no
        // longer correspond to the indexed content.
        if stale != 0 {
            if let Some(q) = &self.reindex_queue {
                q.enqueue(&path);
            }
            return Err(CawError::StaleStub { path });
        }

        let root = self.corpus_root.as_ref().ok_or_else(|| {
            CawError::VectorStore(
                "get_content requires a corpus_root; call with_corpus_root(..) after open"
                    .to_string(),
            )
        })?;
        let full_path = root.join(&path);

        // Stat before read so a changed/missing file is detected without
        // ever handing back stale bytes. We compare full-second mtime because
        // that's what the indexer persisted; sub-second precision on the
        // filesystem side doesn't help if our stored value was truncated.
        let current_mtime = match std::fs::metadata(&full_path) {
            Ok(m) => m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64),
            Err(_) => {
                // Missing / unreadable source → mark stale and schedule
                // reingestion (the worker will see the file is gone and
                // drop the stubs).
                let _ = self.mark_path_stale(&path);
                if let Some(q) = &self.reindex_queue {
                    q.enqueue(&path);
                }
                return Err(CawError::StaleStub { path });
            }
        };
        // stored_mtime == 0 means the indexer didn't record an mtime (e.g.
        // in-memory bench corpora written to a tempdir). Skip the staleness
        // check in that case rather than flagging every stub as stale.
        if stored_mtime != 0 && current_mtime != Some(stored_mtime) {
            let _ = self.mark_path_stale(&path);
            if let Some(q) = &self.reindex_queue {
                q.enqueue(&path);
            }
            return Err(CawError::StaleStub { path });
        }

        let mut file = std::fs::File::open(&full_path)
            .map_err(|e| CawError::VectorStore(format!("open {}: {}", full_path.display(), e)))?;
        file.seek(SeekFrom::Start(offset as u64)).map_err(|e| {
            CawError::VectorStore(format!("seek {} to {}: {}", full_path.display(), offset, e))
        })?;
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf).map_err(|e| {
            CawError::VectorStore(format!(
                "read {} bytes from {} at {}: {}",
                length,
                full_path.display(),
                offset,
                e
            ))
        })?;
        String::from_utf8(buf).map_err(|e| {
            CawError::VectorStore(format!(
                "body at {}:{}+{} is not valid UTF-8: {}",
                full_path.display(),
                offset,
                length,
                e
            ))
        })
    }

    fn get_stub(&self, id: &StubId) -> CawResult<Stub> {
        // Filter stale stubs at the retrieval boundary: returning `StaleStub`
        // here means the `SemanticRetriever` search loop's `Err(_) => continue`
        // arm drops them before scoring, so evicted-but-not-yet-reingested
        // rows don't surface as recall candidates.
        let (stub_json, path, stale, ignored): (String, String, i64, i64) = self
            .conn
            .query_row(
                "SELECT stub_json, path, stale, ignored FROM stubs WHERE id = ?1",
                params![id.0],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => CawError::NotFound(id.0.clone()),
                _ => CawError::VectorStore(format!("Failed to get stub: {}", e)),
            })?;

        if stale != 0 {
            return Err(CawError::StaleStub { path });
        }
        if ignored != 0 {
            return Err(CawError::NotFound(id.0.clone()));
        }

        serde_json::from_str(&stub_json)
            .map_err(|e| CawError::VectorStore(format!("Failed to deserialize stub: {}", e)))
    }

    fn get_by_content_hash(&self, hash: &str) -> CawResult<Option<(Stub, Vec<f32>)>> {
        let result: Result<(String, Vec<u8>), _> = self.conn.query_row(
            "SELECT s.stub_json, e.embedding FROM stubs s
             JOIN embeddings e ON s.id = e.stub_id
             WHERE s.content_hash = ?1
             LIMIT 1",
            params![hash],
            |row| {
                let json: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((json, blob))
            },
        );

        match result {
            Ok((json, blob)) => {
                let stub: Stub = serde_json::from_str(&json).map_err(|e| {
                    CawError::VectorStore(format!("Failed to deserialize stub: {}", e))
                })?;
                let embedding = Self::blob_to_embedding(&blob);
                Ok(Some((stub, embedding)))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(CawError::VectorStore(format!(
                "Failed to query by content hash: {}",
                e
            ))),
        }
    }

    fn save_consolidation(&mut self, stub_id: &StubId, note: &ConsolidationNote) -> CawResult<()> {
        let source_str = match note.source {
            ConsolidationSource::Eviction => "eviction",
            ConsolidationSource::ModelAnnotation => "model_annotation",
        };
        self.conn
            .execute(
                "INSERT INTO consolidation_notes (stub_id, content, source, created_at_secs)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    stub_id.0,
                    note.content,
                    source_str,
                    note.created_at_secs as i64
                ],
            )
            .map_err(|e| {
                CawError::VectorStore(format!("Failed to save consolidation note: {}", e))
            })?;
        Ok(())
    }

    fn load_consolidation(&self, stub_id: &StubId) -> CawResult<Vec<ConsolidationNote>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT content, source, created_at_secs FROM consolidation_notes
                 WHERE stub_id = ?1 ORDER BY created_at_secs ASC",
            )
            .map_err(|e| CawError::VectorStore(format!("Failed to prepare query: {}", e)))?;

        let notes = stmt
            .query_map(params![stub_id.0], |row| {
                let content: String = row.get(0)?;
                let source_str: String = row.get(1)?;
                let created_at_secs: i64 = row.get(2)?;
                let source = match source_str.as_str() {
                    "model_annotation" => ConsolidationSource::ModelAnnotation,
                    _ => ConsolidationSource::Eviction,
                };
                Ok(ConsolidationNote {
                    content,
                    source,
                    created_at_secs: created_at_secs as u64,
                })
            })
            .map_err(|e| CawError::VectorStore(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(notes)
    }

    fn all_embeddings(&self) -> CawResult<Vec<(StubId, Vec<f32>)>> {
        // Exclude stale and ignored rows: neither should appear in an HNSW
        // index that's used for recall.
        let mut stmt = self
            .conn
            .prepare(
                "SELECT e.stub_id, e.embedding FROM embeddings e \
                 JOIN stubs s ON s.id = e.stub_id \
                 WHERE s.stale = 0 AND s.ignored = 0",
            )
            .map_err(|e| CawError::VectorStore(format!("Failed to prepare query: {}", e)))?;

        let results = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((StubId(id), Self::blob_to_embedding(&blob)))
            })
            .map_err(|e| CawError::VectorStore(format!("Query failed: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caw_core::{ChannelReindexQueue, ContentKind};
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    fn temp_root() -> PathBuf {
        // Deterministic, process-scoped, deleted on Drop would be nice but
        // we don't want a tempfile dep; these land under the cargo target
        // dir so `cargo clean` reclaims them.
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("caw-sqlite-test-{}-{}", pid, n));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_file(root: &PathBuf, rel: &str, content: &str) -> u64 {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.sync_all().unwrap();
        drop(f);
        std::fs::metadata(&p)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn mk_stub(id: &str, rel: &str, mtime: u64, len: u64) -> Stub {
        Stub {
            id: StubId(id.to_string()),
            path: rel.to_string(),
            token_estimate: 10,
            kind: ContentKind::Markdown,
            summary: "s".into(),
            outline: vec![],
            content_hash: format!("h-{id}"),
            mtime_unix_secs: mtime,
            byte_offset: 0,
            byte_length: len,
            consolidation_notes: vec![],
        }
    }

    #[test]
    fn get_content_returns_stale_on_mtime_mismatch() {
        let root = temp_root();
        let db_path = root.join("idx.db");
        let content = "hello world";
        let mtime = write_file(&root, "doc.md", content);

        let queue = ChannelReindexQueue::new();
        let mut store = SqliteStubStore::new(db_path.to_str().unwrap(), 3)
            .unwrap()
            .with_corpus_root(root.clone())
            .with_reindex_queue(Arc::new(queue.clone()));

        let stub = mk_stub("s1", "doc.md", mtime, content.len() as u64);
        store.insert(stub, vec![0.1, 0.2, 0.3]).unwrap();

        // Fresh read works.
        assert_eq!(store.get_content(&StubId("s1".into())).unwrap(), content);

        // Edit the file so mtime advances. Sleep past the 1s mtime
        // resolution on every filesystem we care about.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_file(&root, "doc.md", "totally different content here");

        let err = store.get_content(&StubId("s1".into())).unwrap_err();
        match err {
            CawError::StaleStub { path } => assert_eq!(path, "doc.md"),
            other => panic!("expected StaleStub, got {:?}", other),
        }

        // Row is now marked stale, get_stub filters it, queue received it.
        match store.get_stub(&StubId("s1".into())).unwrap_err() {
            CawError::StaleStub { .. } => {}
            other => panic!("expected StaleStub from get_stub, got {:?}", other),
        }
        queue.close();
        let rx = queue.receiver();
        assert_eq!(rx.recv(), Some("doc.md".to_string()));
        assert_eq!(rx.recv(), None);

        assert_eq!(store.stale_paths().unwrap(), vec!["doc.md".to_string()]);
    }

    #[test]
    fn replace_path_clears_stale_and_retrieval_works_again() {
        let root = temp_root();
        let db_path = root.join("idx.db");
        let content = "original";
        let mtime = write_file(&root, "a.md", content);

        let mut store = SqliteStubStore::new(db_path.to_str().unwrap(), 3)
            .unwrap()
            .with_corpus_root(root.clone());

        let stub = mk_stub("s1", "a.md", mtime, content.len() as u64);
        store.insert(stub, vec![1.0, 0.0, 0.0]).unwrap();
        store.mark_path_stale("a.md").unwrap();
        assert!(matches!(
            store.get_stub(&StubId("s1".into())),
            Err(CawError::StaleStub { .. })
        ));

        // Simulate a reindex: overwrite with a new stub for the same path.
        let new_content = "replaced";
        let new_mtime = write_file(&root, "a.md", new_content);
        let new_stub = mk_stub("s2", "a.md", new_mtime, new_content.len() as u64);
        store
            .replace_path("a.md", vec![(new_stub, vec![0.0, 1.0, 0.0])])
            .unwrap();

        // Old id is gone; new id is live and readable.
        assert!(matches!(
            store.get_stub(&StubId("s1".into())),
            Err(CawError::NotFound(_))
        ));
        assert_eq!(
            store.get_content(&StubId("s2".into())).unwrap(),
            new_content
        );
    }

    #[test]
    fn missing_source_file_marks_stale_and_enqueues() {
        let root = temp_root();
        let db_path = root.join("idx.db");
        let mtime = write_file(&root, "gone.md", "soon to vanish");
        let queue = ChannelReindexQueue::new();
        let mut store = SqliteStubStore::new(db_path.to_str().unwrap(), 3)
            .unwrap()
            .with_corpus_root(root.clone())
            .with_reindex_queue(Arc::new(queue.clone()));
        let stub = mk_stub("s1", "gone.md", mtime, 14);
        store.insert(stub, vec![0.1, 0.2, 0.3]).unwrap();

        std::fs::remove_file(root.join("gone.md")).unwrap();

        match store.get_content(&StubId("s1".into())).unwrap_err() {
            CawError::StaleStub { path } => assert_eq!(path, "gone.md"),
            other => panic!("expected StaleStub, got {:?}", other),
        }
        queue.close();
        assert_eq!(queue.receiver().recv(), Some("gone.md".to_string()));
    }
}
