use caw_core::{
    CawError, CawResult, ConsolidationNote, ConsolidationSource, Stub, StubId, StubStore,
};
use rusqlite::{Connection, params};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

pub struct SqliteStubStore {
    conn: Connection,
    dimension: usize,
    /// Source tree root. Stubs store paths relative to this; `get_content`
    /// resolves `corpus_root.join(stub.path)` and reads
    /// `[byte_offset, byte_offset + byte_length)` from the file. When `None`,
    /// `get_content` returns an error — useful for tests and indexing-only
    /// callers that never need to read body text back.
    corpus_root: Option<PathBuf>,
}

impl SqliteStubStore {
    pub fn new(path: &str, dimension: usize) -> CawResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| CawError::VectorStore(format!("Failed to open database: {}", e)))?;

        // WAL + synchronous=NORMAL is ~10-20x faster for bulk inserts than the
        // default rollback-journal + synchronous=FULL, with a minor durability
        // trade-off: a crash in the last second can lose recently-committed
        // transactions but the DB itself stays consistent. Acceptable for a
        // cache of embeddings that can always be re-derived from source.
        // In-memory DBs ignore these PRAGMAs harmlessly.
        for pragma in ["journal_mode=WAL", "synchronous=NORMAL", "temp_store=MEMORY"] {
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
                stub_json TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs table: {}", e)))?;

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
            corpus_root: None,
        })
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
    pub fn indexed_paths(&self) -> CawResult<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT path, mtime_unix_secs FROM stubs")
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
                let stub_json = serde_json::to_string(&stub).map_err(|e| {
                    CawError::VectorStore(format!("serialize stub: {}", e))
                })?;
                let kind_str = format!("{:?}", stub.kind);
                let outline_json = serde_json::to_string(&stub.outline).map_err(|e| {
                    CawError::VectorStore(format!("serialize outline: {}", e))
                })?;

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
                    .map_err(|e| CawError::VectorStore(format!("insert stub {}: {}", stub.id.0, e)))?;
                embed_stmt
                    .execute(params![stub.id.0, Self::embedding_to_blob(&embedding)])
                    .map_err(|e| CawError::VectorStore(format!("insert embedding {}: {}", stub.id.0, e)))?;
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
}

impl StubStore for SqliteStubStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>) -> CawResult<()> {
        self.insert_batch(vec![(stub, embedding)])
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        let (path, offset, length): (String, i64, i64) = self
            .conn
            .query_row(
                "SELECT path, byte_offset, byte_length FROM stubs WHERE id = ?1",
                params![id.0],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => CawError::NotFound(id.0.clone()),
                _ => CawError::VectorStore(format!("Failed to get stub location: {}", e)),
            })?;

        let root = self.corpus_root.as_ref().ok_or_else(|| {
            CawError::VectorStore(
                "get_content requires a corpus_root; call with_corpus_root(..) after open"
                    .to_string(),
            )
        })?;
        let full_path = root.join(&path);

        let mut file = std::fs::File::open(&full_path).map_err(|e| {
            CawError::VectorStore(format!("open {}: {}", full_path.display(), e))
        })?;
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
        let stub_json: String = self
            .conn
            .query_row(
                "SELECT stub_json FROM stubs WHERE id = ?1",
                params![id.0],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => CawError::NotFound(id.0.clone()),
                _ => CawError::VectorStore(format!("Failed to get stub: {}", e)),
            })?;

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
        let mut stmt = self
            .conn
            .prepare("SELECT stub_id, embedding FROM embeddings")
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
