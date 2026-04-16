use caw_core::{CawError, CawResult, ConsolidationNote, ConsolidationSource, Stub, StubId, StubStore};
use rusqlite::{params, Connection};

pub struct SqliteStubStore {
    conn: Connection,
    dimension: usize,
}

impl SqliteStubStore {
    pub fn new(path: &str, dimension: usize) -> CawResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| CawError::VectorStore(format!("Failed to open database: {}", e)))?;

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
                stub_json TEXT NOT NULL
            )",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create stubs table: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS contents (
                stub_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                FOREIGN KEY(stub_id) REFERENCES stubs(id)
            )",
            [],
        )
        .map_err(|e| CawError::VectorStore(format!("Failed to create contents table: {}", e)))?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                stub_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY(stub_id) REFERENCES stubs(id)
            )",
            [],
        )
        .map_err(|e| {
            CawError::VectorStore(format!("Failed to create embeddings table: {}", e))
        })?;

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

        Ok(Self { conn, dimension })
    }

    pub fn in_memory(dimension: usize) -> CawResult<Self> {
        Self::new(":memory:", dimension)
    }

    fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
        embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

impl StubStore for SqliteStubStore {
    fn insert(&mut self, stub: Stub, embedding: Vec<f32>, content: String) -> CawResult<()> {
        if embedding.len() != self.dimension {
            return Err(CawError::VectorStore(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        let stub_json = serde_json::to_string(&stub)
            .map_err(|e| CawError::VectorStore(format!("Failed to serialize stub: {}", e)))?;

        let kind_str = format!("{:?}", stub.kind);
        let outline_json = serde_json::to_string(&stub.outline)
            .map_err(|e| CawError::VectorStore(format!("Failed to serialize outline: {}", e)))?;

        self.conn
            .execute(
                "INSERT OR REPLACE INTO stubs (id, path, token_estimate, kind, summary, outline, content_hash, mtime_unix_secs, stub_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    stub.id.0,
                    stub.path,
                    stub.token_estimate as i64,
                    kind_str,
                    stub.summary,
                    outline_json,
                    stub.content_hash,
                    stub.mtime_unix_secs as i64,
                    stub_json
                ],
            )
            .map_err(|e| CawError::VectorStore(format!("Failed to insert stub: {}", e)))?;

        self.conn
            .execute(
                "INSERT OR REPLACE INTO contents (stub_id, content) VALUES (?1, ?2)",
                params![stub.id.0, content],
            )
            .map_err(|e| CawError::VectorStore(format!("Failed to insert content: {}", e)))?;

        let embedding_blob = Self::embedding_to_blob(&embedding);
        self.conn
            .execute(
                "INSERT OR REPLACE INTO embeddings (stub_id, embedding) VALUES (?1, ?2)",
                params![stub.id.0, embedding_blob],
            )
            .map_err(|e| CawError::VectorStore(format!("Failed to insert embedding: {}", e)))?;

        Ok(())
    }

    fn get_content(&self, id: &StubId) -> CawResult<String> {
        self.conn
            .query_row(
                "SELECT content FROM contents WHERE stub_id = ?1",
                params![id.0],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => CawError::NotFound(id.0.clone()),
                _ => CawError::VectorStore(format!("Failed to get content: {}", e)),
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
                let stub: Stub = serde_json::from_str(&json)
                    .map_err(|e| CawError::VectorStore(format!("Failed to deserialize stub: {}", e)))?;
                let embedding = Self::blob_to_embedding(&blob);
                Ok(Some((stub, embedding)))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(CawError::VectorStore(format!("Failed to query by content hash: {}", e))),
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
                params![stub_id.0, note.content, source_str, note.created_at_secs as i64],
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
