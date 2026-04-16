use caw_core::{CawError, CawResult, ScoredStub, Stub, StubId, VectorStore};
use rusqlite::{params, Connection};
use serde_json;

pub struct SqliteVectorStore {
    conn: Connection,
    dimension: usize,
}

impl SqliteVectorStore {
    pub fn new(path: &str, dimension: usize) -> CawResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| CawError::VectorStore(format!("Failed to open database: {}", e)))?;

        // Create tables
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

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl VectorStore for SqliteVectorStore {
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

    fn search_by_embedding(&self, query_embedding: &[f32], top_k: usize) -> CawResult<Vec<ScoredStub>> {
        if query_embedding.len() != self.dimension {
            return Err(CawError::VectorStore(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                query_embedding.len()
            )));
        }

        let mut stmt = self
            .conn
            .prepare("SELECT stub_id, embedding, stub_json FROM embeddings e JOIN stubs s ON e.stub_id = s.id")
            .map_err(|e| CawError::VectorStore(format!("Failed to prepare query: {}", e)))?;

        let mut results: Vec<ScoredStub> = stmt
            .query_map([], |row| {
                let embedding_blob: Vec<u8> = row.get(1)?;
                let stub_json: String = row.get(2)?;
                Ok((embedding_blob, stub_json))
            })
            .map_err(|e| CawError::VectorStore(format!("Query failed: {}", e)))?
            .filter_map(|result| result.ok())
            .filter_map(|(embedding_blob, stub_json)| {
                let embedding = Self::blob_to_embedding(&embedding_blob);
                let stub: Stub = serde_json::from_str(&stub_json).ok()?;
                let score = Self::cosine_similarity(query_embedding, &embedding);
                Some(ScoredStub { stub, score })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
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
}
