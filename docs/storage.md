# Stub Storage and Vector Index

*Status: Living reference — kept current. See [docs/README.md](README.md) for the docs index.*

The `StubStore` trait handles durability (stubs, embeddings, content, consolidation notes). The `VectorIndex` trait handles similarity search. They are separate so the storage layer can evolve independently of the search layer.

Note: file content is stored as byte-range pointers back into the source, not duplicated in the database.

## SQLite (default)

Single-file database. In-memory variant available for tests. WAL mode, `synchronous=NORMAL`, prepared-statement batch inserts, staleness detection with a reindex queue. Works well up to ~100k stubs without special tuning.

```rust
use caw_index::SqliteStubStore;

let store = SqliteStubStore::new("index.db", 384)?;       // 384 = embedding dimension
let store = SqliteStubStore::in_memory(384)?;             // for tests
```

Consolidation notes persist across sessions via `save_consolidation()` / `load_consolidation()`.

## Qdrant (feature-gated)

Full `qdrant_client` integration with payload indexes. For deployments that outgrow the SQLite store. Enable with `--features qdrant`.

```rust
use caw_index::QdrantStubStore;

let store = QdrantStubStore::local("collection_name")?;
```

## Vector Index

`HnswVectorIndex` via `instant-distance` is the default similarity search layer. Pairs with either `SqliteStubStore` or `QdrantStubStore`.

`FlatVectorIndex` is available for small collections or testing (linear scan, no build overhead).

```rust
use caw_index::{HnswVectorIndex, FlatVectorIndex};

let index = HnswVectorIndex::new();
let index = FlatVectorIndex::new();
```

## Hybrid Retrieval

`HybridRetriever` combines semantic (vector) and BM25 keyword search with min-max normalized score fusion. Default weights: 0.6 semantic / 0.4 BM25. Configurable.

```rust
use caw_index::HybridRetriever;

let retriever = HybridRetriever::new(embedder, store, vector_index);
```
