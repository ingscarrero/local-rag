# ADR-0003: ChromaDB as the text/caption vector store, with caller-supplied embeddings

- **Status:** Accepted
- **Date:** 2026-06-11
- **Affects:** `retrieval/text_store.py`, `config.Settings.chroma_dir`

## Context

Text chunks and figure captions need a persistent, local similarity index with
metadata filtering, for a corpus of tens to low hundreds of PDFs (thousands of
chunks). It must be pip-installable, need no server, and let us pass in
embeddings from the local model rather than computing its own.

Options considered:

| Option | Why not |
|--------|---------|
| **FAISS** | fast, but no persistence or metadata story out of the box — we would be writing a document store around it |
| **Qdrant / Milvus / Weaviate** | server processes (or Docker) for a single-user CLI; overkill until the corpus is orders of magnitude larger |
| **pgvector** | needs PostgreSQL; same objection |
| **SQLite-vec / LanceDB** | viable; less mature metadata/query API at the time, smaller ecosystem |
| **ChromaDB (persistent client)** | embedded, single directory on disk, cosine HNSW, metadata dicts, `upsert` by id, accepts caller-supplied embeddings |

## Decision

ChromaDB `PersistentClient` at `STORAGE_DIR/chroma`, one collection
(`documents`) with `hnsw:space=cosine`. The application always passes its own
embeddings (`models.embed`) for both documents and queries, so Chroma never
loads its default embedding function and the same local model is used
everywhere. Records are distinguished by `metadata.kind ∈ {text, caption}` and
keyed by `"{doc_id}::{kind}::{index}"`, so re-ingestion is an upsert.

## Consequences

- **Positive:** zero operational footprint — `rm -rf storage/chroma` is the
  whole reset procedure. Tests run against a temp directory in milliseconds.
- **Positive:** one query spans prose and figure captions; `kind` is available
  for filtering or display.
- **Negative:** embeddings must be homogeneous. Swapping the embedding model
  without wiping the collection produces a dimension-mismatch error
  ([04 — Setup › Troubleshooting](../04-setup.md#troubleshooting)).
- **Negative:** Chroma's typing is loose (`Optional` result fields, `Mapping`
  metadata); `TextStore` narrows them so the public API stays `list[dict]`.
- **Negative:** Chroma sends anonymised telemetry by default; disabled in CI
  (`ANONYMIZED_TELEMETRY=False`) and documented in
  [00 — Requirements › NFR-S3](../00-requirements.md#privacy-and-security).
- **Migration point:** single-node Chroma is comfortable to ~10 M vectors. The
  `TextStore` interface (`add / query / count / reset`) is the seam for
  Qdrant/pgvector later ([06 — System design › Capacity](../06-system-design.md#capacity-and-scaling)).
