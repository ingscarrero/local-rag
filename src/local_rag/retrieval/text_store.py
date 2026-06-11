"""Text vector store backed by ChromaDB.

Holds two kinds of documents in one collection, distinguished by metadata:
  - kind="text"    : a chunk of page text
  - kind="caption" : a vision-LLM description of an embedded figure

Both are embedded with the same local text-embedding model, so a single
similarity search retrieves prose and figure descriptions together.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from ..config import settings
from .. import models

_COLLECTION = "documents"


@dataclass
class TextHit:
    id: str
    text: str
    score: float  # cosine similarity in [−1, 1]; higher is better
    metadata: dict


class TextStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        # We supply our own embeddings (local model), so no embedding_function here.
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if not ids:
            return
        if embeddings is None:
            embeddings = models.embed(texts)
        self._col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query: str, top_k: int = 5) -> list[TextHit]:
        q_emb = models.embed([query])[0]
        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[TextHit] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(TextHit(id=_id, text=doc, score=1.0 - dist, metadata=meta or {}))
        return hits

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(_COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
