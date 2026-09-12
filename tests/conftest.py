"""Shared fixtures.

Nothing here talks to a model server. The heavy pieces (ChromaDB, torch,
LangGraph) are real, but every LLM / embedding / vision call goes through a
scripted fake of the OpenAI-compatible client so the tests are hermetic and
fast.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Mirror the dummy values CI exports so Settings() never depends on a .env file.
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("LLM_MODEL", "local-chat")
os.environ.setdefault("EMBED_BASE_URL", "http://127.0.0.1:8081/v1")
os.environ.setdefault("EMBED_MODEL", "local-embed")
os.environ.setdefault("VLM_BASE_URL", "http://127.0.0.1:8082/v1")
os.environ.setdefault("VLM_MODEL", "local-vlm")
os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from local_rag import models  # noqa: E402
from local_rag.config import settings  # noqa: E402
from local_rag.retrieval.colpali_store import VisualHit  # noqa: E402
from local_rag.retrieval.text_store import TextHit  # noqa: E402


# ── Fake OpenAI-compatible client ─────────────────────────────────────────────
@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _ChatResponse:
    choices: list[_Choice]


@dataclass
class _EmbeddingDatum:
    embedding: list[float]


@dataclass
class _EmbeddingResponse:
    data: list[_EmbeddingDatum]


def fake_embedding(text: str, dim: int = 32) -> list[float]:
    """Deterministic, content-dependent unit vector so cosine ranking is meaningful.

    Tokens hash (stably, across processes) into buckets; identical texts embed
    identically and texts that share words land closer together than unrelated ones.
    """
    vec = [0.0] * dim
    for word in text.lower().split():
        bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:2], "big") % dim
        vec[bucket] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@dataclass
class FakeOpenAIClient:
    """Scripted stand-in for ``openai.OpenAI``.

    ``replies`` is consumed in order by ``chat.completions.create``; when it runs
    out, ``default_reply`` is returned. Every request is recorded in ``calls``.
    """

    replies: list[str | None] = field(default_factory=list)
    default_reply: str = '{"route": "retrieve"}'
    calls: list[dict[str, Any]] = field(default_factory=list)
    embed_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        client = self

        class _Completions:
            def create(self, **kwargs: Any) -> _ChatResponse:
                client.calls.append(kwargs)
                reply = client.replies.pop(0) if client.replies else client.default_reply
                return _ChatResponse(choices=[_Choice(message=_Message(content=reply))])

        class _Chat:
            completions = _Completions()

        class _Embeddings:
            def create(self, **kwargs: Any) -> _EmbeddingResponse:
                client.embed_calls.append(kwargs)
                inputs = kwargs["input"]
                return _EmbeddingResponse(
                    data=[_EmbeddingDatum(embedding=fake_embedding(t)) for t in inputs]
                )

        self.chat = _Chat()
        self.embeddings = _Embeddings()

    # Convenience for tests that inspect prompts.
    def last_messages(self) -> list[dict[str, Any]]:
        return self.calls[-1]["messages"]

    def system_prompts(self) -> list[str]:
        return [m["content"] for c in self.calls for m in c["messages"] if m["role"] == "system"]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAIClient:
    """Route every ``models._client(base_url)`` call to one scripted fake."""
    client = FakeOpenAIClient()
    monkeypatch.setattr(models, "_client", lambda base_url: client)
    return client


# ── Isolated storage ──────────────────────────────────────────────────────────
@pytest.fixture
def tmp_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every storage path at a fresh temp dir so tests never touch ./storage."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    settings.ensure_dirs()
    return tmp_path


# ── Hit factories ─────────────────────────────────────────────────────────────
def make_text_hit(
    text: str = "some passage",
    *,
    page: int = 1,
    kind: str = "text",
    source: str = "doc.pdf",
    score: float = 0.9,
    hit_id: str | None = None,
) -> TextHit:
    return TextHit(
        id=hit_id or f"doc::{kind}::{page}",
        text=text,
        score=score,
        metadata={"kind": kind, "doc_id": "doc-abc", "source": source, "page_number": page},
    )


def make_visual_hit(page: int = 1, score: float = 10.0, doc_id: str = "doc-abc") -> VisualHit:
    return VisualHit(
        doc_id=doc_id, page_number=page, png_path=f"/tmp/{doc_id}/page-{page:04d}.png", score=score
    )


class FakeTextStore:
    """Duck-typed TextStore: returns canned hits, records the queries it saw."""

    def __init__(self, hits: list[TextHit] | None = None) -> None:
        self.hits = hits or []
        self.queries: list[tuple[str, int]] = []

    def query(self, query: str, top_k: int = 5) -> list[TextHit]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]

    def count(self) -> int:
        return len(self.hits)


class FakeColPaliStore:
    """Duck-typed ColPaliStore with a controllable page count."""

    def __init__(self, hits: list[VisualHit] | None = None) -> None:
        self.hits = hits or []
        self.queries: list[tuple[str, int]] = []

    def query(self, query: str, top_k: int = 3) -> list[VisualHit]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]

    def count(self) -> int:
        return len(self.hits)
