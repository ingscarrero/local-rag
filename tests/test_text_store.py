"""TextStore against a real, temp-dir ChromaDB — only the embedder is faked.

The important assertion is not "a result comes back" but that the *right*
chunk wins the cosine ranking (see CONTRIBUTING: keep retrieval honest).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_rag import models
from local_rag.config import settings
from local_rag.retrieval.text_store import TextStore

# Orthogonal unit vectors keyed by text; the query is nudged toward one of them.
_VECTORS: dict[str, list[float]] = {
    "naive rag has problems": [1.0, 0.0, 0.0, 0.0],
    "[Figure on page 2] bar chart of throughput": [0.0, 1.0, 0.0, 0.0],
    "appendix boilerplate": [0.0, 0.0, 1.0, 0.0],
    "what does the chart show": [0.1, 0.95, 0.0, 0.0],
    "problems of naive rag": [0.95, 0.1, 0.0, 0.0],
}


@pytest.fixture
def store(tmp_storage: Path, monkeypatch: pytest.MonkeyPatch) -> TextStore:
    monkeypatch.setattr(models, "embed", lambda texts: [_VECTORS[t] for t in texts])
    return TextStore()


def _seed(store: TextStore, embeddings: bool = True) -> None:
    texts = list(_VECTORS)[:3]
    store.add(
        ids=["d::text::0", "d::caption::p2-x7", "d::text::1"],
        texts=texts,
        metadatas=[
            {"kind": "text", "doc_id": "d", "source": "s.pdf", "page_number": 1},
            {
                "kind": "caption",
                "doc_id": "d",
                "source": "s.pdf",
                "page_number": 2,
                "image_id": "p2-x7",
            },
            {"kind": "text", "doc_id": "d", "source": "s.pdf", "page_number": 3},
        ],
        embeddings=[_VECTORS[t] for t in texts] if embeddings else None,
    )


def test_fresh_store_is_empty(store: TextStore) -> None:
    assert store.count() == 0
    assert store.query("what does the chart show") == []


def test_add_then_count(store: TextStore) -> None:
    _seed(store)
    assert store.count() == 3


def test_add_with_no_ids_is_a_noop(store: TextStore) -> None:
    store.add(ids=[], texts=[], metadatas=[])
    assert store.count() == 0


def test_add_embeds_locally_when_no_embeddings_are_given(
    store: TextStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return [_VECTORS[t] for t in texts]

    monkeypatch.setattr(models, "embed", recording_embed)
    _seed(store, embeddings=False)
    assert seen == [list(_VECTORS)[:3]]
    assert store.count() == 3


def test_the_right_chunk_wins_the_ranking(store: TextStore) -> None:
    _seed(store)
    hits = store.query("what does the chart show", top_k=3)
    assert [h.id for h in hits][0] == "d::caption::p2-x7"
    assert hits[0].metadata["kind"] == "caption" and hits[0].metadata["page_number"] == 2
    assert hits[0].text.startswith("[Figure on page 2]")
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].score == pytest.approx(0.95 / (0.1**2 + 0.95**2) ** 0.5, abs=1e-3)

    hits = store.query("problems of naive rag", top_k=1)
    assert [h.id for h in hits] == ["d::text::0"]
    assert hits[0].metadata["page_number"] == 1


def test_top_k_limits_results(store: TextStore) -> None:
    _seed(store)
    assert len(store.query("what does the chart show", top_k=2)) == 2


def test_upsert_is_idempotent_by_id(store: TextStore) -> None:
    _seed(store)
    _seed(store)
    assert store.count() == 3, "re-ingesting the same ids must overwrite, not duplicate"


def test_reset_empties_the_collection_but_keeps_it_usable(store: TextStore) -> None:
    _seed(store)
    store.reset()
    assert store.count() == 0
    _seed(store)
    assert store.count() == 3


def test_data_persists_across_instances(store: TextStore) -> None:
    _seed(store)
    again = TextStore()
    assert again.count() == 3
    assert settings.chroma_dir.exists()
