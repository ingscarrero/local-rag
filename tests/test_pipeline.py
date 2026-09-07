"""Ingestion pipeline: parse → chunk → caption → index, with fakes at the store seams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_rag import models
from local_rag.config import settings
from local_rag.ingest import pipeline
from local_rag.ingest.pipeline import ingest_folder, ingest_pdf

from .test_pdf import _make_pdf


class RecordingTextStore:
    def __init__(self) -> None:
        self.adds: list[dict[str, Any]] = []

    def add(
        self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: Any = None
    ) -> None:
        self.adds.append({"ids": ids, "texts": texts, "metadatas": metadatas})


class RecordingColPaliStore:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def index_pages(self, records: list[dict], batch_size: int = 4) -> int:
        self.records.extend(records)
        return len(records)


@pytest.fixture
def pdf(tmp_storage: Path) -> Path:
    return _make_pdf(settings.data_dir / "sample.pdf")


@pytest.fixture
def captioner(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    seen: list[bytes] = []

    def fake_caption(image_bytes: bytes, prompt: str | None = None) -> str:
        seen.append(image_bytes)
        return "A red square figure."

    monkeypatch.setattr(models, "caption_image", fake_caption)
    return seen


def test_ingest_pdf_indexes_text_chunks_captions_and_pages(
    pdf: Path, captioner: list[bytes]
) -> None:
    text_store, colpali = RecordingTextStore(), RecordingColPaliStore()
    stats = ingest_pdf(pdf, text_store, colpali)

    assert stats["source"] == "sample.pdf" and stats["pages"] == 2
    assert stats["text_items"] == 2 and stats["captions"] == 1 and stats["colpali_pages"] == 2
    assert len(captioner) == 1

    add = text_store.adds[0]
    doc_id = stats["doc_id"]
    assert add["ids"] == [
        f"{doc_id}::text::0",
        f"{doc_id}::text::1",
        f"{doc_id}::caption::p2-x{_xref(add)}",
    ]
    assert add["texts"][2] == "[Figure on page 2] A red square figure."
    assert [m["kind"] for m in add["metadatas"]] == ["text", "text", "caption"]
    assert [m["page_number"] for m in add["metadatas"]] == [1, 2, 2]
    assert all(m["doc_id"] == doc_id and m["source"] == "sample.pdf" for m in add["metadatas"])
    assert add["metadatas"][2]["image_id"].startswith("p2-x")

    assert [r["page_number"] for r in colpali.records] == [1, 2]
    assert all(r["doc_id"] == doc_id and Path(r["png_path"]).exists() for r in colpali.records)
    assert all(str(settings.page_image_dir) in r["png_path"] for r in colpali.records)


def _xref(add: dict[str, Any]) -> str:
    return add["metadatas"][2]["image_id"].split("x")[1]


def test_chunk_indices_are_contiguous_across_pages(pdf: Path, captioner: list[bytes]) -> None:
    text_store = RecordingTextStore()
    ingest_pdf(pdf, text_store, None, caption_images=False)
    ids = text_store.adds[0]["ids"]
    assert [i.split("::")[-1] for i in ids] == ["0", "1"]


def test_caption_images_false_skips_the_vlm(pdf: Path, captioner: list[bytes]) -> None:
    text_store = RecordingTextStore()
    stats = ingest_pdf(pdf, text_store, None, caption_images=False)
    assert captioner == []
    assert stats["captions"] == 0 and stats["colpali_pages"] == 0
    assert all(m["kind"] == "text" for m in text_store.adds[0]["metadatas"])


def test_caption_failure_is_logged_and_skipped(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_caption(image_bytes: bytes, prompt: str | None = None) -> str:
        raise ConnectionError("vlm server down")

    monkeypatch.setattr(models, "caption_image", failing_caption)
    text_store = RecordingTextStore()
    stats = ingest_pdf(pdf, text_store, None)
    assert stats["captions"] == 0 and stats["text_items"] == 2, (
        "text indexing survives a VLM outage"
    )


def test_empty_caption_is_not_indexed(pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "caption_image", lambda image_bytes, prompt=None: "   ")
    text_store = RecordingTextStore()
    stats = ingest_pdf(pdf, text_store, None)
    assert stats["captions"] == 0
    assert len(text_store.adds[0]["ids"]) == 2


def test_ingest_folder_with_no_pdfs_returns_empty(
    tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "TextStore", RecordingTextStore)
    assert ingest_folder(settings.data_dir) == []


def test_ingest_folder_walks_recursively_in_sorted_order(
    tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (settings.data_dir / "nested").mkdir()
    b = _make_pdf(settings.data_dir / "b.pdf", big_image=False, tiny_image=False)
    a = _make_pdf(settings.data_dir / "nested" / "a.pdf", big_image=False, tiny_image=False)
    monkeypatch.setattr(pipeline, "TextStore", RecordingTextStore)
    monkeypatch.setattr(pipeline, "ColPaliStore", RecordingColPaliStore)
    calls: list[tuple[Path, Any, Any, bool]] = []

    def fake_ingest_pdf(path: Path, text_store: Any, colpali: Any, *, caption_images: bool) -> dict:
        calls.append((path, text_store, colpali, caption_images))
        return {"source": path.name}

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest_pdf)

    results = ingest_folder(settings.data_dir, caption_images=False, use_colpali=True)
    assert results == [{"source": "b.pdf"}, {"source": "a.pdf"}]
    assert [c[0] for c in calls] == sorted([b, a])
    assert all(isinstance(c[1], RecordingTextStore) for c in calls)
    assert all(isinstance(c[2], RecordingColPaliStore) for c in calls)
    assert calls[0][1] is calls[1][1], "one shared text store per folder run"
    assert all(c[3] is False for c in calls)

    calls.clear()
    ingest_folder(settings.data_dir, use_colpali=False)
    assert all(c[2] is None for c in calls)
