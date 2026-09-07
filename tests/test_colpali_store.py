"""ColPaliStore with the ColQwen2 model swapped for a tiny colour-matching fake.

Real torch tensors, real MaxSim, real on-disk persistence — only the multi-GB
model is replaced, so no weights are downloaded. The fake maps a page image to
its dominant colour and a query string to a colour name, which is enough to
assert that the *right page wins the ranking* rather than merely that a result
comes back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from local_rag.config import settings
from local_rag.retrieval import colpali_store
from local_rag.retrieval.colpali_store import ColPaliStore, VisualHit

_COLOURS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}


class _Batch(dict):
    def to(self, device: str) -> "_Batch":
        return self


class FakeColQwen2:
    device = "cpu"

    def __init__(self) -> None:
        self.image_calls = 0
        self.query_calls = 0

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        if "pixel_values" in inputs:
            self.image_calls += 1
            # [B, 3] colour → [B, 4 patches, 3] multi-vector (4 identical patches)
            return inputs["pixel_values"].unsqueeze(1).repeat(1, 4, 1)
        self.query_calls += 1
        return inputs["query_vec"].unsqueeze(1)  # [1, 1 token, 3]


class FakeProcessor:
    def process_images(self, images: list[Image.Image]) -> _Batch:
        vecs = []
        for im in images:
            r, g, b = im.resize((1, 1)).getpixel((0, 0))[:3]
            vecs.append(torch.tensor([r, g, b], dtype=torch.float32) / 255.0)
        return _Batch(pixel_values=torch.stack(vecs))

    def process_queries(self, queries: list[str]) -> _Batch:
        colour = next(c for c in _COLOURS if c in queries[0].lower())
        return _Batch(query_vec=torch.tensor([_COLOURS[colour]], dtype=torch.float32))

    @staticmethod
    def score_multi_vector(q: torch.Tensor, pages: list[torch.Tensor]) -> torch.Tensor:
        """Genuine MaxSim: sum over query tokens of the max similarity over page patches."""
        scores = [(q[0] @ p.T).max(dim=1).values.sum() for p in pages]
        return torch.stack(scores).unsqueeze(0)  # [1, num_pages]


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeColQwen2:
    model, processor = FakeColQwen2(), FakeProcessor()
    monkeypatch.setattr(colpali_store, "_load_model", lambda: (model, processor))
    return model


def _page_png(dir_: Path, name: str, colour: tuple[float, float, float]) -> str:
    p = dir_ / f"{name}.png"
    Image.new("RGB", (20, 20), tuple(int(c * 255) for c in colour)).save(p)
    return str(p)


@pytest.fixture
def records(tmp_storage: Path) -> list[dict]:
    d = settings.page_image_dir
    return [
        {"doc_id": "doc", "page_number": i + 1, "png_path": _page_png(d, name, rgb)}
        for i, (name, rgb) in enumerate(_COLOURS.items())
    ]


def test_fresh_store_has_no_pages_and_never_loads_the_model(
    tmp_storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> None:
        raise AssertionError("model must not load for an empty index")

    monkeypatch.setattr(colpali_store, "_load_model", boom)
    store = ColPaliStore()
    assert store.count() == 0
    assert store.query("red chart") == []
    assert store.index_pages([]) == 0


def test_index_pages_embeds_persists_and_reports_count(
    records: list[dict], fake_model: FakeColQwen2
) -> None:
    store = ColPaliStore()
    assert store.index_pages(records, batch_size=2) == 3
    assert store.count() == 3
    assert fake_model.image_calls == 2, "3 pages at batch_size=2 → 2 forward passes"
    assert (settings.colpali_dir / "page_embeddings.pt").exists()
    manifest = json.loads((settings.colpali_dir / "manifest.json").read_text())
    assert manifest == records
    assert all(e.dtype == torch.float32 and e.device.type == "cpu" for e in store._embeddings)
    assert all(e.shape == (4, 3) for e in store._embeddings)


def test_index_pages_skips_pages_already_indexed(
    records: list[dict], fake_model: FakeColQwen2
) -> None:
    store = ColPaliStore()
    store.index_pages(records[:2])
    assert store.index_pages(records) == 1, "only the new page is embedded"
    assert store.count() == 3
    assert store.index_pages(records) == 0
    assert fake_model.image_calls == 2


def test_the_right_page_wins_maxsim(records: list[dict], fake_model: FakeColQwen2) -> None:
    store = ColPaliStore()
    store.index_pages(records)
    for colour, page in (("red", 1), ("green", 2), ("blue", 3)):
        hits = store.query(f"a {colour} figure", top_k=3)
        assert [h.page_number for h in hits][0] == page, colour
        assert hits[0].score > hits[1].score
        assert hits[0].doc_id == "doc" and hits[0].png_path == records[page - 1]["png_path"]
        assert isinstance(hits[0], VisualHit) and isinstance(hits[0].score, float)


def test_top_k_is_clamped_to_the_number_of_pages(
    records: list[dict], fake_model: FakeColQwen2
) -> None:
    store = ColPaliStore()
    store.index_pages(records)
    assert len(store.query("red", top_k=10)) == 3
    assert len(store.query("red", top_k=1)) == 1


def test_index_is_reloaded_from_disk_by_a_new_instance(
    records: list[dict], fake_model: FakeColQwen2
) -> None:
    ColPaliStore().index_pages(records)
    reloaded = ColPaliStore()
    assert reloaded.count() == 3
    assert reloaded.query("blue")[0].page_number == 3


def test_partial_files_on_disk_are_ignored(records: list[dict], tmp_storage: Path) -> None:
    (settings.colpali_dir / "manifest.json").write_text("[]")  # no page_embeddings.pt
    assert ColPaliStore().count() == 0
