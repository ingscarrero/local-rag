"""Visual document retrieval with ColPali / ColQwen2.

Instead of OCR-then-chunk, ColPali embeds *page images* directly into a set of
patch-level vectors (late interaction, like ColBERT for images). Retrieval scores
a query's token embeddings against every page's patch embeddings via MaxSim. This
shines on figure/table/chart-heavy pages where OCR pipelines lose layout and visual
structure.

Embeddings are persisted to disk as a list of variable-length tensors plus a JSON
manifest mapping each entry to its (doc_id, page_number, png_path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import torch
from PIL import Image

from ..config import settings

_EMB_FILE = "page_embeddings.pt"
_MANIFEST = "manifest.json"


@dataclass
class VisualHit:
    doc_id: str
    page_number: int
    png_path: str
    score: float


@lru_cache(maxsize=1)
def _load_model():
    """Lazily load ColQwen2 once per process (heavy: multi-GB, GPU/MPS resident)."""
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    device = settings.colpali_device
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model = ColQwen2.from_pretrained(
        settings.colpali_model,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(settings.colpali_model)
    return model, processor


class ColPaliStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._dir = settings.colpali_dir
        self._emb_path = self._dir / _EMB_FILE
        self._manifest_path = self._dir / _MANIFEST
        self._embeddings: list[torch.Tensor] = []
        self._manifest: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._emb_path.exists() and self._manifest_path.exists():
            self._embeddings = torch.load(self._emb_path, map_location="cpu")
            self._manifest = json.loads(self._manifest_path.read_text())

    def _save(self) -> None:
        torch.save(self._embeddings, self._emb_path)
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def index_pages(self, records: list[dict], batch_size: int = 4) -> int:
        """Embed and store page images.

        records: [{"doc_id": str, "page_number": int, "png_path": str}, ...]
        """
        if not records:
            return 0
        model, processor = _load_model()
        existing = {(m["doc_id"], m["page_number"]) for m in self._manifest}
        todo = [r for r in records if (r["doc_id"], r["page_number"]) not in existing]
        if not todo:
            return 0

        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            images = [Image.open(r["png_path"]).convert("RGB") for r in batch]
            inputs = processor.process_images(images).to(model.device)
            with torch.no_grad():
                embs = model(**inputs)  # [B, num_patches, dim]
            for r, emb in zip(batch, embs):
                self._embeddings.append(emb.to(torch.float32).cpu())
                self._manifest.append(r)
        self._save()
        return len(todo)

    def query(self, query: str, top_k: int = 3) -> list[VisualHit]:
        if not self._embeddings:
            return []
        model, processor = _load_model()
        inputs = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            q_emb = model(**inputs)
        # score_multi_vector handles variable-length page embeddings (MaxSim).
        page_embs = [e.to(model.device) for e in self._embeddings]
        scores = processor.score_multi_vector(q_emb, page_embs)[0]  # [num_pages]
        k = min(top_k, len(self._manifest))
        top = torch.topk(scores, k=k)
        hits: list[VisualHit] = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            m = self._manifest[idx]
            hits.append(
                VisualHit(
                    doc_id=m["doc_id"],
                    page_number=m["page_number"],
                    png_path=m["png_path"],
                    score=float(score),
                )
            )
        return hits

    def count(self) -> int:
        return len(self._manifest)
