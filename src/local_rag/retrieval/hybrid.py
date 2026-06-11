"""Hybrid retrieval: run the text store and the ColPali visual store together."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from .text_store import TextStore, TextHit
from .colpali_store import ColPaliStore, VisualHit


@dataclass
class Retrievers:
    text: TextStore
    visual: ColPaliStore | None = None

    @classmethod
    def load(cls, use_visual: bool = True) -> "Retrievers":
        return cls(
            text=TextStore(),
            visual=ColPaliStore() if use_visual else None,
        )

    def retrieve(
        self, query: str, text_k: int | None = None, visual_k: int | None = None
    ) -> tuple[list[TextHit], list[VisualHit]]:
        text_hits = self.text.query(query, top_k=text_k or settings.text_top_k)
        visual_hits: list[VisualHit] = []
        if self.visual is not None and self.visual.count() > 0:
            visual_hits = self.visual.query(query, top_k=visual_k or settings.visual_top_k)
        return text_hits, visual_hits
