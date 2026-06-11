"""Text chunking.

Word-window chunking with overlap. Kept deliberately simple and transparent:
the agent's grading + re-query loop is what compensates for imperfect chunking,
which is the whole point of *agentic* RAG over hand-tuned splitters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_page_text(
    text: str,
    page_number: int,
    *,
    words_per_chunk: int = 220,
    overlap: int = 40,
    start_index: int = 0,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, words_per_chunk - overlap)
    idx = start_index
    for start in range(0, len(words), step):
        window = words[start : start + words_per_chunk]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), page_number=page_number, chunk_index=idx))
        idx += 1
        if start + words_per_chunk >= len(words):
            break
    return chunks
