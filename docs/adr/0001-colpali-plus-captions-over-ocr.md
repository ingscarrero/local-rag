# ADR-0001: ColPali page embeddings plus VLM captions, instead of OCR-then-chunk

- **Status:** Accepted
- **Date:** 2026-06-11
- **Affects:** `retrieval/colpali_store.py`, `models.caption_image`, `ingest/pipeline.py`, `retrieval/hybrid.py`

## Context

Technical PDFs put much of their information in charts, tables and diagrams.
A text-only RAG pipeline never sees it, and an OCR pipeline linearises a chart
into a soup of axis labels that loses the structure the answer depends on. The
goal (see [06 — System design › Goals](../06-system-design.md#goals)) is to find
the *right page* for a visual question and to give the LLM something it can
*read* about the figure.

Three options were on the table:

1. **OCR (or PyMuPDF text) only.** Zero extra models; blind to figures.
2. **Caption-and-embed only.** A local vision-LLM describes each extracted
   figure; the caption is embedded as text. Searchable and readable, but capped
   by the VLM's description quality, and it only sees *embedded* images — a
   table drawn with vector primitives has no image to caption.
3. **ColPali / ColQwen2 only.** Render each page and embed it as patch-level
   multi-vectors (late interaction). Finds the right page without OCR, robust
   to layout, but returns a *page*, not prose to quote, and costs hundreds of
   vectors per page.

## Decision

Do **both 2 and 3**, and keep them as *separate* signals rather than fusing
them into one score:

- Captions go into the same ChromaDB collection as the prose (`kind="caption"`),
  so one cosine query spans text and figures, and the caption text becomes
  quotable, citable context for `generate`.
- ColQwen2 page embeddings live in their own on-disk index and are queried in
  parallel. Their hits are passed to `generate` as "visually relevant pages",
  i.e. *where to look*.

`Retrievers.retrieve` returns both lists unchanged; the agent decides what to
do with each ([03 — Architecture › The agent graph](../03-architecture.md#the-agent-graph)).

## Consequences

- **Positive:** the two methods fail differently, so the combination is more
  robust than either. A wrong caption is compensated by ColPali finding the
  page; a page ColPali scores ambiguously is compensated by a precise caption.
  Vector-drawn tables are covered by ColPali.
- **Positive:** the ingestion pipeline degrades gracefully — `--no-captions`
  and `--no-visual` each remove one path without touching the other.
- **Negative:** two extra models (a VLM and ColQwen2, ≈ 10 GB of downloads) and
  a torch dependency in the application process. Tests replace ColQwen2 with a
  fake ([tests/test_colpali_store.py](../../tests/test_colpali_store.py)).
- **Negative:** brute-force MaxSim is O(pages) per query. Acceptable for the
  target corpus; the migration point to a multi-vector ANN store is documented
  in [06 — System design › Capacity](../06-system-design.md#capacity-and-scaling).
- **Follow-up:** the ColQwen2 adapter can silently fail to load — see
  [ADR-0005](0005-pin-transformers-4x.md).
