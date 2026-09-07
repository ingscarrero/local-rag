# 06 — System Design

[03 — Architecture](03-architecture.md) describes **what** the system is. This
document records **why** it is shaped that way: the goals it optimises for, the
constraints that bound it, the alternatives that were considered and rejected,
where it stops scaling, and how it fails. The individual decisions are recorded
as [ADRs](adr/README.md); this is the connective tissue between them.

## Goals

Ranked — when two conflict, the higher one wins.

1. **Privacy by construction.** Nothing about the documents or the questions
   leaves the machine. Not "configurable to be private" — private with no
   configuration.
2. **Answers that are honest about their evidence.** Every answer is grounded in
   retrieved context, cites page numbers, and the system says "I couldn't find
   it" rather than improvising.
3. **Retrieval that sees figures and tables.** Technical PDFs put their most
   important content in charts. The system must find the right *page* even when
   OCR would mangle it.
4. **Legibility.** A reader should be able to follow the whole control loop from
   a printed trace and from a one-screen graph definition. This is a showcase and
   a teaching artefact as much as a tool.
5. **Reproducibility without a GPU.** Everything except the models must be
   testable in CI on a plain Linux runner.

Explicit **non-goals**: multi-user service, authentication, horizontal scaling,
sub-second latency, leaderboard-grade benchmark numbers.

## Constraints

| Constraint | Consequence |
|------------|-------------|
| Single machine, Apple Silicon first | ColQwen2 runs in-process on MPS; model servers use Metal offload (`-ngl 99`). CUDA/CPU are supported via `COLPALI_DEVICE` but untuned. |
| Small local models (7B–14B class) | Decision prompts must be tiny, return JSON, and be parsed forgivingly. Grade one passage per call, not a batch. Temperature 0 everywhere. |
| `llama-server` is one model per process | Three processes (chat / embed / vision) when using llama.cpp; LM Studio multiplexes them behind one port. Client code is identical either way. |
| Models are multi-GB | Weights are downloaded once and cached; the codebase never bundles them; tests never load them. |
| `transformers` 5.x silently drops ColQwen2's LoRA adapter | Hard pin to `<5.0.0` ([ADR-0005](adr/0005-pin-transformers-4x.md)) and a ranking assertion as the regression test for visual retrieval. |
| Personal / showcase corpus (tens to low hundreds of PDFs) | Brute-force MaxSim and single-node Chroma are acceptable; see *Capacity* for the migration points. |

## Key design decisions (summary)

| Decision | Chosen | Rejected | ADR |
|----------|--------|----------|-----|
| Visual retrieval | ColQwen2 page multi-vectors **plus** VLM captions | OCR-then-chunk only; captions only; ColPali only | [0001](adr/0001-colpali-plus-captions-over-ocr.md) |
| Model serving | Three OpenAI-compatible roles (chat, embed, vision) — three `llama-server` processes or one LM Studio port | one multiplexed llama-server; in-process transformers for everything; a cloud API | [0002](adr/0002-three-model-servers-behind-one-api.md) |
| Text vector store | ChromaDB, persistent, caller-supplied embeddings | FAISS; Qdrant / Milvus / pgvector; SQLite-vec | [0003](adr/0003-chromadb-for-text-index.md) |
| Orchestration | LangGraph `StateGraph` with typed state and conditional edges | hand-rolled loop; LangChain chains; CrewAI / AutoGen | [0004](adr/0004-langgraph-state-machine-over-chain.md) |
| Dependency pin | `transformers>=4.46,<5.0` | float to latest | [0005](adr/0005-pin-transformers-4x.md) |

## Alternatives rejected at the system level

- **A single "do everything" LLM call with the whole document in context.**
  Rejected: a 50-page PDF exceeds practical local context windows, latency is
  proportional to corpus size, and there is no citation granularity. Retrieval
  is the point.
- **Naive RAG (`embed → top-k → generate`).** Rejected: it has no notion of
  retrieval failure and confidently answers from irrelevant chunks. The grading
  node and the rewrite loop exist precisely to attack this
  ([01 — Concepts](01-concepts.md)).
- **Semantic / structure-aware chunking.** Deliberately not done. Fixed 220-word
  windows with 40-word overlap are predictable and testable; the corrective loop
  compensates for boundary misses. Revisit only with evidence from golden
  questions ([02 — Methodology](02-methodology.md#chunking-ingestchunkpy)).
- **A cross-encoder re-ranker.** Deferred, not rejected. It would sit between
  `retrieve` and `grade` and reduce LLM grading calls. Left out to keep the
  local footprint and the graph minimal; it is the first item on the roadmap.
- **Self-RAG reflection tokens.** Rejected in favour of an explicit external
  grader (CRAG): small local models are far more reliable at "is *this one*
  relevant, yes/no" than at critiquing their own generations.
- **A web UI.** Out of scope for the core; a Streamlit `ui` extra is reserved in
  `pyproject.toml` for a later page-image viewer.

## Request lifecycle and where time goes

For `ask` with defaults (`TEXT_TOP_K=5`, `VISUAL_TOP_K=3`,
`MAX_AGENT_ITERATIONS=3`), the LLM-call budget per question is:

| Phase | LLM calls | Notes |
|-------|-----------|-------|
| route | 1 | ≤ 20 output tokens |
| retrieve | 0 | 1 embedding call + 1 ColQwen2 forward + brute-force MaxSim |
| grade | up to 5 per round | ≤ 20 output tokens each; this is the dominant cost |
| rewrite | 1 per failed round | ≤ 64 output tokens |
| generate | 1 | ≤ 768 output tokens |

Common path: **7 calls**. Worst case at the cap: 1 + 3×5 + 2 + 1 = **19 calls**.
The mitigations, in order of payoff: parallel grading (calls are independent),
a smaller model for grading only, and a re-ranker to shrink `TEXT_TOP_K`.

## Capacity and scaling

What breaks first as the corpus grows, and what replaces it. None of these
changes touch the agent graph — that is the point of keeping retrieval behind
`Retrievers`.

| Corpus | Bottleneck | Change |
|--------|------------|--------|
| ≤ ~2 000 pages | none — brute-force MaxSim on MPS stays sub-second; ColPali index ≈ 0.75 GB on disk (384 KB/page), loaded fully into memory on first query | current design |
| ~2 000 – 50 000 pages | ColPali index no longer fits comfortably in memory; per-query MaxSim is O(pages) | multi-vector ANN (Qdrant multi-vector, Vespa, PLAID) and/or token pooling ("Light-ColPali") to cut vectors per page by 3–10× |
| ≥ 10 M chunks | single-node Chroma | Qdrant / Milvus / pgvector; same caller-supplied-embedding contract |
| ingest throughput | serial VLM captioning (one figure per call) | batch captions across pages; run ingest for several PDFs in parallel against the same servers |
| query latency | serial per-passage grading | parallel grading; smaller grading model; re-ranker before grading |

Storage growth per 100 pages (derived, see
[00 — Requirements](00-requirements.md#memory-and-vram-footprint)): ≈ 10 MB of page
renders, ≈ 38 MB of ColPali tensors, ≈ 1 MB of Chroma.

## Failure modes

| Failure | Detected by | Behaviour | Recovery |
|---------|-------------|-----------|----------|
| Chat server down | connection error on the first `route` call | `ask` exits with the OpenAI client's error; nothing is written | start the server (`lms server start` / `serve.sh`) |
| Vision server down during ingest | per-figure exception | figure skipped with a warning; text and ColPali indexing continue | re-ingest after restart; captions are upserted by id |
| Embedding model swapped | Chroma dimension error on next `add`/`query` | ingestion fails loudly | `rm -rf storage/chroma` and re-ingest |
| ColQwen2 adapter silently not applied (`transformers` 5.x) | **not detectable at runtime** — scores stay plausible | visual ranking degrades to noise | version pin + the ranking assertion described in [ADR-0005](adr/0005-pin-transformers-4x.md) |
| Model returns prose instead of JSON | `extract_json` → `{}` | route defaults to *retrieve*; grade defaults to *not relevant* | none needed — safe defaults |
| Nothing relevant after the cap | `decide_after_grade` | `generate` declines explicitly (no LLM call when there are no hits at all) | rephrase; check the PDF was ingested |
| ColPali OOM on MPS | torch error during `index_pages` | ingest aborts for that document | lower `_RENDER_DPI`, or `COLPALI_DEVICE=cpu` |
| Corrupt or exotic PDF | PyMuPDF exception | that document fails; folder ingest stops at it | remove / repair the file; ingestion of earlier files is already persisted |
| Stale entries after editing a PDF | none — new hash means new `doc_id` | old chunks remain searchable alongside new ones | planned: orphan sweep by `source`; today `rm -rf storage/` |

## Observability

- The `trace` field in `AgentState` is the audit log: every node appends one
  human-readable line, the CLI prints it, and the tests assert on it.
- Model servers log to `storage/logs/{chat,embed,vlm}.log` (llama.cpp) or the LM
  Studio app.
- There is deliberately no metrics or tracing backend; the golden-question smoke
  test in [04 — Setup](04-setup.md#smoke-test) is the health check.

## Roadmap (in priority order)

1. Cross-encoder re-ranker between `retrieve` and `grade`.
2. Parallel grading.
3. Orphan sweep for re-ingested documents.
4. Multi-vector ANN backend behind the same `ColPaliStore` interface.
5. RAGAS on the golden set in CI (optional `eval` extra is already declared).
6. Streamlit viewer that shows the cited page image inline.
