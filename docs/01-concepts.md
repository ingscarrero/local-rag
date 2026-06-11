# 01 — Concepts: What "Agentic RAG" Actually Means

## From naive RAG to agentic RAG

**Naive RAG** is a straight line:

```
query → embed → vector search → stuff top-k into prompt → generate
```

It has no notion of whether retrieval *succeeded*. If the top-k chunks are
irrelevant, off-topic, or missing the key passage, the model generates anyway —
often confidently wrong. It also assumes the answer is in the prose. Charts,
tables, and diagrams — where a lot of the real information in technical PDFs
lives — are invisible to it.

**Agentic RAG** turns retrieval into a *control loop*. An LLM-driven agent makes
decisions at each step and can change course:

- **Should I even retrieve?** (Adaptive RAG / routing)
- **Is my query any good?** (query rewriting / decomposition)
- **Are these results actually relevant?** (retrieval grading)
- **Do I have enough to answer, or should I try again?** (corrective loop)
- **Is my answer grounded in the evidence?** (citation / faithfulness)

The agent is allowed to *fail and recover* instead of blindly answering.

## The core patterns

The literature converges on a handful of reusable patterns. This project
implements the ones that matter most for a local, document-grounded assistant.

### 1. Routing / Adaptive RAG
Not every question needs a document search. "Hi" or "summarize our last chat"
shouldn't trigger a vector lookup. A lightweight classifier (here, a small LLM
call) routes the question to either the **retrieval path** or a **direct answer**.

### 2. Query rewriting
The user's phrasing is often a poor *search* query. After a weak retrieval round,
the agent reformulates the question into a keyword-rich query and retries. This
is the cheapest, highest-leverage agentic step.

### 3. Retrieval grading (Corrective RAG / CRAG)
Before generating, the agent asks: *is each retrieved passage relevant to the
question?* Irrelevant chunks are dropped. If nothing survives, the agent doesn't
hallucinate — it rewrites and retries, or admits it can't find the answer. This
"grade-then-decide" step is the defining move of Corrective RAG.

### 4. Iterative / multi-step retrieval
With a bounded loop (`retrieve → grade → rewrite → retrieve …`, capped at N
iterations), the agent can converge on evidence that a single shot would miss —
without risking an infinite loop.

### 5. Grounded generation with citations
The final answer is constrained to the retrieved context and must cite its
sources (`source p.N`). This makes hallucinations visible and the system
auditable — essential for a tool people are meant to trust.

### Related patterns (context)
- **Self-RAG** lets the model emit "reflection tokens" to critique its own output;
  CRAG (used here) externalizes that judgement into an explicit grader node, which
  is more robust on small local models.
- **Re-ranking** (e.g. a cross-encoder) reorders candidates by relevance before
  grading. We rely on the grader instead to keep the local footprint small; a
  re-ranker is a natural drop-in extension.

## The visual dimension: RAG that can *see*

Most RAG tutorials stop at text. But in real documents — papers, reports,
datasheets, slide decks — the answer is frequently in a **figure or table**. Two
complementary techniques close that gap:

### A. Vision-LLM captioning (describe-then-embed)
During ingestion, a local vision-language model looks at each extracted figure and
writes a detailed textual description. That caption is embedded alongside the
prose, so an ordinary text query can retrieve "the bar chart showing throughput
rising then plateauing." Simple, explainable, and it reuses the text pipeline.

### B. ColPali / ColQwen2 visual retrieval (embed-the-page)
[ColPali](https://arxiv.org/abs/2407.01449) skips OCR entirely. It renders each
page to an image and embeds it as a **set of patch-level vectors** using a
vision-language model — the "late interaction" idea from ColBERT, applied to
pixels. At query time, the query's token embeddings are scored against every
page's patch embeddings via **MaxSim**. Because it never linearizes the page into
text, it preserves layout, tables, and chart structure, and consistently beats
OCR-then-chunk pipelines on visually complex documents — at the cost of a larger
index (hundreds of vectors per page) and more compute.

**Why use both?** Captioning makes figures *searchable by meaning* inside the same
text index and gives the LLM something to read; ColPali finds the *right page*
even when OCR would have mangled it. Fusing the two is more robust than either
alone — and demonstrates that you understand the tradeoffs rather than picking a
single fashionable technique.

## How the patterns compose

```
            ┌─────────────┐
question ──►│   ROUTE     │── direct ──────────────► answer
            └──────┬──────┘
                   │ retrieve
                   ▼
            ┌─────────────┐     ┌──────────────────────────────┐
            │  RETRIEVE   │◄────│  text (Chroma) + visual (ColPali)
            └──────┬──────┘     └──────────────────────────────┘
                   ▼
            ┌─────────────┐  relevant==0 & iters<cap   ┌──────────┐
            │   GRADE     │───────────────────────────►│ REWRITE  │
            └──────┬──────┘                            └────┬─────┘
                   │ relevant>0  (or iteration cap)         │
                   ▼                                        │
            ┌─────────────┐                                 │
            │  GENERATE   │  (cite sources)   ◄─────────────┘
            └──────┬──────┘   (loops back to RETRIEVE)
                   ▼
                answer
```

Continue to [02 — Methodology](02-methodology.md) for how each pattern is
implemented here, and [03 — Architecture](03-architecture.md) for the system design.
