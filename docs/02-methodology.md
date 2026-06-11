# 02 — Methodology: How the Patterns Map onto This Build

This document is the bridge between the concepts in [01](01-concepts.md) and the
code. For each design decision it states *what* we do, *why*, and *where it lives*.

## Ingestion methodology

### Parsing (`ingest/pdf.py`)
We use **PyMuPDF** because it gives us all three artifacts we need from one pass:
1. **Page text** (`page.get_text("text")`) — for chunking.
2. **Embedded images** (`page.get_images` + `doc.extract_image`) — the figures we
   caption. Tiny images (< 100×100 px) are skipped as decorative.
3. **A full-page render at 150 DPI** (`page.get_pixmap`) — the input to ColPali.

150 DPI is a deliberate compromise: high enough that ColPali's vision encoder
reads charts and small text, low enough to keep page embeddings and memory sane.

### Chunking (`ingest/chunk.py`)
Fixed **220-word windows with 40-word overlap**. We intentionally *do not* invest
in elaborate semantic chunking. In an agentic system the **grade → rewrite → retry**
loop is what compensates for imperfect chunk boundaries — that's a core thesis of
this project. Simple, predictable chunking + a smart agent beats fragile,
over-tuned splitting + a dumb pipeline.

### Two visual strategies, on purpose
- **Caption-and-embed** (`models.caption_image`): each figure is described by the
  local VLM and stored in Chroma with `kind="caption"`, embedded by the *same*
  text embedder as the prose. One similarity search now spans text and figures.
- **ColPali page embeddings** (`retrieval/colpali_store.py`): every page render is
  embedded as a multi-vector and stored on disk with a manifest. This is a
  *separate* index queried in parallel.

Storing both is the whole point of the "hybrid" claim — see
[03 — Architecture](03-architecture.md) for the data flow.

## Retrieval methodology

### Text + caption search (`retrieval/text_store.py`)
ChromaDB with cosine space. We pass **our own embeddings** (from the local model)
rather than letting Chroma embed — this keeps everything local and lets us reuse
the exact same model for documents and queries. Text chunks and figure captions
live in one collection, separated by a `kind` metadata field, so a single query
returns both.

### Visual search (`retrieval/colpali_store.py`)
At query time the query string is embedded by ColQwen2 and scored against every
stored page embedding with **MaxSim** (`processor.score_multi_vector`). For a
personal/showcase corpus, brute-force MaxSim over all pages is fast enough and
keeps the code legible; the doc on architecture notes how this scales.

### Fusion (`retrieval/hybrid.py`)
The agent gets *both* result sets. Text/caption hits become the graded, cited
evidence the answer is built from; visual hits surface the **page numbers** most
likely to contain the relevant figure/table, which the generation step references.
This division of labour matches each method's strength: text for *what to say*,
ColPali for *where to look*.

## Agent methodology (`agent/`)

The agent is a **LangGraph `StateGraph`**. The shared `AgentState` (see
`agent/state.py`) is the single source of truth that every node reads and writes.

| Node | Pattern | What it does | Why an LLM call |
|------|---------|--------------|-----------------|
| `route` | Adaptive RAG | classify retrieve vs. direct | cheap guard against pointless retrieval |
| `retrieve` | hybrid retrieval | text+caption (Chroma) + visual (ColPali) | — (no LLM) |
| `grade` | Corrective RAG | keep only passages judged relevant | the key agentic judgement |
| `rewrite` | query rewriting | reformulate after a weak round | recovers from poor phrasing |
| `generate` | grounded generation | answer from context, with citations | the synthesis |

**Conditional edges** implement the corrective loop:

- After `route`: `retrieve` or `answer_directly`.
- After `grade` (`decide_after_grade`):
  - any relevant passages → `generate`
  - none, and iterations < cap → `rewrite` → `retrieve` (try again)
  - none, and iterations == cap → `generate` (which honestly reports it can't find it)

The iteration cap (`MAX_AGENT_ITERATIONS`, default 3) guarantees termination — a
must for an agent that can loop.

### Design choices that make this robust on *small local* models
- **Tiny decision prompts** returning JSON (`{"relevant": true}`), parsed with a
  forgiving extractor that tolerates stray prose. Small models follow short,
  structured instructions far more reliably than long ones.
- **Temperature 0** for every decision and for generation — reproducibility and
  fewer flights of fancy.
- **Per-passage grading** rather than grading the whole batch at once: a 7B model
  judges "is *this one* relevant?" much better than "which of these five…?".

## Evaluation methodology (lightweight)

For a showcase, heavyweight eval harnesses are overkill, but you should be able to
*defend* that it works. Two pragmatic levels:

1. **Golden questions**: a handful of Q&A pairs whose answers you know live in
   specific pages/figures. Check the agent retrieves those pages and cites them.
   The sample PDF + the questions in [04 — Setup](04-setup.md) are exactly this.
2. **RAGAS (optional)**: install the `eval` extra to compute *faithfulness*,
   *answer relevance*, and *context precision/recall* — all runnable against your
   local models. See [03 — Architecture](03-architecture.md#evaluation).

The honest framing for an interview: *"I validated retrieval with golden questions
and wired in RAGAS for faithfulness; I did not run a large benchmark because the
contribution is the architecture, not a leaderboard number."*
