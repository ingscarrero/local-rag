# 03 — Architecture

## System overview

Everything below runs on one machine. The three model servers bind to
`127.0.0.1`; the only outbound connection is the one-time model download.

```mermaid
C4Container
    title Container view — local-rag (single machine, no inference egress)

    Person(user, "User", "Asks questions about private PDFs from a terminal")

    System_Boundary(machine, "Local machine — everything binds to 127.0.0.1") {
        Container(cli, "CLI", "Python · Typer + Rich", "local-rag ingest | ask | status")
        Container(ingest, "Ingestion pipeline", "Python · PyMuPDF", "parse → chunk → caption → index")
        Container(agent, "Agent", "Python · LangGraph", "route → retrieve → grade → rewrite ↺ → generate")
        Container(colqwen, "ColQwen2 encoder", "colpali-engine · torch (MPS/CUDA/CPU)", "in-process page + query multi-vectors")
        ContainerDb(chroma, "ChromaDB", "storage/chroma", "text chunks + figure captions, cosine")
        ContainerDb(colpali, "ColPali index", "storage/colpali", "page_embeddings.pt + manifest.json")
        Container(llm, "Chat LLM", "LM Studio / llama-server :8080", "route, grade, rewrite, generate")
        Container(embed, "Embedding model", "LM Studio / llama-server :8081", "nomic-embed-text v1.5")
        Container(vlm, "Vision LLM", "LM Studio / llama-server :8082", "figure captioning at ingest")
    }

    System_Ext(hf, "Hugging Face Hub", "One-time weight download; never contacted at query time")

    Rel(user, cli, "runs")
    Rel(cli, ingest, "ingest")
    Rel(cli, agent, "ask")
    Rel(ingest, vlm, "caption figures", "OpenAI-compatible HTTP")
    Rel(ingest, embed, "embed chunks + captions", "OpenAI-compatible HTTP")
    Rel(ingest, chroma, "upsert by content-hashed id")
    Rel(ingest, colqwen, "encode page renders")
    Rel(colqwen, colpali, "persist / load tensors")
    Rel(agent, llm, "decisions + generation", "OpenAI-compatible HTTP")
    Rel(agent, embed, "embed query")
    Rel(agent, chroma, "cosine top-k")
    Rel(agent, colqwen, "encode query, MaxSim over pages")
    Rel(colqwen, hf, "first run only", "HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Ingestion data flow

One `parse_pdf` pass yields three artefacts per page; two of them end up in
ChromaDB (as text), the third in the ColPali index (as pixels).

```mermaid
flowchart LR
    PDF[("PDFs in data/")] --> P["parse_pdf<br/>PyMuPDF"]
    P --> T["page text"] --> C["chunk_page_text<br/>220 words, 40 overlap"]
    P --> F["embedded figures<br/>≥ 100×100 px"] --> V["caption_image<br/>vision LLM :8082"]
    P --> R["page render @ 150 DPI<br/>storage/page_images/"]
    C --> E["embed<br/>:8081"]
    V --> E
    E --> DB[("ChromaDB<br/>kind = text | caption")]
    R --> Q["ColQwen2 encoder<br/>in-process"]
    Q --> CP[("ColPali index<br/>page_embeddings.pt + manifest.json")]
```

## Component map (code)

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `config.py` | env-driven settings, storage paths |
| Model clients | `models.py` | chat / embed / caption via OpenAI-compatible API |
| Parsing | `ingest/pdf.py` | text, embedded images, page renders (PyMuPDF) |
| Chunking | `ingest/chunk.py` | word-window chunks with overlap |
| Pipeline | `ingest/pipeline.py` | orchestrates parse → caption → index |
| Text index | `retrieval/text_store.py` | ChromaDB wrapper (text + captions) |
| Visual index | `retrieval/colpali_store.py` | ColQwen2 embeddings + MaxSim search |
| Fusion | `retrieval/hybrid.py` | run both retrievers together |
| Agent state | `agent/state.py` | typed `AgentState` shared across nodes |
| Agent nodes | `agent/nodes.py` | route / retrieve / grade / rewrite / generate |
| Agent graph | `agent/graph.py` | LangGraph `StateGraph` wiring |
| CLI | `cli.py` | `ingest`, `ask`, `status` |

## Why three separate model servers?

`llama-server` serves one model per process. Running chat, embeddings, and vision
as **three processes** (ports 8080/8081/8082) means each model loads once and stays
warm — no reload cost between an embedding call and a chat call. They're all behind
the same OpenAI-compatible API, so the client code (`models.py`) is uniform and
could point at OpenAI, LM Studio, or vLLM by changing only the base URLs. With
512 GB of unified memory, holding all three resident simultaneously is trivial.

## Data model

**ChromaDB collection `documents`** — one row per text chunk *or* figure caption:

```
id        = "{doc_id}::text::{i}"   |  "{doc_id}::caption::{image_id}"
document  = chunk text              |  "[Figure on page N] <caption>"
metadata  = {kind, doc_id, source, page_number, [image_id]}
embedding = local text-embedding model (768-d, cosine)
```

`doc_id` is content-hashed (`{stem}-{sha1[:12]}`), making ingestion **idempotent** —
re-ingesting the same file overwrites rather than duplicates.

**ColPali index** — `storage/colpali/`:
- `page_embeddings.pt` — list of variable-length `[num_patches, dim]` tensors
- `manifest.json` — parallel list of `{doc_id, page_number, png_path}`

**Page images** — `storage/page_images/{doc_id}/page-NNNN.png` (the renders, reused
for display and as ColPali inputs).

## The agent graph

The state machine, as wired in `agent/graph.py`:

```mermaid
flowchart TD
    S((START)) --> route
    route -- "route == direct" --> answer_directly --> E((END))
    route -- "route == retrieve" --> retrieve
    retrieve --> grade
    grade -- "relevant > 0<br/>or iterations ≥ cap" --> generate --> E
    grade -- "relevant == 0<br/>and iterations < cap" --> rewrite
    rewrite --> retrieve
```

And the same loop as a sequence of calls, which is what the CLI's `trace`
panel prints:

```mermaid
sequenceDiagram
    autonumber
    actor U as User (CLI)
    participant G as LangGraph agent
    participant L as Chat LLM (:8080)
    participant TS as TextStore (Chroma + embed :8081)
    participant CS as ColPaliStore (ColQwen2)

    U->>G: ask "What does Figure 1 show?"
    G->>L: route? (JSON, max 20 tokens)
    L-->>G: {"route": "retrieve"}

    loop until relevant > 0, capped at MAX_AGENT_ITERATIONS (3)
        G->>TS: query(current query, top_k = TEXT_TOP_K)
        TS-->>G: text + caption hits (cosine)
        G->>CS: query(current query, top_k = VISUAL_TOP_K)
        CS-->>G: page hits (MaxSim)
        G->>L: grade each text hit → {"relevant": bool}
        L-->>G: per-passage verdicts
        break ≥ 1 relevant, or cap reached
            Note over G: proceed to generate
        end
        G->>L: rewrite query (keyword-rich)
        L-->>G: new query
    end

    alt relevant text or visual hits exist
        G->>L: generate from context + visual page refs
        L-->>G: answer with (source p.N) citations
    else no evidence at all
        Note over G: canned "couldn't find evidence" — no LLM call
    end
    G-->>U: answer + trace + text sources + visual matches
```

A greeting takes the short path: `route` returns `direct`, `answer_directly`
makes one LLM call, and no retriever is touched.

`AgentState` carries: `question`, current `query`, `route`, `text_hits`,
`visual_hits`, graded `relevant`, `iterations`, `answer`, and an appendable
`trace` for observability (printed by the CLI).

## Scaling notes (what you'd change for "real")

The current build is tuned for a **personal / showcase corpus** (tens to low
hundreds of PDFs). The honest scaling story:

- **ColPali brute-force MaxSim** is O(pages) per query. Past a few thousand pages,
  swap in an ANN store with multi-vector / late-interaction support (e.g. Vespa,
  Qdrant multi-vector, or PLAID) and/or pooled "Light-ColPali" embeddings to cut
  the per-page vector count.
- **ChromaDB** is fine to tens of millions of vectors single-node; beyond that, a
  distributed store (Qdrant/Milvus/pgvector) is the move.
- **Re-ranking**: insert a cross-encoder between `retrieve` and `grade` to improve
  candidate ordering before the (more expensive) LLM grader runs.
- **Batching captions**: ingestion captions figures serially for clarity; batch
  them across pages for throughput.

## Evaluation

Install the optional extra: `uv sync --extra eval`. RAGAS can score
**faithfulness**, **answer relevancy**, and **context precision/recall** using the
same local chat + embedding models (point RAGAS's LLM/embeddings at the local
endpoints). For a quick smoke test, the golden-question approach in
[04 — Setup](04-setup.md#smoke-test) is enough to prove the pipeline end-to-end.

## Privacy & security posture

- **No egress at inference time.** All three model servers bind to `127.0.0.1`.
  Models are downloaded once from Hugging Face; after that the system runs offline.
- **No secrets.** `llama-server` takes a dummy API key. There is no telemetry.
- **Trust boundary.** The only external input is the PDFs themselves. PDF parsing
  is done by PyMuPDF; we never execute embedded JavaScript or follow links.
