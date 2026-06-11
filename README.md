# local-rag

**Fully-local agentic RAG over PDFs — with hybrid text + visual (ColPali) retrieval.**

Point it at a folder of PDFs. It reads the prose, *looks at* the figures, charts and
tables, and answers questions with citations — using an agent that decides when to
retrieve, grades what it finds, and re-queries when the evidence is weak. **No data
leaves your machine.** Every model runs locally via LM Studio or `llama.cpp`.

```
              ┌──────────────────────── ingestion ────────────────────────┐
   PDFs  ──►  PyMuPDF  ──►  text chunks ─────────────┐
                       └─►  figures ─► vision-LLM caption ─► text embeddings ─► ChromaDB
                       └─►  full-page renders ───────────────► ColPali embeddings ─► disk
              └────────────────────────────────────────────────────────────┘

   question ─►  ┌─ route ─┬─ retrieve (text + visual) ─ grade ─┬─ generate ─► answer + citations
                │         └────────── rewrite & retry ◄────────┘
                └─ direct answer
```

## Why this is different from a typical RAG tutorial

- **Local-first, end to end.** LLM, embeddings, and a vision model all run on your
  hardware through an OpenAI-compatible local server (LM Studio or `llama.cpp`).
  Private documents stay private.
- **It actually sees the pages.** Two complementary visual strategies: figures are
  *captioned* by a vision-LLM and embedded as text, **and** whole pages are embedded with
  **ColPali/ColQwen2** late-interaction visual retrieval — which beats OCR pipelines on
  chart/table-heavy documents.
- **Genuinely agentic.** A LangGraph state machine routes, grades retrieved evidence,
  and self-corrects by rewriting the query and retrying (Corrective / Adaptive RAG) — not
  a single `retrieve → stuff → generate` shot.

## Quickstart

Two interchangeable ways to serve the models — same OpenAI-compatible API, so only
`.env` changes. **LM Studio** (GUI) is the default below; **llama.cpp** (CLI) is the
fully-scripted alternative. Full details in [docs/04-setup.md](docs/04-setup.md).

```bash
# 0. Prereqs: macOS (Apple Silicon recommended), uv
uv sync                       # create .venv and install deps
cp .env.example .env          # set base URLs/model ids for your runtime

# 1a. LM Studio: download a chat, embedding, and vision model, then:
lms server start              # serves all three on :1234, routed by model id
# (or 1b. llama.cpp: ./scripts/serve.sh — chat :8080, embeddings :8081, vision :8082)

# 2. Add PDFs and ingest
cp ~/papers/*.pdf data/
uv run local-rag ingest

# 3. Ask
uv run local-rag ask "What does Figure 1 show about throughput vs. batch size?"
uv run local-rag status
```

With LM Studio, stop the server from the app (or `lms server stop`); with llama.cpp,
`./scripts/stop.sh`.

## Documentation

- [docs/01-concepts.md](docs/01-concepts.md) — what agentic RAG is and the patterns it uses
- [docs/02-methodology.md](docs/02-methodology.md) — how those patterns map onto this build
- [docs/03-architecture.md](docs/03-architecture.md) — system design, data flow, the agent graph
- [docs/04-setup.md](docs/04-setup.md) — detailed setup, model choices, troubleshooting
- [docs/05-interview-prep.md](docs/05-interview-prep.md) — Q&A, tradeoffs, limitations, demo script
- [docs/medium-article.md](docs/medium-article.md) — the write-up, with real output + a debugging war story

## Stack

| Concern            | Choice                                              |
|--------------------|-----------------------------------------------------|
| Model runtime      | LM Studio **or** `llama.cpp` (both OpenAI-compatible) |
| Chat / reasoning   | any local instruct model — e.g. Ministral-3-14B / Qwen2.5-7B |
| Text embeddings    | nomic-embed-text-v1.5                               |
| Figure captioning  | any local VLM — e.g. GLM-4.6V-Flash / Qwen2.5-VL    |
| Visual retrieval   | ColQwen2 via `colpali-engine` (Apple Silicon MPS)   |
| Vector DB          | ChromaDB (local, persistent)                        |
| Agent orchestration| LangGraph                                           |

## License

MIT
