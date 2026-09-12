# 07 — CLI Reference

`local-rag` is a [Typer](https://typer.tiangolo.com/) application installed by
`uv sync` as a console script. Run it as `uv run local-rag …` (or `local-rag …`
inside the activated `.venv`). Output is rendered with Rich.

```
Usage: local-rag [OPTIONS] COMMAND [ARGS]...

  Fully-local agentic RAG over PDFs.

Commands:
  ingest  Parse PDFs and build the text + visual indexes.
  ask     Ask the agent a question over the ingested documents.
  status  Show index counts and configured endpoints.
```

Shell completion is disabled (`add_completion=False`). Every command reads its
configuration from the environment / `.env` (see
[Environment variables](#environment-variables)); no command takes a config
flag.

---

## `local-rag ingest [PATH] [OPTIONS]`

Parse PDFs and build both indexes. Idempotent: a file whose bytes have not
changed gets the same content-hashed `doc_id`, and its chunks, captions and page
embeddings are upserted rather than duplicated.

| Argument / option | Type | Default | Meaning |
|-------------------|------|---------|---------|
| `PATH` | path | `DATA_DIR` (`./data`) | A single `.pdf` **file**, or a **folder** searched recursively (`**/*.pdf`, sorted). |
| `--no-captions` | flag | off | Skip vision-LLM figure captioning. Text chunks and ColPali page embeddings are still built. The VLM server is not contacted. |
| `--no-visual` | flag | off | Skip ColPali page-image indexing. ColQwen2 is not loaded (no first-run download). |
| `--help` | | | Show help and exit. |

What it does, per PDF ([ingestion flow](03-architecture.md#ingestion-data-flow)):

1. `parse_pdf` — page text, embedded images ≥ 100×100 px, a 150-DPI PNG of
   every page under `STORAGE_DIR/page_images/{doc_id}/`.
2. Chunk text into 220-word windows with 40-word overlap.
3. Unless `--no-captions`: caption each figure with the VLM. A failed caption is
   logged in yellow and skipped; ingestion continues.
4. Embed chunks + captions with the embedding model and upsert into ChromaDB.
5. Unless `--no-visual`: embed page renders with ColQwen2 and append to the
   ColPali index (pages already present in the manifest are skipped).

Requires: the **embedding** server always; the **vision** server unless
`--no-captions`; ColQwen2 weights (auto-downloaded, ≈ 5 GB) unless `--no-visual`.
The chat LLM is not used.

Output: one block per document with a stats dict —
`{doc_id, source, pages, text_items, captions, colpali_pages}` — then
`Ingestion complete.`

```bash
uv run local-rag ingest                          # everything under ./data
uv run local-rag ingest ~/papers                 # a different folder
uv run local-rag ingest ~/papers/colpali.pdf     # one file
uv run local-rag ingest --no-captions --no-visual  # text-only, fastest, no VLM / ColQwen2
```

Exit status: `0` on success. A PDF that PyMuPDF cannot open raises and stops a
folder ingest at that file; earlier files are already persisted.

---

## `local-rag ask QUESTION [OPTIONS]`

Run the agent over the ingested documents and print the answer with its
evidence.

| Argument / option | Type | Default | Meaning |
|-------------------|------|---------|---------|
| `QUESTION` | text | *required* | The question. Quote it. |
| `--no-visual` | flag | off | Disable ColPali retrieval for this question; only the text/caption index is searched. ColQwen2 is not loaded. |
| `--show-trace` / `--no-show-trace` | flag | `--show-trace` | Print (or hide) the agent's step trace panel. |
| `--help` | | | Show help and exit. |

Flow ([sequence diagram](03-architecture.md#the-agent-graph)): route → if
`retrieve`: hybrid retrieval → per-passage grading → rewrite-and-retry while
nothing is relevant and `iterations < MAX_AGENT_ITERATIONS` → grounded
generation with `(source p.N)` citations. If `direct`: one short LLM reply, no
retrieval.

Requires: the **chat** server always; the **embedding** server whenever the
route is `retrieve`; ColQwen2 unless `--no-visual` or the ColPali index is empty.

Output sections, in order:

| Panel / table | Present when | Columns |
|---------------|--------------|---------|
| `agent trace` | `--show-trace` and the trace is non-empty | one line per node, e.g. `route → retrieve`, `retrieve → 3 text, 2 visual (iter 1)`, `grade → 1/3 relevant`, `rewrite → '…'`, `generate → answer` |
| `answer` | always | the answer text (green border) |
| `text sources` | ≥ 1 passage survived grading | `source`, `page`, `kind` (`text` \| `caption`) |
| `visual matches (ColPali)` | ≥ 1 visual hit | `doc`, `page`, `score` (MaxSim, 2 dp) |

When nothing relevant is found after the iteration cap and there are no visual
hits, the answer is a fixed "I couldn't find supporting evidence…" message and
no generation call is made.

```bash
uv run local-rag ask "What does Figure 1 show about throughput vs batch size?"
uv run local-rag ask "What problems does naive RAG have?" --no-visual
uv run local-rag ask "hello" --no-show-trace
```

---

## `local-rag status`

Print a two-column table; never contacts a model server.

| Row | Source |
|-----|--------|
| `chroma items` | `TextStore().count()` — text chunks + captions. If Chroma cannot be opened the cell shows `error: <message>` and the command still exits `0`. |
| `colpali pages` | number of entries in `STORAGE_DIR/colpali/manifest.json` (0 if absent) |
| `LLM` | `LLM_MODEL @ LLM_BASE_URL` |
| `embed` | `EMBED_MODEL @ EMBED_BASE_URL` |
| `VLM` | `VLM_MODEL @ VLM_BASE_URL` |
| `ColPali` | `COLPALI_MODEL (COLPALI_DEVICE)` |

```bash
uv run local-rag status
```

---

## Environment variables

Read by `config.Settings` from the process environment and `.env` (environment
wins; names are case-insensitive; unknown names are ignored; a non-integer
`*_TOP_K` or `MAX_AGENT_ITERATIONS` fails fast). Copy `.env.example` to get
started.

| Variable | Default | Used by | Notes |
|----------|---------|---------|-------|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | `ask` | chat / reasoning server |
| `LLM_MODEL` | `local-chat` | `ask` | must match the server's model id (LM Studio routes by it; llama-server ignores it) |
| `EMBED_BASE_URL` | `http://127.0.0.1:8081/v1` | `ingest`, `ask` | embedding server |
| `EMBED_MODEL` | `local-embed` | `ingest`, `ask` | change ⇒ wipe `storage/chroma` and re-ingest |
| `VLM_BASE_URL` | `http://127.0.0.1:8082/v1` | `ingest` | vision server for captions |
| `VLM_MODEL` | `local-vlm` | `ingest` | must be a true vision model |
| `OPENAI_API_KEY` | `not-a-real-key` | all | any non-empty string; local servers ignore it |
| `COLPALI_MODEL` | `vidore/colqwen2-v1.0` | `ingest`, `ask` | Hugging Face id; downloaded on first use |
| `COLPALI_DEVICE` | `mps` | `ingest`, `ask` | `mps` \| `cuda` \| `cpu`; bf16 on GPU, fp32 on CPU |
| `DATA_DIR` | `./data` | `ingest` | default `PATH` |
| `STORAGE_DIR` | `./storage` | all | parent of `chroma/`, `colpali/`, `page_images/`, `logs/`, `pids/` |
| `TEXT_TOP_K` | `5` | `ask` | passages retrieved (and graded) per round |
| `VISUAL_TOP_K` | `3` | `ask` | pages returned by ColPali |
| `MAX_AGENT_ITERATIONS` | `3` | `ask` | hard cap on retrieval rounds |
| `ANONYMIZED_TELEMETRY` | *(unset)* | all | set to `False` to silence ChromaDB's telemetry |

Model-server ports and model choices for the llama.cpp path are set on
`scripts/serve.sh` (`CHAT_HF`, `EMBED_HF`, `VLM_HF`, `NGL`), not in `.env` — see
the [llama.cpp runbook](runbook-llama-cpp.md).
