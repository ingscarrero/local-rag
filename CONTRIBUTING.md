# Contributing

Thank you for your interest in local-rag! Contributions are welcome — bug reports,
improvements to docs, new retrieval strategies, and anything that keeps the system
local-first and reproducible.

## Development setup

```bash
git clone https://github.com/ingscarrero/local-rag.git
cd local-rag
uv sync                # installs all deps including dev extras
cp .env.example .env   # configure for your runtime (LM Studio or llama.cpp)
```

## Running checks locally

```bash
# Lint + format
uv run ruff check src/ scripts/
uv run ruff format --check src/ scripts/

# Type check
uv run mypy src/local_rag --ignore-missing-imports

# Unit tests (no models required)
uv run pytest tests/ -v
```

All three should be green before opening a PR. CI runs the same commands.

## Project layout

```
src/local_rag/
  config.py          # Pydantic Settings — all knobs are env vars
  models.py          # thin OpenAI-client wrappers (chat / embed / caption_image)
  ingest/
    pdf.py           # PyMuPDF parsing → text chunks + figure images
    chunk.py         # word-window chunking
    pipeline.py      # orchestrates parse → embed → store
  retrieval/
    text_store.py    # ChromaDB wrapper
    colpali_store.py # ColQwen2 visual embeddings
    hybrid.py        # fuses text + visual hits
  agent/
    state.py         # LangGraph TypedDict state
    nodes.py         # route / retrieve / grade / rewrite / generate nodes
    graph.py         # StateGraph wiring + answer_question() entrypoint
  cli.py             # typer CLI (ingest / ask / status)
docs/                # concept, methodology, architecture, setup, runbooks
scripts/             # serve.sh / stop.sh (llama.cpp) + make_sample_pdf.py
```

## Contribution guidelines

- **No cloud calls.** Every model call must go through the local OpenAI-compatible
  endpoint (`settings.*_base_url`). Do not add API keys for external services.
- **Env-vars only for config.** Add new knobs to `config.py` as Pydantic fields,
  not as hard-coded values.
- **Keep retrieval honest.** If you add a new retrieval path, add a test that
  asserts the *right page wins the ranking* — not just that a result is returned.
- **Update the relevant doc.** Architecture change → `docs/03-architecture.md`.
  New setup step → `docs/04-setup.md` / the relevant runbook.

## Opening a PR

1. Fork the repo and create a feature branch.
2. Make your changes; run checks (see above).
3. Write a clear description of *what* and *why*.
4. Reference any related issue.

## Reporting bugs

Open a GitHub issue with:
- OS + Apple Silicon / Intel / CUDA
- Runtime: LM Studio version or `llama-server --version`
- `.env` values (redact `OPENAI_API_KEY`)
- Full terminal output of the failing command
