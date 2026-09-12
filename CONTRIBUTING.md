# Contributing

Thank you for your interest in local-rag! Contributions are welcome — bug reports,
improvements to docs, new retrieval strategies, and anything that keeps the system
local-first and reproducible.

## Development setup

```bash
git clone https://github.com/ingscarrero/local-rag.git
cd local-rag
uv sync --extra dev    # runtime deps + pytest/ruff/mypy/pytest-cov
cp .env.example .env   # configure for your runtime (LM Studio or llama.cpp)
```

`uv sync` installs torch and transformers (≈ 1 GB on disk) but **no model
weights** — those download only when you actually run `ingest`/`ask` with
ColPali enabled. The test suite never loads them.

## Running checks locally

```bash
# Lint + format (src, scripts, and tests)
uv run ruff check .
uv run ruff format --check .

# Type check
uv run mypy src/local_rag --ignore-missing-imports

# Unit tests with coverage (no models, no servers, no network; ~3 s)
uv run pytest
```

All four should be green before opening a PR. CI runs the same commands and
fails below **95 %** statement coverage (`--cov-fail-under` in `pyproject.toml`);
`coverage.xml` is uploaded as a CI artifact. Every model call is faked at the
`models._client` seam — see `tests/conftest.py` for the scripted
OpenAI-compatible client and the store fakes, and reuse them rather than
patching deeper.

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
tests/               # hermetic unit tests; conftest.py holds the fakes
docs/                # requirements, concepts, architecture, system design, ADRs, CLI reference, runbooks
scripts/             # serve.sh / stop.sh (llama.cpp) + make_sample_pdf.py
```

## Contribution guidelines

- **No cloud calls.** Every model call must go through the local OpenAI-compatible
  endpoint (`settings.*_base_url`). Do not add API keys for external services.
- **Env-vars only for config.** Add new knobs to `config.py` as Pydantic fields,
  not as hard-coded values.
- **Keep retrieval honest.** If you add a new retrieval path, add a test that
  asserts the *right page wins the ranking* — not just that a result is returned
  (see `tests/test_text_store.py` and `tests/test_colpali_store.py`, and
  [ADR-0005](docs/adr/0005-pin-transformers-4x.md) for why).
- **Don't bump `transformers` past 4.x** without re-running the real-model
  ranking check in ADR-0005.
- **Update the relevant doc.** Architecture change → `docs/03-architecture.md`
  and `docs/06-system-design.md`. New requirement → `docs/00-requirements.md`.
  New flag or env var → `docs/07-cli-reference.md`. New setup step →
  `docs/04-setup.md` / the relevant runbook. A decision that would be costly to
  reverse → a new record in `docs/adr/`. Add a line to `CHANGELOG.md`.

## Opening a PR

1. Fork the repo and create a feature branch.
2. Make your changes; run checks (see above).
3. Write a clear description of *what* and *why*.
4. Reference any related issue.

## Reporting bugs

Open a GitHub issue with:
- OS + Apple Silicon / Intel / CUDA
- Runtime: LM Studio version or `llama-server --version`
- The `*_BASE_URL` / `*_MODEL` / `COLPALI_*` values from your `.env` — never
  paste API keys or the full file
- Full terminal output of the failing command
