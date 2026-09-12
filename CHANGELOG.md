# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- MIT `LICENSE` file (the licence was declared in `pyproject.toml` but not shipped).
- Unit tests for the agent graph and nodes, hybrid/text/ColPali retrieval, the
  ingestion pipeline and PDF parser, configuration, model clients, and the CLI —
  all hermetic, no model servers or weights required. Coverage is enforced at 95%.
- `pytest-cov` in the `dev` extra; CI uploads `coverage.xml` as an artifact.
- Mermaid diagrams (C4 container view, ingestion flowchart, agent sequence diagram)
  in the README and `docs/03-architecture.md`.
- `docs/00-requirements.md` (functional + non-functional requirements),
  `docs/06-system-design.md`, `docs/07-cli-reference.md`, five Architecture
  Decision Records under `docs/adr/`, `SECURITY.md`, and this changelog.
- README badges for CI, coverage, licence, and Python version.

### Changed
- Dependencies re-locked (`uv lock --upgrade`) to clear every Dependabot alert
  that has a fix compatible with the `transformers` 4.x pin (ADR-0005): 61 of 78
  open alerts, across GitPython (dropped from the tree by streamlit 1.63),
  Pillow 12.3.0, aiohttp 3.14.3, starlette 1.6.0, python-multipart 0.0.32,
  langsmith 0.12.4, langchain 1.4.0, pydantic-settings 2.15.0, setuptools 84.0.0
  and accelerate 1.15.0. Direct floors raised to `pillow>=12.3.0` and
  `pydantic-settings>=2.14.2`; `[tool.uv] constraint-dependencies` pins the
  transitive floors so a later re-lock cannot regress below the fixes.
- Ruff configuration made explicit (`select = ["E4", "E7", "E9", "F"]`, Markdown
  excluded) so lint and format results no longer change when ruff widens its
  defaults, as 0.16 did.

### Security
- `SECURITY.md` gains a "Known unresolved advisories" section for the 17 alerts
  that have no fix compatible with this project (`transformers` 4.x, `torch`
  < 2.8 via `colpali-engine`, `chromadb` 1.5.9, and the `eval`-only `ragas` /
  `diskcache`), each with the blocking reason, the mitigation, and the
  re-check trigger.

### Fixed
- mypy errors on the `openai` and `chromadb` call sites that had kept CI red
  since the initial commit.

## [0.1.0] — 2026-06-11

### Added
- Initial release: fully-local agentic RAG over PDFs.
- Ingestion: PyMuPDF parsing, word-window chunking, vision-LLM figure
  captioning, ColQwen2 page embeddings; idempotent via content-hashed `doc_id`.
- Retrieval: ChromaDB text + caption index, ColPali MaxSim visual index, hybrid
  fusion.
- Agent: LangGraph state machine — route → retrieve → grade → rewrite ↺ →
  generate, with an iteration cap and cited answers.
- CLI: `local-rag ingest | ask | status` (Typer + Rich).
- Two runtimes behind one OpenAI-compatible API: LM Studio and `llama.cpp`
  (`scripts/serve.sh` / `stop.sh`), with runbooks for each.
- Docs: concepts, methodology, architecture, setup, interview prep, write-up.

[Unreleased]: https://github.com/ingscarrero/local-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ingscarrero/local-rag/releases/tag/v0.1.0
