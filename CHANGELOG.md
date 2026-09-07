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
