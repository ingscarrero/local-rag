## What this PR does

<!-- One-paragraph summary of the change and why it's needed. -->

## Type of change

- [ ] Bug fix
- [ ] New feature / retrieval strategy
- [ ] Documentation / runbook update
- [ ] Refactor (no behaviour change)
- [ ] Dependency update

## How to test

```bash
# Commands the reviewer should run to verify the change
```

Golden questions should still pass:

```bash
uv run local-rag ask "What does Figure 1 show about throughput versus batch size?"
uv run local-rag ask "What problems does naive RAG have?"
uv run local-rag ask "hello there"
```

## Checklist

- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run mypy src/local_rag --ignore-missing-imports` passes
- [ ] `uv run pytest` passes (coverage ≥ 95 %)
- [ ] Docs updated if behaviour / setup steps changed (`docs/`, `CHANGELOG.md`)
- [ ] No cloud API keys or external model calls introduced
