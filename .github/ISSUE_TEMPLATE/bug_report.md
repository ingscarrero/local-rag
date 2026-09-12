---
name: Bug report
about: Something is broken or behaving unexpectedly
labels: bug
---

## Describe the bug

A clear description of what went wrong.

## Steps to reproduce

```bash
# Paste the exact commands you ran
```

## Expected behaviour

What you expected to happen.

## Actual behaviour / error output

```
Paste the full terminal output here, including tracebacks.
```

## Environment

- OS:
- CPU / GPU: (Apple Silicon M-series / Intel / NVIDIA)
- Runtime: LM Studio `lms --version` _or_ llama.cpp `llama-server --version`
- Python: `uv run python --version`

Relevant settings (only these keys — they contain no secrets):

```ini
LLM_BASE_URL=
LLM_MODEL=
EMBED_BASE_URL=
EMBED_MODEL=
VLM_BASE_URL=
VLM_MODEL=
COLPALI_MODEL=
COLPALI_DEVICE=
```

**Never paste `OPENAI_API_KEY`, any other API key or token, or your full `.env`.**
