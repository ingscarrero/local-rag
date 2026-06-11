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
- Model(s) in use: (chat / embed / vision model ids from `lms ps` or `.env`)
- Python: `uv run python --version`
- ColPali device: (mps / cuda / cpu from `.env`)

## `.env` (redact `OPENAI_API_KEY`)

```ini
# Paste your .env here — remove or replace OPENAI_API_KEY value
```
