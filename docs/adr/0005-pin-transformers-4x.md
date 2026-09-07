# ADR-0005: Pin `transformers` to the 4.x line

- **Status:** Accepted
- **Date:** 2026-06-11
- **Affects:** `pyproject.toml` (`transformers>=4.46.0,<5.0.0`, `colpali-engine>=0.3.4,<0.3.12`), `retrieval/colpali_store.py`

## Context

`vidore/colqwen2-v1.0` ships as a **LoRA adapter** on top of a Qwen2-VL base.
With `transformers` 5.x installed, the first end-to-end run *looked* healthy:
the model loaded, produced embeddings of the right shape, and returned
plausible MaxSim scores. A five-line functional check — embed both sample pages
and ask which page wins for a chart query versus a prose query — showed the
scores were nearly identical regardless of the query and the chart query
picked the wrong page:

```
'a bar chart of throughput versus batch size'  -> p1=12.94 p2=12.88  winner=page1  ✗
'definition and problems of naive RAG'          -> p1=13.50 p2=13.44  winner=page1  ✗
```

Inspecting the weights showed `lora_B` was all zeros: on 5.x the adapter was
silently not applied and the model fell back to the untrained base. No error,
no warning. The full story is in
[the write-up](../medium-article.md#the-bug-that-almost-shipped-silently).

## Decision

Pin `transformers>=4.46.0,<5.0.0` and let the resolver pick the matching
`colpali-engine` (`>=0.3.4,<0.3.12`). Record the reason next to the pin in
`pyproject.toml` so nobody "helpfully" bumps it. After the pin:

```
'a bar chart of throughput versus batch size'  -> p1=6.72  p2=16.12  winner=page2  ✓
'definition and problems of naive RAG'          -> p1=12.06 p2=5.12  winner=page1  ✓
```

## Consequences

- **Positive:** visual retrieval actually retrieves. The unit suite mirrors the
  same assertion with a fake encoder — the *right page must win* — so the
  contract is documented even where the real model is not loaded
  ([tests/test_colpali_store.py](../../tests/test_colpali_store.py)).
- **Negative:** stuck on 4.x until `colpali-engine` supports 5.x adapters
  correctly. Any bump must re-run the ranking check above against the real
  model (`scripts/make_sample_pdf.py` + `local-rag ingest` + the two queries)
  before merging; a shape check is not evidence.
- **Lesson generalised** (and adopted as a contribution rule in
  [CONTRIBUTING.md](../../CONTRIBUTING.md)): "the code ran without error" is not
  evidence that an ML component works. Embeddings always return a vector; only a
  ranking assertion proves a retriever retrieves.
