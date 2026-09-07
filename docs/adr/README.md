# Architecture Decision Records

One file per decision that shaped the system and would be expensive to reverse.
Format: context → decision → consequences, with status and date. These were
written down on 2026-09-07 from the design docs, code comments and commit
history; each record's date is when the decision took effect (the initial
commit), not when it was transcribed.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-colpali-plus-captions-over-ocr.md) | ColPali page embeddings **plus** VLM captions, instead of OCR-then-chunk | Accepted |
| [0002](0002-three-model-servers-behind-one-api.md) | Three model roles behind one OpenAI-compatible API (three `llama-server` processes or one LM Studio port) | Accepted |
| [0003](0003-chromadb-for-text-index.md) | ChromaDB as the text/caption vector store, with caller-supplied embeddings | Accepted |
| [0004](0004-langgraph-state-machine-over-chain.md) | LangGraph `StateGraph` for the agent, not a chain or a hand-rolled loop | Accepted |
| [0005](0005-pin-transformers-4x.md) | Pin `transformers` to the 4.x line | Accepted |

To add one: copy the structure of any record, number it sequentially, link it
here, and reference it from the code or doc it affects.
