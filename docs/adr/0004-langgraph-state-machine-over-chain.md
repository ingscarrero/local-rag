# ADR-0004: LangGraph `StateGraph` for the agent, not a chain or a hand-rolled loop

- **Status:** Accepted
- **Date:** 2026-06-11
- **Affects:** `agent/graph.py`, `agent/nodes.py`, `agent/state.py`

## Context

The control flow is: route → (direct | retrieve → grade → (generate |
rewrite → retrieve …)) with a bounded cycle. It has typed shared state,
conditional edges and one loop. It must be legible to a reader, debuggable
from a trace, and testable without models.

Options considered:

1. **Plain Python loop.** Fewest dependencies. But the state machine ends up
   implicit in `if/while` nesting, the trace is ad-hoc, and adding a node
   (e.g. a re-ranker) means editing the loop body.
2. **LangChain LCEL chains.** Good for linear pipelines; cycles and conditional
   branching are awkward and the corrective retry does not express naturally.
3. **Multi-agent frameworks (CrewAI, AutoGen).** Built for several agents
   negotiating; this is *one* agent with a decision process. Heavy, and the
   abstractions hide exactly the control flow we want to show.
4. **LangGraph `StateGraph`.** Explicit nodes, a `TypedDict` state with
   per-field reducers, conditional edges as plain functions, cycles allowed,
   `get_graph()` for introspection.

## Decision

Option 4. `AgentState` is a `TypedDict(total=False)` so nodes return partial
updates; `trace` uses an append reducer so every node contributes one line.
Nodes are plain functions `State → partial State`; the only closure is
`make_retrieve_node(retrievers)`, which injects the retrievers so the graph can
be built with fakes. `decide_after_grade` and the route lambda are the two
conditional edges.

## Consequences

- **Positive:** the graph definition fits on one screen and mirrors the
  diagram in [03 — Architecture](../03-architecture.md#the-agent-graph)
  one-to-one; the wiring is asserted by a test
  ([tests/test_graph.py](../../tests/test_graph.py)).
- **Positive:** the `trace` is a first-class output — the CLI prints it and the
  tests assert on it — which makes the corrective loop observable.
- **Positive:** inserting a re-ranker or a decomposition node is `add_node` +
  two edges, no loop surgery.
- **Negative:** a sizeable dependency (LangGraph + langchain-core) for what is,
  today, six nodes. Accepted for legibility and extensibility.
- **Negative:** LangGraph's API has moved quickly; the project pins
  `langgraph>=0.2.50` and uses only the stable core (`StateGraph`, `START`,
  `END`, conditional edges).
- **Design rule that fell out of this:** keep every LLM decision in its own
  node with a tiny JSON prompt (`route`, per-passage `grade`, `rewrite`), so a
  7B model can follow it and each decision is individually testable.
