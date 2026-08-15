# AGENT.md — RankFuse Build Instructions

This file is the operating manual for any AI coding agent (Antigravity) working on this repository. Read this before touching any code. Full context docs live alongside this file — read those too, in the order below, but this file is the quick-reference rulebook you should re-check constantly during the build.

## Project in One Line

RankFuse is a pip-installable Python library that adds hybrid (dense + sparse) retrieval with RRF fusion and reranking to any RAG pipeline — no hosting, no server, user brings their own API key.

## Context Docs (read in this order before writing code)

1. `00-START_HERE.md` — reading order + working rules
2. `01-PROJECT_OVERVIEW.md` — what & why
3. `02-ARCHITECTURE.md` — how components connect
4. `03-PROJECT_STRUCTURE.md` — exact file tree
5. `05-API_REFERENCE.md` — fixed public API contract
6. `06-CODING_STYLE.md` — how the code must be written
7. `04-IMPLEMENTATION_PLAN.md` — the phased task list to execute

## Source Reference Project

The original Retryv project — the working RAG pipeline this library generalizes retrieval logic from (hybrid dense+sparse search, RRF fusion, reranking; the source of the 23%→84% recall improvement referenced throughout the docs) — is located at:

```
F:\Retryv
```

Use this only as a **reference**, not something to copy-paste from. When a phase in `04-IMPLEMENTATION_PLAN.md` says "Port from Retryv," it means: read the relevant logic there, understand the approach, then reimplement it generalized and config-driven per `02-ARCHITECTURE.md` and `06-CODING_STYLE.md` — not lift the code as-is. Retryv's code is tightly coupled to its own hardcoded paths/keys/dataset; none of that should carry over.

Phases where this applies: Phase 2 (Chroma store), Phase 3 (BM25 index), Phase 4 (RRF fusion), Phase 6 (rerankers).

## Non-Negotiable Rules

1. **Follow `06-CODING_STYLE.md` on every single line.** Simple, readable, human-sounding code. No over-abstraction, no defensive-coding-everywhere, no comment-per-line, no generic naming (`data`, `handler`, `manager`). This is the single most important rule in this file — re-check it before marking anything done.
2. **One phase at a time**, in the order defined by `04-IMPLEMENTATION_PLAN.md`. Stop after each phase. Do not chain multiple phases together in one pass.
3. **Do not start a phase until the previous phase's Definition of Done is confirmed.** Definition of Done is written in `04-IMPLEMENTATION_PLAN.md` for every phase — check against it literally before saying a phase is finished.
4. **The public API in `05-API_REFERENCE.md` is fixed.** Class names, method names, config fields, exception names — do not rename or restructure. If something there looks wrong, flag it and ask instead of silently changing it.
5. **Every phase that touches logic includes its own tests.** A phase is not done just because it "runs once manually" — its corresponding tests (from `03-PROJECT_STRUCTURE.md` / the phase's task list) must exist and pass.
6. **No scope creep.** Build exactly what the current phase's task list says. Don't add extra features, extra config options, or extra abstractions "while you're in there."
7. **BYOK always.** Never hardcode any API key anywhere in the codebase, examples, or tests. User-supplied key only, via config or environment variable.
8. **No hosting, no server processes.** Everything runs locally on the user's machine — file-based Chroma, no Docker requirement for the library itself (Docker is fine for the separate demo app, not the library).

## Workflow for Each Phase

1. Re-read the phase's task list and Definition of Done in `04-IMPLEMENTATION_PLAN.md`
2. Implement exactly those tasks, following `03-PROJECT_STRUCTURE.md` for file placement and `05-API_REFERENCE.md` for any public-facing signatures
3. Write/run the tests for that phase
4. Self-check against `06-CODING_STYLE.md`'s "Quick Self-Check" section — rewrite anything that fails it
5. Summarize what was built, confirm the Definition of Done is met, and stop — wait for explicit approval before continuing to the next phase

## Commands (once Phase 0 scaffolding exists)

```bash
# install in editable mode
pip install -e ".[dev]"

# run tests
pytest

# run only unit tests (fast, no external calls)
pytest tests/unit

# lint
ruff check .

# format
black .

# run the benchmark
python benchmarks/run_benchmark.py
```

## Definition of Done for the Full Project

See `00-START_HERE.md` → "Definition of Done for the Whole Project (v0.1.0)". Short version: all 10 phases done, `pip install rankfuse` works from PyPI, quickstart runs clean, benchmark is reproducible, CI is green, and the code reads like a human wrote it.