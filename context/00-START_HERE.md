# RankFuse — START HERE (Agent Instructions)

This is the entry point. Read this file first, then read the other docs in this folder in the order listed below. This project is being built by an AI coding agent (Antigravity) under human review — this doc sets the working rules.

## Reading Order

1. **`00-START_HERE.md`** (this file) — working rules
2. **`01-PROJECT_OVERVIEW.md`** — what RankFuse is and why it exists
3. **`02-ARCHITECTURE.md`** — how it works internally, data flow, design decisions
4. **`03-PROJECT_STRUCTURE.md`** — exact file tree to build
5. **`05-API_REFERENCE.md`** — the exact public contract to implement (read before writing any code)
6. **`06-CODING_STYLE.md`** — how the code must be written, at the sentence/function level
7. **`04-IMPLEMENTATION_PLAN.md`** — the phased build order to actually execute

## Working Rules

### 1. One phase at a time
Build exactly one phase from `04-IMPLEMENTATION_PLAN.md` at a time, in order (Phase 0, then Phase 1, etc. — Phases 2-5 may be done in any order relative to each other, but only after Phase 1 is done, and Phase 7 only after 2-6 are all done). Do not start the next phase until the current phase's "Definition of Done" is explicitly confirmed complete.

### 2. Stop and present, don't cascade
After finishing a phase, stop. Summarize what was built, show the Definition of Done being met, and wait for explicit go-ahead before starting the next phase. Do not silently continue through multiple phases in one pass.

### 3. Every line of code follows `06-CODING_STYLE.md`
No exceptions. If a phase's implementation starts drifting toward over-engineering (extra abstraction, defensive code, verbose comments, generic naming) — stop and simplify before presenting the phase as done. Re-read the style guide's "Quick Self-Check" section before marking any phase complete.

### 4. The API in `05-API_REFERENCE.md` is fixed
Class names, method signatures, config field names, and exception names in that doc are the contract. Don't rename or restructure the public API while implementing — if something in that doc seems wrong or needs to change, flag it and ask, don't silently deviate.

### 5. Tests are part of the phase, not an afterthought
Every phase in the implementation plan that touches logic (Phases 1-8) includes its own tests. A phase isn't done until its tests exist and pass — not just until the feature "works when I try it manually."

### 6. When in doubt about scope, stay small
This is a v1 library, not a platform. If a task seems to be growing beyond what a phase's task list describes, that's a signal to stop and check in rather than build extra "nice to have" functionality unprompted.

## What Each Doc Is For (quick reference)

| Doc | Use it when... |
|---|---|
| `01-PROJECT_OVERVIEW.md` | You need to understand *why* a design choice was made, or explain the project to someone new |
| `02-ARCHITECTURE.md` | You're implementing any component and need to know how it connects to the others |
| `03-PROJECT_STRUCTURE.md` | You're creating a new file and need to know exactly where it goes and what it's responsible for |
| `05-API_REFERENCE.md` | You're writing anything in the public API surface — check the exact signature first |
| `06-CODING_STYLE.md` | You're about to write or review any code, at any point |
| `04-IMPLEMENTATION_PLAN.md` | You need to know what to build next and how to know when it's done |

## Definition of "Done" for the Whole Project (v0.1.0)

- All 10 phases in `04-IMPLEMENTATION_PLAN.md` complete
- `pip install rankfuse` works from a clean environment via PyPI
- `examples/quickstart.py` runs end-to-end with no errors
- `benchmarks/run_benchmark.py` produces a documented, reproducible recall improvement
- All tests pass in CI
- Code throughout matches `06-CODING_STYLE.md` — this project is a portfolio piece and must read as competently hand-written
