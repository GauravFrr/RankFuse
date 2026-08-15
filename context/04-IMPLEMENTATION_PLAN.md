# RankFuse — Implementation Plan

Build one phase at a time. Do not start a phase until the previous one's "Definition of Done" is met. Each phase should be reviewed before moving to the next.

---

## Phase 0 — Setup & Scaffolding

**Tasks:**
- Confirm final package name (check PyPI availability)
- Initialize git repo, `.gitignore`, `LICENSE` (MIT)
- Create `pyproject.toml` with: package metadata, Python 3.10+ requirement, dependencies (`chromadb`, `rank-bm25`, `sentence-transformers`, `google-generativeai`, `pydantic`, `pydantic-settings`), dev dependencies (`pytest`, `pytest-cov`, `ruff`, `black`)
- Create full folder structure from `03-PROJECT_STRUCTURE.md` with empty `__init__.py` files
- Set up `.github/workflows/test.yml` (runs `pytest` + `ruff check` on every PR)

**Definition of Done:** `pip install -e .` works locally, `pytest` runs (even with zero tests), CI is green on an empty commit.

---

## Phase 1 — Core Interfaces (Abstract Base Classes)

**Tasks:**
- `stores/base.py` — `VectorStore` ABC: `add(docs, embeddings, ids)`, `query(embedding, top_k) -> list[tuple[id, score]]`, `delete(ids)`
- `embeddings/base.py` — `Embedder` ABC: `embed(texts: list[str]) -> list[list[float]]`
- `reranker/base.py` — `Reranker` ABC: `rerank(query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]`
- `config.py` — `RetrieverConfig` Pydantic model with all settings fields, validation logic
- `exceptions.py` — `RankFuseError` base class, plus `EmbeddingError`, `StoreError`, `RerankerError`, `ConfigError`

**Why first:** everything else implements these contracts. Get the method signatures right before writing logic against them — changing an interface later means touching every implementation.

**Definition of Done:** All ABCs defined with type hints and docstrings. `RetrieverConfig()` raises `ConfigError` on missing required fields (test manually). No concrete implementations yet.

---

## Phase 2 — Dense Store (Chroma)

**Tasks:**
- `stores/chroma_store.py` — `ChromaStore(VectorStore)` implementing add/query/delete against a file-based `chromadb.PersistentClient`
- `tests/unit/test_chroma_store.py` — add documents, query, verify top-k ordering, using a temp directory fixture

**Port from Retryv:** Chroma setup/connection logic, but generalized (persist path becomes a config parameter, not hardcoded).

**Definition of Done:** Can add 10 sample documents and query for the top-3 nearest to a known query vector, in an isolated test using a temp directory. Temp directory is cleaned up after test.

---

## Phase 3 — Sparse Index (BM25)

**Tasks:**
- `sparse/bm25_index.py` — `BM25Index` class: `build(documents)`, `query(query_text, top_k) -> list[tuple[id, score]]`
- `tests/unit/test_bm25.py` — build over sample docs, verify exact keyword match ranks highly

**Port from Retryv:** BM25 wrapper logic around `rank_bm25`, generalized to take arbitrary document lists instead of the fixed dataset.

**Definition of Done:** A query containing an exact rare keyword from one document ranks that document first, in a unit test with no external API calls.

---

## Phase 4 — Fusion (RRF)

**Tasks:**
- `fusion/rrf.py` — `reciprocal_rank_fusion(ranked_list_1, ranked_list_2, k=60) -> list[tuple[id, score]]`
- `tests/unit/test_fusion.py` — hand-constructed small ranked lists with a known expected merged output

**Port from Retryv:** the exact RRF math already proven there — this is largely a direct port since RRF is pure functional logic, no external dependencies.

**Definition of Done:** Unit test with a manually calculable example (e.g., 3 docs in each list) passes, confirming the fusion formula is implemented correctly.

---

## Phase 5 — Embeddings (Gemini)

**Tasks:**
- `embeddings/gemini_embedder.py` — `GeminiEmbedder(Embedder)`, takes API key via config, calls Gemini embedding endpoint
- `tests/unit/test_gemini_embedder.py` — mocked API response, verify shape/type of output (no real API calls in unit tests)

**Definition of Done:** `GeminiEmbedder(api_key=...).embed(["hello", "world"])` returns two vectors of matching, expected dimensionality, tested against a mocked client.

---

## Phase 6 — Reranker (Cross-Encoder + LLM Judge)

**Tasks:**
- `reranker/cross_encoder.py` — `CrossEncoderReranker(Reranker)` using `sentence-transformers` cross-encoder model, runs locally, no API key needed
- `reranker/llm_judge.py` — `LLMJudgeReranker(Reranker)` using Gemini to score relevance of each candidate against the query
- `tests/unit/test_reranker.py` — for both, verify top_k output length and that a clearly relevant doc outranks a clearly irrelevant one

**Port from Retryv:** the reranking flow (top-20 → top-5 precision pass) that produced the recall jump — same two-stage philosophy, generalized to work on arbitrary candidate sets.

**Definition of Done:** Both rerankers pass their unit test independently. Cross-encoder test runs with no network access required (local model).

---

## Phase 7 — Retriever Orchestrator

**Tasks:**
- `chunking.py` — simple `split_text(text, chunk_size, overlap) -> list[str]`
- `retriever.py` — `Retriever` class: `__init__(config)`, `ingest(documents)`, `search(query, top_k) -> list[RetrievalResult]`. Wires embedder → store + sparse index → RRF fusion → reranker, in that order
- `tests/integration/test_retriever_e2e.py` — full flow with a mocked embedder (deterministic fake vectors) and real Chroma/BM25/RRF/cross-encoder

**Definition of Done:** `examples/quickstart.py` runs end-to-end locally: ingest 5-10 sample documents, search a query, get back sensibly ordered results with no errors.

---

## Phase 8 — Benchmark

**Tasks:**
- `benchmarks/datasets/fastapi_docs/` — prepare a public dataset (FastAPI docs) with a set of eval queries and known-relevant document IDs
- `benchmarks/run_benchmark.py` — computes recall for (a) naive dense-only search and (b) full hybrid+RRF+rerank pipeline, prints comparison
- `benchmarks/results.md` — record the actual numbers produced, with methodology explained

**Definition of Done:** Running `python benchmarks/run_benchmark.py` reproduces a clear, documented recall improvement, matching the spirit of Retryv's 23%→84% result on this new dataset (exact numbers will differ — that's expected and fine, document what's actually measured).

---

## Phase 9 — Docs & Examples

**Tasks:**
- `README.md` — install instructions, quickstart code block, benchmark highlight, link to full docs
- `docs/architecture.md`, `docs/api_reference.md` — expanded from the context docs in this set
- `examples/custom_embedder.py`, `examples/fastapi_integration.py`

**Definition of Done:** A developer with zero prior context can read the README, run the quickstart, and understand how to swap in their own embedder, in under 10 minutes.

---

## Phase 10 — CI/CD & PyPI Publish

**Tasks:**
- `.github/workflows/publish.yml` — builds package and publishes to PyPI on GitHub release tag
- Register package name on PyPI (test on TestPyPI first)
- Tag `v0.1.0` release

**Definition of Done:** `pip install rankfuse` works from a clean environment, pulling the real published package from PyPI.

---

## Suggested Order Summary

```
Phase 0: Setup
Phase 1: Interfaces          ┐
Phase 2: Chroma store         │  can be built in parallel
Phase 3: BM25 index           │  once Phase 1 interfaces exist
Phase 4: RRF fusion           │
Phase 5: Gemini embedder      ┘
Phase 6: Rerankers
Phase 7: Retriever orchestrator (needs everything above)
Phase 8: Benchmark
Phase 9: Docs & examples
Phase 10: CI/CD + publish
```

Phases 2-5 are independent of each other once Phase 1's interfaces are locked — they can be handed to Antigravity as separate units even out of strict numeric order, as long as Phase 1 is complete first and Phase 7 waits for all of them.
