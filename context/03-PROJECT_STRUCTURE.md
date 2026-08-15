# RankFuse — Project Structure

Full repository layout with the purpose of every file. Use this as the exact scaffold to generate.

```
rankfuse/
├── src/
│   └── rankfuse/
│       ├── __init__.py              # Public API: exports Retriever, config, exceptions
│       ├── retriever.py             # Retriever class — the orchestrator, main public entry point
│       ├── config.py                # RetrieverConfig (Pydantic) — all tunable settings
│       ├── exceptions.py            # EmbeddingError, StoreError, RerankerError, ConfigError
│       ├── chunking.py              # split_text() — chunk size/overlap document splitting
│       │
│       ├── stores/
│       │   ├── __init__.py
│       │   ├── base.py              # VectorStore ABC — add(), query(), delete()
│       │   └── chroma_store.py      # ChromaStore(VectorStore) — file-based Chroma wrapper
│       │
│       ├── sparse/
│       │   ├── __init__.py
│       │   └── bm25_index.py        # BM25Index — build(), query() over rank_bm25
│       │
│       ├── fusion/
│       │   ├── __init__.py
│       │   └── rrf.py               # reciprocal_rank_fusion(dense_results, sparse_results, k)
│       │
│       ├── reranker/
│       │   ├── __init__.py
│       │   ├── base.py              # Reranker ABC — rerank(query, candidates, top_k)
│       │   ├── cross_encoder.py     # CrossEncoderReranker — local sentence-transformers model
│       │   └── llm_judge.py         # LLMJudgeReranker — Gemini-based relevance scoring
│       │
│       └── embeddings/
│           ├── __init__.py
│           ├── base.py              # Embedder ABC — embed(texts) -> list[list[float]]
│           ├── gemini_embedder.py   # GeminiEmbedder(Embedder)
│           └── openai_embedder.py   # OpenAIEmbedder(Embedder) — added post-v1
│
├── tests/
│   ├── unit/
│   │   ├── test_fusion.py           # RRF correctness with known small examples
│   │   ├── test_bm25.py             # BM25Index build/query, mocked, no real API calls
│   │   ├── test_chroma_store.py     # ChromaStore add/query with temp dir
│   │   ├── test_reranker.py         # Reranker interface + cross-encoder unit test
│   │   └── test_config.py           # Config validation (missing key -> raises)
│   ├── integration/
│   │   └── test_retriever_e2e.py    # Full ingest -> search flow, real Chroma, mocked embedder
│   └── conftest.py                  # Shared pytest fixtures (sample docs, temp dirs, mock embedder)
│
├── benchmarks/
│   ├── run_benchmark.py             # Reproduces recall comparison: naive search vs hybrid+rerank
│   ├── datasets/
│   │   └── fastapi_docs/            # Public FastAPI documentation, chunked, with eval queries
│   └── results.md                  # Published, dated benchmark numbers with methodology
│
├── examples/
│   ├── quickstart.py                # Minimal ingest + search example
│   ├── custom_embedder.py           # How to plug in a different Embedder implementation
│   └── fastapi_integration.py       # Using rankfuse inside a FastAPI endpoint
│
├── docs/
│   ├── architecture.md              # (mirrors this doc set, lives in-repo for GitHub readers)
│   └── api_reference.md             # Public class/method reference
│
├── .github/
│   └── workflows/
│       ├── test.yml                 # Run unit tests + lint on every PR
│       └── publish.yml              # Build + publish to PyPI on GitHub release
│
├── pyproject.toml                   # Package metadata, dependencies, build system (no setup.py)
├── README.md                        # GitHub landing page: install, quickstart, benchmark highlight
├── LICENSE                          # MIT
├── CONTRIBUTING.md                  # How to set up dev env, run tests, submit PRs
└── .gitignore
```

## File-by-File Purpose Notes

**`__init__.py` (top level)** — this defines what `from rankfuse import X` exposes. Keep it minimal: `Retriever`, `RetrieverConfig`, and the exception classes. Everything else (stores, embedders, rerankers) is imported from its submodule directly — keeps the top-level namespace clean.

**`retriever.py`** — the only file most users read. Should be readable top-to-bottom as: construct with config → `ingest(docs)` → `search(query, top_k)`. All the wiring of embedder/store/sparse-index/fusion/reranker happens here, but each piece is a one-line call into its own module — this file should not contain algorithm logic itself, only orchestration.

**`config.py`** — single `RetrieverConfig(BaseSettings)` class. Fields like `embedder_provider`, `api_key`, `reranker_type`, `persist_dir`, `chunk_size`, `chunk_overlap`, `dense_top_k`, `sparse_top_k`, `rerank_top_k`, `rrf_k`. Validates on construction (e.g., raise `ConfigError` if `api_key` missing for the chosen provider).

**`stores/base.py` and `embeddings/base.py` and `reranker/base.py`** — these three ABCs are the extension points of the whole library. Write these first, before any concrete implementation, and get the method signatures right — everything else depends on them.

**`benchmarks/results.md`** — not just numbers, include methodology (dataset, query set, how recall was measured, naive-search baseline vs hybrid+rerank) so it's independently verifiable, not just asserted.

**`tests/conftest.py`** — critical for velocity. Fixtures should include: a small sample document set, a mocked `Embedder` (returns deterministic fake vectors, no real API calls needed for unit tests), and a temp `persist_dir` fixture that auto-cleans after each test.

## Naming Decision Needed Before Scaffolding

Package name — pick before running `pip install` anywhere:
- Check availability on PyPI first (`pip install <name>` should currently fail / show nothing)
- Candidates to check: `rankfuse`, `rrfusion`, `hybrid-retrieve`
- Must match: GitHub repo name, PyPI package name, and the importable module name (`src/<name>/`)
