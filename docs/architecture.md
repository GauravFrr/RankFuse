# Architecture

RankFuse is a retrieval library that wires together dense search, sparse search, RRF fusion, and optional reranking into a single `ingest()` / `search()` interface. This document explains how the pieces fit together and why each design decision was made.

## High-Level Flow

```
User documents
      │
      ▼
 Chunking (chunking.py)
      │
  ┌───┴────────────────────┐
  ▼                         ▼
Embedder              BM25 Index Builder
(Gemini / custom)     (sparse/bm25_index.py)
  │                         │
  ▼                         ▼
Chroma dense store    BM25 on-disk index

      ── search(query) ──

Dense search (top-N)    Sparse search (top-N)
      │                         │
      └──────────┬──────────────┘
                 ▼
           RRF Fusion
           (fusion/rrf.py)
                 │
                 ▼
           Reranker (optional)
           cross-encoder or LLM-judge
           top-20 → top-K
                 │
                 ▼
          Final ranked results
```

## Components

### Chunking (`chunking.py`)

Splits raw document text into smaller chunks before indexing. Configurable chunk size and overlap via `RetrieverConfig`. Chunking happens at ingest time — search always operates on chunks, and results are deduplicated back to document level by the caller if needed.

### Embedder (`embeddings/`)

Abstract base class (`base.py`) with one required method:

```python
def embed(self, texts: list[str]) -> list[list[float]]: ...
```

The built-in implementation is `GeminiEmbedder` (`gemini_embedder.py`), which calls the Gemini embedding API (`gemini-embedding-001` by default). The user's own API key is passed via config — the library never bundles a key.

You can swap in any embedder by subclassing `Embedder`. See [`examples/custom_embedder.py`](../examples/custom_embedder.py).

### Dense Store (`stores/`)

Abstract `VectorStore` base class, implemented by `ChromaStore`. Uses a file-based Chroma persistent client — no server process required. The `persist_dir` config field controls where it writes to disk.

The store handles two operations: `add()` (index embeddings at ingest time) and `query()` (nearest-neighbor retrieval at search time).

### Sparse Index (`sparse/bm25_index.py`)

Wraps `rank_bm25`. Builds a keyword-frequency index over the same document chunks the dense store holds. At query time, it tokenizes the query with the same stopword-filtered tokenizer used during indexing, so common words like `to`, `and`, `how` don't inflate BM25 scores.

This index catches exact-term matches (IDs, names, version numbers, jargon) that dense embeddings sometimes miss when the embedding space doesn't capture rare tokens well.

### Fusion (`fusion/rrf.py`)

Reciprocal Rank Fusion merges the dense-search ranked list and sparse-search ranked list into a single list. The formula is:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
```

The `k` constant (default: 60, from the original IR literature) prevents top-ranked documents from dominating too strongly. Because RRF works on rank position rather than raw scores, it doesn't require score normalization — dense cosine similarity and BM25 scores live on different scales, but their rank positions are comparable.

### Reranker (`reranker/`)

Takes the fused top-N candidates (default: top-20) and applies a more expensive but more accurate scoring pass before returning the final top-K:

- **`CrossEncoderReranker`** — uses `ms-marco-MiniLM-L-12-v2` locally via `sentence-transformers`. Free, runs offline, ~200MB model download on first use.
- **`LLMJudgeReranker`** — uses Gemini (`gemini-flash-latest` by default) to score each candidate against the query. Costs API calls; more accurate on semantic edge cases.
- Set `reranker_type="none"` to skip reranking entirely and return the RRF-fused list directly.

### Retriever (`retriever.py`)

The orchestrator — the only class most users touch. Wires all components together and exposes two methods: `ingest()` and `search()`. Config is passed at construction time; component initialization happens there, not lazily during the first search.

### Config (`config.py`)

A single Pydantic settings object. Raises `ConfigError` at construction if a required field is missing. All tuning parameters (top_k values, chunk size, RRF k constant) are here, not scattered across the codebase.

## Key Design Decisions

| Decision | Why |
|---|---|
| Abstract base classes for Store, Embedder, Reranker | The library's whole point is swappability — FAISS, pgvector, OpenAI, Cohere can all be added later without changing `retriever.py` |
| `src/` layout | Standard Python packaging; forces testing against the installed package |
| RRF over score normalization | RRF doesn't need dense/sparse scores on the same scale — simpler, more robust, well-documented in IR literature |
| File-based Chroma (no server) | Zero infrastructure for end users — install and run |
| BYOK (bring your own key) | Library is free to run; same pattern as LangChain, LlamaIndex |
| Two-stage retrieval (fusion → rerank) | Cheap stage casts a wide net; expensive reranker only touches a small candidate set (cost control) |
| Fail loud at config time | Missing API key raises immediately at `Retriever()` construction, not mid-search when it's harder to debug |

## Data Contracts

What flows between components:

- **Into `ingest()`:** `list[dict]` with `{"id": str, "text": str, "metadata": dict}`
- **Embedder out:** `list[list[float]]` — one vector per text
- **Dense/Sparse search out:** `list[tuple[doc_id, score]]` — ranked list
- **RRF out:** `list[tuple[doc_id, fused_score]]` — single merged ranked list
- **Reranker out:** `list[RetrievalResult]` — final ordering
- **`.search()` returns:** `list[RetrievalResult]` — typed, ranked

## Error Handling

- All errors surface as one of the project's own exception types (`EmbeddingError`, `StoreError`, `RerankerError`, `ConfigError`), all inheriting from `RankFuseError`. Raw provider exceptions don't leak to callers.
- Config problems raise at `Retriever()` construction time, not during search.
- Sparse and dense failures raise — they don't silently return empty results.
