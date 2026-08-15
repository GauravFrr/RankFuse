# RankFuse — Architecture

## High-Level Flow

```
                         ┌─────────────────┐
                         │   User's Docs    │
                         └────────┬─────────┘
                                  │  .ingest(documents)
                                  ▼
                   ┌──────────────────────────┐
                   │        Chunking           │  (chunking.py)
                   └──────────────┬────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                  ▼
      ┌──────────────────┐              ┌──────────────────────┐
      │  Embedder          │             │   BM25 Index Builder  │
      │  (Gemini/OpenAI)   │             │   (sparse/bm25_index) │
      └─────────┬─────────┘              └───────────┬───────────┘
                 ▼                                    ▼
      ┌──────────────────┐              ┌──────────────────────┐
      │   Chroma Store     │             │   BM25 Index (disk)   │
      │  (dense vectors)   │             │                        │
      └─────────┬─────────┘              └───────────┬───────────┘
                 │                                    │
                 │        .search(query, top_k=5)     │
                 ▼                                    ▼
      ┌──────────────────┐              ┌──────────────────────┐
      │  Dense Search      │             │   Sparse Search        │
      │  (top-N results)   │             │   (top-N results)      │
      └─────────┬─────────┘              └───────────┬───────────┘
                 └────────────────┬───────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   RRF Fusion             │  (fusion/rrf.py)
                     │   merges both ranked      │
                     │   lists into one           │
                     └────────────┬─────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   Reranker                │  (reranker/*)
                     │   cross-encoder or         │
                     │   LLM-as-judge              │
                     │   top-20 → top-K precise   │
                     └────────────┬─────────────┘
                                  ▼
                        Final ranked results
                        returned to the user
```

## Component Breakdown

### 1. Chunking (`chunking.py`)
Splits raw documents into retrievable units before indexing. Simple, configurable (chunk size, overlap). Not the focus of v1 — sensible defaults, pluggable later.

### 2. Embedder (`embeddings/`)
Abstract interface (`base.py`) with concrete implementations (`gemini_embedder.py`, later `openai_embedder.py`). Converts text into vector embeddings for dense storage/search. This is where the user's own API key is used — the library never embeds its own key.

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

### 3. Dense Store (`stores/`)
Abstract `VectorStore` interface, with `ChromaStore` as the v1 implementation. File-based Chroma persistence — no server process required. Handles `add()` (index documents + embeddings) and `query()` (nearest-neighbor search).

```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, docs, embeddings, ids): ...
    @abstractmethod
    def query(self, embedding, top_k): ...
```

### 4. Sparse Index (`sparse/bm25_index.py`)
Wraps `rank_bm25`. Builds a keyword-frequency index over the same chunked documents. Catches exact-term matches dense embeddings can miss (IDs, names, jargon, numbers).

### 5. Fusion (`fusion/rrf.py`)
Reciprocal Rank Fusion — combines the dense-search ranked list and sparse-search ranked list into a single merged ranking, without needing to normalize different similarity score scales (RRF works on rank position, not raw score — this is what makes it robust).

```
RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
```

### 6. Reranker (`reranker/`)
Takes the fused top-N (e.g., top-20) candidates and re-scores them with a more expensive but more accurate model — either a local cross-encoder (`sentence-transformers`) or an LLM-as-judge call (Gemini). Returns the final top-K (e.g., top-5). This is the precision pass that gave Retryv its biggest recall/precision jump.

### 7. Retriever (`retriever.py`)
The orchestrator — the only class most users will ever touch directly. Wires embedder + stores + fusion + reranker together behind two methods: `ingest()` and `search()`. All the complexity above is hidden behind this single entry point.

### 8. Config (`config.py`)
A single Pydantic settings object controlling: which embedder to use, which reranker to use, chunk size, top_k at each stage, persist directory, RRF's `k` constant, etc. Passed once at `Retriever()` construction time.

## Key Architectural Decisions & Why

| Decision | Why |
|---|---|
| Abstract base classes for Store/Embedder/Reranker | Lets the library grow (pgvector, FAISS, OpenAI, Cohere) without breaking the public API or rewriting `retriever.py` |
| `src/` layout | Standard Python packaging practice; forces testing against the installed package, avoids import path bugs |
| RRF over score-normalization fusion | RRF doesn't require dense/sparse scores to be on the same scale — simpler, more robust, well-established in IR literature |
| File-based Chroma (no server) | Zero infra for the end user — matches the "no hosting" design principle |
| BYOK (bring your own key) | Keeps the library free to run for the author; standard pattern (LangChain, LlamaIndex do the same) |
| Reranking as a separate pluggable stage | Different users have different cost/latency/accuracy tradeoffs — local cross-encoder is free but slower to set up; LLM-as-judge is more accurate but costs API calls |
| Two-stage retrieval (broad fusion → narrow rerank) | Cheap stage casts a wide net (top-20), expensive stage only runs on a small candidate set (cost control) |

## Data Flow Contracts (what gets passed between components)

- **Documents in:** `list[str]` or `list[dict]` with `{"id": ..., "text": ..., "metadata": {...}}`
- **Embedder out:** `list[list[float]]` — one vector per input text
- **Dense/Sparse search out:** `list[tuple[doc_id, score]]` — ranked list
- **RRF out:** `list[tuple[doc_id, fused_score]]` — single merged ranked list
- **Reranker out:** `list[tuple[doc_id, rerank_score]]` — final ordering
- **`.search()` returns:** `list[RetrievalResult]` — a typed object with `doc_id`, `text`, `metadata`, `score`

## Error Handling Philosophy

- Custom exceptions in `exceptions.py` (e.g., `EmbeddingError`, `StoreNotFoundError`, `RerankerError`) rather than letting raw provider exceptions leak to the user
- Fail loud and early on config problems (e.g., missing API key) at `Retriever()` construction time, not mid-search
- Sparse and dense search failures should not silently return empty results — raise, don't swallow
