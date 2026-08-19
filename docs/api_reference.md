# API Reference

## Installation

```bash
pip install rankfuse
```

## `RetrieverConfig`

Pydantic settings object. Pass it to `Retriever()` at construction time.

```python
from rankfuse import RetrieverConfig

config = RetrieverConfig(
    embedder_provider="gemini",
    api_key="your-gemini-api-key",  # or set GEMINI_API_KEY env var
    persist_dir="./my_index",
    reranker_type="cross_encoder",
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `embedder_provider` | `str` | `"gemini"` | Which embedder to use. `"gemini"` is the only built-in option in v0.1.0. |
| `api_key` | `str` | required | Your Gemini API key. Can also be set via `GEMINI_API_KEY` environment variable. |
| `persist_dir` | `str` | `"./rankfuse_index"` | Directory where Chroma vector store and BM25 index files are saved. |
| `reranker_type` | `str` | `"cross_encoder"` | `"cross_encoder"` (local, free) \| `"llm_judge"` (Gemini API) \| `"none"` (no reranking). |
| `chunk_size` | `int` | `500` | Characters per chunk during ingest. |
| `chunk_overlap` | `int` | `50` | Overlap between adjacent chunks in characters. |
| `dense_top_k` | `int` | `20` | How many candidates to pull from dense search before fusion. |
| `sparse_top_k` | `int` | `20` | How many candidates to pull from BM25 before fusion. |
| `rerank_top_k` | `int` | `5` | Final number of results returned after reranking. |
| `rrf_k` | `int` | `60` | RRF rank constant. 60 is the standard value from IR literature. |

Raises `ConfigError` at construction if `api_key` is missing and not found in the environment.

---

## `Retriever`

The main entry point. Constructs the full retrieval pipeline from a config.

```python
from rankfuse import Retriever, RetrieverConfig

retriever = Retriever(config)
```

To pass a custom embedder:

```python
retriever = Retriever(config, embedder=MyEmbedder())
```

### `.ingest(documents)`

```python
retriever.ingest(documents: list[dict]) -> None
```

Indexes a list of documents into both the dense store and the BM25 sparse index. Each document must be a dict with:

- `"id"` (str, required) — unique identifier
- `"text"` (str, required) — document content
- `"metadata"` (dict, optional) — any additional fields

Documents are chunked, embedded, and stored. Re-ingesting a document with the same `id` updates it in-place — it does not create a duplicate.

Raises `EmbeddingError` if the embedding API call fails.

### `.search(query, top_k=None)`

```python
retriever.search(query: str, top_k: int | None = None) -> list[RetrievalResult]
```

Runs dense search + sparse search in parallel, merges via RRF, applies the configured reranker, and returns the final ranked list.

- `top_k` overrides `config.rerank_top_k` for this call if provided.
- Returns an empty list (not an error) if the index is empty.
- Raises `StoreError` or `RerankerError` on underlying failures.

### `.delete(ids)`

```python
retriever.delete(ids: list[str]) -> None
```

Removes documents from both the dense store and the sparse index by their IDs.

---

## `RetrievalResult`

```python
@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    metadata: dict
```

Returned in ranked order (highest relevance first) from `.search()`. The `score` is the reranker score if a reranker is configured, or the RRF fused score otherwise.

---

## Exceptions

All exceptions inherit from `RankFuseError`, so you can catch everything with a single `except RankFuseError` if needed.

```python
from rankfuse.exceptions import RankFuseError, EmbeddingError, StoreError, RerankerError, ConfigError
```

| Exception | When it's raised |
|---|---|
| `ConfigError` | Missing or invalid config at `Retriever()` construction time |
| `EmbeddingError` | Embedding API call fails during `ingest()` or `search()` |
| `StoreError` | Vector store read/write failure |
| `RerankerError` | Reranking model or API failure |

---

## Extension Points

### Custom Embedder

Subclass `Embedder` and pass an instance to `Retriever()`:

```python
from rankfuse.embeddings.base import Embedder

class MyEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # return one float vector per input text
        ...

retriever = Retriever(config, embedder=MyEmbedder())
```

See [`examples/custom_embedder.py`](../examples/custom_embedder.py) for a full working example using a local sentence-transformers model.

### Custom Vector Store

Subclass `VectorStore`:

```python
from rankfuse.stores.base import VectorStore

class MyStore(VectorStore):
    def add(self, docs, embeddings, ids): ...
    def query(self, embedding, top_k): ...
    def delete(self, ids): ...
```

### Custom Reranker

Subclass `Reranker`:

```python
from rankfuse.reranker.base import Reranker, RetrievalResult

class MyReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        ...
```

---

## Versioning

The stable public API in v0.1.0:

- `Retriever`, `RetrieverConfig`, `RetrievalResult`
- The four exception classes
- The three base classes (`Embedder`, `VectorStore`, `Reranker`)

Internal implementations (`ChromaStore`, `GeminiEmbedder`, `BM25Index`, `CrossEncoderReranker`, `LLMJudgeReranker`) are not part of the stability guarantee until v1.0.0 — they may change between minor versions.
