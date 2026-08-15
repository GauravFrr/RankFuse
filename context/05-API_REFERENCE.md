# RankFuse — Public API Reference

This is the exact contract the library must expose. Treat this as the spec — implementation details can vary, but these signatures and behaviors must hold.

## Installation (target)

```bash
pip install rankfuse
```

## Quickstart (target usage)

```python
from rankfuse import Retriever, RetrieverConfig

config = RetrieverConfig(
    embedder_provider="gemini",
    api_key="your-gemini-api-key",   # or read from GEMINI_API_KEY env var
    persist_dir="./my_index",
    reranker_type="cross_encoder",   # or "llm_judge"
)

retriever = Retriever(config)

retriever.ingest([
    {"id": "doc1", "text": "Refunds are processed within 5-7 business days.", "metadata": {"source": "faq"}},
    {"id": "doc2", "text": "To reset your password, go to Settings > Security.", "metadata": {"source": "faq"}},
    # ... more documents
])

results = retriever.search("what is the refund policy?", top_k=5)

for r in results:
    print(r.doc_id, r.score, r.text[:80])
```

## `RetrieverConfig`

Pydantic settings object. All fields below.

| Field | Type | Default | Notes |
|---|---|---|---|
| `embedder_provider` | `str` | `"gemini"` | `"gemini"` \| `"openai"` (openai post-v1) |
| `api_key` | `str` | required | User's own key. Can also be read from env var (e.g. `GEMINI_API_KEY`) |
| `persist_dir` | `str` | `"./rankfuse_index"` | Where Chroma + BM25 index files live on disk |
| `reranker_type` | `str` | `"cross_encoder"` | `"cross_encoder"` \| `"llm_judge"` \| `"none"` |
| `chunk_size` | `int` | `500` | Characters per chunk during ingest, if auto-chunking is used |
| `chunk_overlap` | `int` | `50` | Overlap between adjacent chunks |
| `dense_top_k` | `int` | `20` | Candidates pulled from dense search before fusion |
| `sparse_top_k` | `int` | `20` | Candidates pulled from BM25 before fusion |
| `rerank_top_k` | `int` | `5` | Final result count after reranking |
| `rrf_k` | `int` | `60` | RRF constant (standard default from IR literature) |

Raises `ConfigError` at construction time if `api_key` is missing and not found in environment.

## `Retriever`

### `Retriever(config: RetrieverConfig)`
Constructs the retriever. Initializes embedder, vector store, and reranker based on config. Does not require documents yet.

### `.ingest(documents: list[dict]) -> None`
- `documents`: list of `{"id": str, "text": str, "metadata": dict (optional)}`
- Chunks (if needed), embeds, and indexes into both the dense store and the sparse index
- Idempotent for the same `id` — re-ingesting a document with the same id updates it, does not duplicate
- Raises `EmbeddingError` if the embedding API call fails

### `.search(query: str, top_k: int | None = None) -> list[RetrievalResult]`
- Runs dense search + sparse search, fuses via RRF, reranks, returns final ranked list
- `top_k` overrides `config.rerank_top_k` for this call if provided
- Returns empty list (not an error) if index is empty
- Raises `StoreError` / `RerankerError` on underlying failures

### `.delete(ids: list[str]) -> None`
Removes documents from both dense and sparse indices by id.

## `RetrievalResult`

```python
@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    metadata: dict
```

Returned in ranked order (highest relevance first) from `.search()`.

## Extension Points (for advanced users)

### Custom Embedder

```python
from rankfuse.embeddings.base import Embedder

class MyEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

config = RetrieverConfig(embedder_provider="custom", ...)
retriever = Retriever(config, embedder=MyEmbedder())
```

### Custom Vector Store

```python
from rankfuse.stores.base import VectorStore

class MyStore(VectorStore):
    def add(self, docs, embeddings, ids): ...
    def query(self, embedding, top_k): ...
    def delete(self, ids): ...
```

### Custom Reranker

```python
from rankfuse.reranker.base import Reranker

class MyReranker(Reranker):
    def rerank(self, query, candidates, top_k): ...
```

## Exceptions

All inherit from `RankFuseError`:

- `ConfigError` — invalid or missing configuration
- `EmbeddingError` — embedding API/model failure
- `StoreError` — vector store read/write failure
- `RerankerError` — reranking model/API failure

## Versioning & Stability Notes for v0.1.0

- Public API surface for v0.1.0: `Retriever`, `RetrieverConfig`, `RetrievalResult`, the four exception classes, and the three base classes (`Embedder`, `VectorStore`, `Reranker`)
- Everything under `stores/`, `embeddings/`, `reranker/`, `sparse/`, `fusion/` other than the base classes should be considered internal/swappable and not part of the stability guarantee until v1.0.0
