# RankFuse

Hybrid retrieval (dense + sparse + RRF fusion + optional reranking) for RAG pipelines.

Plug it in with two method calls — `ingest()` and `search()` — and your pipeline gets keyword-aware hybrid search with Reciprocal Rank Fusion on top of your existing dense embeddings.

## Install

```bash
pip install rankfuse
```

Lightweight install (~20MB): pulls in `chromadb`, `rank-bm25`, `google-genai`, and `pydantic`. No PyTorch or GPU binaries. Uses Gemini for embeddings and the optional LLM-judge reranker.

To use the local cross-encoder reranker (adds `sentence-transformers` + PyTorch, ~1-2GB on first use):

```bash
pip install rankfuse[cross-encoder]
```

Requires Python 3.10+. You'll need a [Gemini API key](https://aistudio.google.com/app/apikey) for embeddings.

## Quickstart

```python
from rankfuse import Retriever, RetrieverConfig

config = RetrieverConfig(
    embedder_provider="gemini",
    api_key="your-gemini-api-key",  # or set GEMINI_API_KEY env var
    persist_dir="./my_index",
    reranker_type="cross_encoder",  # or "llm_judge", or "none"
)

retriever = Retriever(config)

retriever.ingest([
    {"id": "doc1", "text": "Refunds are processed within 5-7 business days.", "metadata": {"source": "faq"}},
    {"id": "doc2", "text": "To reset your password, go to Settings > Security.", "metadata": {"source": "faq"}},
    {"id": "doc3", "text": "Check order status by logging into your dashboard.", "metadata": {"source": "faq"}},
])

results = retriever.search("what is the refund policy?", top_k=5)

for r in results:
    print(r.doc_id, r.score, r.text[:80])
```

Run the full working example:

```bash
GEMINI_API_KEY=your-key python examples/quickstart.py
```

## What it does

Standard RAG pipelines search with dense embeddings only. Dense embeddings are good at semantic similarity but miss exact keyword matches — a query for `"RFC 7231"` won't reliably surface a document that only contains the literal text `"RFC 7231"` unless the embedding space happens to capture it.

RankFuse adds a BM25 sparse index alongside the dense index, runs both in parallel, and merges the results using Reciprocal Rank Fusion (RRF). RRF works on rank position rather than raw scores, so it doesn't require normalization between the two score scales — it's simple and robust.

Optionally, a reranker (local cross-encoder or Gemini LLM-judge) does a precision pass over the fused top-N candidates before returning the final results.

## Reranker options

| `reranker_type` | What it uses | Cost |
|---|---|---|
| `"cross_encoder"` | `ms-marco-MiniLM-L-12-v2` (local, no API) | Free, ~200MB model download on first use |
| `"llm_judge"` | Gemini (API call per candidate) | API quota cost |
| `"none"` | No reranking, returns RRF-fused results directly | Free |

## Benchmark

Evaluated on 30 queries over the full FastAPI documentation corpus (154 documents). Hybrid search with stopword-filtered BM25 closes the candidate recall gap versus dense-only search — hybrid RRF-only matches dense-only at Recall@5 (0.90) while also providing exact-term coverage dense alone misses.

The standard equal-weight RRF configuration doesn't improve Recall@1 on this particular corpus — the release notes document causes keyword concentration that inflates BM25 scores for non-tutorial results. See [`benchmarks/results.md`](benchmarks/results.md) for the full methodology, diagnostic breakdown, and honest discussion of where hybrid search helps and where it doesn't.

## Swap in your own embedder or store

```python
from rankfuse.embeddings.base import Embedder

class MyEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # your embedding logic here
        ...

retriever = Retriever(config, embedder=MyEmbedder())
```

See [`examples/custom_embedder.py`](examples/custom_embedder.py) for a full working example.

## Docs

- [Architecture](docs/architecture.md) — how the components connect
- [API Reference](docs/api_reference.md) — full config fields and method signatures
- [Benchmark Results](benchmarks/results.md) — methodology and findings
- [FastAPI integration example](examples/fastapi_integration.py)

## License

MIT
