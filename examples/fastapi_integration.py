"""
fastapi_integration.py — shows how to wire RankFuse into a FastAPI app.

The retriever is initialized once at startup (using the lifespan context) and
stored in app.state so each request handler can use it without rebuilding the
index on every call.

Run with:
    pip install fastapi uvicorn
    GEMINI_API_KEY=your-key python examples/fastapi_integration.py

Then query it:
    curl "http://localhost:8000/search?q=refund+policy&top_k=3"
"""

import os
import shutil
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from rankfuse import Retriever, RetrieverConfig
from rankfuse.exceptions import RankFuseError

PERSIST_DIR = "./fastapi_example_index"

SAMPLE_DOCS = [
    {"id": "faq-1", "text": "Refunds are processed within 5-7 business days of approval.", "metadata": {"source": "faq"}},
    {"id": "faq-2", "text": "To reset your password, go to Settings > Security > Reset Password.", "metadata": {"source": "faq"}},
    {"id": "faq-3", "text": "Track your order status by logging into your account dashboard.", "metadata": {"source": "faq"}},
    {"id": "faq-4", "text": "Contact support at support@example.com for billing questions.", "metadata": {"source": "faq"}},
    {"id": "faq-5", "text": "Shipping takes 3-5 business days for standard delivery.", "metadata": {"source": "faq"}},
    {"id": "faq-6", "text": "To cancel a subscription, go to Account > Billing > Cancel.", "metadata": {"source": "faq"}},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index once at startup
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    config = RetrieverConfig(
        embedder_provider="gemini",
        api_key=api_key,
        persist_dir=PERSIST_DIR,
        reranker_type="cross_encoder",
    )

    retriever = Retriever(config)
    retriever.ingest(SAMPLE_DOCS)
    print(f"Index ready: {len(SAMPLE_DOCS)} documents ingested.")

    app.state.retriever = retriever
    yield

    # Cleanup on shutdown
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)


app = FastAPI(
    title="RankFuse Search API",
    description="Hybrid retrieval example powered by RankFuse",
    lifespan=lifespan,
)


@app.get("/search")
def search(
    q: str = Query(..., description="The search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """Search the document index using hybrid retrieval."""
    retriever: Retriever = app.state.retriever
    try:
        results = retriever.search(q, top_k=top_k)
    except RankFuseError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "query": q,
        "results": [
            {"id": r.doc_id, "score": round(r.score, 4), "text": r.text, "metadata": r.metadata}
            for r in results
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
