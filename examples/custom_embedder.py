"""
custom_embedder.py — shows how to swap in a local sentence-transformers embedder
instead of the default Gemini one.

This is useful if you want zero API cost, fully offline embeddings, or a
domain-specific model. The custom embedder just needs to implement one method:
embed(texts) -> list[list[float]].

Run with:
    pip install sentence-transformers
    python examples/custom_embedder.py
"""

import shutil

from sentence_transformers import SentenceTransformer

from rankfuse import Retriever, RetrieverConfig
from rankfuse.embeddings.base import Embedder


class LocalEmbedder(Embedder):
    """Embedder backed by a local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


def main():
    persist_dir = "./custom_embedder_index"
    shutil.rmtree(persist_dir, ignore_errors=True)

    # api_key is required by RetrieverConfig even when using a custom embedder,
    # because the config validator checks for it. Pass a placeholder if you're
    # not using Gemini at all — it won't be used for embedding.
    config = RetrieverConfig(
        embedder_provider="gemini",
        api_key="not-used",
        persist_dir=persist_dir,
        reranker_type="cross_encoder",
    )

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    retriever = Retriever(config, embedder=embedder)

    print("Ingesting documents with local embedder...")
    retriever.ingest([
        {"id": "a", "text": "RankFuse adds hybrid search to any RAG pipeline.", "metadata": {}},
        {"id": "b", "text": "BM25 catches exact keyword matches that dense embeddings miss.", "metadata": {}},
        {"id": "c", "text": "Reciprocal Rank Fusion merges ranked lists without score normalization.", "metadata": {}},
        {"id": "d", "text": "The cross-encoder reranker runs fully offline with no API key.", "metadata": {}},
    ])

    query = "how does fusion work?"
    print(f"\nSearching: '{query}'")
    results = retriever.search(query, top_k=3)

    print("\nResults:")
    for r in results:
        print(f"  [{r.doc_id}] score={r.score:.4f}  {r.text}")

    shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
