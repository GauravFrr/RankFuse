from sentence_transformers import CrossEncoder

from rankfuse.exceptions import RerankerError
from rankfuse.reranker.base import Reranker, RetrievalResult


class CrossEncoderReranker(Reranker):
    """Local Cross-Encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazily initialize the CrossEncoder model."""
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                raise RerankerError(
                    f"Failed to load CrossEncoder model '{self.model_name}': {e}"
                )
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank a list of candidate documents using the Cross-Encoder model.

        Args:
            query: The search query.
            candidates: List of RetrievalResult candidates to rerank.
            top_k: Number of top reranked candidates to return.

        Returns:
            A list of top_k RetrievalResult documents, sorted by descending relevance.
        """
        if not candidates:
            return []

        try:
            pairs = [(query, c.text) for c in candidates]
            scores = self.model.predict(pairs)

            reranked = []
            for c, score in zip(candidates, scores):
                reranked.append(
                    RetrievalResult(
                        doc_id=c.doc_id,
                        text=c.text,
                        score=float(score),
                        metadata=c.metadata,
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]
        except Exception as e:
            raise RerankerError(f"CrossEncoder reranking failed: {e}")
