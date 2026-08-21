from rankfuse.exceptions import RerankerError
from rankfuse.reranker.base import Reranker, RetrievalResult


class CrossEncoderReranker(Reranker):
    """Local Cross-Encoder reranker using sentence-transformers.

    Requires the cross-encoder extra:
        pip install rankfuse[cross-encoder]
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError:
            raise ImportError(
                "CrossEncoderReranker requires the 'cross-encoder' extra. "
                "Install it with: pip install rankfuse[cross-encoder]"
            )
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazily initialize the CrossEncoder model on first rerank call."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "CrossEncoderReranker requires the 'cross-encoder' extra. "
                    "Install it with: pip install rankfuse[cross-encoder]"
                )
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
            pairs = []
            for c in candidates:
                # Retrieve human-readable context from metadata if available
                context = ""
                if c.metadata:
                    for key in ["title", "source"]:
                        val = c.metadata.get(key)
                        if val:
                            context = f"Document Title: {val}\n"
                            break

                # Graceful fallback: use raw text if no metadata title/source exists
                passage = f"{context}Content: {c.text}" if context else c.text
                pairs.append((query, passage))

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
