from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a single retrieved document chunk with its relevance score."""

    doc_id: str
    text: str
    score: float
    metadata: dict


class Reranker(ABC):
    """Abstract Base Class for document reranking models."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank a list of candidate documents based on their relevance to the query.

        Args:
            query: The search query string.
            candidates: A list of RetrievalResult candidate documents.
            top_k: The number of top documents to return after reranking.

        Returns:
            A list of top_k RetrievalResult documents, sorted by relevance.
        """
        pass
