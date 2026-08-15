from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract Base Class for dense vector databases."""

    @abstractmethod
    def add(
        self,
        docs: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents and their embeddings to the vector store.

        Args:
            docs: A list of document texts.
            embeddings: A list of vector embeddings corresponding to the documents.
            ids: A list of unique identifiers for the documents.
            metadatas: Optional list of metadata dicts corresponding to the documents.
        """
        pass

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Query the vector store for the nearest neighbors of the given embedding.

        Args:
            embedding: The query vector.
            top_k: The number of nearest neighbors to return.

        Returns:
            A list of tuples, where each tuple contains the document ID and its
            similarity score.
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents from the vector store by their IDs.

        Args:
            ids: A list of document IDs to delete.
        """
        pass

    @abstractmethod
    def get(self, ids: list[str]) -> list[dict]:
        """Retrieve documents and their metadata by their IDs.

        Args:
            ids: A list of document IDs.

        Returns:
            A list of dicts, where each dict contains 'id', 'text', and 'metadata'.
        """
        pass
