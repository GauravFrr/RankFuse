from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract Base Class for document and query embedding generators."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A list of float lists, where each float list is the vector embedding
            for the corresponding input text.
        """
        pass
