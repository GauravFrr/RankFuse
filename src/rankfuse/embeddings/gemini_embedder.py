import google.generativeai as genai

from rankfuse.embeddings.base import Embedder
from rankfuse.exceptions import EmbeddingError


class GeminiEmbedder(Embedder):
    """Embedder implementation using the Gemini Generative AI SDK."""

    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name

        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            raise EmbeddingError(f"Failed to configure Gemini SDK: {e}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            A list of embedding vectors (list of lists of floats).
        """
        if not texts:
            return []

        try:
            response = genai.embed_content(
                model=self.model_name,
                content=texts,
            )

            if not response or "embedding" not in response:
                raise EmbeddingError(
                    "Gemini API returned an invalid response structure."
                )

            embeddings = response["embedding"]

            # Handle case where SDK returns a flat list for a single text input
            if len(texts) == 1 and isinstance(embeddings[0], (int, float)):
                return [embeddings]

            return embeddings
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise e
            raise EmbeddingError(f"Gemini embedding API call failed: {e}")
