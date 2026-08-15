from google import genai

from rankfuse.embeddings.base import Embedder
from rankfuse.exceptions import EmbeddingError


class GeminiEmbedder(Embedder):
    """Embedder implementation using the modern Google GenAI SDK."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Google GenAI Client: {e}")

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
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
            )

            if not response or not response.embeddings:
                raise EmbeddingError("Gemini API returned empty or invalid response.")

            # Handle both ContentEmbedding objects (with .values) and raw floats/lists
            first_item = response.embeddings[0]
            if isinstance(first_item, (int, float)):
                embeddings = response.embeddings
            elif hasattr(first_item, "values"):
                embeddings = [emb.values for emb in response.embeddings]
            else:
                embeddings = response.embeddings

            # Handle case where response might be flat for a single text input
            if len(texts) == 1 and isinstance(embeddings[0], (int, float)):
                return [embeddings]

            return embeddings
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise e
            raise EmbeddingError(f"Gemini embedding API call failed: {e}")
