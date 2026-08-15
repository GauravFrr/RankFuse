from unittest.mock import MagicMock, patch

import pytest

from rankfuse.embeddings.gemini_embedder import GeminiEmbedder
from rankfuse.exceptions import EmbeddingError


def test_gemini_embedder_success():
    mock_embedding1 = MagicMock()
    mock_embedding1.values = [0.1, 0.2, 0.3]
    mock_embedding2 = MagicMock()
    mock_embedding2.values = [0.4, 0.5, 0.6]

    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding1, mock_embedding2]

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = GeminiEmbedder(api_key="test-key")
        results = embedder.embed(["hello", "world"])

        mock_client.models.embed_content.assert_called_once_with(
            model="text-embedding-004", contents=["hello", "world"]
        )

        assert results == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert len(results) == 2


def test_gemini_embedder_single_flat_list_handling():
    mock_response = MagicMock()
    mock_response.embeddings = [0.1, 0.2, 0.3]

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = GeminiEmbedder(api_key="test-key")
        results = embedder.embed(["hello"])

        mock_client.models.embed_content.assert_called_once_with(
            model="text-embedding-004", contents=["hello"]
        )

        assert results == [[0.1, 0.2, 0.3]]
        assert len(results) == 1


def test_gemini_embedder_failure():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception(
            "API connection refused"
        )
        mock_client_cls.return_value = mock_client

        embedder = GeminiEmbedder(api_key="test-key")
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])

        assert "Gemini embedding API call failed" in str(exc_info.value)
        assert "API connection refused" in str(exc_info.value)


def test_gemini_embedder_invalid_response():
    mock_response = MagicMock()
    mock_response.embeddings = None

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = GeminiEmbedder(api_key="test-key")
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])

        assert "Gemini API returned empty or invalid response" in str(exc_info.value)
