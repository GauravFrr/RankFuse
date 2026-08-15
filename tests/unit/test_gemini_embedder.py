from unittest.mock import patch

import pytest

from rankfuse.embeddings.gemini_embedder import GeminiEmbedder
from rankfuse.exceptions import EmbeddingError


def test_gemini_embedder_success():
    embedder = GeminiEmbedder(api_key="test-key")

    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_response = {"embedding": mock_embeddings}

    with patch(
        "google.generativeai.embed_content", return_value=mock_response
    ) as mock_embed:
        results = embedder.embed(["hello", "world"])

        mock_embed.assert_called_once_with(
            model="models/text-embedding-004", content=["hello", "world"]
        )

        assert results == mock_embeddings
        assert len(results) == 2


def test_gemini_embedder_single_flat_list_handling():
    embedder = GeminiEmbedder(api_key="test-key")
    mock_response = {"embedding": [0.1, 0.2, 0.3]}

    with patch(
        "google.generativeai.embed_content", return_value=mock_response
    ) as mock_embed:
        results = embedder.embed(["hello"])

        mock_embed.assert_called_once_with(
            model="models/text-embedding-004", content=["hello"]
        )

        assert results == [[0.1, 0.2, 0.3]]
        assert len(results) == 1


def test_gemini_embedder_failure():
    embedder = GeminiEmbedder(api_key="test-key")

    with patch(
        "google.generativeai.embed_content",
        side_effect=Exception("API connection refused"),
    ):
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])

        assert "Gemini embedding API call failed" in str(exc_info.value)
        assert "API connection refused" in str(exc_info.value)


def test_gemini_embedder_invalid_response():
    embedder = GeminiEmbedder(api_key="test-key")

    with patch("google.generativeai.embed_content", return_value={"wrong_key": []}):
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["test"])

        assert "Gemini API returned an invalid response structure" in str(
            exc_info.value
        )
