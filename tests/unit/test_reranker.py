from unittest.mock import MagicMock, patch

from rankfuse.reranker.base import RetrievalResult
from rankfuse.reranker.cross_encoder import CrossEncoderReranker
from rankfuse.reranker.llm_judge import LLMJudgeReranker


def test_cross_encoder_reranker_real():
    # Use standard small MS-MARCO model
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "python programming language"
    c1 = RetrievalResult(
        doc_id="doc1",
        text="Python is a general-purpose programming language commonly used in AI.",
        score=0.1,
        metadata={},
    )
    c2 = RetrievalResult(
        doc_id="doc2",
        text="The domestic cat is a small carnivorous mammal.",
        score=0.2,
        metadata={},
    )

    # Rerank and request top 1
    results = reranker.rerank(query, [c2, c1], top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "doc1"  # Relevant doc should outrank irrelevant one
    # MS-MARCO outputs raw logits which can be negative
    assert results[0].score > -10.0


def test_cross_encoder_reranker_empty():
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.rerank("query", [], top_k=5) == []


def test_llm_judge_reranker_mocked():
    # Mocking the client and generator responses
    mock_response_relevant = MagicMock()
    mock_response_relevant.text = '{"score": 0.95}'

    mock_response_irrelevant = MagicMock()
    mock_response_irrelevant.text = '{"score": 0.05}'

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        # side_effect to return relevant response first, then irrelevant response
        mock_client.models.generate_content.side_effect = [
            mock_response_irrelevant,
            mock_response_relevant,
        ]
        mock_client_cls.return_value = mock_client

        reranker = LLMJudgeReranker(api_key="test-key")

        c1 = RetrievalResult(
            doc_id="doc1",
            text="Irrelevant text chunk.",
            score=0.5,
            metadata={},
        )
        c2 = RetrievalResult(
            doc_id="doc2",
            text="Highly relevant text chunk.",
            score=0.5,
            metadata={},
        )

        results = reranker.rerank("test query", [c1, c2], top_k=1)

        # Verify calls
        assert mock_client.models.generate_content.call_count == 2

        # Verify c2 outranks c1 due to LLM score (0.95 vs 0.05) and respects top_k=1
        assert len(results) == 1
        assert results[0].doc_id == "doc2"
        assert results[0].score == 0.95


def test_llm_judge_reranker_empty():
    reranker = LLMJudgeReranker(api_key="test-key")
    assert reranker.rerank("query", [], top_k=5) == []


def test_cross_encoder_reranker_handles_missing_metadata():
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    query = "python programming language"
    c1 = RetrievalResult(
        doc_id="doc1",
        text="Python is a general-purpose programming language commonly used in AI.",
        score=0.1,
        metadata=None,  # No metadata dict at all
    )
    c2 = RetrievalResult(
        doc_id="doc2",
        text="The domestic cat is a small carnivorous mammal.",
        score=0.2,
        metadata={},  # Empty metadata dict
    )
    
    results = reranker.rerank(query, [c2, c1], top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[0].score > -10.0
