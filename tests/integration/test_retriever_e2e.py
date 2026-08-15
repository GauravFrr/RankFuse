from unittest.mock import MagicMock, patch

from rankfuse.config import RetrieverConfig
from rankfuse.retriever import Retriever


def make_mock_embedding(text):
    # Deterministic vectors based on keywords
    val = [0.1, 0.1, 0.1]
    lower_text = text.lower()
    if "python" in lower_text:
        val = [0.9, 0.05, 0.05]
    elif "cat" in lower_text:
        val = [0.05, 0.9, 0.05]
    elif "database" in lower_text or "chroma" in lower_text:
        val = [0.05, 0.05, 0.9]

    emb = MagicMock()
    emb.values = val
    return emb


def mock_embed_content(model, contents):
    mock_response = MagicMock()
    mock_response.embeddings = [make_mock_embedding(txt) for txt in contents]
    return mock_response


def test_retriever_e2e_flow(temp_dir):
    config = RetrieverConfig(
        api_key="test-api-key",
        persist_dir=temp_dir,
        chunk_size=100,
        chunk_overlap=10,
        dense_top_k=5,
        sparse_top_k=5,
        rerank_top_k=2,
        rrf_k=60,
    )

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = mock_embed_content
        mock_client_cls.return_value = mock_client

        retriever = Retriever(config)

        # Ingest documents
        documents = [
            {
                "id": "python_doc",
                "text": "Python is a programming language widely used in ML.",
                "metadata": {"category": "coding"},
            },
            {
                "id": "cat_doc",
                "text": "The cat is a domestic species of small carnivorous mammal.",
                "metadata": {"category": "animals"},
            },
            {
                "id": "chroma_doc",
                "text": "ChromaDB is an open-source AI vector database for embeddings.",
                "metadata": {"category": "database"},
            },
        ]

        retriever.ingest(documents)

        # Search for Python
        results_python = retriever.search("python programming", top_k=1)
        assert len(results_python) == 1
        assert results_python[0].doc_id == "python_doc"
        assert "ML" in results_python[0].text
        assert results_python[0].metadata["category"] == "coding"

        # Search for database
        results_db = retriever.search("vector database", top_k=1)
        assert len(results_db) == 1
        assert results_db[0].doc_id == "chroma_doc"
        assert "vector database" in results_db[0].text

        # Delete python_doc
        retriever.delete(["python_doc"])

        # Search for Python again, python_doc should be gone
        results_after = retriever.search("python programming", top_k=2)
        doc_ids = [r.doc_id for r in results_after]
        assert "python_doc" not in doc_ids
