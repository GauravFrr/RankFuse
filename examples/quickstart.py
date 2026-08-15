import os
import shutil
from unittest.mock import MagicMock, patch

from rankfuse import Retriever, RetrieverConfig

# Temporary directory for persisting index
persist_dir = "./quickstart_index"
shutil.rmtree(persist_dir, ignore_errors=True)

# 1. Setup API key and config
api_key = os.environ.get("GEMINI_API_KEY")
is_mocked = False
if not api_key:
    api_key = "dummy-api-key"
    is_mocked = True

config = RetrieverConfig(
    embedder_provider="gemini",
    api_key=api_key,
    persist_dir=persist_dir,
    reranker_type="cross_encoder",
)


# 2. Mocking embedder helper if using dummy key
def make_mock_embedding(text):
    val = [0.1, 0.1, 0.1]
    lower_text = text.lower()
    if "refund" in lower_text:
        val = [0.9, 0.05, 0.05]
    elif "password" in lower_text or "reset" in lower_text:
        val = [0.05, 0.9, 0.05]

    emb = MagicMock()
    emb.values = val
    return emb


def mock_embed_content(model, contents):
    mock_response = MagicMock()
    mock_response.embeddings = [make_mock_embedding(txt) for txt in contents]
    return mock_response


def run_retriever():
    retriever = Retriever(config)

    print("Ingesting sample documents...")
    retriever.ingest(
        [
            {
                "id": "doc1",
                "text": "Refunds are processed within 5-7 business days.",
                "metadata": {"source": "faq"},
            },
            {
                "id": "doc2",
                "text": "To reset your password, go to Settings > Security.",
                "metadata": {"source": "faq"},
            },
            {
                "id": "doc3",
                "text": "Check order status by logging into your dashboard.",
                "metadata": {"source": "faq"},
            },
        ]
    )
    print("Ingestion complete.")

    query = "what is the refund policy?"
    print(f"\nSearching for: '{query}'")
    results = retriever.search(query, top_k=2)

    print("\nSearch results:")
    for r in results:
        print(f"- ID: {r.doc_id} | Score: {r.score:.4f} | Text: {r.text}")

    # Cleanup
    shutil.rmtree(persist_dir, ignore_errors=True)


if is_mocked:
    print("No GEMINI_API_KEY environment variable detected.")
    print("Running quickstart with mocked Google GenAI client...\n")
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = mock_embed_content
        mock_client_cls.return_value = mock_client
        run_retriever()
else:
    print("GEMINI_API_KEY detected. Running quickstart with real Gemini API...\n")
    run_retriever()
