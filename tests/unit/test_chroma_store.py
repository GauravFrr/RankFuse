import pytest

from rankfuse.exceptions import StoreError
from rankfuse.stores.chroma_store import ChromaStore


def test_chroma_store_lifecycle(temp_dir):
    store = ChromaStore(persist_dir=temp_dir)

    # Verify initial empty query
    assert store.query([0.1, 0.2, 0.3], top_k=5) == []

    # Add sample documents
    docs = [
        "Python is an interpreted programming language.",
        "ChromaDB is an open-source AI vector database.",
        "Deep learning models require vector embeddings.",
    ]
    # Simple embeddings (3-dimensional)
    embeddings = [
        [0.9, 0.1, 0.1],  # Python
        [0.1, 0.9, 0.1],  # Chroma
        [0.1, 0.1, 0.9],  # Deep learning
    ]
    ids = ["doc1", "doc2", "doc3"]
    metadatas = [
        {"topic": "python"},
        {"topic": "database"},
        {"topic": "ai"},
    ]

    store.add(docs=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)

    # Verify get by ID
    retrieved = store.get(ids=["doc1", "doc2"])
    assert len(retrieved) == 2
    assert retrieved[0]["id"] == "doc1"
    assert retrieved[0]["text"] == docs[0]
    assert retrieved[0]["metadata"] == metadatas[0]

    assert retrieved[1]["id"] == "doc2"
    assert retrieved[1]["text"] == docs[1]
    assert retrieved[1]["metadata"] == metadatas[1]

    # Query matching doc1
    query_vector = [0.8, 0.1, 0.1]
    results = store.query(query_vector, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "doc1"
    assert results[0][1] > 0.9

    # Delete doc2
    store.delete(ids=["doc2"])

    # Verify deleted doc cannot be retrieved
    assert store.get(ids=["doc2"]) == []

    # Query again, doc2 should be gone
    query_vector_db = [0.1, 0.9, 0.1]
    results_after = store.query(query_vector_db, top_k=2)
    assert "doc2" not in [r[0] for r in results_after]


def test_chroma_store_invalid_init():
    # Attempting to initialize with an invalid path format to trigger StoreError
    # Using a path with null byte which is illegal on all filesystems
    invalid_path = "\0invalid_path"
    with pytest.raises(StoreError):
        ChromaStore(persist_dir=invalid_path)
