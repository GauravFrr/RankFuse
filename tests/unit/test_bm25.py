from rankfuse.sparse.bm25_index import BM25Index


def test_bm25_index_rare_keyword_ranking(temp_dir):
    index = BM25Index(persist_dir=temp_dir)

    # Initial query on empty index
    assert index.query("python", top_k=5) == []

    # Build index over sample documents
    documents = [
        {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
        {
            "id": "doc2",
            "text": "Python is an interpreted, high-level programming language.",
        },
        {
            "id": "doc3",
            "text": "ChromaDB is a database designed for AI vector embeddings.",
        },
        {
            "id": "doc4",
            "text": "Rare xiphoid cartilage issues can cause chest discomfort.",
        },
    ]

    index.build(documents)

    # Query for exact rare keyword "xiphoid"
    results = index.query("xiphoid", top_k=5)
    assert len(results) == 1
    # doc4 must be ranked first
    assert results[0][0] == "doc4"
    assert results[0][1] > 0.0

    # Query for multiple words, verify rank order
    results_multi = index.query("Python programming database", top_k=5)
    # doc2 contains both "Python" and "programming", doc3 contains "database"
    assert len(results_multi) == 2
    assert results_multi[0][0] == "doc2"
    assert results_multi[1][0] == "doc3"


def test_bm25_index_persistence(temp_dir):
    # Build and save
    index1 = BM25Index(persist_dir=temp_dir)
    documents = [
        {"id": "doc1", "text": "Python is a snake and a programming language."},
        {"id": "doc2", "text": "Java is another popular language."},
        {"id": "doc3", "text": "C++ is a compiled programming language."},
    ]
    index1.build(documents)

    # Load from the same persist directory
    index2 = BM25Index(persist_dir=temp_dir)
    results = index2.query("Java", top_k=5)
    assert len(results) == 1
    assert results[0][0] == "doc2"


def test_bm25_index_filters_stopwords(temp_dir):
    index = BM25Index(persist_dir=temp_dir)
    documents = [
        {"id": "doc1", "text": "to be or not to be that is the question."}
    ]
    index.build(documents)
    # Querying for only stopwords should return empty results
    assert index.query("to be or", top_k=5) == []
