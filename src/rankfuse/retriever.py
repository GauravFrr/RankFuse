from rankfuse.chunking import split_text
from rankfuse.config import RetrieverConfig
from rankfuse.exceptions import ConfigError
from rankfuse.fusion.rrf import reciprocal_rank_fusion
from rankfuse.reranker.base import RetrievalResult


class Retriever:
    """Hybrid retriever orchestrating dense, sparse, fusion, and reranking."""

    def __init__(self, config: RetrieverConfig):
        self.config = config

        # Initialize the Embedder
        if self.config.embedder_provider.lower() == "gemini":
            from rankfuse.embeddings.gemini_embedder import GeminiEmbedder

            self.embedder = GeminiEmbedder(api_key=self.config.api_key)
        else:
            raise ConfigError(
                f"Unsupported embedder provider '{self.config.embedder_provider}'."
            )

        # Initialize the Vector Store
        from rankfuse.stores.chroma_store import ChromaStore

        self.vector_store = ChromaStore(persist_dir=self.config.persist_dir)

        # Initialize the BM25 Sparse Index
        from rankfuse.sparse.bm25_index import BM25Index

        self.sparse_index = BM25Index(persist_dir=self.config.persist_dir)

        # Initialize the Reranker
        reranker_type = self.config.reranker_type.lower()
        if reranker_type == "cross_encoder":
            from rankfuse.reranker.cross_encoder import CrossEncoderReranker

            self.reranker = CrossEncoderReranker()
        elif reranker_type == "llm_judge":
            from rankfuse.reranker.llm_judge import LLMJudgeReranker

            self.reranker = LLMJudgeReranker(api_key=self.config.api_key)
        elif reranker_type == "none":
            self.reranker = None
        else:
            raise ConfigError(f"Unsupported reranker type '{reranker_type}'.")

    def ingest(self, documents: list[dict]) -> None:
        """Ingest a list of documents into both dense and sparse indexes.

        Args:
            documents: A list of dicts, each with 'id', 'text', and optional 'metadata'.
        """
        if not documents:
            return

        all_texts = []
        all_ids = []
        all_metadatas = []

        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc.get("metadata") or {}

            # Make ingest idempotent for the same doc_id
            self.delete([doc_id])

            chunks = split_text(text, self.config.chunk_size, self.config.chunk_overlap)
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_meta = metadata.copy()
                chunk_meta["doc_id"] = doc_id
                chunk_meta["chunk_index"] = i

                all_texts.append(chunk_text)
                all_ids.append(chunk_id)
                all_metadatas.append(chunk_meta)

        if all_texts:
            embeddings = self.embedder.embed(all_texts)
            self.vector_store.add(
                docs=all_texts,
                embeddings=embeddings,
                ids=all_ids,
                metadatas=all_metadatas,
            )

        # Rebuild the BM25 index over all chunks currently in vector store.
        # NOTE: Rebuilding the full BM25 index on every ingest() call is a v1
        # limitation where ingest cost grows with the total corpus size over time.
        all_stored = self.vector_store.get()
        bm25_docs = [{"id": item["id"], "text": item["text"]} for item in all_stored]
        self.sparse_index.build(bm25_docs)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Perform hybrid search + fusion + reranking on the query.

        Args:
            query: The search query.
            top_k: Number of final results to return (overrides config if set).

        Returns:
            A list of RetrievalResult documents.
        """
        if not query:
            return []

        final_top_k = top_k if top_k is not None else self.config.rerank_top_k

        # 1. Dense retrieve
        query_emb = self.embedder.embed([query])[0]
        dense_results = self.vector_store.query(
            query_emb, top_k=self.config.dense_top_k
        )

        # 2. Sparse retrieve
        sparse_results = self.sparse_index.query(query, top_k=self.config.sparse_top_k)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_results = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )

        if not fused_results:
            return []

        # 4. Fetch document texts and metadata for all fused candidate chunk IDs
        fused_ids = [doc_id for doc_id, _ in fused_results]
        fused_details = self.vector_store.get(fused_ids)
        details_map = {item["id"]: item for item in fused_details}

        candidates = []
        for doc_id, fused_score in fused_results:
            details = details_map.get(doc_id)
            if details:
                orig_id = details["metadata"].get("doc_id", doc_id)
                candidates.append(
                    RetrievalResult(
                        doc_id=orig_id,
                        text=details["text"],
                        score=fused_score,
                        metadata=details["metadata"],
                    )
                )

        # 5. Two-stage reranking
        if self.reranker is not None:
            # Rerank all candidate chunks first
            reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))
        else:
            reranked = candidates

        # 6. Deduplicate by doc_id after reranking (keeping highest-scoring chunk per document)
        final_results = []
        seen_doc_ids = set()
        for res in reranked:
            if res.doc_id not in seen_doc_ids:
                seen_doc_ids.add(res.doc_id)
                final_results.append(res)
                if len(final_results) == final_top_k:
                    break

        return final_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents and their chunks from indexes by original document IDs.

        Args:
            ids: A list of original document IDs.
        """
        if not ids:
            return

        all_stored = self.vector_store.get()
        chunk_ids_to_delete = []

        for item in all_stored:
            doc_id = item["metadata"].get("doc_id")
            if doc_id in ids:
                chunk_ids_to_delete.append(item["id"])

        if chunk_ids_to_delete:
            self.vector_store.delete(chunk_ids_to_delete)

            # Rebuild sparse index over remaining documents.
            # NOTE: Rebuilding the full BM25 index on every delete() call is a v1
            # limitation where delete cost grows with the total corpus size over time.
            remaining_stored = self.vector_store.get()
            bm25_docs = [
                {"id": item["id"], "text": item["text"]} for item in remaining_stored
            ]
            self.sparse_index.build(bm25_docs)
