import os

import chromadb

from rankfuse.exceptions import StoreError
from rankfuse.stores.base import VectorStore


class ChromaStore(VectorStore):
    """File-based ChromaDB vector store implementation."""

    def __init__(self, persist_dir: str, collection_name: str = "rankfuse_collection"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        try:
            # Chroma client requires the path to exist before initialization
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise StoreError(
                f"Failed to initialize ChromaStore at '{persist_dir}': {e}"
            )

    def add(
        self,
        docs: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        if not docs or not embeddings or not ids:
            return

        try:
            self.collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        except Exception as e:
            raise StoreError(f"Failed to add documents to ChromaStore: {e}")

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["distances"],
            )
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            ids = results["ids"][0]
            distances = (
                results["distances"][0]
                if results.get("distances")
                else [0.0] * len(ids)
            )

            # Cosine similarity = 1.0 - Cosine distance
            return [(doc_id, 1.0 - float(dist)) for doc_id, dist in zip(ids, distances)]
        except Exception as e:
            raise StoreError(f"Failed to query ChromaStore: {e}")

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            raise StoreError(f"Failed to delete documents from ChromaStore: {e}")

    def get(self, ids: list[str] | None = None) -> list[dict]:
        try:
            if ids is not None:
                results = self.collection.get(
                    ids=ids, include=["documents", "metadatas"]
                )
            else:
                results = self.collection.get(include=["documents", "metadatas"])

            retrieved = []
            if results and results.get("ids"):
                for doc_id, doc_text, meta in zip(
                    results["ids"],
                    results["documents"],
                    results.get("metadatas") or [None] * len(results["ids"]),
                ):
                    retrieved.append(
                        {
                            "id": doc_id,
                            "text": doc_text,
                            "metadata": meta or {},
                        }
                    )
            return retrieved
        except Exception as e:
            raise StoreError(f"Failed to retrieve documents from ChromaStore: {e}")
