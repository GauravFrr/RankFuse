import json
import os
import shutil
import sys
import threading
import time

from dotenv import load_dotenv

# Ensure project root is in sys.path when running from benchmarks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai

from rankfuse.config import RetrieverConfig
from rankfuse.embeddings.gemini_embedder import GeminiEmbedder
from rankfuse.exceptions import EmbeddingError
from rankfuse.retriever import Retriever


class RotatingGeminiEmbedder(GeminiEmbedder):
    """Benchmark helper to handle free tier rate limits via key rotation and caching."""
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        super().__init__(api_key, model_name)
        self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self.key_idx = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_idx])
        self._lock = threading.Lock()

        # Load persistent cache
        self.cache_file = os.path.join("benchmarks", "embeddings_cache.json")
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}")

    def _save_cache(self):
        try:
            temp_file = self.cache_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
            os.replace(temp_file, self.cache_file)
        except Exception:
            pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Fast path for single text caching
        if len(texts) == 1 and texts[0] in self.cache:
            return [self.cache[texts[0]]]

        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Check cache for elements in batch
            batch_to_embed = []
            cached_map = {}
            for t in batch:
                if t in self.cache:
                    cached_map[t] = self.cache[t]
                else:
                    batch_to_embed.append(t)

            embeddings = []
            if batch_to_embed:
                # Sleep 4.5 seconds between batches to stay under IP-based RPM limits
                if i > 0:
                    time.sleep(4.5)

                max_attempts = len(self.api_keys) * 2
                attempts = 0
                while attempts < max_attempts:
                    try:
                        response = self.client.models.embed_content(
                            model=self.model_name,
                            contents=batch_to_embed,
                        )
                        break
                    except Exception as e:
                        should_rotate = False
                        status_code = getattr(e, "status_code", None)
                        err_str = str(e).lower()
                        if status_code in (403, 429):
                            should_rotate = True
                        elif any(p in err_str for p in ["429", "resource_exhausted", "403", "permission_denied", "suspended", "api key", "unauthorized"]):
                            should_rotate = True

                        if should_rotate and attempts < max_attempts - 1:
                            attempts += 1
                            self.key_idx = (self.key_idx + 1) % len(self.api_keys)
                            sleep_time = 3.0 if ("429" in err_str or "resource_exhausted" in err_str) else 0.5
                            print(f"[RotatingEmbedder] Error hit. Sleeping {sleep_time}s and rotating to key index {self.key_idx}...")
                            self.client = genai.Client(api_key=self.api_keys[self.key_idx])
                            time.sleep(sleep_time)
                        else:
                            raise e

                if not response or not response.embeddings:
                    raise EmbeddingError("Gemini API returned empty or invalid response.")

                first_item = response.embeddings[0]
                if isinstance(first_item, (int, float)):
                    embeddings = response.embeddings
                elif hasattr(first_item, "values"):
                    embeddings = [emb.values for emb in response.embeddings]
                else:
                    embeddings = response.embeddings

                if len(batch_to_embed) == 1 and isinstance(embeddings[0], (int, float)):
                    embeddings = [embeddings]

                # Update cache
                for t, emb in zip(batch_to_embed, embeddings):
                    self.cache[t] = emb
                self._save_cache()

            # Reconstruct full batch embeddings in original order
            batch_embeddings = []
            embed_idx = 0
            for t in batch:
                if t in cached_map:
                    batch_embeddings.append(cached_map[t])
                else:
                    batch_embeddings.append(embeddings[embed_idx])
                    embed_idx += 1

            all_embeddings.extend(batch_embeddings)
        return all_embeddings

from google.genai import types

from rankfuse.exceptions import RerankerError
from rankfuse.reranker.base import RetrievalResult
from rankfuse.reranker.llm_judge import LLMJudgeReranker, RelevanceScore


class RotatingLLMJudgeReranker(LLMJudgeReranker):
    """Benchmark helper to handle LLM judge requests using Gemini with key rotation and candidate dedup."""
    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash-lite"):
        super().__init__(api_key, model_name)
        self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self.key_idx = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_idx])
        self._lock = threading.Lock()

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []

        # Deduplicate candidates at doc level to minimize API calls
        unique_candidates = []
        seen_docs = set()
        for c in candidates:
            if c.doc_id not in seen_docs:
                seen_docs.add(c.doc_id)
                unique_candidates.append(c)

        # Rerank at most top-5 candidates to be token-efficient
        candidates_to_eval = unique_candidates[:5]

        try:
            reranked = []
            for c in candidates_to_eval:
                prompt = (
                    "Analyze how relevant the document is to the query.\n"
                    "Score from 0.0 (irrelevant) to 1.0 (highly relevant).\n\n"
                    f"Query: {query}\n"
                    f"Document: {c.text}"
                )

                time.sleep(0.4) # Brief pause to help respect rate limits

                response = None
                max_attempts = len(self.api_keys) * 3
                attempts = 0
                while attempts < max_attempts:
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=RelevanceScore,
                                temperature=0.0,
                            ),
                        )
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        status_code = getattr(e, "status_code", None)

                        # Handle rate limit (429) or temporary service unavailable (503)
                        should_rotate = (status_code in (403, 429, 503))
                        if not should_rotate:
                            should_rotate = any(p in err_str for p in ["429", "resource_exhausted", "403", "permission_denied", "suspended", "api key", "unauthorized", "503", "unavailable", "high demand"])

                        if should_rotate and attempts < max_attempts - 1:
                            attempts += 1
                            self.key_idx = (self.key_idx + 1) % len(self.api_keys)
                            is_heavy = any(x in err_str for x in ["429", "resource_exhausted", "503", "unavailable", "high demand"])
                            sleep_time = 4.0 if is_heavy else 0.5
                            print(f"[RotatingLLM] Error hit ({status_code}) on key index {self.key_idx} for model '{self.model_name}': {e}. Sleeping {sleep_time}s and rotating...")
                            self.client = genai.Client(api_key=self.api_keys[self.key_idx])
                            time.sleep(sleep_time)
                        else:
                            # Re-raise unexpected exceptions or when all retry attempts are exhausted
                            raise e

                if not response or not response.text:
                    raise RerankerError("LLM judge returned an empty response.")

                score_data = RelevanceScore.model_validate_json(response.text)
                score = score_data.score

                reranked.append(
                    RetrievalResult(
                        doc_id=c.doc_id,
                        text=c.text,
                        score=float(score),
                        metadata=c.metadata,
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]

        except Exception as e:
            if isinstance(e, RerankerError):
                raise e
            raise RerankerError(f"LLM judge reranking failed: {e}")

def main():
    # 1. Load API Key
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        # Fallback to Retryv .env if running on user machine
        load_dotenv(r"F:\Retryv\.env")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in environment or F:\\Retryv\\.env")
        sys.exit(1)

    # Pass all keys if comma-separated to support rotation
    pass

    print("API Key loaded successfully.")

    # 2. Paths
    dataset_dir = os.path.join("benchmarks", "datasets", "fastapi_docs")
    queries_file = os.path.join("benchmarks", "datasets", "eval_queries.json")
    persist_dir = "./benchmark_index"

    # Cleanup previous index if any
    shutil.rmtree(persist_dir, ignore_errors=True)

    # 3. Load Queries
    with open(queries_file, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)
    print(f"Loaded {len(eval_queries)} evaluation queries.")

    # 4. Load Corpus Documents
    documents = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_dir).replace("\\", "/")
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append({
                    "id": rel_path,
                    "text": text,
                    "metadata": {
                        "title": rel_path,
                        "source": rel_path
                    }
                })
    print(f"Loaded {len(documents)} corpus documents.")

    # 5. Initialize Retriever and test for rate limits / select active model
    active_model = "gemini-embedding-001"
    try:
        test_embedder = RotatingGeminiEmbedder(api_key=api_key, model_name=active_model)
        test_embedder.embed(["test"])
        print(f"Using embedding model: {active_model}")
    except Exception as e:
        print(f"ERROR: {active_model} is rate-limited or failed ({e}).")
        sys.exit(1)

    config = RetrieverConfig(
        embedder_provider="gemini",
        api_key=api_key,
        persist_dir=persist_dir,
        dense_top_k=20,
        sparse_top_k=20,
        rerank_top_k=5,
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        reranker_type="cross_encoder"
    )
    retriever = Retriever(config)
    retriever.embedder = RotatingGeminiEmbedder(api_key=api_key, model_name=active_model)

    # 6. Ingest Documents
    print("Ingesting corpus into vector store and sparse index...")
    start_time = time.time()
    retriever.ingest(documents)
    print(f"Ingestion completed in {time.time() - start_time:.2f}s.")

    # Instantiate LLM Judge reranker
    llm_reranker = RotatingLLMJudgeReranker(api_key=api_key)

    # 7. Evaluate
    dense_recalls = {1: 0, 3: 0, 5: 0}
    rrf_recalls = {1: 0, 3: 0, 5: 0}
    pipeline_recalls = {1: 0, 3: 0, 5: 0}
    llm_recalls = {1: 0, 3: 0, 5: 0}

    print("\nEvaluating queries...")
    for idx, eq in enumerate(eval_queries):
        query_text = eq["query"]
        ground_truth = eq["relevant_source"]

        # --- Baseline: Naive Dense-only ---
        query_emb = retriever.embedder.embed([query_text])[0]
        dense_results = retriever.vector_store.query(query_emb, top_k=20)

        # Deduplicate dense results at document level
        dense_ids = [doc_id for doc_id, _ in dense_results]
        fused_details = retriever.vector_store.get(dense_ids)
        details_map = {item["id"]: item for item in fused_details}

        dense_docs = []
        seen_dense = set()
        for doc_id, _ in dense_results:
            details = details_map.get(doc_id)
            if details:
                orig_id = details["metadata"].get("doc_id", doc_id)
                if orig_id not in seen_dense:
                    seen_dense.add(orig_id)
                    dense_docs.append(orig_id)

        # --- Condition 2: Hybrid RRF only (No Rerank) ---
        original_reranker = retriever.reranker
        retriever.reranker = None
        rrf_results = retriever.search(query_text, top_k=5)
        rrf_docs = [r.doc_id for r in rrf_results]
        retriever.reranker = original_reranker

        # --- Condition 3: Full RankFuse Pipeline (Hybrid RRF + Cross-Encoder) ---
        pipeline_results = retriever.search(query_text, top_k=5)
        pipeline_docs = [r.doc_id for r in pipeline_results]

        # --- Condition 4: RankFuse Pipeline (LLM Judge) ---
        retriever.reranker = llm_reranker
        llm_results = retriever.search(query_text, top_k=5)
        llm_docs = [r.doc_id for r in llm_results]
        retriever.reranker = original_reranker

        # Compute recalls
        for k in [1, 3, 5]:
            if ground_truth in dense_docs[:k]:
                dense_recalls[k] += 1
            if ground_truth in rrf_docs[:k]:
                rrf_recalls[k] += 1
            if ground_truth in pipeline_docs[:k]:
                pipeline_recalls[k] += 1
            if ground_truth in llm_docs[:k]:
                llm_recalls[k] += 1

        # print progress
        print(f"[{idx+1}/{len(eval_queries)}] Query: '{query_text}'")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Dense-only top-3: {dense_docs[:3]}")
        print(f"  RRF-only top-3  : {rrf_docs[:3]}")
        print(f"  Cross-Enc top-3 : {pipeline_docs[:3]}")
        print(f"  LLM-Judge top-3 : {llm_docs[:3]}")
        time.sleep(1.0)

    # Calculate average recalls
    num_queries = len(eval_queries)
    for k in [1, 3, 5]:
        dense_recalls[k] /= num_queries
        rrf_recalls[k] /= num_queries
        pipeline_recalls[k] /= num_queries
        llm_recalls[k] /= num_queries

    # Cleanup
    shutil.rmtree(persist_dir, ignore_errors=True)

    # 8. Print Results
    print("\n" + "=" * 105)
    print("BENCHMARK RESULTS - RETRIEVAL RECALL COMPARISON")
    print("=" * 105)
    print(f"{'Metric':<15} | {'Naive Dense-Only':<18} | {'Hybrid RRF-Only':<18} | {'RankFuse Cross-Enc':<18} | {'RankFuse LLM-Judge':<18}")
    print("-" * 105)
    print(f"{'Recall@1':<15} | {dense_recalls[1]:<18.4f} | {rrf_recalls[1]:<18.4f} | {pipeline_recalls[1]:<18.4f} | {llm_recalls[1]:<18.4f}")
    print(f"{'Recall@3':<15} | {dense_recalls[3]:<18.4f} | {rrf_recalls[3]:<18.4f} | {pipeline_recalls[3]:<18.4f} | {llm_recalls[3]:<18.4f}")
    print(f"{'Recall@5':<15} | {dense_recalls[5]:<18.4f} | {rrf_recalls[5]:<18.4f} | {pipeline_recalls[5]:<18.4f} | {llm_recalls[5]:<18.4f}")
    print("=" * 105)
    print("\nDone.")

if __name__ == "__main__":
    main()
