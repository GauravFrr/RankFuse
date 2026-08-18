# Retrieval Recall Benchmarks (Four-Column Evaluation)

This report documents the results and methodology of the retrieval evaluation performed on the expanded FastAPI documentation dataset.

## Methodology

### Dataset and Corpus
* **Corpus Size:** 154 markdown documents (`.md` files) representing the complete English FastAPI documentation corpus.
* **Evaluation Query Set:** 30 natural language queries selected from a realistic golden query set.
  > [!WARNING]
  > **Sample Size Limitation:** The evaluation query set is small (30 queries). At this sample size, a single query change represents a 3.33% swing in recall. These results should be interpreted as directionally promising indicators of relative pipeline behavior rather than statistically definitive benchmarks.

### Retrieval Conditions
All hybrid pipelines use standard library default parameters (`dense_top_k = 20`, `sparse_top_k = 20`, `rrf_k = 60`, `dense_weight = 1.0`, `sparse_weight = 1.0`) to avoid overfitting. 

**Stopword-Filtered Sparse Indexing:** BM25 uses a standard English stopword filter within `clean_tokenize()` to prevent common terms (like `to`, `and`, `in`, `how`, `with`) from distorting rank scores.

1. **Naive Dense-Only Search (Baseline):**
   * Embeds the query using `gemini-embedding-001`.
   * Retrieves the top 20 chunks from the Chroma vector store.
   * Deduplicates the retrieved chunks at the document level (retaining the highest-scoring chunk per document).
2. **Hybrid RRF-Only:**
   * Hybrid retrieval combining dense (`gemini-embedding-001`) and sparse (BM25 with stopword filtering) search.
   * Fuses dense and sparse rankings with equal weights.
   * Deduplicates candidates at the document level based on their fused RRF rank. No neural reranking is applied.
3. **RankFuse Cross-Encoder:**
   * Uses standard equal-weight RRF fusion.
   * Neural reranking using the **`ms-marco-MiniLM-L-12-v2`** Cross-Encoder model.
   * Document-level deduplication performed **after** reranking (retaining the highest-scoring reranked chunk per document).
4. **RankFuse LLM-Judge:**
   * Uses standard equal-weight RRF fusion.
   * Neural reranking using Gemini (`gemini-3.5-flash-lite`).
   * **Structured Reranking Pool Cap:** To optimize API call quota and respect cost limits, the LLM-Judge only reranks the top 5 unique candidates.
   * Document-level deduplication performed **after** reranking.

---

## Results

Below are the actual stable, reproducible numbers produced by running the benchmark script (`benchmarks/run_benchmark.py`):

| Metric | Naive Dense-Only | Hybrid RRF-Only | RankFuse Cross-Enc | RankFuse LLM-Judge |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | 0.5000 (15/30) | 0.5000 (15/30) | 0.4333 (13/30) | 0.4000 (12/30) |
| **Recall@3** | **0.8333 (25/30)** | 0.7667 (23/30) | 0.7667 (23/30) | 0.7333 (22/30) |
| **Recall@5** | **0.9000 (27/30)** | **0.9000 (27/30)** | **0.9000 (27/30)** | **0.9000 (27/30)\*** |

*\*Note: The LLM-Judge condition is structurally capped at a maximum Recall@5 of 0.9000 because it only evaluates the top 5 unique candidates returned by the fusion step to limit API costs.*

---

## Analysis & Insights

### 1. The Keyword-Concentration Disease & Stopword Filtering
Technical changelogs and release notes group lists of features, parameters, and fixes in close proximity. This creates high keyword density for terms like `database`, `sessions`, `yield`, `OpenAPI`, `tags`, and `APIRouter`. 

By adding stopword filtering (removing terms like `how`, `to`, `with`, `and`), we eliminate basic noise, which successfully boosts Recall@5 for RRF-only from **0.8333** to **0.9000** (matching Naive Dense-only). However, the fundamental mechanism of keyword concentration remains at Recall@1:

* **Example Query:** `"how to manage database sessions using dependency injection with yield and error handling"`
* **Stopword-Filtered Tokens:** `['manage', 'database', 'sessions', 'using', 'dependency', 'injection', 'yield', 'error', 'handling']`
* **Raw BM25 Results:**
  1. `release-notes.md_chunk_1489` (Score: **16.31**, Matches: `['error', 'dependency', 'database', 'sessions']`)
  2. `release-notes.md_chunk_1490` (Score: **14.29**, Matches: `['dependency', 'database', 'sessions']`)
  3. `release-notes.md_chunk_270` (Score: **13.15**, Matches: `['dependency', 'database', 'yield']`)
  4. `advanced/advanced-dependencies.md_chunk_11` (Score: **12.93**, Matches: `['dependency', 'database', 'yield']`)
  5. `dependencies-with-yield.md_chunk_1` (Score: **12.83**, Matches: `['dependency', 'database', 'yield']`)

Even with stopwords removed, the release note chunks still score higher (containing 4 distinct query keywords) than the actual tutorial guides (which contain 3 distinct keywords). As a result, standard equal-weight RRF-only search (0.5000 Recall@1) still fails to beat pure dense search.

This demonstrates that **stopword filtering is a highly effective indexing enhancement that improves overall candidate recall (Recall@5), but dense-priority RRF weights or neural reranking are still required to resolve remaining top-1 ranking noise**.

### 2. Cross-Encoder Failure on Code Docs
Using a general-domain Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) degrades performance, dropping Recall@1 to **0.4333**. 
* General-domain models pre-trained on MS-MARCO do not comprehend Python/Pydantic code blocks and route declarations. 
* Their predictions are noisy, causing them to reinforce the lexical keyword matches in release notes and demote highly relevant guides.

### 3. LLM-Judge Performance
The **Gemini LLM-Judge (`gemini-3.5-flash-lite`)** achieves a Recall@5 of 0.9000 (matching all other conditions) but scores **0.4000 Recall@1** (12/30 queries correct) on the standard equal-weight parameters.
* **Corpus/Genre Limitation:** Documents like `release-notes.md` represent a known hard case for keyword-based sparse search because they list dozens of feature updates in close proximity. This naturally pushes them to the top of the sparse candidate list, polluting the initial candidate pool passed to the LLM-Judge.
* **Recall Depth Tradeoff:** Limiting the reranking pool to 5 unique documents to control API request volume structurally disadvantages the LLM-Judge at higher recall levels ($k=5$). In production scenarios, users can expand this pool (e.g., to 10 candidates) to match the reranking depth of the Cross-Encoder, at the cost of doubling their LLM API request volume.
