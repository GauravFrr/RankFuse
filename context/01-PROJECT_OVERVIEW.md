# RankFuse — Project Overview

## What This Is

`rankfuse` is an open-source Python library that gives any Retrieval-Augmented Generation (RAG) pipeline production-grade retrieval quality out of the box — hybrid dense + sparse search, fused with Reciprocal Rank Fusion (RRF), and refined with a reranking pass — without the developer needing to build any of that themselves.

It is a **pip-installable library**, not a hosted service. It runs entirely on the end user's own machine, using their own API keys and their own local vector store. There is nothing for the author to host, scale, or pay for.

## Origin

This library generalizes the retrieval engine originally built for **Retryv**, a project where hybrid retrieval + cross-encoder reranking took recall from 23% to 84% on a real dataset. Retryv was a single-purpose application with hardcoded config. `rankfuse` extracts that proven approach and rebuilds it as a reusable, provider-agnostic, config-driven library.

The core algorithms (RRF math, BM25 keyword matching, cross-encoder reranking flow) carry over conceptually. The code itself is refactored — not copy-pasted — into modular, testable, swappable components.

## Problem It Solves

Every RAG developer eventually hits the same wall: naive vector similarity search alone returns too many irrelevant chunks. This happens because:
- Pure semantic search misses exact keyword/term matches (e.g., product codes, names, numbers)
- Pure keyword search misses paraphrased/semantically similar content
- Even combined, raw top-K results aren't precision-sorted for the LLM's context window

Fixing this properly requires implementing hybrid search + rank fusion + reranking — usually 2-3 days of work, and most developers either skip it (bad results) or reach for a heavy full framework (LangChain) just to get this one piece.

`rankfuse` is the focused, drop-in fix for exactly this problem — nothing more.

## Who It's For

- RAG/LLM app developers who want better retrieval without adopting a full framework
- Developers already using ChromaDB who want hybrid search bolted on
- Anyone building chatbots, document Q&A, internal search, or agent memory systems

## Core Design Principles

1. **Zero hosting, zero servers** — file-based Chroma, runs fully local
2. **Bring your own key (BYOK)** — user supplies their own embedding/LLM API key; author never pays for user usage
3. **Provider-agnostic** — not locked to Gemini; embeddings and rerankers are swappable via abstract interfaces
4. **Reproducible benchmarks** — recall numbers are provable by anyone via a script in the repo, not just claimed in the README
5. **Library, not application** — small, focused public API; no opinions about the rest of the user's stack

## What Success Looks Like

- `pip install rankfuse` works and the quickstart example runs in under 5 minutes
- Benchmark script reproduces a recall improvement similar to Retryv's on a public dataset (FastAPI docs)
- Clean enough code and structure to be a credible portfolio/resume piece ("author of an open-source RAG retrieval library")
- Usable inside the author's own JobPilot project as the first real dogfooding consumer

## Non-Goals (v1)

- No hosted API / SaaS wrapper around this library
- No UI (a Streamlit demo may exist separately, but it's not the package)
- No support for every vector store on day one — start with Chroma, design the interface so others (pgvector, FAISS) can be added later without breaking changes
- No fine-tuning or training of custom models — reuse existing embedding/reranker models via API or local inference
