# RankFuse — Coding Style Guide

This doc exists for one reason: code should read like a competent developer wrote it by hand — not like an AI generated it. Antigravity (or anyone else touching this codebase) must follow these rules for every file, every function, every commit.

## The Core Rule

**Simple > Clever.** If a problem can be solved with a basic `if/else`, a `for` loop, and a couple of variables — do that. Do not reach for design patterns, abstraction layers, decorators, metaclasses, or "elegant" one-liners unless the problem genuinely requires it. Most of this codebase doesn't.

A human developer solving these problems under normal time pressure writes straightforward code first, and only adds complexity when something actually forces their hand (a real bug, a real perf issue, a real need to swap implementations). Write it that way from the start.

## Signs of "AI-written" code — avoid these

- **Over-abstracting things that only have one implementation.** Don't build a `Factory` or `Strategy` pattern for something that's used exactly once. (Exception: the ABCs in `stores/`, `embeddings/`, `reranker/` — those genuinely need the interface because the project's whole design depends on swappable implementations. That's the one place abstraction is the point, not a smell.)
- **Excessive defensive coding.** Don't wrap every single line in try/except "just in case." Catch errors where they can realistically happen and where you can actually do something useful about them (see `exceptions.py` in the architecture doc).
- **Comment-per-line explaining the obvious.** `# increment counter` above `count += 1` is AI-tell #1. Comments should explain *why*, not *what* — and only where the *why* isn't obvious from the code itself.
- **Needlessly clever one-liners.** List comprehensions nested three deep, `functools.reduce` where a for-loop is clearer, ternaries stacked on ternaries. If you have to pause and mentally unpack it, rewrite it as normal lines.
- **Perfectly uniform function length and structure everywhere.** Real codebases are a little uneven — some functions are 4 lines, some are 40, because the problem sizes differ. Don't artificially chop things up just to make every function "look tidy," and don't pad short functions with unnecessary structure either.
- **Over-generic naming for things that aren't generic.** `data`, `result`, `item`, `obj`, `handler`, `manager`, `processor` used everywhere with no context. Name things after what they actually are.
- **Docstrings on every single private helper function.** Public API (`Retriever`, `RetrieverConfig`, the ABCs) gets real docstrings. Small internal helpers get a docstring only if the name + code aren't already self-explanatory.
- **Type-hinting to the point of noise.** Type hints on function signatures: yes, always (helps IDEs, helps users, and this is normal modern Python practice, not an AI-tell). But don't over-annotate every local variable inline when it's obvious from the assignment.
- **Boilerplate "enterprise" scaffolding for a small library.** No `AbstractFactoryProvider`, no dependency-injection framework, no plugin registry system. This is a focused library — keep the machinery proportional to the actual size of the problem.

## Naming Conventions

Names should sound like normal engineering language, not like a thesaurus was involved.

**Good:**
```python
def search(query, top_k=5): ...
dense_results = store.query(embedding, top_k)
sparse_results = bm25.query(query, top_k)
fused = reciprocal_rank_fusion(dense_results, sparse_results)
```

**Avoid:**
```python
def executeSearchOperation(queryInput, topKParameter=5): ...
denseSearchResultSet = vectorStoreInstance.performQuery(embeddingVector, topKParameter)
```

Rules:
- `snake_case` for functions/variables, `PascalCase` for classes — standard Python, no exceptions
- No Hungarian notation, no type suffixes (`_list`, `_dict`, `_str`) unless the name is genuinely ambiguous without it
- No unnecessary verbosity — `top_k` not `numberOfTopResultsToReturn`
- No unnecessary abbreviation either — `query` not `qry`, `config` not `cfg` (except extremely common short forms like `id`, `idx`, `db` which any developer reads instantly)
- Boolean variables/functions read as yes/no questions: `is_valid`, `has_results`, `should_rerank` — not `valid_flag` or `check_results()`
- Match the domain language already established in the architecture doc: `dense`, `sparse`, `fusion`, `rerank`, `ingest` — don't invent synonyms for concepts already named elsewhere in the project (e.g. don't suddenly call embeddings "vectors" in one file and "embeddings" in another — pick one and stay consistent, matching what `02-ARCHITECTURE.md` already uses)

## Comments — when and how

Write a comment when:
- The *why* behind a non-obvious decision needs explaining (e.g., "RRF uses rank position, not raw score, so dense/sparse results don't need score normalization")
- A workaround exists for a specific bug/quirk in a dependency (e.g., "Chroma's persistent client needs the dir to exist before init, hence the mkdir here")
- A magic number needs justification (e.g., "60 is the standard RRF k constant from the original paper — keeps well-established ranks from dominating")

Don't write a comment when:
- The code already says it (`# create the config object` above `config = RetrieverConfig(...)`)
- It restates the function name in prose

## Function & File Size — keep it natural

- No hard rule like "functions must be under 20 lines." Let function size match the actual complexity of what it's doing.
- If a file grows past ~300 lines and covers more than one real responsibility, that's a genuine signal to split it — not an arbitrary line-count target.
- Don't split a cohesive piece of logic into 5 tiny functions just to "look modular." If it's one linear sequence of steps that's only ever called from one place, one function is fine.

## Error Handling — proportional, not paranoid

- Catch exceptions where the failure is real and expected (API call fails, file not found, invalid config) — and raise one of the project's own exception types (`EmbeddingError`, `StoreError`, etc.) so callers get a consistent, meaningful error.
- Don't catch generic `Exception` and silently pass — that hides real bugs.
- Don't wrap trivial, unlikely-to-fail operations (like a dict lookup on a key you just set yourself) in try/except.

## Tests — write them like a developer testing their own work

- Test names describe the actual scenario: `test_rrf_merges_two_ranked_lists`, not `test_case_1` or `test_function_works`
- Don't write a test for every possible input permutation "for coverage's sake." Cover the realistic cases: normal input, empty input, one clearly-broken input (e.g., mismatched list lengths). That's usually enough for a function this size.
- Assertions should check the actual thing that matters (e.g., correct ordering, correct top-k count) — not superficial things like "did it return a list" when the real question is "is it the *right* list."

## Formatting & Tooling

- `black` for formatting, `ruff` for linting — configured in `pyproject.toml`, run in CI (`test.yml`). Don't hand-format against these tools' defaults.
- Line length: default black (88 chars) — don't override this to something unusual.
- Imports: standard library, then third-party, then local — grouped with a blank line between, no manual sorting needed if `ruff` handles it.

## Git Commit Style

- Commit messages describe what changed and why in plain language, present tense: `Add RRF fusion logic`, not `Implemented comprehensive reciprocal rank fusion algorithm with configurable k parameter`
- One logical change per commit where practical — don't bundle "add BM25 index + fix unrelated typo in README" into one commit
- No AI-flavored commit messages like "This commit implements X to enhance Y functionality as per requirements" — just say what happened

## Quick Self-Check Before Committing Any File

Ask:
1. Would a mid-level developer, not an AI, plausibly write this exact code under normal deadline pressure?
2. Is every abstraction here actually justified by something real (multiple implementations, genuine reuse), or did I add it because it "looked more professional"?
3. Could I delete a third of the comments and lose nothing?
4. Do the names sound like normal spoken engineering language?

If any answer is "no," simplify before moving on.
