def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Perform Reciprocal Rank Fusion (RRF) on dense and sparse search rankings.

    Args:
        dense_results: Ranked list of (doc_id, score) from dense search.
        sparse_results: Ranked list of (doc_id, score) from sparse search.
        k: The RRF rank smoothing constant (defaults to 60).

    Returns:
        A merged list of (doc_id, fused_score) sorted by descending fused score.
    """
    fused_scores = {}

    for rank, (doc_id, _) in enumerate(dense_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(sparse_results):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
