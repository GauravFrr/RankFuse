import pytest

from rankfuse.fusion.rrf import reciprocal_rank_fusion


def test_rrf_merges_two_ranked_lists():
    # Hand-calculable test case with k = 60
    # Dense results (ranks: doc1=1, doc2=2, doc3=3)
    dense_results = [("doc1", 0.95), ("doc2", 0.8), ("doc3", 0.6)]
    # Sparse results (ranks: doc2=1, doc4=2, doc1=3)
    sparse_results = [("doc2", 12.5), ("doc4", 8.2), ("doc1", 5.1)]

    # Expected values:
    # doc2 score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # doc1 score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # doc4 score = 1/(60+2) = 1/62 = 0.016129
    # doc3 score = 1/(60+3) = 1/63 = 0.015873

    expected_doc2_score = 1 / 62 + 1 / 61
    expected_doc1_score = 1 / 61 + 1 / 63
    expected_doc4_score = 1 / 62
    expected_doc3_score = 1 / 63

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    assert len(fused) == 4

    # Verify order: doc2 > doc1 > doc4 > doc3
    assert fused[0][0] == "doc2"
    assert fused[1][0] == "doc1"
    assert fused[2][0] == "doc4"
    assert fused[3][0] == "doc3"

    # Verify scores using approx
    assert fused[0][1] == pytest.approx(expected_doc2_score)
    assert fused[1][1] == pytest.approx(expected_doc1_score)
    assert fused[2][1] == pytest.approx(expected_doc4_score)
    assert fused[3][1] == pytest.approx(expected_doc3_score)


def test_rrf_empty_inputs():
    # Empty inputs should return empty lists gracefully
    assert reciprocal_rank_fusion([], []) == []
    assert reciprocal_rank_fusion([("doc1", 0.9)], []) == [("doc1", 1 / 61)]
