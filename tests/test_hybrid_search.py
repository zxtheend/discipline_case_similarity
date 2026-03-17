import unittest
from datetime import datetime, timezone

from app.core.hybrid_search import reciprocal_rank_fusion
from app.models.domain import SearchCandidate
from app.models.request import IdentifyRequest


def make_candidate(case_id, reported_persons):
    return SearchCandidate(
        case_id=case_id,
        location="太原市",
        reported_persons=reported_persons,
        description_text="test",
        create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        dense_score=0.8,
        sparse_score=0.7,
    )


class HybridSearchTests(unittest.TestCase):
    def test_rrf_merges_dense_and_sparse_rankings(self):
        dense_hits = [make_candidate("CASE-1", ["王建国"]), make_candidate("CASE-2", ["李海峰"])]
        sparse_hits = [make_candidate("CASE-2", ["李海峰"]), make_candidate("CASE-1", ["王建国"])]
        request = IdentifyRequest(
            reported_persons=["王建国"],
            reporter="张某",
            location="太原市",
            description="王建国收礼",
        )

        merged = reciprocal_rank_fusion(dense_hits, sparse_hits, request, k=60, limit=10)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].case_id, "CASE-1")
        self.assertGreater(merged[0].hybrid_score, merged[1].hybrid_score)


if __name__ == "__main__":
    unittest.main()
