import unittest
from datetime import datetime, timezone

from app.config import Settings
from app.core.hybrid_search import HybridSearchEngine, reciprocal_rank_fusion
from app.models.domain import QueryEmbedding, SparseEmbedding
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
        self.assertEqual({candidate.case_id for candidate in merged}, {"CASE-1", "CASE-2"})
        self.assertAlmostEqual(merged[0].hybrid_score, merged[1].hybrid_score)


class FakeEmbeddingService:
    def __init__(self, embedding):
        self.embedding = embedding

    async def embed_text(self, text):
        return self.embedding


class FakeQdrantService:
    def __init__(self, dense_hits=None, sparse_hits=None, fallback_hits=None):
        self.dense_calls = 0
        self.sparse_calls = 0
        self.filtered_calls = 0
        self._dense_hits = dense_hits or []
        self._sparse_hits = sparse_hits or []
        self._fallback_hits = fallback_hits or []

    async def search_dense(self, embedding, query_filter, limit):
        self.dense_calls += 1
        return list(self._dense_hits)

    async def search_sparse(self, embedding, query_filter, limit):
        self.sparse_calls += 1
        return list(self._sparse_hits)

    async def fetch_filtered_candidates(self, query_filter, limit):
        self.filtered_calls += 1
        return list(self._fallback_hits)[:limit]


class HybridSearchEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_calls_dense_and_sparse_when_sparse_query_exists(self):
        qdrant_service = FakeQdrantService()
        engine = HybridSearchEngine(
            settings=Settings(prompt_dir="prompts", state_dir="data/app_state"),
            qdrant_service=qdrant_service,
            embedding_service=FakeEmbeddingService(
                QueryEmbedding(
                    dense_vector=[0.1, 0.2],
                    sparse_vector=SparseEmbedding(indices=[1], values=[0.5]),
                )
            ),
        )

        await engine.search(
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收礼",
            )
        )

        self.assertEqual(qdrant_service.dense_calls, 1)
        self.assertEqual(qdrant_service.sparse_calls, 1)

    async def test_search_triggers_fallback_when_merged_candidates_too_few(self):
        primary_candidate = make_candidate("CASE-1", ["王建国"])
        fallback_candidate = make_candidate("CASE-2", ["王建国"])
        qdrant_service = FakeQdrantService(
            dense_hits=[primary_candidate],
            sparse_hits=[primary_candidate],
            fallback_hits=[primary_candidate, fallback_candidate],
        )
        engine = HybridSearchEngine(
            settings=Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
                fallback_min_candidates=2,
                fallback_max_fetch=2,
            ),
            qdrant_service=qdrant_service,
            embedding_service=FakeEmbeddingService(
                QueryEmbedding(
                    dense_vector=[0.1, 0.2],
                    sparse_vector=SparseEmbedding(indices=[1], values=[0.5]),
                )
            ),
        )

        results = await engine.search(
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收礼",
            )
        )

        self.assertEqual(qdrant_service.filtered_calls, 1)
        self.assertEqual([candidate.case_id for candidate in results], ["CASE-1", "CASE-2"])

    async def test_search_fallback_deduplicates_existing_candidates(self):
        primary_candidate = make_candidate("CASE-1", ["王建国"])
        fallback_candidate = make_candidate("CASE-2", ["王建国"])
        extra_candidate = make_candidate("CASE-3", ["王建国"])
        qdrant_service = FakeQdrantService(
            dense_hits=[primary_candidate],
            sparse_hits=[primary_candidate],
            fallback_hits=[primary_candidate, fallback_candidate, extra_candidate],
        )
        engine = HybridSearchEngine(
            settings=Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
                fallback_min_candidates=3,
                fallback_max_fetch=3,
            ),
            qdrant_service=qdrant_service,
            embedding_service=FakeEmbeddingService(
                QueryEmbedding(
                    dense_vector=[0.1, 0.2],
                    sparse_vector=SparseEmbedding(indices=[1], values=[0.5]),
                )
            ),
        )

        results = await engine.search(
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收礼",
            )
        )

        self.assertEqual([candidate.case_id for candidate in results], ["CASE-1", "CASE-2", "CASE-3"])

    async def test_search_skips_fallback_when_merged_candidates_sufficient(self):
        first_candidate = make_candidate("CASE-1", ["王建国"])
        second_candidate = make_candidate("CASE-2", ["王建国"])
        qdrant_service = FakeQdrantService(
            dense_hits=[first_candidate, second_candidate],
            sparse_hits=[],
            fallback_hits=[make_candidate("CASE-3", ["王建国"])],
        )
        engine = HybridSearchEngine(
            settings=Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
                fallback_min_candidates=2,
                fallback_max_fetch=2,
            ),
            qdrant_service=qdrant_service,
            embedding_service=FakeEmbeddingService(
                QueryEmbedding(
                    dense_vector=[0.1, 0.2],
                    sparse_vector=SparseEmbedding(indices=[1], values=[0.5]),
                )
            ),
        )

        results = await engine.search(
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收礼",
            )
        )

        self.assertEqual(qdrant_service.filtered_calls, 0)
        self.assertEqual([candidate.case_id for candidate in results], ["CASE-1", "CASE-2"])


if __name__ == "__main__":
    unittest.main()
