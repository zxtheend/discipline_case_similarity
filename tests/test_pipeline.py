import unittest
from datetime import datetime, timezone

from app.config import Settings
from app.core.pipeline import IdentifyPipeline
from app.models.domain import SearchCandidate
from app.models.request import ClueMiningRequest, IdentifyRequest


class FakeHybridSearchEngine:
    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    async def search(self, request):
        self.calls += 1
        return list(self._candidates)


class FakeRerankEngine:
    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    async def rerank(self, query, candidates):
        self.calls += 1
        return list(self._candidates)


class FakeLLMJudgeEngine:
    def __init__(self, clue_result):
        self._clue_result = clue_result
        self.mine_clues_calls = []
        self.judge_calls = 0

    async def judge(self, request, candidates):
        self.judge_calls += 1
        raise AssertionError("identify should not call LLM judge")

    async def mine_clues(self, request, similar_cases):
        self.mine_clues_calls.append(
            {
                "request": request,
                "similar_cases": similar_cases,
            }
        )
        return self._clue_result


class FakeAuditLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, request_id, **details):
        self.events.append(
            {
                "event_type": event_type,
                "request_id": request_id,
                "details": details,
            }
        )


class FakeClueResult:
    def __init__(self, new_clues):
        self.new_clues = new_clues


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
            judge_top_n=5,
        )

    async def test_identify_returns_ranked_similar_cases_without_llm(self):
        candidate = SearchCandidate(
            case_id="CASE-001",
            location="太原市",
            location_district="小店区",
            reported_persons=["王建国"],
            reporter="张某",
            description_text="王建国收受礼金并安排亲属承揽工程。",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            rerank_score=0.87,
        )
        llm_judge_engine = FakeLLMJudgeEngine(FakeClueResult([]))
        audit_logger = FakeAuditLogger()
        pipeline = IdentifyPipeline(
            settings=self.settings,
            hybrid_search_engine=FakeHybridSearchEngine([candidate]),
            rerank_engine=FakeRerankEngine([candidate]),
            llm_judge_engine=llm_judge_engine,
            audit_logger=audit_logger,
        )

        response = await pipeline.identify(
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收受礼金",
            ),
            request_id="req-1",
        )

        self.assertEqual(llm_judge_engine.judge_calls, 0)
        self.assertEqual(len(response.similar_cases), 1)
        self.assertEqual(response.similar_cases[0].case_id, "CASE-001")
        self.assertEqual(response.similar_cases[0].similarity_score, 87)
        self.assertEqual(response.similar_cases[0].reported_persons, ["王建国"])
        self.assertEqual(audit_logger.events[0]["event_type"], "identify_completed")
        self.assertEqual(audit_logger.events[0]["details"]["similar_case_count"], 1)

    async def test_mine_clues_uses_client_supplied_similar_cases(self):
        llm_judge_engine = FakeLLMJudgeEngine(
            FakeClueResult(
                [
                    {
                        "source_case_id": "CASE-001",
                        "clue_type": "关系",
                        "description": "亲属承揽工程",
                        "risk_level": "高",
                    }
                ]
            )
        )
        audit_logger = FakeAuditLogger()
        pipeline = IdentifyPipeline(
            settings=self.settings,
            hybrid_search_engine=FakeHybridSearchEngine([]),
            rerank_engine=FakeRerankEngine([]),
            llm_judge_engine=llm_judge_engine,
            audit_logger=audit_logger,
        )

        response = await pipeline.mine_clues(
            ClueMiningRequest(
                reported_persons=["王建国"],
                reporter="张某",
                location="太原市",
                description="王建国收受礼金",
                similar_cases=[
                    {
                        "case_id": "CASE-001",
                        "similarity_score": 91,
                        "rank": 1,
                        "reason": "事实高度接近",
                        "location": "太原市",
                        "reported_persons": ["王建国"],
                        "reporter": "李某",
                        "description_text": "历史案件提到亲属承包工程。",
                    }
                ],
            ),
            request_id="req-2",
        )

        self.assertEqual(len(llm_judge_engine.mine_clues_calls), 1)
        self.assertEqual(
            llm_judge_engine.mine_clues_calls[0]["similar_cases"][0].case_id,
            "CASE-001",
        )
        self.assertEqual(response.new_clues[0].source_case_id, "CASE-001")
        self.assertEqual(audit_logger.events[0]["event_type"], "clue_mining_completed")
        self.assertEqual(audit_logger.events[0]["details"]["clue_count"], 1)


if __name__ == "__main__":
    unittest.main()
