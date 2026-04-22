import unittest

from app.config import Settings
from app.core.llm_judge import LLMJudgeEngine
from app.models.domain import SearchCandidate
from app.models.request import IdentifyRequest


class FakeLLMService:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def complete_json(self, system_prompt, user_prompt):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self._responses.pop(0)


class LLMJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_filters_unknown_case_ids(self):
        settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
        )
        llm_service = FakeLLMService(
            [
                '{"is_duplicate": true, "ranked_cases": [{"case_id": "CASE-001", "similarity_score": 91, "rank": 1, "reason": "事实重合"}, {"case_id": "UNKNOWN", "similarity_score": 99, "rank": 2, "reason": "无效"}]}',
                '{"new_clues": [{"source_case_id": "CASE-001", "clue_type": "关系", "description": "亲属承包工程", "risk_level": "高"}]}',
            ]
        )
        engine = LLMJudgeEngine(settings=settings, llm_service=llm_service)
        request = IdentifyRequest(
            reported_persons=["王建国"],
            reporter="张某",
            location="太原市",
            description="王建国收受礼金",
        )
        candidates = [
            SearchCandidate(
                case_id="CASE-001",
                location="太原市",
                reported_persons=["王建国"],
                description_text="test",
                create_time="2024-01-01T00:00:00+00:00",
            )
        ]

        duplicate_result, clue_result = await engine.judge(request, candidates)

        self.assertTrue(duplicate_result.is_duplicate)
        self.assertEqual(len(duplicate_result.ranked_cases), 1)
        self.assertEqual(duplicate_result.ranked_cases[0].case_id, "CASE-001")
        self.assertEqual(clue_result.new_clues[0].source_case_id, "CASE-001")
        self.assertEqual(clue_result.new_clues[0].risk_level, "高")
        self.assertEqual(len(llm_service.calls), 2)
        self.assertIn("重复案件：", llm_service.calls[1]["user_prompt"])
        self.assertNotIn("UNKNOWN", llm_service.calls[1]["user_prompt"])

    async def test_judge_skips_clue_generation_for_non_duplicate(self):
        settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
        )
        llm_service = FakeLLMService(
            [
                '{"is_duplicate": false, "ranked_cases": [{"case_id": "CASE-001", "similarity_score": 40, "rank": 1, "reason": "不构成重复"}]}'
            ]
        )
        engine = LLMJudgeEngine(settings=settings, llm_service=llm_service)
        request = IdentifyRequest(
            reported_persons=["王建国"],
            reporter="张某",
            location="太原市",
            description="王建国收受礼金",
        )
        candidates = [
            SearchCandidate(
                case_id="CASE-001",
                location="太原市",
                reported_persons=["王建国"],
                description_text="test",
                create_time="2024-01-01T00:00:00+00:00",
            )
        ]

        duplicate_result, clue_result = await engine.judge(request, candidates)

        self.assertFalse(duplicate_result.is_duplicate)
        self.assertEqual(clue_result.new_clues, [])
        self.assertEqual(len(llm_service.calls), 1)

    async def test_judge_discards_clues_with_invalid_source_case_id(self):
        settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
        )
        llm_service = FakeLLMService(
            [
                '{"is_duplicate": true, "ranked_cases": [{"case_id": "CASE-001", "similarity_score": 91, "rank": 1, "reason": "事实重合"}]}',
                '{"new_clues": [{"source_case_id": "CASE-001", "clue_type": "关系", "description": "亲属承包工程", "risk_level": "高"}, {"source_case_id": "CASE-999", "clue_type": "金额", "description": "收受现金", "risk_level": "高"}]}',
            ]
        )
        engine = LLMJudgeEngine(settings=settings, llm_service=llm_service)
        request = IdentifyRequest(
            reported_persons=["王建国"],
            reporter="张某",
            location="太原市",
            description="王建国收受礼金",
        )
        candidates = [
            SearchCandidate(
                case_id="CASE-001",
                location="太原市",
                reported_persons=["王建国"],
                description_text="test",
                create_time="2024-01-01T00:00:00+00:00",
            ),
            SearchCandidate(
                case_id="CASE-002",
                location="太原市",
                reported_persons=["王建国"],
                description_text="test2",
                create_time="2024-01-01T00:00:00+00:00",
            ),
        ]

        duplicate_result, clue_result = await engine.judge(request, candidates)

        self.assertTrue(duplicate_result.is_duplicate)
        self.assertEqual(len(clue_result.new_clues), 1)
        self.assertEqual(clue_result.new_clues[0].source_case_id, "CASE-001")


if __name__ == "__main__":
    unittest.main()
