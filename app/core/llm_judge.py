import asyncio
import json
from typing import List, Tuple

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import ClueMiningResult, DuplicateJudgeResult, SearchCandidate
from app.models.request import IdentifyRequest
from app.services.llm_service import LLMService


class LLMJudgeEngine:
    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self._settings = settings
        self._llm_service = llm_service
        self._duplicate_prompt = settings.duplicate_prompt_path.read_text(encoding="utf-8")
        self._clue_prompt = settings.clue_prompt_path.read_text(encoding="utf-8")

    async def judge(
        self,
        request: IdentifyRequest,
        candidates: List[SearchCandidate],
    ) -> Tuple[DuplicateJudgeResult, ClueMiningResult]:
        if not candidates:
            return DuplicateJudgeResult(is_duplicate=False, ranked_cases=[]), ClueMiningResult()

        new_case_json = json.dumps(
            {
                "reported_persons": request.reported_persons,
                "reporter": request.reporter,
                "location": request.location,
                "description": request.description,
                "time_range_years": request.time_range_years,
            },
            ensure_ascii=False,
        )
        candidates_json = json.dumps(
            [
                {
                    "case_id": candidate.case_id,
                    "location": candidate.location,
                    "location_district": candidate.location_district,
                    "reported_persons": candidate.reported_persons,
                    "reporter": candidate.reporter,
                    "description_text": candidate.description_text,
                    "hybrid_score": candidate.hybrid_score,
                    "rerank_score": candidate.rerank_score,
                }
                for candidate in candidates
            ],
            ensure_ascii=False,
            default=str,
        )
        duplicate_prompt = self._duplicate_prompt.format(
            new_case=new_case_json,
            candidates=candidates_json,
            top_n=self._settings.judge_top_n,
        )
        clue_prompt = self._clue_prompt.format(
            new_case=new_case_json,
            candidates=candidates_json,
        )
        duplicate_raw, clue_raw = await asyncio.gather(
            self._llm_service.complete_json(
                system_prompt="你是纪委信访案件重复件研判助手，只输出合法 JSON。",
                user_prompt=duplicate_prompt,
            ),
            self._llm_service.complete_json(
                system_prompt="你是纪委信访新线索挖掘助手，只输出合法 JSON。",
                user_prompt=clue_prompt,
            ),
        )
        try:
            duplicate_result = DuplicateJudgeResult.model_validate_json(duplicate_raw)
            clue_result = ClueMiningResult.model_validate_json(clue_raw)
        except Exception as exc:
            raise ServiceError(
                error_code="llm_invalid_json",
                message="LLM response could not be parsed as schema-compliant JSON.",
                status_code=502,
                retryable=True,
                details={"error": str(exc)},
            )

        allowed_case_ids = {candidate.case_id for candidate in candidates}
        duplicate_result.ranked_cases = [
            case
            for case in duplicate_result.ranked_cases
            if case.case_id in allowed_case_ids
        ][: self._settings.judge_top_n]
        if not duplicate_result.is_duplicate:
            clue_result = ClueMiningResult(new_clues=[])
        return duplicate_result, clue_result
