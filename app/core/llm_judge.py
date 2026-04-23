import json
from typing import Dict, List, Tuple

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import ClueMiningResult, DuplicateJudgeResult, SearchCandidate
from app.models.request import IdentifyRequest
from app.models.response import SimilarCase
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

        new_case_json = self._new_case_payload_json(request)
        candidates_json = json.dumps(
            [self._candidate_payload(candidate) for candidate in candidates],
            ensure_ascii=False,
            default=str,
        )
        duplicate_prompt = self._duplicate_prompt.format(
            new_case=new_case_json,
            candidates=candidates_json,
            top_n=self._settings.judge_top_n,
        )
        duplicate_raw = await self._llm_service.complete_json(
            system_prompt="你是纪委信访案件重复件研判助手，只输出合法 JSON。",
            user_prompt=duplicate_prompt,
        )
        try:
            duplicate_result = DuplicateJudgeResult.model_validate_json(duplicate_raw)
        except Exception as exc:
            raise ServiceError(
                error_code="llm_invalid_json",
                message="LLM response could not be parsed as schema-compliant JSON.",
                status_code=502,
                retryable=True,
                details={"error": str(exc)},
            )

        candidate_map: Dict[str, SearchCandidate] = {
            candidate.case_id: candidate for candidate in candidates
        }
        allowed_case_ids = set(candidate_map)
        filtered_ranked_cases = [
            case
            for case in duplicate_result.ranked_cases
            if case.case_id in allowed_case_ids
        ][: self._settings.judge_top_n]
        duplicate_result.ranked_cases = self._enrich_similar_cases(
            filtered_ranked_cases,
            candidate_map,
        )
        if not duplicate_result.is_duplicate:
            return duplicate_result, ClueMiningResult()

        duplicate_candidates = [
            candidate_map[case.case_id]
            for case in duplicate_result.ranked_cases
            if case.case_id in candidate_map
        ]
        if not duplicate_candidates:
            return duplicate_result, ClueMiningResult()

        clue_result = await self.mine_clues(
            request=request,
            similar_cases=self._enrich_similar_cases(
                duplicate_result.ranked_cases,
                candidate_map,
            ),
        )
        return duplicate_result, clue_result

    async def mine_clues(
        self,
        request: IdentifyRequest,
        similar_cases: List[SimilarCase],
    ) -> ClueMiningResult:
        if not similar_cases:
            return ClueMiningResult()

        new_case_json = self._new_case_payload_json(request)
        allowed_cases = similar_cases[: self._settings.judge_top_n]
        duplicate_cases_json = json.dumps(
            [self._similar_case_payload(case) for case in allowed_cases],
            ensure_ascii=False,
            default=str,
        )
        clue_prompt = self._clue_prompt.format(
            new_case=new_case_json,
            duplicate_cases=duplicate_cases_json,
            max_clues=3,
        )
        clue_raw = await self._llm_service.complete_json(
            system_prompt="你是纪委信访新线索挖掘助手，只输出合法 JSON。",
            user_prompt=clue_prompt,
        )
        try:
            clue_result = ClueMiningResult.model_validate_json(clue_raw)
        except Exception as exc:
            raise ServiceError(
                error_code="llm_invalid_json",
                message="LLM response could not be parsed as schema-compliant JSON.",
                status_code=502,
                retryable=True,
                details={"error": str(exc)},
            )

        allowed_case_ids = {case.case_id for case in allowed_cases}
        clue_result.incremental_clues = [
            clue
            for clue in clue_result.incremental_clues
            if clue.source_case_id in allowed_case_ids
        ]
        clue_result.supplemental_clues = [
            clue
            for clue in clue_result.supplemental_clues
            if clue.source_case_id in allowed_case_ids
        ]
        return clue_result

    def _new_case_payload_json(self, request: IdentifyRequest) -> str:
        return json.dumps(
            {
                "reported_persons": request.reported_persons,
                "reporter": request.reporter,
                "location": request.location,
                "description": request.description,
                "time_range_years": request.time_range_years,
            },
            ensure_ascii=False,
        )

    def _enrich_similar_cases(
        self,
        ranked_cases: List[SimilarCase],
        candidate_map: Dict[str, SearchCandidate],
    ) -> List[SimilarCase]:
        enriched_cases = []
        for case in ranked_cases:
            candidate = candidate_map.get(case.case_id)
            if candidate is None:
                continue
            enriched_cases.append(
                case.model_copy(
                    update={
                        "location": candidate.location,
                        "location_district": candidate.location_district,
                        "reported_persons": candidate.reported_persons,
                        "reporter": candidate.reporter,
                        "description_text": candidate.description_text,
                        "create_time": candidate.create_time,
                        "updated_at": candidate.updated_at,
                    }
                )
            )
        return enriched_cases

    def _candidate_payload(self, candidate: SearchCandidate) -> Dict[str, object]:
        return {
            "case_id": candidate.case_id,
            "location": candidate.location,
            "location_district": candidate.location_district,
            "reported_persons": candidate.reported_persons,
            "reporter": candidate.reporter,
            "description_text": candidate.description_text,
            "hybrid_score": candidate.hybrid_score,
            "rerank_score": candidate.rerank_score,
        }

    def _similar_case_payload(self, similar_case: SimilarCase) -> Dict[str, object]:
        return {
            "case_id": similar_case.case_id,
            "location": similar_case.location,
            "location_district": similar_case.location_district,
            "reported_persons": similar_case.reported_persons,
            "reporter": similar_case.reporter,
            "description_text": similar_case.description_text or "",
            "similarity_score": similar_case.similarity_score,
            "rank": similar_case.rank,
        }
