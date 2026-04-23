from time import perf_counter

from app.config import Settings
from app.core.hybrid_search import HybridSearchEngine
from app.core.llm_judge import LLMJudgeEngine
from app.core.rerank import RerankEngine
from app.models.request import ClueMiningRequest, IdentifyRequest
from app.models.response import ClueMiningResponse, IdentifyResponse, SimilarCase
from app.utils.audit import AuditLogger


class IdentifyPipeline:
    def __init__(
        self,
        settings: Settings,
        hybrid_search_engine: HybridSearchEngine,
        rerank_engine: RerankEngine,
        llm_judge_engine: LLMJudgeEngine,
        audit_logger: AuditLogger,
    ) -> None:
        self._settings = settings
        self._hybrid_search_engine = hybrid_search_engine
        self._rerank_engine = rerank_engine
        self._llm_judge_engine = llm_judge_engine
        self._audit_logger = audit_logger

    async def identify(self, request: IdentifyRequest, request_id: str) -> IdentifyResponse:
        started = perf_counter()
        hybrid_candidates = await self._hybrid_search_engine.search(request)
        if not hybrid_candidates:
            response = IdentifyResponse(
                similar_cases=[],
                processing_time_ms=int((perf_counter() - started) * 1000),
                request_id=request_id,
            )
            self._audit_logger.log_event(
                "identify_completed",
                request_id=request_id,
                similar_case_ids=[],
                similar_case_count=0,
            )
            return response

        reranked_candidates = await self._rerank_engine.rerank(
            query=request.description,
            candidates=hybrid_candidates,
        )
        response = IdentifyResponse(
            similar_cases=self._to_similar_cases(reranked_candidates),
            processing_time_ms=int((perf_counter() - started) * 1000),
            request_id=request_id,
        )
        self._audit_logger.log_event(
            "identify_completed",
            request_id=request_id,
            similar_case_ids=[item.case_id for item in response.similar_cases],
            similar_case_count=len(response.similar_cases),
        )
        return response

    async def mine_clues(
        self,
        request: ClueMiningRequest,
        request_id: str,
    ) -> ClueMiningResponse:
        started = perf_counter()
        identify_request = IdentifyRequest(
            reported_persons=request.reported_persons,
            reporter=request.reporter,
            location=request.location,
            description=request.description,
            time_range_years=request.time_range_years,
        )
        clue_result = await self._llm_judge_engine.mine_clues(
            request=identify_request,
            similar_cases=[SimilarCase(**request.similar_case.model_dump())],
        )
        response = ClueMiningResponse(
            incremental_clues=clue_result.incremental_clues,
            supplemental_clues=clue_result.supplemental_clues,
            processing_time_ms=int((perf_counter() - started) * 1000),
            request_id=request_id,
        )
        self._audit_logger.log_event(
            "clue_mining_completed",
            request_id=request_id,
            similar_case_ids=[request.similar_case.case_id],
            incremental_clue_count=len(response.incremental_clues),
            supplemental_clue_count=len(response.supplemental_clues),
        )
        return response

    def _to_similar_cases(self, candidates):
        similar_cases = []
        for rank, candidate in enumerate(candidates[: self._settings.judge_top_n], start=1):
            rerank_score = candidate.rerank_score or 0.0
            similarity_score = max(0, min(100, int(round(rerank_score * 100))))
            similar_cases.append(
                SimilarCase(
                    case_id=candidate.case_id,
                    similarity_score=similarity_score,
                    rank=rank,
                    location=candidate.location,
                    location_district=candidate.location_district,
                    reported_persons=candidate.reported_persons,
                    reporter=candidate.reporter,
                    description_text=candidate.description_text,
                    create_time=candidate.create_time,
                    updated_at=candidate.updated_at,
                )
            )
        return similar_cases
