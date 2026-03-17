from time import perf_counter

from app.config import Settings
from app.core.hybrid_search import HybridSearchEngine
from app.core.llm_judge import LLMJudgeEngine
from app.core.rerank import RerankEngine
from app.models.request import IdentifyRequest
from app.models.response import IdentifyResponse
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
                is_duplicate=False,
                similar_cases=[],
                new_clues=[],
                processing_time_ms=int((perf_counter() - started) * 1000),
                request_id=request_id,
            )
            self._audit_logger.log_event(
                "identify_completed",
                request_id=request_id,
                is_duplicate=False,
                similar_case_ids=[],
            )
            return response

        reranked_candidates = await self._rerank_engine.rerank(
            query=request.description,
            candidates=hybrid_candidates,
        )
        duplicate_result, clue_result = await self._llm_judge_engine.judge(
            request=request,
            candidates=reranked_candidates,
        )
        response = IdentifyResponse(
            is_duplicate=duplicate_result.is_duplicate,
            similar_cases=duplicate_result.ranked_cases[: self._settings.judge_top_n],
            new_clues=clue_result.new_clues,
            processing_time_ms=int((perf_counter() - started) * 1000),
            request_id=request_id,
        )
        self._audit_logger.log_event(
            "identify_completed",
            request_id=request_id,
            is_duplicate=response.is_duplicate,
            similar_case_ids=[item.case_id for item in response.similar_cases],
            clue_count=len(response.new_clues),
        )
        return response
