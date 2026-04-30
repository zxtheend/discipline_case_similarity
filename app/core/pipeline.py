from time import perf_counter

from app.config import Settings
from app.core.hybrid_search import HybridSearchEngine
from app.core.llm_judge import LLMJudgeEngine
from app.core.rerank import RerankEngine
from app.core.similar_case_mapper import (
    map_search_candidates_to_similar_cases,
    resolve_identify_top_n,
)
from app.models.request import ClueMiningRequest, IdentifyRequest
from app.models.response import ClueMiningResponse, IdentifyResponse, SimilarCase
from app.utils.audit import AuditLogger


class IdentifyFlow:
    def __init__(
        self,
        settings: Settings,
        hybrid_search_engine: HybridSearchEngine,
        rerank_engine: RerankEngine,
        audit_logger: AuditLogger,
    ) -> None:
        self._settings = settings
        self._hybrid_search_engine = hybrid_search_engine
        self._rerank_engine = rerank_engine
        self._audit_logger = audit_logger

    async def execute(self, request: IdentifyRequest, request_id: str) -> IdentifyResponse:
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
            similar_cases=map_search_candidates_to_similar_cases(
                reranked_candidates,
                top_n=resolve_identify_top_n(self._settings),
            ),
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


class ClueMiningFlow:
    def __init__(
        self,
        llm_judge_engine: LLMJudgeEngine,
        audit_logger: AuditLogger,
    ) -> None:
        self._llm_judge_engine = llm_judge_engine
        self._audit_logger = audit_logger

    async def execute(
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
        )
        clue_result = await self._llm_judge_engine.mine_clues(
            request=identify_request,
            similar_cases=[
                SimilarCase(
                    case_id=request.similar_case.case_id,
                    petition_id=None,
                    similarity_score=0,
                    rank=1,
                    location=request.similar_case.location,
                    reported_persons=request.similar_case.reported_persons,
                    reporter=request.similar_case.reporter,
                    description_text=request.similar_case.description_text,
                )
            ],
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


class IdentifyPipeline:
    def __init__(
        self,
        settings: Settings,
        hybrid_search_engine: HybridSearchEngine,
        rerank_engine: RerankEngine,
        llm_judge_engine: LLMJudgeEngine,
        audit_logger: AuditLogger,
    ) -> None:
        self._identify_flow = IdentifyFlow(
            settings=settings,
            hybrid_search_engine=hybrid_search_engine,
            rerank_engine=rerank_engine,
            audit_logger=audit_logger,
        )
        self._clue_mining_flow = ClueMiningFlow(
            llm_judge_engine=llm_judge_engine,
            audit_logger=audit_logger,
        )

    async def identify(self, request: IdentifyRequest, request_id: str) -> IdentifyResponse:
        return await self._identify_flow.execute(request, request_id)

    async def mine_clues(
        self,
        request: ClueMiningRequest,
        request_id: str,
    ) -> ClueMiningResponse:
        return await self._clue_mining_flow.execute(request, request_id)
