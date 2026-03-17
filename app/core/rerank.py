from typing import List

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import SearchCandidate
from app.services.rerank_service import RerankService


class RerankEngine:
    def __init__(self, settings: Settings, rerank_service: RerankService) -> None:
        self._settings = settings
        self._rerank_service = rerank_service

    async def rerank(self, query: str, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        if not candidates:
            return []

        documents = [candidate.rerank_document for candidate in candidates]
        results = await self._rerank_service.rerank(
            query=query,
            documents=documents,
            top_n=min(self._settings.rerank_top_n, len(documents)),
        )
        if not results:
            raise ServiceError(
                error_code="rerank_empty",
                message="Rerank service returned no results.",
                status_code=502,
                retryable=True,
            )

        reranked = []
        for item in results:
            candidate = candidates[item["index"]].model_copy(
                update={"rerank_score": item["score"]}
            )
            reranked.append(candidate)
        return sorted(reranked, key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
