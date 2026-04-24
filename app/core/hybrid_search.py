import asyncio
from typing import Dict, List

from app.config import Settings
from app.core.filter import build_case_filter
from app.models.domain import QueryEmbedding, SearchCandidate
from app.models.request import IdentifyRequest
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


def build_query_text(request: IdentifyRequest) -> str:
    weighted_names = " ".join(request.reported_persons * 2)
    location_tokens = " ".join([request.location] * 2)
    parts = [
        "属地: {0}".format(location_tokens),
        "举报人: {0}".format(request.reporter or ""),
        "被举报人: {0}".format(weighted_names),
        "案情描述: {0}".format(request.description),
    ]
    return "\n".join(item for item in parts if item.strip())


def reciprocal_rank_fusion(
    dense_hits: List[SearchCandidate],
    sparse_hits: List[SearchCandidate],
    request: IdentifyRequest,
    k: int,
    limit: int,
) -> List[SearchCandidate]:
    merged: Dict[str, SearchCandidate] = {}
    for hits, score_field in ((dense_hits, "dense_score"), (sparse_hits, "sparse_score")):
        for rank, candidate in enumerate(hits, start=1):
            current = merged.get(candidate.case_id)
            if current is None:
                current = candidate.model_copy()
                current.hybrid_score = 0.0
                merged[candidate.case_id] = current
            setattr(current, score_field, getattr(candidate, score_field))
            current.hybrid_score += 1.0 / (k + rank)

    for candidate in merged.values():
        if candidate.location == request.location:
            candidate.hybrid_score += 0.03

    return sorted(
        merged.values(),
        key=lambda item: (item.hybrid_score, item.rerank_score or 0.0),
        reverse=True,
    )[:limit]


class HybridSearchEngine:
    def __init__(
        self,
        settings: Settings,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._settings = settings
        self._qdrant_service = qdrant_service
        self._embedding_service = embedding_service

    async def search(self, request: IdentifyRequest) -> List[SearchCandidate]:
        query_text = build_query_text(request)
        query_embedding = await self._embedding_service.embed_text(query_text)
        query_filter = build_case_filter(
            request.reported_persons,
            request.start_time,
            request.end_time,
        )
        dense_task = self._qdrant_service.search_dense(
            embedding=query_embedding,
            query_filter=query_filter,
            limit=self._settings.hybrid_limit,
        )
        sparse_task = self._qdrant_service.search_sparse(
            embedding=query_embedding,
            query_filter=query_filter,
            limit=self._settings.hybrid_limit,
        )
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        return reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            request=request,
            k=self._settings.rrf_k,
            limit=self._settings.hybrid_limit,
        )
