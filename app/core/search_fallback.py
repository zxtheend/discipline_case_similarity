from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from app.config import Settings
from app.models.domain import SearchCandidate

if TYPE_CHECKING:
    from app.services.qdrant_service import QdrantService


@dataclass(frozen=True)
class FallbackCompletion:
    candidates: list[SearchCandidate]
    triggered: bool
    added_count: int


class FallbackCandidateStrategy:
    def __init__(
        self,
        settings: Settings,
        qdrant_service: QdrantService,
    ) -> None:
        self._settings = settings
        self._qdrant_service = qdrant_service

    async def complete(
        self,
        merged_hits: Sequence[SearchCandidate],
        query_filter: Any,
    ) -> FallbackCompletion:
        fallback_triggered = len(merged_hits) < self._settings.fallback_min_candidates
        if not fallback_triggered:
            return FallbackCompletion(
                candidates=list(merged_hits),
                triggered=False,
                added_count=0,
            )

        target_count = max(self._settings.fallback_min_candidates, len(merged_hits))
        fallback_candidates = await self._qdrant_service.fetch_filtered_candidates(
            query_filter=query_filter,
            limit=max(self._settings.fallback_max_fetch, target_count),
        )
        existing_case_ids = {candidate.case_id for candidate in merged_hits}
        final_hits = list(merged_hits)
        added_count = 0
        for candidate in fallback_candidates:
            if candidate.case_id in existing_case_ids:
                continue
            final_hits.append(candidate)
            existing_case_ids.add(candidate.case_id)
            added_count += 1
            if len(final_hits) >= target_count:
                break

        return FallbackCompletion(
            candidates=final_hits,
            triggered=True,
            added_count=added_count,
        )
