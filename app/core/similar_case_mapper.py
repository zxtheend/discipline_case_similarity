from collections.abc import Sequence
from typing import Any

from app.models.domain import SearchCandidate
from app.models.response import SimilarCase


def resolve_identify_top_n(settings: Any) -> int:
    identify_top_n = getattr(settings, "identify_top_n", None)
    if identify_top_n is not None:
        return identify_top_n
    return settings.judge_top_n


def map_search_candidates_to_similar_cases(
    candidates: Sequence[SearchCandidate],
    top_n: int,
) -> list[SimilarCase]:
    similar_cases = []
    for rank, candidate in enumerate(candidates[:top_n], start=1):
        rerank_score = candidate.rerank_score or 0.0
        similarity_score = max(0, min(100, int(round(rerank_score * 100))))
        similar_cases.append(
            SimilarCase(
                case_id=candidate.case_id,
                petition_id=candidate.petition_id,
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
