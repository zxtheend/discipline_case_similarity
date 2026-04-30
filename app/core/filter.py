from datetime import datetime

from qdrant_client.http import models

from app.utils.time_utils import normalize_to_utc


def _format_qdrant_datetime(value: datetime) -> str:
    return normalize_to_utc(value).isoformat()


def build_case_filter(
    reported_persons: list[str],
    start_time: datetime,
    end_time: datetime,
) -> models.Filter:
    normalized_persons = [item.strip() for item in reported_persons if item and item.strip()]
    return models.Filter(
        must=[
            models.FieldCondition(
                key="reported_persons",
                match=models.MatchAny(any=normalized_persons),
            ),
            models.FieldCondition(
                key="create_time",
                range=models.DatetimeRange(
                    gte=_format_qdrant_datetime(start_time),
                    lte=_format_qdrant_datetime(end_time),
                ),
            ),
        ]
    )
