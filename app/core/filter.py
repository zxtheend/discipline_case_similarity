from datetime import datetime, timedelta, timezone

from qdrant_client.http import models


def cutoff_datetime(years: int, now: datetime = None) -> datetime:
    active_now = now or datetime.now(timezone.utc)
    return active_now - timedelta(days=365 * years)


def build_case_filter(reported_persons: list[str], time_range_years: int) -> models.Filter:
    cutoff = cutoff_datetime(time_range_years)
    normalized_persons = [item.strip() for item in reported_persons if item and item.strip()]
    return models.Filter(
        must=[
            models.FieldCondition(
                key="reported_persons",
                match=models.MatchAny(any=normalized_persons),
            ),
            models.FieldCondition(
                key="create_time",
                range=models.DatetimeRange(gte=cutoff.isoformat()),
            ),
        ]
    )
