from datetime import datetime, timedelta, timezone

from qdrant_client.http import models


def cutoff_datetime(years: int, now: datetime = None) -> datetime:
    active_now = now or datetime.now(timezone.utc)
    return active_now - timedelta(days=365 * years)


def build_case_filter(location: str, time_range_years: int) -> models.Filter:
    cutoff = cutoff_datetime(time_range_years)
    return models.Filter(
        must=[
            models.FieldCondition(
                key="location",
                match=models.MatchValue(value=location),
            ),
            models.FieldCondition(
                key="create_time",
                range=models.DatetimeRange(gte=cutoff.isoformat()),
            ),
        ]
    )
