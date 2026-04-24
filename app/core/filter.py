from datetime import datetime

from qdrant_client.http import models


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
                    gte=start_time.isoformat(),
                    lte=end_time.isoformat(),
                ),
            ),
        ]
    )
