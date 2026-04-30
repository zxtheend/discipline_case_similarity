from datetime import datetime, timezone
from typing import Union


def normalize_to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def serialize_utc_datetime(value: datetime) -> str:
    return normalize_to_utc(value).isoformat()


def parse_utc_datetime(value: Union[datetime, str]) -> datetime:
    if isinstance(value, datetime):
        return normalize_to_utc(value)
    if isinstance(value, str):
        return normalize_to_utc(datetime.fromisoformat(value))
    raise TypeError("Unsupported datetime value type")
