from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


DEFAULT_IDENTIFY_WINDOW_YEARS = 5
REQUEST_MODEL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)
NonBlankStr = Annotated[str, StringConstraints(min_length=1)]
OptionalNonBlankStr = Optional[Annotated[str, StringConstraints(min_length=1)]]


class RequestModel(BaseModel):
    @field_validator("reported_persons", mode="before", check_fields=False)
    @classmethod
    def _normalize_reported_persons(cls, value):
        if value is None or not isinstance(value, (list, tuple)):
            return value

        normalized = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    normalized.append(item)
            else:
                normalized.append(item)

        if not normalized:
            raise ValueError("reported_persons must contain at least one non-blank value")
        return normalized

    @staticmethod
    def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class IdentifyRequest(RequestModel):
    model_config = REQUEST_MODEL_CONFIG

    reported_persons: List[str] = Field(min_length=1)
    reporter: Optional[str] = None
    location: NonBlankStr
    description: NonBlankStr
    start_time: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("start_time", "startTime"),
    )
    end_time: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("end_time", "endTime"),
    )

    @model_validator(mode="after")
    def normalize_time_range(self):
        normalized_end = self._normalize_datetime(self.end_time) or datetime.now(timezone.utc)
        normalized_start = self._normalize_datetime(self.start_time) or (
            normalized_end - timedelta(days=365 * DEFAULT_IDENTIFY_WINDOW_YEARS)
        )
        if normalized_start > normalized_end:
            raise ValueError("start_time must be earlier than or equal to end_time")

        self.start_time = normalized_start
        self.end_time = normalized_end
        return self


class ClueSimilarCaseRequest(RequestModel):
    model_config = REQUEST_MODEL_CONFIG

    case_id: NonBlankStr
    location: OptionalNonBlankStr = None
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    description_text: NonBlankStr


class ClueMiningRequest(RequestModel):
    model_config = REQUEST_MODEL_CONFIG

    reported_persons: List[str] = Field(min_length=1)
    reporter: Optional[str] = None
    location: NonBlankStr
    description: NonBlankStr
    similar_case: ClueSimilarCaseRequest


class RebuildRowRequest(RequestModel):
    model_config = REQUEST_MODEL_CONFIG

    case_id: NonBlankStr
    source_wtxx_bh: NonBlankStr
    petition_id: NonBlankStr
    location: OptionalNonBlankStr = None
    encrypted_reported_persons: Optional[str] = None
    encrypted_reporter: Optional[str] = None
    encrypted_description: Optional[str] = None
    create_time: datetime = Field(
        validation_alias=AliasChoices("create_time", "createTime"),
    )

    @field_validator("create_time", mode="before")
    @classmethod
    def normalize_create_time(cls, value) -> datetime:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timestamp = float(value)
            if abs(timestamp) >= 1_000_000_000_000:
                timestamp /= 1000.0
            value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        elif isinstance(value, str):
            if value.endswith("Z"):
                value = "{0}+00:00".format(value[:-1])
            value = datetime.fromisoformat(value)
        return cls._normalize_datetime(value)
