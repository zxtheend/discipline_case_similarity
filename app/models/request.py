from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_IDENTIFY_WINDOW_YEARS = 5


class IdentifyRequest(BaseModel):
    reported_persons: List[str] = Field(default_factory=list, min_length=1)
    reporter: Optional[str] = None
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

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

    @staticmethod
    def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class ClueSimilarCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    location: Optional[str] = None
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    description_text: str = Field(min_length=1)


class ClueMiningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reported_persons: List[str] = Field(default_factory=list, min_length=1)
    reporter: Optional[str] = None
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)
    similar_case: ClueSimilarCaseRequest


class RebuildRowRequest(BaseModel):
    case_id: str = Field(min_length=1)
    source_wtxx_bh: str = Field(min_length=1)
    petition_id: str = Field(min_length=1)
    location: Optional[str] = None
    encrypted_reported_persons: Optional[str] = None
    encrypted_reporter: Optional[str] = None
    encrypted_description: Optional[str] = None
    create_time: datetime
