from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    case_id: str
    petition_id: Optional[str] = None
    similarity_score: int = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    location: Optional[str] = None
    location_district: Optional[str] = None
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    description_text: Optional[str] = None
    create_time: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class NewClue(BaseModel):
    source_case_id: str = Field(
        description="Historical case identifier used as the comparison baseline for this clue."
    )
    clue_type: str
    description: str
    risk_level: str


class IdentifyResponse(BaseModel):
    similar_cases: List[SimilarCase] = Field(default_factory=list)
    processing_time_ms: int = Field(ge=0)
    request_id: str


class ClueMiningResponse(BaseModel):
    incremental_clues: List[NewClue] = Field(default_factory=list)
    supplemental_clues: List[NewClue] = Field(default_factory=list)
    processing_time_ms: int = Field(ge=0)
    request_id: str


class ApiErrorResponse(BaseModel):
    request_id: str
    error_code: str
    error_message: str
    retryable: bool = False


class DependencyStatus(BaseModel):
    name: str
    healthy: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class ReadyResponse(BaseModel):
    status: str
    dependencies: List[DependencyStatus]


class SyncResponse(BaseModel):
    started_at: datetime
    finished_at: datetime
    mode: str
    total_read: int = 0
    total_upserted: int = 0
    batches: int = 0
    last_updated_at: Optional[datetime] = None
    last_case_id: Optional[str] = None
