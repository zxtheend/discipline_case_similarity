from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class IdentifyRequest(BaseModel):
    reported_persons: List[str] = Field(default_factory=list, min_length=1)
    reporter: Optional[str] = None
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)
    time_range_years: int = Field(default=5, ge=1, le=20)


class ClueSimilarCaseRequest(BaseModel):
    case_id: str = Field(min_length=1)
    similarity_score: int = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    reason: str = Field(min_length=1)
    location: Optional[str] = None
    location_district: Optional[str] = None
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    description_text: Optional[str] = None
    create_time: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ClueMiningRequest(IdentifyRequest):
    similar_cases: List[ClueSimilarCaseRequest] = Field(default_factory=list)


class RebuildRowRequest(BaseModel):
    case_id: str = Field(min_length=1)
    source_wtxx_bh: str = Field(min_length=1)
    petition_id: str = Field(min_length=1)
    location: Optional[str] = None
    encrypted_reported_persons: Optional[str] = None
    encrypted_reporter: Optional[str] = None
    encrypted_description: Optional[str] = None
    create_time: datetime
