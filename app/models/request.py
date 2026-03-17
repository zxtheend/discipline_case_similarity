from typing import List, Optional

from pydantic import BaseModel, Field


class IdentifyRequest(BaseModel):
    reported_persons: List[str] = Field(default_factory=list, min_length=1)
    reporter: Optional[str] = None
    location: str = Field(min_length=1)
    description: str = Field(min_length=1)
    time_range_years: int = Field(default=5, ge=1, le=20)
