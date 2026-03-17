from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.response import NewClue, SimilarCase


class SparseEmbedding(BaseModel):
    indices: List[int] = Field(default_factory=list)
    values: List[float] = Field(default_factory=list)


class QueryEmbedding(BaseModel):
    dense_vector: List[float]
    sparse_vector: SparseEmbedding = Field(default_factory=SparseEmbedding)


class SourceCase(BaseModel):
    case_id: str
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    location: str
    location_district: Optional[str] = None
    description_text: str
    create_time: datetime
    updated_at: datetime
    status: str = "ACTIVE"
    extra: Dict[str, Any] = Field(default_factory=dict)

    @property
    def document_text(self) -> str:
        parts = [
            "属地: {0}".format(self.location),
            "区县: {0}".format(self.location_district or ""),
            "举报人: {0}".format(self.reporter or ""),
            "被举报人: {0}".format("、".join(self.reported_persons)),
            "案情: {0}".format(self.description_text),
        ]
        return "\n".join(item for item in parts if item.strip())


class JoinedSourceRow(BaseModel):
    case_id: str
    source_xfj_bh: Optional[str] = None
    petition_id: Optional[str] = None
    encrypted_reported_persons: Any = None
    encrypted_reporter: Any = None
    encrypted_description: Any = None
    location: Optional[str] = None
    create_time: Optional[datetime] = None
    w_updated_at: Optional[datetime] = None
    x_create_time: Optional[datetime] = None
    x_updated_at: Optional[datetime] = None

    @property
    def updated_at(self) -> Optional[datetime]:
        candidates = [
            value
            for value in (
                self.w_updated_at,
                self.create_time,
                self.x_updated_at,
                self.x_create_time,
            )
            if value is not None
        ]
        if not candidates:
            return None
        return max(candidates)


class RowDecryptionResult(BaseModel):
    case_id: str
    reported_persons_text: Optional[str] = None
    reporter_text: Optional[str] = None
    description_text: Optional[str] = None
    error_message: Optional[str] = None


class SearchCandidate(BaseModel):
    case_id: str
    location: str
    location_district: Optional[str] = None
    reported_persons: List[str] = Field(default_factory=list)
    reporter: Optional[str] = None
    description_text: str
    create_time: datetime
    updated_at: Optional[datetime] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hybrid_score: float = 0.0
    rerank_score: Optional[float] = None
    reason: Optional[str] = None

    @property
    def rerank_document(self) -> str:
        parts = [
            "案件编号: {0}".format(self.case_id),
            "属地: {0}".format(self.location),
            "区县: {0}".format(self.location_district or ""),
            "举报人: {0}".format(self.reporter or ""),
            "被举报人: {0}".format("、".join(self.reported_persons)),
            "案情摘要: {0}".format(self.description_text),
        ]
        return "\n".join(parts)


class DuplicateJudgeResult(BaseModel):
    is_duplicate: bool
    ranked_cases: List[SimilarCase] = Field(default_factory=list)


class ClueMiningResult(BaseModel):
    new_clues: List[NewClue] = Field(default_factory=list)


class SyncCursor(BaseModel):
    last_updated_at: Optional[datetime] = None
    last_case_id: Optional[str] = None


class SyncRunResult(BaseModel):
    started_at: datetime
    finished_at: datetime
    mode: str
    total_read: int = 0
    total_upserted: int = 0
    batches: int = 0
    cursor: SyncCursor = Field(default_factory=SyncCursor)
