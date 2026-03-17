import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import (
    JoinedSourceRow,
    QueryEmbedding,
    RowDecryptionResult,
    SourceCase,
    SyncCursor,
    SyncRunResult,
)
from app.services.decrypt_service import DecryptProvider
from app.services.embedding_service import EmbeddingService
from app.services.mysql_service import MySQLService
from app.services.qdrant_service import QdrantService
from app.utils.audit import AuditLogger
from app.utils.logger import get_logger


_REPORTED_PERSON_SPLIT_PATTERN = re.compile(r"[,，]")


def split_reported_persons(value: Optional[str]) -> List[str]:
    if not value:
        return []
    seen = set()
    persons = []
    for raw_item in _REPORTED_PERSON_SPLIT_PATTERN.split(value):
        item = raw_item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        persons.append(item)
    return persons


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


class SyncStateStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> SyncCursor:
        if not self._path.exists():
            return SyncCursor()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return SyncCursor.model_validate(payload)

    def save(self, cursor: SyncCursor) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(
            cursor.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)

    def reset(self) -> None:
        self.save(SyncCursor())


class DataSyncService:
    def __init__(
        self,
        settings: Settings,
        mysql_service: MySQLService,
        decrypt_provider: DecryptProvider,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        audit_logger: AuditLogger,
    ) -> None:
        self._settings = settings
        self._mysql_service = mysql_service
        self._decrypt_provider = decrypt_provider
        self._embedding_service = embedding_service
        self._qdrant_service = qdrant_service
        self._audit_logger = audit_logger
        self._state_store = SyncStateStore(settings.sync_state_path)
        self._logger = get_logger("data_sync")

    async def full_sync(self, request_id: str) -> SyncRunResult:
        await self._qdrant_service.recreate_collection()
        started_at = datetime.now(timezone.utc)
        total_read = 0
        total_upserted = 0
        batches = 0
        last_case_id: Optional[str] = None
        last_processed_row: Optional[JoinedSourceRow] = None

        while True:
            rows = await self._mysql_service.fetch_joined_source_rows(
                limit=self._settings.sync_batch_size,
                last_case_id=last_case_id,
            )
            if not rows:
                break

            decrypted_rows = await self._decrypt_provider.decrypt_rows(rows)
            source_cases = self._map_rows_to_source_cases(
                rows=rows,
                decrypted_rows=decrypted_rows,
                request_id=request_id,
            )
            upserted = 0
            if source_cases:
                embeddings = await self._embed_rows(source_cases)
                upserted = await self._qdrant_service.upsert_cases(source_cases, embeddings)

            total_read += len(rows)
            total_upserted += upserted
            batches += 1
            last_processed_row = rows[-1]
            last_case_id = last_processed_row.case_id
            self._audit_logger.log_event(
                "sync_batch_completed",
                request_id=request_id,
                mode="full",
                batch_size=len(rows),
                source_cases=len(source_cases),
                upserted=upserted,
                last_case_id=last_case_id,
            )

        finished_at = datetime.now(timezone.utc)
        return SyncRunResult(
            started_at=started_at,
            finished_at=finished_at,
            mode="full",
            total_read=total_read,
            total_upserted=total_upserted,
            batches=batches,
            cursor=SyncCursor(
                last_updated_at=last_processed_row.updated_at if last_processed_row else None,
                last_case_id=last_processed_row.case_id if last_processed_row else None,
            ),
        )

    async def incremental_sync(self, request_id: str) -> SyncRunResult:
        raise ServiceError(
            error_code="incremental_not_supported",
            message="Current version only supports full sync.",
            status_code=501,
        )

    async def _embed_rows(self, rows: List[SourceCase]) -> List[QueryEmbedding]:
        documents = [row.document_text for row in rows]
        return await self._embedding_service.embed_texts(documents)

    def _map_rows_to_source_cases(
        self,
        rows: Sequence[JoinedSourceRow],
        decrypted_rows: Sequence[RowDecryptionResult],
        request_id: str,
    ) -> List[SourceCase]:
        if len(rows) != len(decrypted_rows):
            raise ServiceError(
                error_code="decrypt_result_mismatch",
                message="Decrypt provider returned unexpected item count.",
                status_code=500,
            )

        source_cases = []
        for row, decrypted_row in zip(rows, decrypted_rows):
            if row.case_id != decrypted_row.case_id:
                raise ServiceError(
                    error_code="decrypt_result_order_mismatch",
                    message="Decrypt provider returned rows out of order.",
                    status_code=500,
                )

            if decrypted_row.error_message:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="decrypt_failed",
                    details=decrypted_row.error_message,
                )
                continue

            if not row.petition_id:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="missing_joined_xfj",
                )
                continue

            location = clean_text(row.location)
            if not location:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="missing_location",
                )
                continue

            description_text = clean_text(decrypted_row.description_text)
            if not description_text:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="missing_description",
                )
                continue

            if row.create_time is None:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="missing_create_time",
                )
                continue

            updated_at = row.updated_at
            if updated_at is None:
                self._log_skipped_row(
                    request_id=request_id,
                    row=row,
                    reason="missing_updated_at",
                )
                continue

            source_cases.append(
                SourceCase(
                    case_id=row.case_id,
                    reported_persons=split_reported_persons(decrypted_row.reported_persons_text),
                    reporter=clean_text(decrypted_row.reporter_text),
                    location=location,
                    description_text=description_text,
                    create_time=row.create_time,
                    updated_at=updated_at,
                    extra={
                        "petition_id": row.petition_id,
                        "source_xfj_bh": row.source_xfj_bh,
                    },
                )
            )
        return source_cases

    def _log_skipped_row(
        self,
        request_id: str,
        row: JoinedSourceRow,
        reason: str,
        details: Optional[str] = None,
    ) -> None:
        self._logger.warning(
            "sync_row_skipped",
            extra={
                "request_id": request_id,
                "case_id": row.case_id,
                "petition_id": row.petition_id,
                "reason": reason,
                "details": details,
            },
        )
