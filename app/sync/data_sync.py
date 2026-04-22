import json
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
    SourceTableRow,
    SyncCursor,
    SyncRunResult,
)
from app.services.decrypt_service import DecryptProvider
from app.services.embedding_service import EmbeddingService
from app.services.mysql_service import MySQLService
from app.services.qdrant_service import QdrantService
from app.utils.audit import AuditLogger
from app.utils.logger import get_logger

def split_reported_persons(value: Optional[str]) -> List[str]:
    if not value:
        return []
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip()
    if not normalized:
        return []

    parsed_names = _extract_reported_person_names_from_json(normalized)
    return parsed_names or []


def _extract_reported_person_names_from_json(value: str) -> Optional[List[str]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None

    seen = set()
    names: List[str] = []

    def append_name(raw_value: object) -> None:
        cleaned = clean_text(raw_value if isinstance(raw_value, str) else None)
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        names.append(cleaned)

    if isinstance(payload, dict):
        append_name(payload.get("mc"))
        return names

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                append_name(item.get("mc"))
            elif isinstance(item, str):
                append_name(item)
        return names

    return None


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
        last_processed_row: Optional[SourceTableRow] = None

        while True:
            rows = await self._mysql_service.fetch_source_rows(
                limit=self._settings.sync_batch_size,
                last_case_id=last_case_id,
            )
            if not rows:
                break

            upserted = 0
            source_case_count = 0
            for row in rows:
                processed = await self._process_source_row(
                    request_id=request_id,
                    row=row,
                    fail_fast=False,
                )
                source_case_count += processed
                upserted += processed

            total_read += len(rows)
            total_upserted += upserted
            batches += 1
            last_processed_row = rows[-1]
            last_case_id = last_processed_row.case_id
            self._audit_logger.log_event(
                "sync_batch_completed",
                request_id=request_id,
                mode="full",
                batch_number=batches,
                batch_size=len(rows),
                source_cases=source_case_count,
                upserted=upserted,
                total_read=total_read,
                total_upserted=total_upserted,
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

    async def rebuild_row(self, request_id: str, row: SourceTableRow) -> SyncRunResult:
        started_at = datetime.now(timezone.utc)
        upserted = await self._process_source_row(
            request_id=request_id,
            row=row,
            fail_fast=True,
        )
        finished_at = datetime.now(timezone.utc)
        self._audit_logger.log_event(
            "sync_single_row_completed",
            request_id=request_id,
            mode="rebuild-row",
            case_id=row.case_id,
            upserted=upserted,
        )
        return SyncRunResult(
            started_at=started_at,
            finished_at=finished_at,
            mode="rebuild-row",
            total_read=1,
            total_upserted=upserted,
            batches=1,
            cursor=SyncCursor(
                last_updated_at=row.updated_at,
                last_case_id=row.case_id,
            ),
        )

    async def incremental_sync(self, request_id: str) -> SyncRunResult:
        raise ServiceError(
            error_code="incremental_not_supported",
            message="Current version uses /admin/sync/rebuild-row for incremental updates.",
            status_code=501,
        )

    async def _embed_rows(self, rows: List[SourceCase]) -> List[QueryEmbedding]:
        documents = [row.document_text for row in rows]
        return await self._embedding_service.embed_texts(documents)

    async def _process_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        fail_fast: bool,
    ) -> int:
        try:
            decrypted_rows = await self._decrypt_provider.decrypt_rows([row])
            source_cases = self._map_source_rows_to_source_cases(
                rows=[row],
                decrypted_rows=decrypted_rows,
                request_id=request_id,
                fail_fast=fail_fast,
            )
            if not source_cases:
                return 0

            embeddings = await self._embed_rows(source_cases)
            return await self._qdrant_service.upsert_cases(source_cases, embeddings)
        except ServiceError as exc:
            if fail_fast:
                raise
            self._log_skipped_source_row(
                request_id=request_id,
                row=row,
                reason=exc.error_code or "row_processing_failed",
                details=exc.message,
            )
            return 0

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
                        "source_wtxx_bh": row.source_wtxx_bh,
                    },
                )
            )
        return source_cases

    def _map_source_rows_to_source_cases(
        self,
        rows: Sequence[SourceTableRow],
        decrypted_rows: Sequence[RowDecryptionResult],
        request_id: str,
        fail_fast: bool = False,
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

            source_case = self._build_source_case_from_source_row(
                row=row,
                decrypted_row=decrypted_row,
                request_id=request_id,
                fail_fast=fail_fast,
            )
            if source_case is not None:
                source_cases.append(source_case)
        return source_cases

    def _build_source_case_from_source_row(
        self,
        row: SourceTableRow,
        decrypted_row: RowDecryptionResult,
        request_id: str,
        fail_fast: bool,
    ) -> Optional[SourceCase]:
        if decrypted_row.error_message:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="decrypt_failed",
                details=decrypted_row.error_message,
                fail_fast=fail_fast,
            )

        case_id = clean_text(row.case_id)
        if not case_id:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_case_id",
                fail_fast=fail_fast,
            )

        source_wtxx_bh = clean_text(row.source_wtxx_bh)
        if not source_wtxx_bh:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_source_wtxx_bh",
                fail_fast=fail_fast,
            )

        petition_id = clean_text(row.petition_id)
        if not petition_id:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_petition_id",
                fail_fast=fail_fast,
            )

        reported_persons = split_reported_persons(decrypted_row.reported_persons_text)
        if not reported_persons:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_reported_persons",
                fail_fast=fail_fast,
            )

        description_text = clean_text(decrypted_row.description_text)
        if not description_text:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_description",
                fail_fast=fail_fast,
            )

        if row.create_time is None:
            return self._handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_create_time",
                fail_fast=fail_fast,
            )

        return SourceCase(
            case_id=case_id,
            reported_persons=reported_persons,
            reporter=clean_text(decrypted_row.reporter_text),
            location=clean_text(row.location),
            description_text=description_text,
            create_time=row.create_time,
            updated_at=row.updated_at or row.create_time,
            extra={
                "petition_id": petition_id,
                "source_wtxx_bh": source_wtxx_bh,
            },
        )

    def _handle_invalid_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        reason: str,
        fail_fast: bool,
        details: Optional[str] = None,
    ) -> Optional[SourceCase]:
        if fail_fast:
            raise ServiceError(
                error_code=reason,
                message="Invalid source row for case_id={0}".format(row.case_id),
                status_code=400,
                details={"case_id": row.case_id, "reason": reason, "details": details},
            )
        self._log_skipped_source_row(
            request_id=request_id,
            row=row,
            reason=reason,
            details=details,
        )
        return None

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

    def _log_skipped_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        reason: str,
        details: Optional[str] = None,
    ) -> None:
        self._logger.warning(
            "sync_source_row_skipped",
            extra={
                "request_id": request_id,
                "case_id": row.case_id,
                "petition_id": row.petition_id,
                "reason": reason,
                "details": details,
            },
        )
