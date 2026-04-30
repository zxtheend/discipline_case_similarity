from datetime import datetime, timezone
from typing import List, Optional, Sequence

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import (
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
from app.sync.decrypt import DecryptionCoordinator
from app.sync.mapping import SourceCaseMapper
from app.sync.telemetry import SyncTelemetry
from app.sync.text import split_reported_persons
from app.sync.upsert import SourceCaseUpsertCoordinator
from app.utils.audit import AuditLogger


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
        self._qdrant_service = qdrant_service
        self._telemetry = SyncTelemetry(audit_logger=audit_logger)
        self._decryption_coordinator = DecryptionCoordinator(decrypt_provider=decrypt_provider)
        self._source_case_mapper = SourceCaseMapper(telemetry=self._telemetry)
        self._upsert_coordinator = SourceCaseUpsertCoordinator(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            decryption_coordinator=self._decryption_coordinator,
            source_case_mapper=self._source_case_mapper,
            telemetry=self._telemetry,
        )

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

            upserted = await self._process_source_rows(
                request_id=request_id,
                rows=rows,
                fail_fast=False,
            )
            source_case_count = upserted

            total_read += len(rows)
            total_upserted += upserted
            batches += 1
            last_processed_row = rows[-1]
            last_case_id = last_processed_row.case_id
            self._telemetry.log_batch_completed(
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
        self._telemetry.log_single_row_completed(
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
            message=(
                "/admin/sync/incremental is deprecated and not supported. "
                "Use /admin/sync/rebuild-row for single-row updates."
            ),
            status_code=501,
        )

    async def _embed_rows(self, rows: List[SourceCase]) -> List[QueryEmbedding]:
        return await self._upsert_coordinator.embed_rows(rows)

    async def _process_source_rows(
        self,
        request_id: str,
        rows: Sequence[SourceTableRow],
        fail_fast: bool,
    ) -> int:
        return await self._upsert_coordinator.process_source_rows(
            request_id=request_id,
            rows=rows,
            fail_fast=fail_fast,
        )

    async def _process_source_cases_individually(
        self,
        request_id: str,
        rows: Sequence[SourceTableRow],
        source_cases: Sequence[SourceCase],
        fail_fast: bool,
    ) -> int:
        return await self._upsert_coordinator.process_source_cases_individually(
            request_id=request_id,
            rows=rows,
            source_cases=source_cases,
            fail_fast=fail_fast,
        )

    async def _process_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        fail_fast: bool,
    ) -> int:
        return await self._upsert_coordinator.process_source_row(
            request_id=request_id,
            row=row,
            fail_fast=fail_fast,
        )

    def _map_source_rows_to_source_cases(
        self,
        rows: Sequence[SourceTableRow],
        decrypted_rows: Sequence[RowDecryptionResult],
        request_id: str,
        fail_fast: bool = False,
    ) -> List[SourceCase]:
        self._decryption_coordinator.validate_results(
            rows=rows,
            decrypted_rows=decrypted_rows,
        )
        return self._source_case_mapper.map_source_rows_to_source_cases(
            rows=rows,
            decrypted_rows=decrypted_rows,
            request_id=request_id,
            fail_fast=fail_fast,
        )

    def _build_source_case_from_source_row(
        self,
        row: SourceTableRow,
        decrypted_row: RowDecryptionResult,
        request_id: str,
        fail_fast: bool,
    ) -> Optional[SourceCase]:
        return self._source_case_mapper.build_source_case_from_source_row(
            row=row,
            decrypted_row=decrypted_row,
            request_id=request_id,
            fail_fast=fail_fast,
        )

    def _handle_invalid_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        reason: str,
        fail_fast: bool,
        details: Optional[str] = None,
    ) -> Optional[SourceCase]:
        return self._source_case_mapper.handle_invalid_source_row(
            request_id=request_id,
            row=row,
            reason=reason,
            fail_fast=fail_fast,
            details=details,
        )

    def _log_skipped_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        reason: str,
        details: Optional[str] = None,
    ) -> None:
        self._telemetry.log_skipped_source_row(
            request_id=request_id,
            row=row,
            reason=reason,
            details=details,
        )
