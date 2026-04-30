from typing import Optional

from app.models.domain import SourceTableRow
from app.utils.audit import AuditLogger
from app.utils.logger import get_logger


class SyncTelemetry:
    def __init__(self, audit_logger: AuditLogger) -> None:
        self._audit_logger = audit_logger
        self._logger = get_logger("data_sync")

    def log_batch_completed(
        self,
        request_id: str,
        mode: str,
        batch_number: int,
        batch_size: int,
        source_cases: int,
        upserted: int,
        total_read: int,
        total_upserted: int,
        last_case_id: Optional[str],
    ) -> None:
        self._audit_logger.log_event(
            "sync_batch_completed",
            request_id=request_id,
            mode=mode,
            batch_number=batch_number,
            batch_size=batch_size,
            source_cases=source_cases,
            upserted=upserted,
            total_read=total_read,
            total_upserted=total_upserted,
            last_case_id=last_case_id,
        )

    def log_single_row_completed(
        self,
        request_id: str,
        mode: str,
        case_id: str,
        upserted: int,
    ) -> None:
        self._audit_logger.log_event(
            "sync_single_row_completed",
            request_id=request_id,
            mode=mode,
            case_id=case_id,
            upserted=upserted,
        )

    def log_batch_fallback(
        self,
        request_id: str,
        batch_size: int,
        valid_source_cases: int,
        reason: str,
        details: Optional[str] = None,
    ) -> None:
        self._logger.warning(
            "sync_batch_fallback_to_row_mode",
            extra={
                "request_id": request_id,
                "batch_size": batch_size,
                "valid_source_cases": valid_source_cases,
                "reason": reason,
                "details": details,
            },
        )

    def log_skipped_source_row(
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
