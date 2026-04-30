from typing import List, Optional, Sequence

from app.errors import ServiceError
from app.models.domain import RowDecryptionResult, SourceCase, SourceTableRow

from app.sync.telemetry import SyncTelemetry
from app.sync.text import clean_text, split_reported_persons
from app.utils.time_utils import normalize_to_utc


class SourceCaseMapper:
    def __init__(self, telemetry: SyncTelemetry) -> None:
        self._telemetry = telemetry

    def map_source_rows_to_source_cases(
        self,
        rows: Sequence[SourceTableRow],
        decrypted_rows: Sequence[RowDecryptionResult],
        request_id: str,
        fail_fast: bool = False,
    ) -> List[SourceCase]:
        source_cases = []
        for row, decrypted_row in zip(rows, decrypted_rows):
            source_case = self.build_source_case_from_source_row(
                row=row,
                decrypted_row=decrypted_row,
                request_id=request_id,
                fail_fast=fail_fast,
            )
            if source_case is not None:
                source_cases.append(source_case)
        return source_cases

    def build_source_case_from_source_row(
        self,
        row: SourceTableRow,
        decrypted_row: RowDecryptionResult,
        request_id: str,
        fail_fast: bool,
    ) -> Optional[SourceCase]:
        if decrypted_row.error_message:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="decrypt_failed",
                details=decrypted_row.error_message,
                fail_fast=fail_fast,
            )

        case_id = clean_text(row.case_id)
        if not case_id:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_case_id",
                fail_fast=fail_fast,
            )

        source_wtxx_bh = clean_text(row.source_wtxx_bh)
        if not source_wtxx_bh:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_source_wtxx_bh",
                fail_fast=fail_fast,
            )

        petition_id = clean_text(row.petition_id)
        if not petition_id:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_petition_id",
                fail_fast=fail_fast,
            )

        reported_persons = split_reported_persons(decrypted_row.reported_persons_text)
        if not reported_persons:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_reported_persons",
                fail_fast=fail_fast,
            )

        description_text = clean_text(decrypted_row.description_text)
        if not description_text:
            return self.handle_invalid_source_row(
                request_id=request_id,
                row=row,
                reason="missing_description",
                fail_fast=fail_fast,
            )

        if row.create_time is None:
            return self.handle_invalid_source_row(
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
            create_time=normalize_to_utc(row.create_time),
            updated_at=normalize_to_utc(row.updated_at or row.create_time),
            extra={
                "petition_id": petition_id,
                "source_wtxx_bh": source_wtxx_bh,
            },
        )

    def handle_invalid_source_row(
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
        self._telemetry.log_skipped_source_row(
            request_id=request_id,
            row=row,
            reason=reason,
            details=details,
        )
        return None
