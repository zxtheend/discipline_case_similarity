from typing import List, Sequence

from app.errors import ServiceError
from app.models.domain import QueryEmbedding, SourceCase, SourceTableRow
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

from app.sync.decrypt import DecryptionCoordinator
from app.sync.mapping import SourceCaseMapper
from app.sync.telemetry import SyncTelemetry


class SourceCaseUpsertCoordinator:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        decryption_coordinator: DecryptionCoordinator,
        source_case_mapper: SourceCaseMapper,
        telemetry: SyncTelemetry,
    ) -> None:
        self._embedding_service = embedding_service
        self._qdrant_service = qdrant_service
        self._decryption_coordinator = decryption_coordinator
        self._source_case_mapper = source_case_mapper
        self._telemetry = telemetry

    async def embed_rows(self, rows: Sequence[SourceCase]) -> List[QueryEmbedding]:
        documents = [row.document_text for row in rows]
        return await self._embedding_service.embed_texts(documents)

    async def process_source_rows(
        self,
        request_id: str,
        rows: Sequence[SourceTableRow],
        fail_fast: bool,
    ) -> int:
        if not rows:
            return 0

        try:
            decrypted_rows = await self._decryption_coordinator.decrypt_rows(rows)
            source_cases = self._source_case_mapper.map_source_rows_to_source_cases(
                rows=rows,
                decrypted_rows=decrypted_rows,
                request_id=request_id,
                fail_fast=fail_fast,
            )
            if not source_cases:
                return 0

            try:
                embeddings = await self.embed_rows(source_cases)
                return await self._qdrant_service.upsert_cases(source_cases, embeddings)
            except ServiceError as exc:
                if fail_fast or len(source_cases) == 1:
                    raise

                self._telemetry.log_batch_fallback(
                    request_id=request_id,
                    batch_size=len(rows),
                    valid_source_cases=len(source_cases),
                    reason=exc.error_code or "batch_processing_failed",
                    details=exc.message,
                )
                return await self.process_source_cases_individually(
                    request_id=request_id,
                    rows=rows,
                    source_cases=source_cases,
                    fail_fast=fail_fast,
                )
        except ServiceError as exc:
            if fail_fast:
                raise
            if len(rows) == 1:
                self._telemetry.log_skipped_source_row(
                    request_id=request_id,
                    row=rows[0],
                    reason=exc.error_code or "row_processing_failed",
                    details=exc.message,
                )
                return 0
            raise

    async def process_source_row(
        self,
        request_id: str,
        row: SourceTableRow,
        fail_fast: bool,
    ) -> int:
        return await self.process_source_rows(
            request_id=request_id,
            rows=[row],
            fail_fast=fail_fast,
        )

    async def process_source_cases_individually(
        self,
        request_id: str,
        rows: Sequence[SourceTableRow],
        source_cases: Sequence[SourceCase],
        fail_fast: bool,
    ) -> int:
        row_by_case_id = {row.case_id: row for row in rows}
        upserted = 0
        for source_case in source_cases:
            try:
                embeddings = await self.embed_rows([source_case])
                upserted += await self._qdrant_service.upsert_cases([source_case], embeddings)
            except ServiceError as exc:
                if fail_fast:
                    raise
                row = row_by_case_id.get(source_case.case_id)
                if row is not None:
                    self._telemetry.log_skipped_source_row(
                        request_id=request_id,
                        row=row,
                        reason=exc.error_code or "row_processing_failed",
                        details=exc.message,
                    )
        return upserted
