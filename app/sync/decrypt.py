from typing import Sequence

from app.errors import ServiceError
from app.models.domain import RowDecryptionResult, SourceTableRow
from app.services.decrypt_service import DecryptProvider


class DecryptionCoordinator:
    def __init__(self, decrypt_provider: DecryptProvider) -> None:
        self._decrypt_provider = decrypt_provider

    async def decrypt_rows(
        self,
        rows: Sequence[SourceTableRow],
    ) -> Sequence[RowDecryptionResult]:
        decrypted_rows = await self._decrypt_provider.decrypt_rows(rows)
        self.validate_results(rows=rows, decrypted_rows=decrypted_rows)
        return decrypted_rows

    def validate_results(
        self,
        rows: Sequence[SourceTableRow],
        decrypted_rows: Sequence[RowDecryptionResult],
    ) -> None:
        if len(rows) != len(decrypted_rows):
            raise ServiceError(
                error_code="decrypt_result_mismatch",
                message="Decrypt provider returned unexpected item count.",
                status_code=500,
            )

        for row, decrypted_row in zip(rows, decrypted_rows):
            if row.case_id != decrypted_row.case_id:
                raise ServiceError(
                    error_code="decrypt_result_order_mismatch",
                    message="Decrypt provider returned rows out of order.",
                    status_code=500,
                )
