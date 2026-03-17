from typing import Any, List, Optional, Protocol, Sequence

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import JoinedSourceRow, RowDecryptionResult


class DecryptProvider(Protocol):
    async def decrypt_rows(
        self,
        rows: Sequence[JoinedSourceRow],
    ) -> List[RowDecryptionResult]:
        ...


class NoopDecryptProvider:
    async def decrypt_rows(
        self,
        rows: Sequence[JoinedSourceRow],
    ) -> List[RowDecryptionResult]:
        return [
            RowDecryptionResult(
                case_id=row.case_id,
                reported_persons_text=self._normalize_value(row.encrypted_reported_persons),
                reporter_text=self._normalize_value(row.encrypted_reporter),
                description_text=self._normalize_value(row.encrypted_description),
            )
            for row in rows
        ]

    def _normalize_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip() or None
        return str(value).strip() or None


def build_decrypt_provider(settings: Settings) -> DecryptProvider:
    provider_name = settings.decrypt_provider.strip().lower()
    if provider_name in {"noop", "passthrough"}:
        return NoopDecryptProvider()
    raise ServiceError(
        error_code="decrypt_provider_unsupported",
        message="Unsupported decrypt provider: {0}".format(settings.decrypt_provider),
        status_code=500,
    )
