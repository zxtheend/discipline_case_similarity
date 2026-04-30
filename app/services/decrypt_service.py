import base64
import binascii
import json
import re
from typing import Any, List, Optional, Protocol, Sequence

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import RowDecryptionResult, SourceTableRow


class DecryptProvider(Protocol):
    async def decrypt_rows(
        self,
        rows: Sequence[SourceTableRow],
    ) -> List[RowDecryptionResult]:
        ...


class NoopDecryptProvider:
    _hex_pattern = re.compile(r"^[0-9A-Fa-f]+$")
    _base64_pattern = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

    async def decrypt_rows(
        self,
        rows: Sequence[SourceTableRow],
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
            normalized = value.decode("utf-8", errors="replace").strip() or None
        else:
            normalized = str(value).strip() or None

        if normalized is None:
            return None

        base64_decoded = self._decode_base64_wrapped_value(normalized)
        if base64_decoded is not None:
            return self._normalize_text_value(base64_decoded)

        return self._normalize_text_value(normalized)

    def _normalize_text_value(self, value: str) -> Optional[str]:
        normalized = value.strip() or None
        if normalized is None:
            return None

        decrypted_json = self._decrypt_json_value(normalized)
        if decrypted_json is not None:
            return decrypted_json

        return self._decode_hex_cipher_text(normalized)

    def _decode_base64_wrapped_value(self, value: str) -> Optional[str]:
        candidate = value.strip()
        if len(candidate) < 8 or len(candidate) % 4 != 0:
            return None
        if not self._base64_pattern.fullmatch(candidate):
            return None

        try:
            decoded = base64.b64decode(candidate, validate=True)
        except (binascii.Error, ValueError):
            return None

        if not decoded:
            return None
        try:
            decoded_text = decoded.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        if not decoded_text or decoded_text == candidate:
            return None

        if self._looks_like_encrypted_payload(decoded_text):
            return decoded_text
        return None

    def _looks_like_encrypted_payload(self, value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if stripped.startswith(("{", "[")):
            return True
        compact = "".join(stripped.split())
        if len(compact) >= 4 and len(compact) % 2 == 0 and self._hex_pattern.fullmatch(compact):
            return True
        return False

    def _decrypt_json_value(self, value: str) -> Optional[str]:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None

        decrypted = self._decrypt_json_payload(payload)
        return json.dumps(decrypted, ensure_ascii=False, separators=(",", ":"))

    def _decrypt_json_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            decrypted = {}
            for key, value in payload.items():
                if key == "mc" and isinstance(value, str):
                    decrypted[key] = self._decode_hex_cipher_text(value)
                else:
                    decrypted[key] = self._decrypt_json_payload(value)
            return decrypted
        if isinstance(payload, list):
            return [self._decrypt_json_payload(item) for item in payload]
        return payload

    def _decode_hex_cipher_text(self, value: str) -> str:
        compact = "".join(value.split())
        if not compact:
            return value
        if len(compact) % 2 != 0:
            return value
        if not self._hex_pattern.fullmatch(compact):
            return value
        if not any(char.isalpha() for char in compact) and " " not in value:
            return value

        try:
            encrypted_bytes = bytes.fromhex(compact)
        except ValueError:
            return value

        decrypted_bytes = bytes((byte - 1) % 256 for byte in encrypted_bytes)
        try:
            return decrypted_bytes.decode("gbk").strip() or value
        except UnicodeDecodeError:
            return value


def build_decrypt_provider(settings: Settings) -> DecryptProvider:
    provider_name = settings.decrypt_provider.strip().lower()
    if provider_name in {"noop", "passthrough"}:
        return NoopDecryptProvider()
    raise ServiceError(
        error_code="decrypt_provider_unsupported",
        message="Unsupported decrypt provider: {0}".format(settings.decrypt_provider),
        status_code=500,
    )
