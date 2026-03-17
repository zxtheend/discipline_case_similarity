from typing import Any

from app.utils.logger import get_logger


class AuditLogger:
    def __init__(self) -> None:
        self._logger = get_logger("audit")

    def log_event(self, event_type: str, request_id: str, **details: Any) -> None:
        self._logger.info(
            "audit_event",
            extra={
                "event_type": event_type,
                "request_id": request_id,
                "details": details,
            },
        )
