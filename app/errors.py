from typing import Any, Dict, Optional


class ServiceError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 500,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}
