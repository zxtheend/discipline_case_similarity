import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class SuppressSuccessfulEmbeddingsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "httpx":
            return True
        message = record.getMessage()
        if "HTTP Request:" not in message:
            return True
        return not _is_successful_httpx_request(message)


def _is_successful_httpx_request(message: str) -> bool:
    success_markers = (
        " 200 OK",
        '"200 OK"',
        " 201 Created",
        '"201 Created"',
        " 202 Accepted",
        '"202 Accepted"',
        " 204 No Content",
        '"204 No Content"',
        " 301 Moved Permanently",
        '"301 Moved Permanently"',
        " 302 Found",
        '"302 Found"',
        " 304 Not Modified",
        '"304 Not Modified"',
    )
    return any(marker in message for marker in success_markers)


class JsonFormatter(logging.Formatter):
    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._reserved or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _ensure_httpx_filter() -> None:
    httpx_logger = logging.getLogger("httpx")
    for current_filter in httpx_logger.filters:
        if isinstance(current_filter, SuppressSuccessfulEmbeddingsFilter):
            return
    httpx_logger.addFilter(SuppressSuccessfulEmbeddingsFilter())


def configure_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    _ensure_httpx_filter()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(JsonFormatter())
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
