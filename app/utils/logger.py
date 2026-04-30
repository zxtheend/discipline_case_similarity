import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


BUSINESS_LOG_CHANNEL = "business"
SYNC_REBUILD_LOG_CHANNEL = "sync_rebuild"
SYNC_FULL_LOG_CHANNEL = "sync_full"

_LOG_CHANNEL = ContextVar("sx_log_channel", default=None)


class SuppressSuccessfulHTTPXRequestsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "httpx":
            return True
        message = record.getMessage()
        if "HTTP Request:" not in message:
            return True
        return not _is_successful_httpx_request(message)


# Backward-compatible alias for existing imports/tests.
SuppressSuccessfulEmbeddingsFilter = SuppressSuccessfulHTTPXRequestsFilter


class LogChannelFilter(logging.Filter):
    def __init__(self, allowed_channel: str) -> None:
        super().__init__()
        self._allowed_channel = allowed_channel

    def filter(self, record: logging.LogRecord) -> bool:
        return _current_log_channel(record) == self._allowed_channel


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
        log_channel = _current_log_channel(record)
        if log_channel is not None:
            payload["log_channel"] = log_channel
        for key, value in record.__dict__.items():
            if key in self._reserved or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def set_log_channel(channel: Optional[str]) -> Token:
    return _LOG_CHANNEL.set(channel)


def reset_log_channel(token: Token) -> None:
    _LOG_CHANNEL.reset(token)


def current_log_channel() -> Optional[str]:
    return _LOG_CHANNEL.get()


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    default_channel: Optional[str] = None,
) -> None:
    root_logger = logging.getLogger()
    formatter = JsonFormatter()
    _ensure_httpx_filter()
    _configure_uvicorn_loggers()

    if default_channel is not None:
        _LOG_CHANNEL.set(default_channel)

    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler._sx_managed = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)

    _remove_managed_file_handlers(root_logger)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for channel, filename in (
            (BUSINESS_LOG_CHANNEL, "business.log"),
            (SYNC_REBUILD_LOG_CHANNEL, "sync_rebuild.log"),
            (SYNC_FULL_LOG_CHANNEL, "sync_full.log"),
        ):
            file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(LogChannelFilter(channel))
            file_handler._sx_managed = True  # type: ignore[attr-defined]
            file_handler._sx_file_handler = True  # type: ignore[attr-defined]
            root_logger.addHandler(file_handler)

    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _remove_managed_file_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, "_sx_file_handler", False):
            root_logger.removeHandler(handler)
            handler.close()


def _ensure_httpx_filter() -> None:
    httpx_logger = logging.getLogger("httpx")
    for current_filter in httpx_logger.filters:
        if isinstance(current_filter, SuppressSuccessfulHTTPXRequestsFilter):
            return
    httpx_logger.addFilter(SuppressSuccessfulHTTPXRequestsFilter())


def _configure_uvicorn_loggers() -> None:
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True


def _current_log_channel(record: logging.LogRecord) -> Optional[str]:
    value = getattr(record, "log_channel", None)
    if value is not None:
        return value
    return _LOG_CHANNEL.get()
