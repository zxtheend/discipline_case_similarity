import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


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


def configure_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
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
