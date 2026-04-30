import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from fastapi import Request
from pydantic import ValidationError

from app.container import ApplicationContainer
from app.models.domain import SourceTableRow
from app.models.request import RebuildRowRequest
from app.utils.logger import get_logger


logger = get_logger("api_dependencies")

_LEGACY_ALIAS_KEYS_BY_ROUTE = {
    "/identify": ("startTime", "endTime"),
    "/admin/sync/rebuild-row": ("createTime",),
}
_SENSITIVE_FIELD_MARKERS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "description",
    "reporter",
    "reported_persons",
)
_MAX_LOG_VALUE_LENGTH = 256


@dataclass(frozen=True)
class RequestContext:
    container: ApplicationContainer
    request_id: str
    request: Request


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def get_request_context(request: Request) -> RequestContext:
    await _log_legacy_alias_usage(request)
    return RequestContext(
        container=get_container(request),
        request_id=get_request_id(request),
        request=request,
    )


async def resolve_request_context(
    request: Request,
    context: Optional[RequestContext],
) -> RequestContext:
    if isinstance(context, RequestContext):
        return context
    return await get_request_context(request)


def build_source_row_from_rebuild_request(payload: RebuildRowRequest) -> SourceTableRow:
    return SourceTableRow.model_validate(payload.model_dump())


def build_rebuild_request_from_payload(payload: Any) -> RebuildRowRequest:
    normalized_payload = extract_rebuild_row_payload(payload)
    return RebuildRowRequest.model_validate(normalized_payload)


def extract_rebuild_row_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError.from_exception_data(
            "RebuildRowRequest",
            [
                {
                    "type": "model_type",
                    "loc": ("body",),
                    "msg": "Input should be a valid dictionary or object to extract fields from",
                    "input": payload,
                    "ctx": {"class_name": "RebuildRowRequest"},
                }
            ],
        )

    if "BinLogPosition" in payload or "OldData" in payload:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"BinLogPosition", "OldData"}
        }
    return payload


def sanitize_for_logging(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_field(str(key), item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, bytes):
        return _truncate(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _truncate(value)
    return value


def sanitize_payload_json(value: Any) -> str:
    return json.dumps(
        sanitize_for_logging(value),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def sanitize_request_body(raw_body: bytes) -> str:
    if not raw_body:
        return ""
    try:
        payload = json.loads(raw_body)
    except (TypeError, ValueError):
        return _truncate(raw_body.decode("utf-8", errors="replace"))
    return sanitize_payload_json(payload)


async def _log_legacy_alias_usage(request: Request) -> None:
    legacy_keys = await _extract_legacy_alias_keys(request)
    if not legacy_keys:
        return

    logger.warning(
        "legacy_request_alias_used",
        extra={
            "request_id": get_request_id(request),
            "legacy_keys": legacy_keys,
            "path": _get_request_path(request),
        },
    )


async def _extract_legacy_alias_keys(request: Request) -> Sequence[str]:
    route_path = _get_route_path(request)
    candidate_keys = _get_legacy_alias_candidates(route_path)
    if not candidate_keys:
        return ()

    json_reader = getattr(request, "json", None)
    if not callable(json_reader):
        return ()

    payload = await json_reader()
    if not isinstance(payload, dict):
        return ()

    return tuple(key for key in candidate_keys if key in payload)


def _get_route_path(request: Request) -> str:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        route = scope.get("route")
        route_path = getattr(route, "path", "")
        if isinstance(route_path, str):
            return route_path
    return ""


def _get_request_path(request: Request) -> str:
    url = getattr(request, "url", None)
    path = getattr(url, "path", "")
    return path if isinstance(path, str) else ""


def _get_legacy_alias_candidates(route_path: str) -> Sequence[str]:
    for candidate_route, candidate_keys in _LEGACY_ALIAS_KEYS_BY_ROUTE.items():
        if route_path == candidate_route or route_path.endswith(candidate_route):
            return candidate_keys
    return ()


def _sanitize_field(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]" if value is not None else None
    return sanitize_for_logging(value)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_FIELD_MARKERS)


def _truncate(value: str) -> str:
    if len(value) <= _MAX_LOG_VALUE_LENGTH:
        return value
    return "{0}...<truncated>".format(value[:_MAX_LOG_VALUE_LENGTH])
