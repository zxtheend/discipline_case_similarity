from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.dependencies import (
    RequestContext,
    build_rebuild_request_from_payload,
    build_source_row_from_rebuild_request,
    get_request_context,
    resolve_request_context,
    sanitize_payload_json,
)
from app.errors import ServiceError
from app.models.response import ApiErrorResponse, SyncResponse
from app.utils.logger import get_logger

router = APIRouter(prefix="/admin/sync", tags=["sync"])
logger = get_logger("sync_api")


def _build_sync_response(result) -> SyncResponse:
    return SyncResponse(
        started_at=result.started_at,
        finished_at=result.finished_at,
        mode=result.mode,
        total_read=result.total_read,
        total_upserted=result.total_upserted,
        batches=result.batches,
        last_updated_at=result.cursor.last_updated_at,
        last_case_id=result.cursor.last_case_id,
    )


@router.post(
    "/full",
    response_model=SyncResponse,
    responses={500: {"model": ApiErrorResponse}},
)
async def trigger_full_sync(
    request: Request,
    context: Optional[RequestContext] = Depends(get_request_context),
) -> SyncResponse:
    request_context = await resolve_request_context(request, context)
    result = await request_context.container.sync_service.full_sync(
        request_id=request_context.request_id,
    )
    return _build_sync_response(result)


@router.post(
    "/rebuild-row",
    response_model=SyncResponse,
    responses={
        400: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
)
async def trigger_rebuild_row(
    request: Request,
    context: Optional[RequestContext] = Depends(get_request_context),
) -> SyncResponse:
    request_context = await resolve_request_context(request, context)
    request_id = request_context.request_id
    raw_payload = await request.json()
    try:
        payload = build_rebuild_request_from_payload(raw_payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(), body=raw_payload) from exc
    try:
        result = await request_context.container.sync_service.rebuild_row(
            request_id=request_id,
            row=build_source_row_from_rebuild_request(payload),
        )
        return _build_sync_response(result)
    except Exception as exc:
        payload_json = sanitize_payload_json(payload.model_dump(mode="json"))
        extra = {
            "request_id": request_id,
            "payload_json": payload_json,
        }
        if isinstance(exc, ServiceError):
            extra["error_code"] = exc.error_code
            logger.error(
                "rebuild_row_failed request_id=%s payload_json=%s",
                request_id,
                payload_json,
                extra=extra,
            )
        else:
            logger.exception(
                "rebuild_row_failed request_id=%s payload_json=%s",
                request_id,
                payload_json,
                extra=extra,
            )
        raise


@router.post(
    "/incremental",
    deprecated=True,
    responses={
        501: {"model": ApiErrorResponse},
    },
)
async def trigger_incremental_sync():
    raise ServiceError(
        error_code="incremental_not_supported",
        message=(
            "/admin/sync/incremental is deprecated and not supported. "
            "Use /admin/sync/rebuild-row for single-row updates."
        ),
        status_code=501,
        retryable=False,
    )
