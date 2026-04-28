from fastapi import APIRouter, Request

from app.api.dependencies import get_container
from app.errors import ServiceError
from app.models.domain import SourceTableRow
from app.models.request import RebuildRowRequest
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
async def trigger_full_sync(request: Request) -> SyncResponse:
    container = get_container(request)
    request_id = request.state.request_id
    result = await container.sync_service.full_sync(request_id=request_id)
    return _build_sync_response(result)


@router.post(
    "/rebuild-row",
    response_model=SyncResponse,
    responses={
        400: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
)
async def trigger_rebuild_row(payload: RebuildRowRequest, request: Request) -> SyncResponse:
    container = get_container(request)
    request_id = request.state.request_id
    try:
        result = await container.sync_service.rebuild_row(
            request_id=request_id,
            row=SourceTableRow.model_validate(payload.model_dump()),
        )
        return _build_sync_response(result)
    except Exception as exc:
        payload_json = payload.model_dump_json()
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
    responses={
        429: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        501: {"model": ApiErrorResponse},
    },
)
async def trigger_incremental_sync():
    raise ServiceError(
        error_code="incremental_not_supported",
        message="Current version uses /admin/sync/rebuild-row for incremental updates.",
        status_code=501,
        retryable=False,
    )
