import asyncio

from fastapi import APIRouter, Request

from app.api.dependencies import get_container
from app.errors import ServiceError
from app.models.domain import SourceTableRow
from app.models.request import RebuildRowRequest
from app.models.response import ApiErrorResponse, SyncResponse

router = APIRouter(prefix="/admin/sync", tags=["sync"])


async def _try_acquire(lock: asyncio.Lock, timeout_seconds: float) -> bool:
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds)
        return True
    except asyncio.TimeoutError:
        return False


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


async def _acquire_sync_guards(request: Request):
    container = get_container(request)
    timeout_seconds = container.settings.lock_timeout_seconds
    runtime_acquired = await _try_acquire(container.runtime_lock, timeout_seconds)
    if not runtime_acquired:
        raise ServiceError(
            error_code="sync_busy",
            message="The service is busy with another identify or sync task.",
            status_code=429,
            retryable=True,
        )

    sync_acquired = await _try_acquire(container.sync_lock, timeout_seconds)
    if not sync_acquired:
        container.runtime_lock.release()
        raise ServiceError(
            error_code="sync_busy",
            message="Another sync task is already running.",
            status_code=429,
            retryable=True,
        )
    return container


def _release_sync_guards(container) -> None:
    if container.sync_lock.locked():
        container.sync_lock.release()
    if container.runtime_lock.locked():
        container.runtime_lock.release()


@router.post(
    "/full",
    response_model=SyncResponse,
    responses={429: {"model": ApiErrorResponse}, 500: {"model": ApiErrorResponse}},
)
async def trigger_full_sync(request: Request) -> SyncResponse:
    container = await _acquire_sync_guards(request)
    request_id = request.state.request_id
    try:
        result = await container.sync_service.full_sync(request_id=request_id)
        return _build_sync_response(result)
    finally:
        _release_sync_guards(container)


@router.post(
    "/rebuild-row",
    response_model=SyncResponse,
    responses={
        400: {"model": ApiErrorResponse},
        429: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
)
async def trigger_rebuild_row(payload: RebuildRowRequest, request: Request) -> SyncResponse:
    container = await _acquire_sync_guards(request)
    request_id = request.state.request_id
    try:
        result = await container.sync_service.rebuild_row(
            request_id=request_id,
            row=SourceTableRow.model_validate(payload.model_dump()),
        )
        return _build_sync_response(result)
    finally:
        _release_sync_guards(container)


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
