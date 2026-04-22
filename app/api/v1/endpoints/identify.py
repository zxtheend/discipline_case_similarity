import asyncio

from fastapi import APIRouter, Request

from app.api.dependencies import get_container
from app.errors import ServiceError
from app.models.request import ClueMiningRequest, IdentifyRequest
from app.models.response import ApiErrorResponse, ClueMiningResponse, IdentifyResponse

router = APIRouter()


async def _try_acquire(lock: asyncio.Lock, timeout_seconds: float) -> bool:
    try:
        await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds)
        return True
    except asyncio.TimeoutError:
        return False


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    responses={429: {"model": ApiErrorResponse}, 500: {"model": ApiErrorResponse}},
)
async def identify_case(payload: IdentifyRequest, request: Request) -> IdentifyResponse:
    container = get_container(request)
    request_id = request.state.request_id
    acquired = await _try_acquire(
        container.runtime_lock,
        timeout_seconds=container.settings.lock_timeout_seconds,
    )
    if not acquired:
        raise ServiceError(
            error_code="identify_busy",
            message="The service is busy with another identify or sync task.",
            status_code=429,
            retryable=True,
        )
    try:
        return await container.pipeline.identify(payload, request_id=request_id)
    finally:
        container.runtime_lock.release()


@router.post(
    "/clues",
    response_model=ClueMiningResponse,
    responses={429: {"model": ApiErrorResponse}, 500: {"model": ApiErrorResponse}},
)
async def mine_clues(payload: ClueMiningRequest, request: Request) -> ClueMiningResponse:
    container = get_container(request)
    request_id = request.state.request_id
    acquired = await _try_acquire(
        container.runtime_lock,
        timeout_seconds=container.settings.lock_timeout_seconds,
    )
    if not acquired:
        raise ServiceError(
            error_code="identify_busy",
            message="The service is busy with another identify or sync task.",
            status_code=429,
            retryable=True,
        )
    try:
        return await container.pipeline.mine_clues(payload, request_id=request_id)
    finally:
        container.runtime_lock.release()
