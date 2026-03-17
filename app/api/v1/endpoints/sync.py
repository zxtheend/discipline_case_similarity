from fastapi import APIRouter

from app.errors import ServiceError
from app.models.response import ApiErrorResponse

router = APIRouter(prefix="/admin/sync", tags=["sync"])


@router.post(
    "/incremental",
    responses={
        409: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        501: {"model": ApiErrorResponse},
    },
)
async def trigger_incremental_sync():
    raise ServiceError(
        error_code="incremental_not_supported",
        message="Current version only supports full sync.",
        status_code=501,
        retryable=False,
    )
