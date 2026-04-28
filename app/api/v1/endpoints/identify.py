from fastapi import APIRouter, Request

from app.api.dependencies import get_container
from app.models.request import ClueMiningRequest, IdentifyRequest
from app.models.response import ClueMiningResponse, IdentifyResponse

router = APIRouter()


@router.post(
    "/identify",
    response_model=IdentifyResponse,
)
async def identify_case(payload: IdentifyRequest, request: Request) -> IdentifyResponse:
    container = get_container(request)
    request_id = request.state.request_id
    return await container.pipeline.identify(payload, request_id=request_id)


@router.post(
    "/clues",
    response_model=ClueMiningResponse,
)
async def mine_clues(payload: ClueMiningRequest, request: Request) -> ClueMiningResponse:
    container = get_container(request)
    request_id = request.state.request_id
    return await container.pipeline.mine_clues(payload, request_id=request_id)
