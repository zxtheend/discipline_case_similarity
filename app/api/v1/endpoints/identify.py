from typing import Optional

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import RequestContext, get_request_context, resolve_request_context
from app.models.request import ClueMiningRequest, IdentifyRequest
from app.models.response import ClueMiningResponse, IdentifyResponse

router = APIRouter()


@router.post(
    "/identify",
    response_model=IdentifyResponse,
)
async def identify_case(
    payload: IdentifyRequest,
    request: Request,
    context: Optional[RequestContext] = Depends(get_request_context),
) -> IdentifyResponse:
    request_context = await resolve_request_context(request, context)
    return await request_context.container.pipeline.identify(
        payload,
        request_id=request_context.request_id,
    )


@router.post(
    "/clues",
    response_model=ClueMiningResponse,
)
async def mine_clues(
    payload: ClueMiningRequest,
    request: Request,
    context: Optional[RequestContext] = Depends(get_request_context),
) -> ClueMiningResponse:
    request_context = await resolve_request_context(request, context)
    return await request_context.container.pipeline.mine_clues(
        payload,
        request_id=request_context.request_id,
    )
