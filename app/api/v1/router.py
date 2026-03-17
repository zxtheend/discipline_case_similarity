from fastapi import APIRouter

from app.api.v1.endpoints.identify import router as identify_router
from app.api.v1.endpoints.sync import router as sync_router

api_router = APIRouter()
api_router.include_router(identify_router, tags=["identify"])
api_router.include_router(sync_router)
