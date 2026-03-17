import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.bootstrap import build_container, close_container
from app.config import get_settings
from app.errors import ServiceError
from app.models.response import ApiErrorResponse, DependencyStatus, HealthResponse, ReadyResponse
from app.utils.logger import configure_logging, get_logger


logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.container = await build_container(settings)
    try:
        yield
    finally:
        await close_container(app.state.container)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        logger.warning(
            "service_error",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "error_code": exc.error_code,
                "details": exc.details,
            },
        )
        payload = ApiErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code=exc.error_code,
            error_message=exc.message,
            retryable=exc.retryable,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception(
            "unhandled_error",
            extra={"request_id": getattr(request.state, "request_id", "unknown")},
        )
        payload = ApiErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code="internal_error",
            error_message="Unexpected internal server error.",
            retryable=False,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(request: Request) -> ReadyResponse:
        container = request.app.state.container
        dependencies = await _check_dependencies(
            [
                ("qdrant", container.qdrant_service.check_ready()),
                ("embedding", container.embedding_service.check_ready()),
                ("rerank", container.rerank_service.check_ready()),
                ("llm", container.llm_service.check_ready()),
            ]
        )
        return ReadyResponse(
            status="ok" if all(item.healthy for item in dependencies) else "degraded",
            dependencies=dependencies,
        )

    @app.get("/ready/sync", response_model=ReadyResponse, tags=["ops"])
    async def ready_sync(request: Request) -> ReadyResponse:
        container = request.app.state.container
        dependencies = await _check_dependencies(
            [
                ("mysql", container.mysql_service.check_ready()),
                ("qdrant", container.qdrant_service.check_ready()),
                ("embedding", container.embedding_service.check_ready()),
            ]
        )
        return ReadyResponse(
            status="ok" if all(item.healthy for item in dependencies) else "degraded",
            dependencies=dependencies,
        )

    return app


async def _check_dependencies(checks):
    results = []
    for name, coroutine in checks:
        try:
            await coroutine
            results.append(DependencyStatus(name=name, healthy=True))
        except Exception as exc:
            results.append(DependencyStatus(name=name, healthy=False, detail=str(exc)))
    return results


app = create_app()
