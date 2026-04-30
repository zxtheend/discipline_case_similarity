import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.dependencies import get_container, sanitize_for_logging, sanitize_request_body
from app.api.v1.router import api_router
from app.bootstrap import build_container, close_container
from app.container import ApplicationContainer, ReadinessProbe
from app.config import get_settings
from app.errors import ServiceError
from app.models.response import ApiErrorResponse, DependencyStatus, HealthResponse, ReadyResponse
from app.utils.logger import (
    BUSINESS_LOG_CHANNEL,
    SYNC_FULL_LOG_CHANNEL,
    SYNC_REBUILD_LOG_CHANNEL,
    configure_logging,
    get_logger,
    reset_log_channel,
    set_log_channel,
)


logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
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
        token = None
        log_channel = _resolve_request_log_channel(str(request.url.path), settings.api_v1_prefix)
        if log_channel is not None:
            token = set_log_channel(log_channel)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            if token is not None:
                reset_log_channel(token)

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        logger.warning(
            "service_error",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "error_code": exc.error_code,
                "details": sanitize_for_logging(exc.details),
            },
        )
        payload = ApiErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error_code=exc.error_code,
            error_message=exc.message,
            retryable=exc.retryable,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        raw_body = await request.body()
        request_id = getattr(request.state, "request_id", "unknown")
        body_json = sanitize_request_body(raw_body)
        logger.warning(
            "request_validation_error request_id=%s body_json=%s",
            request_id,
            body_json,
            extra={
                "request_id": request_id,
                "details": sanitize_for_logging(exc.errors()),
                "body_json": body_json,
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "request_id": request_id,
            },
        )

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
        container = get_container(request)
        dependencies = await _check_readiness_registry(container, "default")
        return ReadyResponse(
            status="ok" if all(item.healthy for item in dependencies) else "degraded",
            dependencies=dependencies,
        )

    @app.get("/ready/sync", response_model=ReadyResponse, tags=["ops"])
    async def ready_sync(request: Request) -> ReadyResponse:
        container = get_container(request)
        dependencies = await _check_readiness_registry(container, "sync")
        return ReadyResponse(
            status="ok" if all(item.healthy for item in dependencies) else "degraded",
            dependencies=dependencies,
        )

    return app


async def _check_readiness_registry(
    container: ApplicationContainer,
    registry_name: str,
) -> list[DependencyStatus]:
    probes = container.get_readiness_probes(registry_name)
    return list(await asyncio.gather(*(_check_dependency(probe) for probe in probes)))


async def _check_dependency(probe: ReadinessProbe) -> DependencyStatus:
    try:
        await probe.check()
        return DependencyStatus(name=probe.name, healthy=True)
    except Exception as exc:
        return DependencyStatus(name=probe.name, healthy=False, detail=str(exc))


app = create_app()


def _resolve_request_log_channel(request_path: str, api_v1_prefix: str) -> Optional[str]:
    business_paths = {
        "/health",
        "/ready",
        "/ready/sync",
        "{0}/identify".format(api_v1_prefix),
        "{0}/clues".format(api_v1_prefix),
    }
    if request_path in business_paths:
        return BUSINESS_LOG_CHANNEL
    if request_path == "{0}/admin/sync/rebuild-row".format(api_v1_prefix):
        return SYNC_REBUILD_LOG_CHANNEL
    if request_path == "{0}/admin/sync/full".format(api_v1_prefix):
        return SYNC_FULL_LOG_CHANNEL
    return None
