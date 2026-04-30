from typing import Optional

from app.config import Settings, get_settings
from app.container import ApplicationContainer, ReadinessProbe
from app.core.hybrid_search import HybridSearchEngine
from app.core.llm_judge import LLMJudgeEngine
from app.core.pipeline import IdentifyPipeline
from app.core.rerank import RerankEngine
from app.services.decrypt_service import build_decrypt_provider
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.mysql_service import MySQLService
from app.services.qdrant_service import QdrantService
from app.services.rerank_service import RerankService
from app.sync.data_sync import DataSyncService
from app.utils.audit import AuditLogger
from app.utils.logger import get_logger


async def build_container(settings: Optional[Settings] = None) -> ApplicationContainer:
    active_settings = settings or get_settings()
    logger = get_logger("bootstrap")
    audit_logger = AuditLogger()

    qdrant_service = QdrantService(active_settings)
    embedding_service = EmbeddingService(
        base_url=active_settings.embedding_base_url,
        model_name=active_settings.embedding_model,
        api_key=active_settings.model_api_key,
        timeout_seconds=active_settings.http_timeout_seconds,
        retry_attempts=active_settings.embedding_retry_attempts,
        retry_backoff_seconds=active_settings.embedding_retry_backoff_seconds,
        enable_sparse=active_settings.embedding_enable_sparse,
    )
    rerank_service = RerankService(
        base_url=active_settings.rerank_base_url,
        model_name=active_settings.rerank_model,
        api_key=active_settings.model_api_key,
        timeout_seconds=active_settings.http_timeout_seconds,
    )
    llm_service = LLMService(
        base_url=active_settings.llm_base_url,
        model_name=active_settings.llm_model,
        api_key=active_settings.model_api_key,
        timeout_seconds=active_settings.http_timeout_seconds,
    )
    mysql_service = MySQLService(active_settings)
    decrypt_provider = build_decrypt_provider(active_settings)

    hybrid_search_engine = HybridSearchEngine(
        settings=active_settings,
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
    )
    rerank_engine = RerankEngine(
        settings=active_settings,
        rerank_service=rerank_service,
    )
    llm_judge_engine = LLMJudgeEngine(
        settings=active_settings,
        llm_service=llm_service,
    )
    sync_service = DataSyncService(
        settings=active_settings,
        mysql_service=mysql_service,
        decrypt_provider=decrypt_provider,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        audit_logger=audit_logger,
    )
    pipeline = IdentifyPipeline(
        settings=active_settings,
        hybrid_search_engine=hybrid_search_engine,
        rerank_engine=rerank_engine,
        llm_judge_engine=llm_judge_engine,
        audit_logger=audit_logger,
    )

    logger.info(
        "container_built",
        extra={
            "app_env": active_settings.app_env,
            "qdrant_collection": active_settings.qdrant_collection,
        },
    )
    return ApplicationContainer(
        settings=active_settings,
        sync_service=sync_service,
        pipeline=pipeline,
        readiness_registry={
            "default": (
                ReadinessProbe(name="qdrant", check=qdrant_service.check_ready),
                ReadinessProbe(name="embedding", check=embedding_service.check_ready),
                ReadinessProbe(name="rerank", check=rerank_service.check_ready),
                ReadinessProbe(name="llm", check=llm_service.check_ready),
            ),
            "sync": (
                ReadinessProbe(name="mysql", check=mysql_service.check_ready),
                ReadinessProbe(name="qdrant", check=qdrant_service.check_ready),
                ReadinessProbe(name="embedding", check=embedding_service.check_ready),
            ),
        },
        shutdown_callbacks=(
            ("qdrant", qdrant_service.close),
            ("embedding", embedding_service.close),
            ("rerank", rerank_service.close),
            ("llm", llm_service.close),
            ("mysql", mysql_service.close),
        ),
    )


async def close_container(container: ApplicationContainer) -> None:
    logger = get_logger("bootstrap")
    for component, close_callback in container.shutdown_callbacks:
        try:
            await close_callback()
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            logger.warning(
                "container_shutdown_failed",
                extra={
                    "component": component,
                    "error_type": exc.__class__.__name__,
                    "detail": str(exc),
                },
            )
