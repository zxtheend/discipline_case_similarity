import asyncio
from dataclasses import dataclass

from app.config import Settings
from app.core.hybrid_search import HybridSearchEngine
from app.core.llm_judge import LLMJudgeEngine
from app.core.pipeline import IdentifyPipeline
from app.core.rerank import RerankEngine
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.mysql_service import MySQLService
from app.services.qdrant_service import QdrantService
from app.services.rerank_service import RerankService
from app.sync.data_sync import DataSyncService
from app.utils.audit import AuditLogger


@dataclass
class ApplicationContainer:
    settings: Settings
    audit_logger: AuditLogger
    qdrant_service: QdrantService
    embedding_service: EmbeddingService
    rerank_service: RerankService
    llm_service: LLMService
    mysql_service: MySQLService
    hybrid_search_engine: HybridSearchEngine
    rerank_engine: RerankEngine
    llm_judge_engine: LLMJudgeEngine
    sync_service: DataSyncService
    pipeline: IdentifyPipeline
    runtime_lock: asyncio.Lock
    sync_lock: asyncio.Lock
