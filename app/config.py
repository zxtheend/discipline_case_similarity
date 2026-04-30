from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "sx-case-similarity"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    state_dir: Path = Field(default=Path("./data/app_state"))
    log_dir: Path = Field(default=Path("./data/logs"))
    prompt_dir: Path = Field(default=Path("./prompts"))

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_https: bool = False
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "xinfang_cases"
    qdrant_dense_vector_name: str = "dense_vector"
    qdrant_sparse_vector_name: str = "sparse_vector"
    qdrant_dense_vector_size: int = 1024

    llm_base_url: str = "http://localhost:9000/v1"
    embedding_base_url: str = "http://localhost:9001/v1"
    rerank_base_url: str = "http://localhost:9002/v1"
    llm_model: str = "qwen3-8b-awq"
    embedding_model: str = "bge-m3"
    rerank_model: str = "bge-reranker-v2-m3"
    model_api_key: Optional[str] = "EMPTY"
    http_timeout_seconds: float = 60.0
    embedding_retry_attempts: int = 3
    embedding_retry_backoff_seconds: float = 1.0
    embedding_enable_sparse: bool = True

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_db: str = "sjw"
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_source_table: str = "case_similarity_source"
    mysql_wtxx_table: str = "t_xf_wtxx"
    mysql_xfj_table: str = "t_xf_xfj"
    mysql_active_statuses: str = "ACTIVE"
    mysql_pool_min_size: int = 1
    mysql_pool_max_size: int = 5
    mysql_connect_timeout_seconds: int = 5
    decrypt_provider: str = "noop"

    filter_years: int = 5
    rrf_k: int = 60
    hybrid_limit: int = 50
    hybrid_location_boost: float = 0.03
    rerank_top_n: int = 20
    judge_top_n: int = 5
    identify_top_n: int = 5
    fallback_min_candidates: int = 10
    fallback_max_fetch: int = 10
    sync_batch_size: int = 32

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    @property
    def duplicate_prompt_path(self) -> Path:
        return self.prompt_dir / "duplicate_judge.txt"

    @property
    def clue_prompt_path(self) -> Path:
        return self.prompt_dir / "clue_mining.txt"

    @property
    def mysql_status_list(self) -> List[str]:
        values = [item.strip() for item in self.mysql_active_statuses.split(",")]
        return [item for item in values if item]

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
