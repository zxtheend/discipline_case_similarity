import re
from typing import Any, List, Optional

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import SourceTableRow

try:  # pragma: no cover - dependency is optional during local scaffolding
    import aiomysql
except ImportError:  # pragma: no cover - dependency is optional during local scaffolding
    aiomysql = None


class MySQLService:
    _table_pattern = re.compile(r"^[A-Za-z0-9_]+$")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = None

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def check_ready(self) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()

    async def fetch_source_rows(
        self,
        limit: int,
        last_case_id: Optional[str] = None,
    ) -> List[SourceTableRow]:
        pool = await self._ensure_pool()
        query, params = self._build_source_fetch_query(
            limit=limit,
            last_case_id=last_case_id,
        )
        async with pool.acquire() as connection:
            async with connection.cursor(aiomysql.DictCursor) as db_cursor:
                await db_cursor.execute(query, params)
                rows = await db_cursor.fetchall()
        return [self._parse_source_row(row) for row in rows]

    async def fetch_source_row_by_case_id(self, case_id: str) -> Optional[SourceTableRow]:
        pool = await self._ensure_pool()
        query, params = self._build_source_fetch_by_case_id_query(case_id)
        async with pool.acquire() as connection:
            async with connection.cursor(aiomysql.DictCursor) as db_cursor:
                await db_cursor.execute(query, params)
                row = await db_cursor.fetchone()
        if row is None:
            return None
        return self._parse_source_row(row)

    async def _ensure_pool(self):
        if aiomysql is None:
            raise ServiceError(
                error_code="mysql_driver_missing",
                message="aiomysql is not installed. Install dependencies before using sync features.",
                status_code=500,
            )
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                host=self._settings.mysql_host,
                port=self._settings.mysql_port,
                db=self._settings.mysql_db,
                user=self._settings.mysql_user,
                password=self._settings.mysql_password,
                minsize=self._settings.mysql_pool_min_size,
                maxsize=self._settings.mysql_pool_max_size,
                autocommit=True,
                connect_timeout=self._settings.mysql_connect_timeout_seconds,
                charset="utf8mb4",
            )
        return self._pool

    def _build_source_fetch_query(
        self,
        limit: int,
        last_case_id: Optional[str],
    ):
        source_table = self._validate_table_name(self._settings.mysql_source_table, "mysql_source_table")
        query = """
            SELECT
                case_id,
                source_wtxx_bh,
                petition_id,
                location,
                encrypted_reported_persons,
                encrypted_reporter,
                encrypted_description,
                create_time
            FROM {source_table}
            WHERE (%s IS NULL OR case_id > %s)
            ORDER BY case_id ASC
            LIMIT %s
        """.format(source_table=source_table)
        return query, [last_case_id, last_case_id, limit]

    def _build_source_fetch_by_case_id_query(self, case_id: str):
        source_table = self._validate_table_name(self._settings.mysql_source_table, "mysql_source_table")
        query = """
            SELECT
                case_id,
                source_wtxx_bh,
                petition_id,
                location,
                encrypted_reported_persons,
                encrypted_reporter,
                encrypted_description,
                create_time
            FROM {source_table}
            WHERE case_id = %s
            LIMIT 1
        """.format(source_table=source_table)
        return query, [case_id]

    def _parse_source_row(self, row: Any) -> SourceTableRow:
        return SourceTableRow(
            case_id=str(row["case_id"]),
            source_wtxx_bh=row.get("source_wtxx_bh"),
            petition_id=row.get("petition_id"),
            encrypted_reported_persons=row.get("encrypted_reported_persons"),
            encrypted_reporter=row.get("encrypted_reporter"),
            encrypted_description=row.get("encrypted_description"),
            location=row.get("location"),
            create_time=row.get("create_time"),
        )

    def _validate_table_name(self, table_name: str, field_name: str) -> str:
        if not self._table_pattern.fullmatch(table_name):
            raise ServiceError(
                error_code="mysql_invalid_table",
                message="{0} contains unsupported characters.".format(field_name),
                status_code=500,
            )
        return table_name
