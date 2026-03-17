import re
from typing import Any, List, Optional

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import JoinedSourceRow

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

    async def fetch_joined_source_rows(
        self,
        limit: int,
        last_case_id: Optional[str] = None,
    ) -> List[JoinedSourceRow]:
        pool = await self._ensure_pool()
        query, params = self._build_joined_fetch_query(
            limit=limit,
            last_case_id=last_case_id,
        )
        async with pool.acquire() as connection:
            async with connection.cursor(aiomysql.DictCursor) as db_cursor:
                await db_cursor.execute(query, params)
                rows = await db_cursor.fetchall()
        return [self._parse_joined_row(row) for row in rows]

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

    def _build_joined_fetch_query(
        self,
        limit: int,
        last_case_id: Optional[str],
    ):
        wtxx_table = self._validate_table_name(self._settings.mysql_wtxx_table, "mysql_wtxx_table")
        xfj_table = self._validate_table_name(self._settings.mysql_xfj_table, "mysql_xfj_table")

        conditions = ["1 = 1"]
        params: List[Any] = []
        if last_case_id:
            conditions.append("w.C_BH > %s")
            params.append(last_case_id)
        where_clause = " AND ".join(conditions)
        query = """
            SELECT
                w.C_BH AS case_id,
                w.C_XFJ_BH AS source_xfj_bh,
                w.LC_YJMS AS encrypted_description,
                w.DT_CJSJ AS create_time,
                w.DT_ZHXGSJ AS w_updated_at,
                x.C_BH AS petition_id,
                x.C_BFYR_XX AS encrypted_reported_persons,
                x.C_FYR_XX AS encrypted_reporter,
                x.C_WTSD_QC AS location,
                x.DT_CJSJ AS x_create_time,
                x.DT_ZHXGSJ AS x_updated_at
            FROM {wtxx_table} AS w
            LEFT JOIN {xfj_table} AS x
                ON w.C_XFJ_BH = x.C_BH
            WHERE {where_clause}
            ORDER BY w.C_BH ASC
            LIMIT %s
        """.format(
            wtxx_table=wtxx_table,
            xfj_table=xfj_table,
            where_clause=where_clause,
        )
        params.append(limit)
        return query, params

    def _parse_joined_row(self, row: Any) -> JoinedSourceRow:
        return JoinedSourceRow(
            case_id=str(row["case_id"]),
            source_xfj_bh=row.get("source_xfj_bh"),
            petition_id=row.get("petition_id"),
            encrypted_reported_persons=row.get("encrypted_reported_persons"),
            encrypted_reporter=row.get("encrypted_reporter"),
            encrypted_description=row.get("encrypted_description"),
            location=row.get("location"),
            create_time=row.get("create_time"),
            w_updated_at=row.get("w_updated_at"),
            x_create_time=row.get("x_create_time"),
            x_updated_at=row.get("x_updated_at"),
        )

    def _validate_table_name(self, table_name: str, field_name: str) -> str:
        if not self._table_pattern.fullmatch(table_name):
            raise ServiceError(
                error_code="mysql_invalid_table",
                message="{0} contains unsupported characters.".format(field_name),
                status_code=500,
            )
        return table_name
