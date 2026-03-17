import unittest
from datetime import datetime, timezone
from typing import Optional

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import JoinedSourceRow, QueryEmbedding, RowDecryptionResult
from app.services.mysql_service import MySQLService
from app.sync.data_sync import DataSyncService, split_reported_persons
from app.utils.audit import AuditLogger


def make_joined_row(
    case_id: str,
    petition_id: Optional[str] = "XFJ-001",
    location: Optional[str] = "太原市",
    create_time: Optional[datetime] = None,
    w_updated_at: Optional[datetime] = None,
    x_updated_at: Optional[datetime] = None,
) -> JoinedSourceRow:
    created = create_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return JoinedSourceRow(
        case_id=case_id,
        source_xfj_bh=petition_id,
        petition_id=petition_id,
        encrypted_reported_persons="cipher-persons",
        encrypted_reporter="cipher-reporter",
        encrypted_description=b"cipher-description",
        location=location,
        create_time=created,
        w_updated_at=w_updated_at or datetime(2024, 1, 2, tzinfo=timezone.utc),
        x_create_time=created,
        x_updated_at=x_updated_at or datetime(2024, 1, 3, tzinfo=timezone.utc),
    )


class FakeMySQLService:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = []

    async def fetch_joined_source_rows(self, limit, last_case_id=None):
        self.calls.append((limit, last_case_id))
        if not self._batches:
            return []
        return self._batches.pop(0)


class FakeDecryptProvider:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def decrypt_rows(self, rows):
        self.calls.append([row.case_id for row in rows])
        return [self._responses[row.case_id] for row in rows]


class FakeEmbeddingService:
    def __init__(self):
        self.calls = []

    async def embed_texts(self, texts):
        self.calls.append(texts)
        return [
            QueryEmbedding(
                dense_vector=[0.1, 0.2],
            )
            for _ in texts
        ]


class FakeQdrantService:
    def __init__(self):
        self.recreated = False
        self.upsert_calls = []

    async def recreate_collection(self):
        self.recreated = True

    async def upsert_cases(self, cases, embeddings):
        self.upsert_calls.append((cases, embeddings))
        return len(cases)


class SyncServiceTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self, mysql_service, decrypt_provider):
        settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
            sync_batch_size=8,
        )
        embedding_service = FakeEmbeddingService()
        qdrant_service = FakeQdrantService()
        service = DataSyncService(
            settings=settings,
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            audit_logger=AuditLogger(),
        )
        return service, embedding_service, qdrant_service

    async def test_full_sync_maps_joined_rows_and_upserts_valid_cases(self):
        mysql_service = FakeMySQLService(
            [
                [
                    make_joined_row("CASE-001"),
                    make_joined_row("CASE-002", petition_id=None),
                ],
                [],
            ]
        )
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-001": RowDecryptionResult(
                    case_id="CASE-001",
                    reported_persons_text="王建国, 李四，王建国",
                    reporter_text=" 张某 ",
                    description_text=" 反映王建国收受礼金 ",
                ),
                "CASE-002": RowDecryptionResult(
                    case_id="CASE-002",
                    reported_persons_text="赵某",
                    reporter_text="李某",
                    description_text="内容",
                ),
            }
        )
        service, embedding_service, qdrant_service = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        result = await service.full_sync(request_id="req-001")

        self.assertTrue(qdrant_service.recreated)
        self.assertEqual(result.mode, "full")
        self.assertEqual(result.total_read, 2)
        self.assertEqual(result.total_upserted, 1)
        self.assertEqual(result.batches, 1)
        self.assertEqual(result.cursor.last_case_id, "CASE-002")
        self.assertEqual(mysql_service.calls, [(8, None), (8, "CASE-002")])
        self.assertEqual(decrypt_provider.calls, [["CASE-001", "CASE-002"]])
        self.assertEqual(len(qdrant_service.upsert_calls), 1)
        upserted_cases, _ = qdrant_service.upsert_calls[0]
        self.assertEqual(len(upserted_cases), 1)
        self.assertEqual(upserted_cases[0].case_id, "CASE-001")
        self.assertEqual(upserted_cases[0].reported_persons, ["王建国", "李四"])
        self.assertEqual(upserted_cases[0].reporter, "张某")
        self.assertEqual(upserted_cases[0].location, "太原市")
        self.assertEqual(upserted_cases[0].description_text, "反映王建国收受礼金")
        self.assertEqual(
            upserted_cases[0].extra,
            {"petition_id": "XFJ-001", "source_xfj_bh": "XFJ-001"},
        )
        self.assertEqual(
            embedding_service.calls,
            [[upserted_cases[0].document_text]],
        )

    async def test_map_rows_to_source_cases_skips_missing_location_and_description(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider({})
        service, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        rows = [
            make_joined_row("CASE-010", location=" "),
            make_joined_row("CASE-011"),
        ]
        decrypted_rows = [
            RowDecryptionResult(
                case_id="CASE-010",
                reported_persons_text="王建国",
                reporter_text="张某",
                description_text="有效内容",
            ),
            RowDecryptionResult(
                case_id="CASE-011",
                reported_persons_text="王建国",
                reporter_text="张某",
                description_text=" ",
            ),
        ]

        cases = service._map_rows_to_source_cases(
            rows=rows,
            decrypted_rows=decrypted_rows,
            request_id="req-002",
        )

        self.assertEqual(cases, [])

    async def test_incremental_sync_service_raises_not_supported(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider({})
        service, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        with self.assertRaises(ServiceError) as context:
            await service.incremental_sync(request_id="req-003")

        self.assertEqual(context.exception.status_code, 501)


class MySQLServiceTests(unittest.TestCase):
    def test_build_joined_fetch_query_uses_join_and_cursor(self):
        service = MySQLService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )

        query, params = service._build_joined_fetch_query(limit=16, last_case_id="CASE-100")

        self.assertIn("FROM t_xf_wtxx AS w", query)
        self.assertIn("LEFT JOIN t_xf_xfj AS x", query)
        self.assertIn("ON w.C_XFJ_BH = x.C_BH", query)
        self.assertIn("ORDER BY w.C_BH ASC", query)
        self.assertEqual(params, ["CASE-100", 16])

    def test_split_reported_persons_supports_chinese_and_english_commas(self):
        persons = split_reported_persons("王建国, 李四，王建国, 赵六")

        self.assertEqual(persons, ["王建国", "李四", "赵六"])


class SyncEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_endpoint_raises_not_supported(self):
        try:
            from app.api.v1.endpoints.sync import trigger_incremental_sync
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        with self.assertRaises(ServiceError) as context:
            await trigger_incremental_sync()

        self.assertEqual(context.exception.status_code, 501)
        self.assertEqual(context.exception.error_code, "incremental_not_supported")


if __name__ == "__main__":
    unittest.main()
