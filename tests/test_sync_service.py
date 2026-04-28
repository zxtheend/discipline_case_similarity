import asyncio
import base64
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

from pydantic import ValidationError
from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import QueryEmbedding, RowDecryptionResult, SourceTableRow, SyncRunResult
from app.models.request import ClueMiningRequest, IdentifyRequest, RebuildRowRequest
from app.services.decrypt_service import NoopDecryptProvider
from app.services.mysql_service import MySQLService
from app.services.qdrant_service import QdrantService
from app.sync.data_sync import DataSyncService, split_reported_persons
from app.utils.audit import AuditLogger


def make_source_row(
    case_id: str,
    source_wtxx_bh: Optional[str] = "XFJ-001",
    petition_id: Optional[str] = "XFJ-001",
    location: Optional[str] = "太原市",
    encrypted_reported_persons: Optional[str] = "cipher-persons",
    encrypted_reporter: Optional[str] = "cipher-reporter",
    create_time: Optional[datetime] = None,
    encrypted_description: Optional[bytes] = b"cipher-description",
) -> SourceTableRow:
    created = create_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return SourceTableRow(
        case_id=case_id,
        source_wtxx_bh=source_wtxx_bh,
        petition_id=petition_id,
        encrypted_reported_persons=encrypted_reported_persons,
        encrypted_reporter=encrypted_reporter,
        encrypted_description=encrypted_description,
        location=location,
        create_time=created,
    )


class FakeMySQLService:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = []
        self.row_by_case_id = {}

    async def fetch_source_rows(self, limit, last_case_id=None):
        self.calls.append((limit, last_case_id))
        if not self._batches:
            return []
        return self._batches.pop(0)

    async def fetch_source_row_by_case_id(self, case_id):
        return self.row_by_case_id.get(case_id)


class FakeDecryptProvider:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    async def decrypt_rows(self, rows):
        self.calls.append([row.case_id for row in rows])
        return [self._responses[row.case_id] for row in rows]


class FakeEmbeddingService:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = failures or {}

    async def embed_texts(self, texts):
        self.calls.append(texts)
        for text in texts:
            error = self.failures.get(text)
            if error is not None:
                raise error
        return [
            QueryEmbedding(
                dense_vector=[0.1, 0.2],
                sparse_vector={"indices": [1, 2], "values": [0.5, 0.25]},
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


class FakeAuditLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type: str, request_id: str, **details):
        self.events.append(
            {
                "event_type": event_type,
                "request_id": request_id,
                "details": details,
            }
        )


class SyncServiceTests(unittest.IsolatedAsyncioTestCase):
    def make_service(
        self,
        mysql_service,
        decrypt_provider,
        embedding_failures=None,
        audit_logger=None,
    ):
        settings = Settings(
            prompt_dir="prompts",
            state_dir="data/app_state",
            sync_batch_size=8,
        )
        embedding_service = FakeEmbeddingService(failures=embedding_failures)
        qdrant_service = FakeQdrantService()
        audit = audit_logger or AuditLogger()
        service = DataSyncService(
            settings=settings,
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            audit_logger=audit,
        )
        return service, embedding_service, qdrant_service, audit

    async def test_full_sync_maps_joined_rows_and_upserts_valid_cases(self):
        mysql_service = FakeMySQLService(
            [
                [
                    make_source_row("CASE-001"),
                    make_source_row("CASE-002", location=" "),
                ],
                [],
            ]
        )
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-001": RowDecryptionResult(
                    case_id="CASE-001",
                    reported_persons_text='[{"mc":"王建国"},{"mc":"李四"},{"mc":"王建国"}]',
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
        service, embedding_service, qdrant_service, _ = self.make_service(
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
            {"petition_id": "XFJ-001", "source_wtxx_bh": "XFJ-001"},
        )
        self.assertEqual(
            embedding_service.calls,
            [[upserted_cases[0].document_text]],
        )
        _, embeddings = qdrant_service.upsert_calls[0]
        self.assertEqual(embeddings[0].sparse_vector.indices, [1, 2])
        self.assertEqual(embeddings[0].sparse_vector.values, [0.5, 0.25])

    async def test_full_sync_continues_when_single_row_embedding_fails(self):
        mysql_service = FakeMySQLService(
            [
                [
                    make_source_row("CASE-101"),
                    make_source_row("CASE-102"),
                ],
                [],
            ]
        )
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-101": RowDecryptionResult(
                    case_id="CASE-101",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="张某",
                    description_text="可正常写入",
                ),
                "CASE-102": RowDecryptionResult(
                    case_id="CASE-102",
                    reported_persons_text='[{"mc":"李四"}]',
                    reporter_text="李某",
                    description_text="超长文本导致 embedding 失败",
                ),
            }
        )
        normal_case = service_case = None
        service, _, qdrant_service, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )
        cases = service._map_source_rows_to_source_cases(
            rows=[make_source_row("CASE-101"), make_source_row("CASE-102")],
            decrypted_rows=[
                decrypt_provider._responses["CASE-101"],
                decrypt_provider._responses["CASE-102"],
            ],
            request_id="req-preview",
        )
        normal_case, service_case = cases
        embedding_failures = {
            service_case.document_text: ServiceError(
                error_code="embedding_failed",
                message="Embedding request failed",
                status_code=502,
                retryable=True,
            )
        }
        service, embedding_service, qdrant_service, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
            embedding_failures=embedding_failures,
        )

        result = await service.full_sync(request_id="req-continue")

        self.assertEqual(result.total_read, 2)
        self.assertEqual(result.total_upserted, 1)
        self.assertEqual(result.batches, 1)
        self.assertEqual(len(qdrant_service.upsert_calls), 1)
        upserted_cases, _ = qdrant_service.upsert_calls[0]
        self.assertEqual([case.case_id for case in upserted_cases], ["CASE-101"])
        self.assertEqual(decrypt_provider.calls, [["CASE-101", "CASE-102"]])
        self.assertEqual(
            embedding_service.calls,
            [
                [normal_case.document_text, service_case.document_text],
                [normal_case.document_text],
                [service_case.document_text],
            ],
        )

    async def test_map_source_rows_to_source_cases_allows_missing_location_but_skips_missing_description(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider({})
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        rows = [
            make_source_row("CASE-010", location=" "),
            make_source_row("CASE-011"),
        ]
        decrypted_rows = [
            RowDecryptionResult(
                case_id="CASE-010",
                reported_persons_text='[{"mc":"王建国"}]',
                reporter_text="张某",
                description_text="有效内容",
            ),
            RowDecryptionResult(
                case_id="CASE-011",
                reported_persons_text='[{"mc":"王建国"}]',
                reporter_text="张某",
                description_text=" ",
            ),
        ]

        cases = service._map_source_rows_to_source_cases(
            rows=rows,
            decrypted_rows=decrypted_rows,
            request_id="req-002",
        )

        self.assertEqual(len(cases), 1)
        self.assertIsNone(cases[0].location)
        self.assertEqual(cases[0].case_id, "CASE-010")

    async def test_full_sync_skips_row_when_source_wtxx_bh_missing(self):
        mysql_service = FakeMySQLService([[make_source_row("CASE-030", source_wtxx_bh=" ")], []])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-030": RowDecryptionResult(
                    case_id="CASE-030",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人甲",
                    description_text="有效内容",
                )
            }
        )
        service, _, qdrant_service, _ = self.make_service(mysql_service, decrypt_provider)

        result = await service.full_sync(request_id="req-missing-source-wtxx")

        self.assertEqual(result.total_upserted, 0)
        self.assertEqual(qdrant_service.upsert_calls, [])

    async def test_full_sync_skips_row_when_petition_id_missing(self):
        mysql_service = FakeMySQLService([[make_source_row("CASE-031", petition_id=" ")], []])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-031": RowDecryptionResult(
                    case_id="CASE-031",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人甲",
                    description_text="有效内容",
                )
            }
        )
        service, _, qdrant_service, _ = self.make_service(mysql_service, decrypt_provider)

        result = await service.full_sync(request_id="req-missing-petition")

        self.assertEqual(result.total_upserted, 0)
        self.assertEqual(qdrant_service.upsert_calls, [])

    async def test_full_sync_skips_row_when_reported_persons_empty_after_decrypt(self):
        mysql_service = FakeMySQLService([[make_source_row("CASE-032")], []])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-032": RowDecryptionResult(
                    case_id="CASE-032",
                    reported_persons_text='{"zj":"","mc":""}',
                    reporter_text="举报人甲",
                    description_text="有效内容",
                )
            }
        )
        service, _, qdrant_service, _ = self.make_service(mysql_service, decrypt_provider)

        result = await service.full_sync(request_id="req-empty-reported-persons")

        self.assertEqual(result.total_upserted, 0)
        self.assertEqual(qdrant_service.upsert_calls, [])

    async def test_rebuild_row_allows_null_location_and_reporter(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-033": RowDecryptionResult(
                    case_id="CASE-033",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text=None,
                    description_text="反映收受礼金",
                )
            }
        )
        service, _, qdrant_service, _ = self.make_service(mysql_service, decrypt_provider)

        result = await service.rebuild_row(
            request_id="req-null-location",
            row=make_source_row("CASE-033", location=None, encrypted_reporter=None),
        )

        self.assertEqual(result.total_upserted, 1)
        upserted_cases, _ = qdrant_service.upsert_calls[0]
        self.assertIsNone(upserted_cases[0].location)
        self.assertIsNone(upserted_cases[0].reporter)

    async def test_rebuild_row_upserts_single_case(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-020": RowDecryptionResult(
                    case_id="CASE-020",
                    reported_persons_text='[{"mc":"王建国"},{"mc":"李四"}]',
                    reporter_text="举报人甲",
                    description_text="反映收受礼金",
                )
            }
        )
        service, embedding_service, qdrant_service, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        result = await service.rebuild_row(
            request_id="req-rebuild",
            row=make_source_row("CASE-020"),
        )

        self.assertEqual(result.mode, "rebuild-row")
        self.assertEqual(result.total_read, 1)
        self.assertEqual(result.total_upserted, 1)
        self.assertEqual(result.cursor.last_case_id, "CASE-020")
        self.assertFalse(qdrant_service.recreated)
        self.assertEqual(len(qdrant_service.upsert_calls), 1)
        upserted_cases, _ = qdrant_service.upsert_calls[0]
        self.assertEqual(upserted_cases[0].case_id, "CASE-020")
        self.assertEqual(
            embedding_service.calls,
            [[upserted_cases[0].document_text]],
        )

    async def test_rebuild_row_accepts_base64_wrapped_streamsets_payload_from_log(self):
        service, _, qdrant_service, _ = self.make_service(
            mysql_service=FakeMySQLService([[]]),
            decrypt_provider=NoopDecryptProvider(),
        )

        result = await service.rebuild_row(
            request_id="req-streamsets",
            row=SourceTableRow(
                case_id="00fd0146941bfdf4066e",
                source_wtxx_bh="00fd0146941bfdf4066b",
                petition_id="00fd0146941bfdf4066b",
                location="太原市杏花岭区",
                encrypted_reported_persons=(
                    "eyJ6aiI6IjE3IiwibWMiOiJjMmY2IGI0ZTggY2FhZSAifQ=="
                ),
                encrypted_reporter=None,
                encrypted_description=(
                    "MzMgMzEgMzIgMzMgYzVlYiBkM2YzIGIzYzcgY2ZmMiBiNGY3IGQwZDcgY2ZjYiBjZGUzIGI3ZjkgYmNiNCBkM2NhIGIzYzcgY2ZmMiBiM2MwIGQ0ZWMgYjZlOCBjNWQ1IGIzYzAgYTRhZCBjZmYwIGQzYjYgYjNjMCBkMGUxIGJhYjUgYmVlMiBhNGFkIGNjYmUgZDdjNyBiN2Q1IGQ1YjIgYmFhNSBkNGM0IGQxY2QgYTRhZCBiN2Y5IGM4ZDMgYmFjZSBkNGI3IGIyYTQgYjJlYiBiOGM4IGI4YTkgYmZkMSBiZWZjIGIzYTMgYzZiOCBiNWYzIGQ1YjIgYmFhNSBhNGFkIGIzYTMgYzhkMyBiNWU3IGQ1ZGIgY2ViNiBjY2IxIGMzYWEgY2NiMSBhNGFkIGQ0ZDEgYzVkYiBjZWUzIGMyYmUgY2RhOSBiOGZmIGNmZjIgYzdmOCBhNGFkIGM1ZGIgYjNjMCBiOGZmIGNmZjIgYzdmOCBjZmFiIGQ2ZTcgY2JiNiBiNmM1IGE0YWQgYjdmOSBjZWUzIGI4ZmYgY2ZmMiBkOGE5IGM0YzYgYjlmOSBjY2IxIGNmZjIgYmZkNyBiM2VhIGNiYjIgZDRjNCBhNGFkIGJjYmEgZDRkMSBiN2UxIGM0ZmMgYzhlYSBjOWNjIGE0YTkgZDhkZCBiZmFlIGMxZWUgYjRjMyBiMmE3IGM4ZGEgYmJjZSBjNGMxIGM5ZGUgYjNjMCBiZmFlIGMxZWUgYTRhYSBhNGFkIGIzYTMgYzhkMyBkNWY5IGJhY2UgZDRiNyBjY2M1IGI1YTkgYjZjNSBkM2JjIGM0ZmMgY2VjYyBjZmVhIGJmZmQgYzljYyBjZmFiIGIyYTQgZjBkYiBjNWM0IGMwYjQgYjZiNyBjZWZmIGQxYjMgY2NiMSBjZmYyIGJmZDcgYjVhNyBiNGE1IGE0YWQgYzBjNyBiNGE1IA=="
                ),
                create_time=datetime(2014, 6, 13, 23, 23, 20, 948000, tzinfo=timezone.utc),
            ),
        )

        self.assertEqual(result.total_upserted, 1)
        upserted_case = qdrant_service.upsert_calls[0][0][0]
        self.assertEqual(upserted_case.reported_persons, ["刘崇森"])
        self.assertTrue(upserted_case.description_text.startswith("2012年因财务出现问题"))

    async def test_rebuild_row_raises_for_invalid_payload(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-021": RowDecryptionResult(
                    case_id="CASE-021",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人乙",
                    description_text=" ",
                )
            }
        )
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        with self.assertRaises(ServiceError) as context:
            await service.rebuild_row(
                request_id="req-rebuild-invalid",
                row=make_source_row("CASE-021"),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.error_code, "missing_description")

    async def test_rebuild_row_raises_for_missing_source_wtxx_bh(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-022": RowDecryptionResult(
                    case_id="CASE-022",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人乙",
                    description_text="有效内容",
                )
            }
        )
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        with self.assertRaises(ServiceError) as context:
            await service.rebuild_row(
                request_id="req-rebuild-missing-source-wtxx",
                row=make_source_row("CASE-022", source_wtxx_bh=" "),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.error_code, "missing_source_wtxx_bh")

    async def test_rebuild_row_raises_for_missing_petition_id(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-023": RowDecryptionResult(
                    case_id="CASE-023",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人乙",
                    description_text="有效内容",
                )
            }
        )
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        with self.assertRaises(ServiceError) as context:
            await service.rebuild_row(
                request_id="req-rebuild-missing-petition",
                row=make_source_row("CASE-023", petition_id=" "),
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.error_code, "missing_petition_id")

    async def test_incremental_sync_service_raises_not_supported(self):
        mysql_service = FakeMySQLService([[]])
        decrypt_provider = FakeDecryptProvider({})
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
        )

        with self.assertRaises(ServiceError) as context:
            await service.incremental_sync(request_id="req-003")

        self.assertEqual(context.exception.status_code, 501)

    async def test_full_sync_audit_event_includes_batch_counters(self):
        mysql_service = FakeMySQLService(
            [
                [make_source_row("CASE-201")],
                [],
            ]
        )
        decrypt_provider = FakeDecryptProvider(
            {
                "CASE-201": RowDecryptionResult(
                    case_id="CASE-201",
                    reported_persons_text='[{"mc":"王建国"}]',
                    reporter_text="举报人甲",
                    description_text="有效内容",
                )
            }
        )
        audit_logger = FakeAuditLogger()
        service, _, _, _ = self.make_service(
            mysql_service=mysql_service,
            decrypt_provider=decrypt_provider,
            audit_logger=audit_logger,
        )

        await service.full_sync(request_id="req-audit")

        batch_event = next(
            event for event in audit_logger.events if event["event_type"] == "sync_batch_completed"
        )
        self.assertEqual(batch_event["details"]["batch_number"], 1)
        self.assertEqual(batch_event["details"]["total_read"], 1)
        self.assertEqual(batch_event["details"]["total_upserted"], 1)


class MySQLServiceTests(unittest.TestCase):
    def test_build_source_fetch_query_uses_case_id_cursor(self):
        service = MySQLService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )

        query, params = service._build_source_fetch_query(limit=16, last_case_id="CASE-100")

        self.assertIn("FROM case_similarity_source", query)
        self.assertIn("ORDER BY case_id ASC", query)
        self.assertIn("case_id > %s", query)
        self.assertEqual(params, ["CASE-100", "CASE-100", 16])

    def test_split_reported_persons_returns_empty_when_json_parse_fails(self):
        persons = split_reported_persons("王建国, 李四，王建国, 赵六")

        self.assertEqual(persons, [])

    def test_split_reported_persons_extracts_mc_from_json_object(self):
        persons = split_reported_persons('{"zj":"13","mc":"王建国"}')

        self.assertEqual(persons, ["王建国"])

    def test_split_reported_persons_extracts_mc_from_json_array(self):
        persons = split_reported_persons(
            '[{"zj":"13","mc":"王建国"},{"zj":"14","mc":"李四"},{"zj":"15","mc":"王建国"}]'
        )

        self.assertEqual(persons, ["王建国", "李四"])

    def test_split_reported_persons_returns_empty_for_json_with_blank_mc(self):
        persons = split_reported_persons('{"zj":"","mc":""}')

        self.assertEqual(persons, [])


class DecryptProviderTests(unittest.TestCase):
    def test_noop_decrypt_provider_decodes_hex_cipher_text(self):
        provider = NoopDecryptProvider()

        value = provider._normalize_value("badd c1ee bced c3d3 cfcb cde3 a2a4")

        self.assertEqual(value, "管理混乱问题。")

    def test_noop_decrypt_provider_decodes_mc_inside_json(self):
        provider = NoopDecryptProvider()

        value = provider._normalize_value('{"zj":"13","mc":"c3bb b1b3 c0f4 d3b6 bdb0 cec6"}')

        self.assertEqual(value, '{"zj":"13","mc":"潞安矿业集团"}')

    def test_noop_decrypt_provider_decodes_base64_wrapped_json_from_streamsets(self):
        provider = NoopDecryptProvider()
        wrapped = base64.b64encode(
            '{"zj":"17","mc":"c2f6 b4e8 caae "}'.encode("utf-8")
        ).decode("ascii")

        value = provider._normalize_value(wrapped)

        self.assertEqual(value, '{"zj":"17","mc":"刘崇森"}')

    def test_noop_decrypt_provider_decodes_streamsets_payload_from_log(self):
        provider = NoopDecryptProvider()

        reported_persons = provider._normalize_value(
            "eyJ6aiI6IjE3IiwibWMiOiJjMmY2IGI0ZTggY2FhZSAifQ=="
        )
        description = provider._normalize_value(
            "MzMgMzEgMzIgMzMgYzVlYiBkM2YzIGIzYzcgY2ZmMiBiNGY3IGQwZDcgY2ZjYiBjZGUzIGI3ZjkgYmNiNCBkM2NhIGIzYzcgY2ZmMiBiM2MwIGQ0ZWMgYjZlOCBjNWQ1IGIzYzAgYTRhZCBjZmYwIGQzYjYgYjNjMCBkMGUxIGJhYjUgYmVlMiBhNGFkIGNjYmUgZDdjNyBiN2Q1IGQ1YjIgYmFhNSBkNGM0IGQxY2QgYTRhZCBiN2Y5IGM4ZDMgYmFjZSBkNGI3IGIyYTQgYjJlYiBiOGM4IGI4YTkgYmZkMSBiZWZjIGIzYTMgYzZiOCBiNWYzIGQ1YjIgYmFhNSBhNGFkIGIzYTMgYzhkMyBiNWU3IGQ1ZGIgY2ViNiBjY2IxIGMzYWEgY2NiMSBhNGFkIGQ0ZDEgYzVkYiBjZWUzIGMyYmUgY2RhOSBiOGZmIGNmZjIgYzdmOCBhNGFkIGM1ZGIgYjNjMCBiOGZmIGNmZjIgYzdmOCBjZmFiIGQ2ZTcgY2JiNiBiNmM1IGE0YWQgYjdmOSBjZWUzIGI4ZmYgY2ZmMiBkOGE5IGM0YzYgYjlmOSBjY2IxIGNmZjIgYmZkNyBiM2VhIGNiYjIgZDRjNCBhNGFkIGJjYmEgZDRkMSBiN2UxIGM0ZmMgYzhlYSBjOWNjIGE0YTkgZDhkZCBiZmFlIGMxZWUgYjRjMyBiMmE3IGM4ZGEgYmJjZSBjNGMxIGM5ZGUgYjNjMCBiZmFlIGMxZWUgYTRhYSBhNGFkIGIzYTMgYzhkMyBkNWY5IGJhY2UgZDRiNyBjY2M1IGI1YTkgYjZjNSBkM2JjIGM0ZmMgY2VjYyBjZmVhIGJmZmQgYzljYyBjZmFiIGIyYTQgZjBkYiBjNWM0IGMwYjQgYjZiNyBjZWZmIGQxYjMgY2NiMSBjZmYyIGJmZDcgYjVhNyBiNGE1IGE0YWQgYzBjNyBiNGE1IA=="
        )

        self.assertEqual(reported_persons, '{"zj":"17","mc":"刘崇森"}')
        self.assertEqual(split_reported_persons(reported_persons), ["刘崇森"])
        self.assertTrue(description.startswith("2012年因财务出现问题"))

    def test_noop_decrypt_provider_decodes_base64_wrapped_hex_text_from_streamsets(self):
        provider = NoopDecryptProvider()
        wrapped = base64.b64encode("badd c1ee bced c3d3 cfcb cde3 a2a4".encode("utf-8")).decode(
            "ascii"
        )

        value = provider._normalize_value(wrapped)

        self.assertEqual(value, "管理混乱问题。")


class RequestModelTests(unittest.TestCase):
    def test_identify_request_defaults_to_recent_five_year_window(self):
        payload = IdentifyRequest(
            reported_persons=["王建国"],
            reporter="张三",
            location="太原市",
            description="测试描述",
        )

        self.assertIsNotNone(payload.start_time)
        self.assertIsNotNone(payload.end_time)
        self.assertLess(payload.start_time, payload.end_time)
        self.assertGreaterEqual((payload.end_time - payload.start_time).days, 365 * 4)

    def test_identify_request_rejects_inverted_time_range(self):
        with self.assertRaises(ValidationError):
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张三",
                location="太原市",
                description="测试描述",
                start_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_rebuild_row_request_requires_source_wtxx_bh_and_petition_id(self):
        with self.assertRaises(ValidationError):
            RebuildRowRequest(
                case_id="CASE-040",
                location=None,
                encrypted_reported_persons="cipher",
                encrypted_description="cipher",
                create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_rebuild_row_request_allows_null_location(self):
        payload = RebuildRowRequest(
            case_id="CASE-041",
            source_wtxx_bh="XFJ-041",
            petition_id="XFJ-041",
            location=None,
            encrypted_reported_persons="cipher",
            encrypted_description="cipher",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        self.assertIsNone(payload.location)

    def test_clue_mining_request_rejects_removed_legacy_fields(self):
        with self.assertRaises(ValidationError):
            ClueMiningRequest(
                reported_persons=["王建国"],
                reporter="张三",
                location="太原市",
                description="测试描述",
                start_time=datetime(2021, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                similar_case={
                    "case_id": "CASE-001",
                    "similarity_score": 91,
                    "rank": 1,
                    "location": "太原市",
                    "reported_persons": ["王建国"],
                    "reporter": "李四",
                    "description_text": "历史案件正文",
                },
            )

    def test_clue_mining_request_accepts_minimal_new_shape(self):
        payload = ClueMiningRequest(
            reported_persons=["王建国"],
            reporter="张三",
            location="太原市",
            description="测试描述",
            similar_case={
                "case_id": "CASE-001",
                "location": "太原市",
                "reported_persons": ["王建国"],
                "reporter": "李四",
                "description_text": "历史案件正文",
            },
        )

        self.assertEqual(payload.similar_case.case_id, "CASE-001")
        self.assertEqual(payload.similar_case.reporter, "李四")


class QdrantServiceTests(unittest.TestCase):
    def test_point_to_candidate_accepts_null_location(self):
        service = QdrantService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )
        point = qdrant_models.ScoredPoint(
            id="point-1",
            version=1,
            score=0.8,
            payload={
                "case_id": "CASE-050",
                "location": None,
                "reported_persons": ["王建国"],
                "reporter": None,
                "description_text": "有效内容",
                "create_time": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            },
        )

        candidate = service._point_to_candidate(point, score_field="dense_score")

        self.assertIsNone(candidate.location)
        self.assertIn("属地: ", candidate.rerank_document)


class QdrantServiceAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_sparse_short_circuits_when_query_sparse_is_empty(self):
        service = QdrantService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )

        class ClientShouldNotBeCalled:
            async def query_points(self, **kwargs):
                raise AssertionError("query_points should not be called for empty sparse queries")

        service._client = ClientShouldNotBeCalled()

        result = await service.search_sparse(
            QueryEmbedding(dense_vector=[0.1, 0.2]),
            query_filter=qdrant_models.Filter(),
            limit=10,
        )

        self.assertEqual(result, [])

    async def test_fetch_filtered_candidates_returns_candidates_from_scroll(self):
        service = QdrantService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )
        point = qdrant_models.Record(
            id="point-1",
            payload={
                "case_id": "CASE-200",
                "location": "太原市",
                "reported_persons": ["王建国"],
                "reporter": "张某",
                "description_text": "有效内容",
                "create_time": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            },
            vector=None,
        )

        class ScrollClient:
            async def scroll(self, **kwargs):
                return ([point], None)

        service._client = ScrollClient()

        results = await service.fetch_filtered_candidates(
            query_filter=qdrant_models.Filter(),
            limit=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].case_id, "CASE-200")


class SyncEndpointTests(unittest.IsolatedAsyncioTestCase):
    def make_request(self, container, request_id="req-endpoint"):
        return SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(container=container)),
            state=SimpleNamespace(request_id=request_id),
        )

    async def test_incremental_endpoint_raises_not_supported(self):
        try:
            from app.api.v1.endpoints.sync import trigger_incremental_sync
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        with self.assertRaises(ServiceError) as context:
            await trigger_incremental_sync()

        self.assertEqual(context.exception.status_code, 501)
        self.assertEqual(context.exception.error_code, "incremental_not_supported")

    async def test_full_sync_endpoint_returns_sync_response(self):
        try:
            from app.api.v1.endpoints.sync import trigger_full_sync
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        class FakeSyncService:
            async def full_sync(self, request_id):
                return SyncRunResult(
                    started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    finished_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    mode="full",
                    total_read=2,
                    total_upserted=2,
                    batches=1,
                )

        container = SimpleNamespace(
            settings=SimpleNamespace(lock_timeout_seconds=0.05),
            runtime_lock=asyncio.Lock(),
            sync_lock=asyncio.Lock(),
            sync_service=FakeSyncService(),
        )

        response = await trigger_full_sync(self.make_request(container))

        self.assertEqual(response.mode, "full")
        self.assertEqual(response.total_upserted, 2)

    async def test_rebuild_row_endpoint_returns_sync_response(self):
        try:
            from app.api.v1.endpoints.sync import trigger_rebuild_row
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        class FakeSyncService:
            async def rebuild_row(self, request_id, row):
                self.last_row = row
                return SyncRunResult(
                    started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    finished_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    mode="rebuild-row",
                    total_read=1,
                    total_upserted=1,
                    batches=1,
                )

        sync_service = FakeSyncService()
        container = SimpleNamespace(
            settings=SimpleNamespace(lock_timeout_seconds=0.05),
            runtime_lock=asyncio.Lock(),
            sync_lock=asyncio.Lock(),
            sync_service=sync_service,
        )
        payload = RebuildRowRequest(
            case_id="CASE-030",
            source_wtxx_bh="XFJ-030",
            petition_id="XFJ-030",
            location="太原市",
            encrypted_reported_persons="王建国",
            encrypted_reporter="举报人丙",
            encrypted_description="案情内容",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        response = await trigger_rebuild_row(payload, self.make_request(container))

        self.assertEqual(response.mode, "rebuild-row")
        self.assertEqual(response.total_upserted, 1)
        self.assertEqual(sync_service.last_row.case_id, "CASE-030")

    async def test_rebuild_row_endpoint_logs_payload_when_service_fails(self):
        try:
            from app.api.v1.endpoints.sync import trigger_rebuild_row
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            self.skipTest(str(exc))

        class FakeSyncService:
            async def rebuild_row(self, request_id, row):
                raise ServiceError(
                    error_code="embedding_failed",
                    message="Embedding request failed",
                    status_code=502,
                )

        container = SimpleNamespace(
            settings=SimpleNamespace(lock_timeout_seconds=0.05),
            runtime_lock=asyncio.Lock(),
            sync_lock=asyncio.Lock(),
            sync_service=FakeSyncService(),
        )
        payload = RebuildRowRequest(
            case_id="CASE-031",
            source_wtxx_bh="XFJ-031",
            petition_id="XFJ-031",
            location="太原市",
            encrypted_reported_persons="王建国",
            encrypted_reporter="举报人丁",
            encrypted_description="案情内容",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with self.assertLogs("sync_api", level="ERROR") as captured:
            with self.assertRaises(ServiceError):
                await trigger_rebuild_row(payload, self.make_request(container, request_id="req-log"))

        combined_logs = "\n".join(captured.output)
        self.assertIn("rebuild_row_failed", combined_logs)
        self.assertIn('"case_id":"CASE-031"', combined_logs)
        self.assertIn("req-log", combined_logs)


if __name__ == "__main__":
    unittest.main()
