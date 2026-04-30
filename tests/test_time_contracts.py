import unittest
from datetime import datetime, timezone

from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.models.domain import SourceCase
from app.services.qdrant_service import QdrantService
from app.sync.mapping import SourceCaseMapper
from app.sync.telemetry import SyncTelemetry
from app.utils.time_utils import normalize_to_utc, parse_utc_datetime, serialize_utc_datetime


class TimeUtilsTests(unittest.TestCase):
    def test_normalize_to_utc_interprets_naive_as_utc(self):
        value = normalize_to_utc(datetime(2024, 1, 1, 8, 0))

        self.assertEqual(value, datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc))

    def test_normalize_to_utc_converts_offset_datetime(self):
        value = normalize_to_utc(datetime.fromisoformat("2024-01-01T08:00:00+08:00"))

        self.assertEqual(value, datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))

    def test_serialize_and_parse_round_trip_to_utc_aware_datetime(self):
        serialized = serialize_utc_datetime(datetime(2024, 1, 1, 8, 0))
        parsed = parse_utc_datetime(serialized)

        self.assertEqual(serialized, "2024-01-01T08:00:00+00:00")
        self.assertEqual(parsed, datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc))


class QdrantTimeContractTests(unittest.TestCase):
    def setUp(self):
        self.service = QdrantService(
            Settings(
                prompt_dir="prompts",
                state_dir="data/app_state",
            )
        )

    def test_build_payload_serializes_naive_datetimes_as_utc_strings(self):
        payload = self.service._build_payload(
            SourceCase(
                case_id="CASE-001",
                reported_persons=["王建国"],
                description_text="有效内容",
                create_time=datetime(2024, 1, 1, 8, 0),
                updated_at=datetime(2024, 1, 2, 9, 30),
            )
        )

        self.assertEqual(payload["create_time"], "2024-01-01T08:00:00+00:00")
        self.assertEqual(payload["updated_at"], "2024-01-02T09:30:00+00:00")

    def test_point_to_candidate_normalizes_aware_payload_datetime(self):
        point = qdrant_models.ScoredPoint(
            id="point-1",
            version=1,
            score=0.8,
            payload={
                "case_id": "CASE-001",
                "location": "太原市",
                "reported_persons": ["王建国"],
                "reporter": "张某",
                "description_text": "有效内容",
                "create_time": "2024-01-01T08:00:00+00:00",
                "updated_at": "2024-01-01T09:00:00+00:00",
            },
        )

        candidate = self.service._point_to_candidate(point, score_field="dense_score")

        self.assertEqual(candidate.create_time, datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc))
        self.assertEqual(candidate.updated_at, datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc))

    def test_point_to_candidate_normalizes_naive_payload_datetime(self):
        point = qdrant_models.ScoredPoint(
            id="point-2",
            version=1,
            score=0.8,
            payload={
                "case_id": "CASE-002",
                "location": "太原市",
                "reported_persons": ["王建国"],
                "reporter": "张某",
                "description_text": "有效内容",
                "create_time": "2024-01-01T08:00:00",
                "updated_at": "2024-01-01T09:00:00",
            },
        )

        candidate = self.service._point_to_candidate(point, score_field="dense_score")

        self.assertEqual(candidate.create_time, datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc))
        self.assertEqual(candidate.updated_at, datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc))


class MappingTimeContractTests(unittest.TestCase):
    def test_source_case_mapper_normalizes_source_row_times_to_utc(self):
        mapper = SourceCaseMapper(telemetry=SyncTelemetry(audit_logger=object()))

        source_case = mapper.build_source_case_from_source_row(
            row=type(
                "Row",
                (),
                {
                    "case_id": "CASE-003",
                    "source_wtxx_bh": "XFJ-003",
                    "petition_id": "PET-003",
                    "location": "太原市",
                    "create_time": datetime(2024, 1, 1, 8, 0),
                    "updated_at": datetime.fromisoformat("2024-01-02T08:00:00+08:00"),
                },
            )(),
            decrypted_row=type(
                "DecryptRow",
                (),
                {
                    "error_message": None,
                    "reported_persons_text": '[{"mc":"王建国"}]',
                    "reporter_text": "张某",
                    "description_text": "有效内容",
                },
            )(),
            request_id="req-time",
            fail_fast=True,
        )

        self.assertEqual(source_case.create_time, datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc))
        self.assertEqual(source_case.updated_at, datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc))
