import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from app.models.request import ClueMiningRequest, IdentifyRequest, RebuildRowRequest


class RequestModelHardeningTests(unittest.TestCase):
    def test_identify_request_accepts_camel_case_time_aliases_and_dumps_snake_case(self):
        payload = IdentifyRequest(
            reported_persons=[" 王建国 ", " ", "\t"],
            reporter=" 张三 ",
            location=" 太原市 ",
            description=" 反映收受礼金问题 ",
            startTime="2024-01-02T08:00:00+08:00",
            endTime="2024-01-03T08:00:00+08:00",
        )

        self.assertEqual(payload.reported_persons, ["王建国"])
        self.assertEqual(payload.reporter, "张三")
        self.assertEqual(payload.location, "太原市")
        self.assertEqual(payload.description, "反映收受礼金问题")
        self.assertEqual(payload.start_time, datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(payload.end_time, datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc))
        self.assertIn("start_time", payload.model_dump())
        self.assertIn("end_time", payload.model_dump())
        self.assertNotIn("startTime", payload.model_dump())
        self.assertNotIn("endTime", payload.model_dump())

    def test_identify_request_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            IdentifyRequest(
                reported_persons=["王建国"],
                reporter="张三",
                location="太原市",
                description="测试描述",
                extra_field="unexpected",
            )

    def test_identify_request_rejects_blank_required_strings(self):
        cases = (
            {"location": " \n ", "description": "测试描述"},
            {"location": "太原市", "description": " \t "},
        )

        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValidationError):
                IdentifyRequest(
                    reported_persons=["王建国"],
                    reporter="张三",
                    **case,
                )

    def test_identify_request_rejects_blank_only_reported_persons(self):
        with self.assertRaises(ValidationError):
            IdentifyRequest(
                reported_persons=[" ", "\t"],
                reporter="张三",
                location="太原市",
                description="测试描述",
            )

    def test_rebuild_row_request_accepts_create_time_alias_and_normalizes_to_utc(self):
        payload = RebuildRowRequest(
            case_id=" CASE-001 ",
            source_wtxx_bh=" XFJ-001 ",
            petition_id=" PET-001 ",
            location=" 太原市 ",
            encrypted_description=" cipher ",
            createTime="2024-01-01T08:00:00+08:00",
        )

        self.assertEqual(payload.case_id, "CASE-001")
        self.assertEqual(payload.source_wtxx_bh, "XFJ-001")
        self.assertEqual(payload.petition_id, "PET-001")
        self.assertEqual(payload.location, "太原市")
        self.assertEqual(payload.encrypted_description, "cipher")
        self.assertEqual(payload.create_time, datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
        self.assertIn("create_time", payload.model_dump())
        self.assertNotIn("createTime", payload.model_dump())

    def test_rebuild_row_request_accepts_millisecond_timestamp_create_time(self):
        payload = RebuildRowRequest(
            case_id="CASE-001",
            source_wtxx_bh="XFJ-001",
            petition_id="PET-001",
            location="太原市",
            encrypted_description="cipher",
            create_time=1777230053020,
        )

        self.assertEqual(
            payload.create_time,
            datetime.fromtimestamp(1777230053020 / 1000.0, tz=timezone.utc),
        )

    def test_rebuild_row_request_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            RebuildRowRequest(
                case_id="CASE-001",
                source_wtxx_bh="XFJ-001",
                petition_id="PET-001",
                create_time="2024-01-01T00:00:00+00:00",
                unknown_field="unexpected",
            )

    def test_rebuild_row_request_rejects_blank_location(self):
        with self.assertRaises(ValidationError):
            RebuildRowRequest(
                case_id="CASE-001",
                source_wtxx_bh="XFJ-001",
                petition_id="PET-001",
                location="   ",
                create_time="2024-01-01T00:00:00+00:00",
            )

    def test_clue_mining_request_rejects_blank_description_text_and_blank_reported_persons(self):
        with self.assertRaises(ValidationError):
            ClueMiningRequest(
                reported_persons=[" 王建国 ", " "],
                reporter="张三",
                location="太原市",
                description="测试描述",
                similar_case={
                    "case_id": "CASE-001",
                    "location": "太原市",
                    "reported_persons": [" ", "\t"],
                    "reporter": "李四",
                    "description_text": "历史案件正文",
                },
            )

        with self.assertRaises(ValidationError):
            ClueMiningRequest(
                reported_persons=["王建国"],
                reporter="张三",
                location="太原市",
                description="测试描述",
                similar_case={
                    "case_id": "CASE-001",
                    "location": "太原市",
                    "reported_persons": ["王建国"],
                    "reporter": "李四",
                    "description_text": "   ",
                },
            )


if __name__ == "__main__":
    unittest.main()
