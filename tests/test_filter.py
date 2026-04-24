import unittest
from datetime import datetime, timezone

from app.core.filter import build_case_filter


class FilterTests(unittest.TestCase):
    def test_build_case_filter_contains_reported_persons_and_time(self):
        qdrant_filter = build_case_filter(
            [" 王建国 ", "李四", ""],
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 12, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(len(qdrant_filter.must), 2)
        self.assertEqual(qdrant_filter.must[0].key, "reported_persons")
        self.assertEqual(qdrant_filter.must[0].match.any, ["王建国", "李四"])
        self.assertEqual(qdrant_filter.must[1].key, "create_time")
        self.assertEqual(
            qdrant_filter.must[1].range.gte,
            datetime(2021, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(
            qdrant_filter.must[1].range.lte,
            datetime(2026, 12, 31, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
