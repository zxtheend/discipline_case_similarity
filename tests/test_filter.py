import unittest
from datetime import datetime, timezone

from app.core.filter import build_case_filter, cutoff_datetime


class FilterTests(unittest.TestCase):
    def test_cutoff_datetime_subtracts_year_window(self):
        now = datetime(2026, 3, 12, tzinfo=timezone.utc)
        cutoff = cutoff_datetime(5, now=now)
        self.assertEqual(cutoff.year, 2021)
        self.assertEqual(cutoff.month, 3)

    def test_build_case_filter_contains_reported_persons_and_time(self):
        qdrant_filter = build_case_filter([" 王建国 ", "李四", ""], 5)
        self.assertEqual(len(qdrant_filter.must), 2)
        self.assertEqual(qdrant_filter.must[0].key, "reported_persons")
        self.assertEqual(qdrant_filter.must[0].match.any, ["王建国", "李四"])
        self.assertEqual(qdrant_filter.must[1].key, "create_time")


if __name__ == "__main__":
    unittest.main()
