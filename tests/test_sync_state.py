import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.models.domain import SyncCursor
from app.sync.data_sync import SyncStateStore


class SyncStateStoreTests(unittest.TestCase):
    def test_save_and_load_cursor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sync_state.json"
            store = SyncStateStore(path)
            cursor = SyncCursor(
                last_updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                last_case_id="CASE-001",
            )

            store.save(cursor)
            loaded = store.load()

            self.assertEqual(loaded.last_case_id, "CASE-001")
            self.assertEqual(loaded.last_updated_at, cursor.last_updated_at)


if __name__ == "__main__":
    unittest.main()
