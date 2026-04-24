import unittest

import httpx

from app.main import create_app


class MainAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_validation_error_logs_raw_json_body(self):
        app = create_app()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with self.assertLogs("main", level="WARNING") as captured:
                response = await client.post(
                    "/api/v1/admin/sync/rebuild-row",
                    headers={"X-Request-ID": "req-422"},
                    json={
                        "case_id": "CASE-100",
                        "source_wtxx_bh": "XFJ-100",
                        "location": "太原市",
                        "encrypted_description": "案情内容",
                    },
                )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["request_id"], "req-422")
        combined_logs = "\n".join(captured.output)
        self.assertIn("request_validation_error", combined_logs)
        self.assertIn("req-422", combined_logs)
        self.assertIn('"case_id":"CASE-100"', combined_logs)


if __name__ == "__main__":
    unittest.main()
