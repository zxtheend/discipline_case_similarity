import unittest

import httpx

from app.errors import ServiceError
from app.services.llm_service import LLMService


class LLMServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_json_disables_thinking_mode(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = request.content.decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"ok": true}',
                            }
                        }
                    ]
                },
            )

        service = LLMService(
            base_url="http://testserver/v1",
            model_name="qwen3.5-27b-awq",
            api_key="EMPTY",
            timeout_seconds=10,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        content = await service.complete_json("system", "user")

        self.assertEqual(content, '{"ok": true}')
        self.assertIn('"chat_template_kwargs":{"enable_thinking":false}', captured["json"])
        await service.close()

    async def test_complete_json_translates_timeout_to_service_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        service = LLMService(
            base_url="http://testserver/v1",
            model_name="qwen3.5-27b-awq",
            api_key="EMPTY",
            timeout_seconds=10,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ServiceError) as context:
            await service.complete_json("system", "user")

        self.assertEqual(context.exception.error_code, "llm_timeout")
        self.assertEqual(context.exception.status_code, 504)
        await service.close()


if __name__ == "__main__":
    unittest.main()
