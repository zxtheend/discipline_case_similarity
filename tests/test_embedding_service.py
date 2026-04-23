import unittest

import httpx

from app.errors import ServiceError
from app.services.embedding_service import EmbeddingService


class EmbeddingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_texts_retries_after_transport_error(self):
        attempts = {"count": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.RemoteProtocolError("server disconnected")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "embedding": [0.1, 0.2],
                            "sparse_embedding": {"indices": [1], "values": [0.5]},
                        }
                    ]
                },
            )

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=2,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        embeddings = await service.embed_texts(["案情内容"])

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].dense_vector, [0.1, 0.2])
        await service.close()

    async def test_embed_texts_translates_transport_error_after_retries(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("server disconnected")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=2,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ServiceError) as context:
            await service.embed_texts(["案情内容"])

        self.assertEqual(context.exception.error_code, "embedding_transport_error")
        self.assertEqual(context.exception.status_code, 502)
        self.assertTrue(context.exception.retryable)
        await service.close()


if __name__ == "__main__":
    unittest.main()
