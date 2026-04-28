import unittest

import httpx

from app.errors import ServiceError
from app.models.domain import SparseEmbedding
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
        self.assertEqual(embeddings[0].sparse_vector, SparseEmbedding(indices=[1], values=[0.5]))
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

    async def test_embed_texts_fetches_sparse_from_pooling_and_tokenize(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/embeddings", "/embeddings"}:
                return httpx.Response(
                    200,
                    json={"data": [{"embedding": [0.1, 0.2]}]},
                )
            if request.url.path in {"/v1/pooling", "/pooling"}:
                return httpx.Response(
                    200,
                    json={"data": [{"data": [0.0, 0.5, 0.25, 0.75, 0.0]}]},
                )
            if request.url.path in {"/v1/tokenize", "/tokenize"}:
                return httpx.Response(
                    200,
                    json={
                        "tokens": [0, 101, 102, 101, 2],
                        "token_strs": ["<s>", "王", "建", "王", "</s>"],
                    },
                )
            raise AssertionError(f"Unexpected path: {request.url.path}")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=1,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        embeddings = await service.embed_texts(["案情内容"])

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].dense_vector, [0.1, 0.2])
        self.assertEqual(embeddings[0].sparse_vector.indices, [101, 102])
        self.assertEqual(embeddings[0].sparse_vector.values, [1.25, 0.25])
        await service.close()

    async def test_embed_texts_raises_when_pooling_and_tokenize_lengths_differ(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/embeddings", "/embeddings"}:
                return httpx.Response(
                    200,
                    json={"data": [{"embedding": [0.1, 0.2]}]},
                )
            if request.url.path in {"/v1/pooling", "/pooling"}:
                return httpx.Response(
                    200,
                    json={"data": [{"data": [0.1, 0.2]}]},
                )
            if request.url.path in {"/v1/tokenize", "/tokenize"}:
                return httpx.Response(
                    200,
                    json={"tokens": [11], "token_strs": ["王"]},
                )
            raise AssertionError(f"Unexpected path: {request.url.path}")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=1,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ServiceError) as context:
            await service.embed_texts(["案情内容"])

        self.assertEqual(context.exception.error_code, "embedding_sparse_length_mismatch")
        await service.close()

    async def test_embed_texts_raises_when_sparse_is_missing_and_enabled(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/embeddings", "/embeddings"}:
                return httpx.Response(
                    200,
                    json={"data": [{"embedding": [0.1, 0.2]}]},
                )
            if request.url.path in {"/v1/pooling", "/pooling"}:
                return httpx.Response(
                    200,
                    json={"data": [{"data": "not-a-score-list"}]},
                )
            if request.url.path in {"/v1/tokenize", "/tokenize"}:
                return httpx.Response(
                    200,
                    json={"tokens": [11], "token_strs": ["王"]},
                )
            raise AssertionError(f"Unexpected path: {request.url.path}")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=1,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ServiceError) as context:
            await service.embed_texts(["案情内容"])

        self.assertEqual(context.exception.error_code, "embedding_sparse_invalid_payload")
        await service.close()

    async def test_embed_texts_can_disable_sparse_for_diagnostics(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/embeddings", "/embeddings"}:
                return httpx.Response(
                    200,
                    json={"data": [{"embedding": [0.1, 0.2]}]},
                )
            raise AssertionError(f"Unexpected path: {request.url.path}")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=1,
            retry_backoff_seconds=0,
            enable_sparse=False,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        embeddings = await service.embed_texts(["案情内容"])

        self.assertEqual(embeddings[0].dense_vector, [0.1, 0.2])
        self.assertEqual(embeddings[0].sparse_vector.indices, [])
        self.assertEqual(embeddings[0].sparse_vector.values, [])
        await service.close()

    async def test_embed_texts_translates_sparse_pooling_transport_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in {"/v1/embeddings", "/embeddings"}:
                return httpx.Response(
                    200,
                    json={"data": [{"embedding": [0.1, 0.2]}]},
                )
            if request.url.path in {"/v1/pooling", "/pooling"}:
                raise httpx.RemoteProtocolError("server disconnected")
            raise AssertionError(f"Unexpected path: {request.url.path}")

        service = EmbeddingService(
            base_url="http://testserver/v1",
            model_name="bge-m3",
            api_key="EMPTY",
            timeout_seconds=10,
            retry_attempts=1,
            retry_backoff_seconds=0,
        )
        service._client = httpx.AsyncClient(
            base_url="http://testserver/v1",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ServiceError) as context:
            await service.embed_texts(["案情内容"])

        self.assertEqual(context.exception.error_code, "embedding_sparse_transport_error")
        self.assertEqual(context.exception.status_code, 502)
        await service.close()


if __name__ == "__main__":
    unittest.main()
