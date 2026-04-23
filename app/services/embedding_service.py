import asyncio
from collections.abc import Iterable
from typing import Any, Dict, List

import httpx

from app.errors import ServiceError
from app.models.domain import QueryEmbedding, SparseEmbedding
from app.services.base_http import BaseHTTPService


class EmbeddingService(BaseHTTPService):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None,
        timeout_seconds: float,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self._retry_attempts = max(1, retry_attempts)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    async def embed_text(self, text: str) -> QueryEmbedding:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[QueryEmbedding]:
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        response = await self._post_embeddings(payload)
        if response.is_error:
            raise ServiceError(
                error_code="embedding_failed",
                message="Embedding request failed: {0}".format(response.text[:300]),
                status_code=502,
                retryable=True,
            )
        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise ServiceError(
                error_code="embedding_mismatch",
                message="Embedding service returned unexpected item count.",
                status_code=502,
                retryable=True,
            )
        return [self._parse_embedding_item(item) for item in data]

    async def _post_embeddings(self, payload: Dict[str, Any]) -> httpx.Response:
        last_error: ServiceError | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return await self._client.post("/embeddings", json=payload)
            except httpx.TimeoutException as exc:
                last_error = ServiceError(
                    error_code="embedding_timeout",
                    message="Embedding request timed out.",
                    status_code=504,
                    retryable=True,
                    details={"error": str(exc), "attempt": attempt},
                )
            except httpx.HTTPError as exc:
                last_error = ServiceError(
                    error_code="embedding_transport_error",
                    message="Embedding transport failed: {0}".format(str(exc)),
                    status_code=502,
                    retryable=True,
                    details={"error": str(exc), "attempt": attempt},
                )

            if attempt < self._retry_attempts and self._retry_backoff_seconds > 0:
                await asyncio.sleep(self._retry_backoff_seconds * attempt)

        assert last_error is not None
        raise last_error

    def _parse_embedding_item(self, item: Dict[str, Any]) -> QueryEmbedding:
        dense_vector = item.get("embedding")
        if not isinstance(dense_vector, list):
            raise ServiceError(
                error_code="embedding_invalid_payload",
                message="Embedding response missing dense vector.",
                status_code=502,
            )
        sparse_raw = (
            item.get("sparse_embedding")
            or item.get("sparse")
            or item.get("sparse_vector")
            or {}
        )
        return QueryEmbedding(
            dense_vector=[float(value) for value in dense_vector],
            sparse_vector=self._parse_sparse_vector(sparse_raw),
        )

    def _parse_sparse_vector(self, raw: Any) -> SparseEmbedding:
        if not raw:
            return SparseEmbedding()

        if isinstance(raw, dict) and "indices" in raw and "values" in raw:
            return SparseEmbedding(
                indices=[int(value) for value in raw["indices"]],
                values=[float(value) for value in raw["values"]],
            )

        if isinstance(raw, dict):
            ordered = sorted((int(key), float(value)) for key, value in raw.items())
            return SparseEmbedding(
                indices=[index for index, _ in ordered],
                values=[value for _, value in ordered],
            )

        if isinstance(raw, Iterable):
            pairs = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((int(item[0]), float(item[1])))
            return SparseEmbedding(
                indices=[index for index, _ in pairs],
                values=[value for _, value in pairs],
            )

        raise ServiceError(
            error_code="embedding_invalid_sparse",
            message="Embedding response contains unsupported sparse vector shape.",
            status_code=502,
        )
