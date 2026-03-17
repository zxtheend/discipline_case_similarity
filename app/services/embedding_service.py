from collections.abc import Iterable
from typing import Any, Dict, List

from app.errors import ServiceError
from app.models.domain import QueryEmbedding, SparseEmbedding
from app.services.base_http import BaseHTTPService


class EmbeddingService(BaseHTTPService):
    async def embed_text(self, text: str) -> QueryEmbedding:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[QueryEmbedding]:
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        response = await self._client.post("/embeddings", json=payload)
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
