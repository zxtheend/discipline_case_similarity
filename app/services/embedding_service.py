import asyncio
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.errors import ServiceError
from app.models.domain import QueryEmbedding, SparseEmbedding
from app.services.base_http import BaseHTTPService


class EmbeddingService(BaseHTTPService):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        timeout_seconds: float,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        enable_sparse: bool = True,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self._retry_attempts = max(1, retry_attempts)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._enable_sparse = enable_sparse
        self._openai_base_url = base_url.rstrip("/")
        self._server_root_url = self._derive_server_root_url(base_url)

    async def embed_text(self, text: str) -> QueryEmbedding:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: List[str]) -> List[QueryEmbedding]:
        if not texts:
            return []

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
        if any(self._item_contains_sparse(item) for item in data):
            return [self._parse_embedding_item(item) for item in data]

        if not self._enable_sparse:
            return [
                QueryEmbedding(dense_vector=self._parse_dense_vector(item), sparse_vector=SparseEmbedding())
                for item in data
            ]

        sparse_embeddings = await self._fetch_sparse_embeddings(texts)
        return [
            QueryEmbedding(
                dense_vector=self._parse_dense_vector(item),
                sparse_vector=sparse_embedding,
            )
            for item, sparse_embedding in zip(data, sparse_embeddings)
        ]

    async def _post_embeddings(self, payload: Dict[str, Any]) -> httpx.Response:
        return await self._post_with_retries(
            url=self._build_openai_url("/embeddings"),
            payload=payload,
            timeout_error_code="embedding_timeout",
            timeout_message="Embedding request timed out.",
            transport_error_code="embedding_transport_error",
            transport_message="Embedding transport failed: {0}",
        )

    async def _post_pooling_sparse(self, payload: Dict[str, Any]) -> httpx.Response:
        return await self._post_with_retries(
            url=self._build_server_root_url("/pooling"),
            payload=payload,
            timeout_error_code="embedding_sparse_timeout",
            timeout_message="Sparse pooling request timed out.",
            transport_error_code="embedding_sparse_transport_error",
            transport_message="Sparse pooling transport failed: {0}",
        )

    async def _post_tokenize(self, payload: Dict[str, Any]) -> httpx.Response:
        return await self._post_with_retries(
            url=self._build_server_root_url("/tokenize"),
            payload=payload,
            timeout_error_code="embedding_tokenize_timeout",
            timeout_message="Tokenize request timed out.",
            transport_error_code="embedding_tokenize_transport_error",
            transport_message="Tokenize transport failed: {0}",
        )

    async def _post_with_retries(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_error_code: str,
        timeout_message: str,
        transport_error_code: str,
        transport_message: str,
    ) -> httpx.Response:
        last_error: Optional[ServiceError] = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return await self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last_error = ServiceError(
                    error_code=timeout_error_code,
                    message=timeout_message,
                    status_code=504,
                    retryable=True,
                    details={"error": str(exc), "attempt": attempt},
                )
            except httpx.HTTPError as exc:
                last_error = ServiceError(
                    error_code=transport_error_code,
                    message=transport_message.format(str(exc)),
                    status_code=502,
                    retryable=True,
                    details={"error": str(exc), "attempt": attempt},
                )

            if attempt < self._retry_attempts and self._retry_backoff_seconds > 0:
                await asyncio.sleep(self._retry_backoff_seconds * attempt)

        assert last_error is not None
        raise last_error

    async def _fetch_sparse_embeddings(self, texts: Sequence[str]) -> List[SparseEmbedding]:
        pooling_payload = {
            "model": self.model_name,
            "input": list(texts),
            "task": "token_classify",
        }
        pooling_response = await self._post_pooling_sparse(pooling_payload)
        if pooling_response.is_error:
            raise ServiceError(
                error_code="embedding_sparse_failed",
                message="Sparse pooling request failed: {0}".format(pooling_response.text[:300]),
                status_code=502,
                retryable=True,
            )

        pooling_data = pooling_response.json().get("data", [])
        if len(pooling_data) != len(texts):
            raise ServiceError(
                error_code="embedding_sparse_mismatch",
                message="Sparse pooling returned unexpected item count.",
                status_code=502,
                retryable=True,
            )

        tokenize_tasks = [
            self._fetch_tokenized_text(text)
            for text in texts
        ]
        tokenized_items = await asyncio.gather(*tokenize_tasks)
        return self._build_sparse_embeddings_from_vllm(pooling_data, tokenized_items)

    async def _fetch_tokenized_text(self, text: str) -> Dict[str, Any]:
        # vLLM /tokenize flattens batched prompts into one token list, so we tokenize each text
        # individually to preserve per-input boundaries.
        payload = {
            "model": self.model_name,
            "prompt": text,
            "return_token_strs": True,
            "add_special_tokens": False,
        }
        response = await self._post_tokenize(payload)
        if response.is_error:
            raise ServiceError(
                error_code="embedding_tokenize_failed",
                message="Tokenize request failed: {0}".format(response.text[:300]),
                status_code=502,
                retryable=True,
            )
        body = response.json()
        if not isinstance(body.get("tokens"), list):
            raise ServiceError(
                error_code="embedding_tokenize_invalid_payload",
                message="Tokenize response missing token ids.",
                status_code=502,
            )
        return body

    def _parse_embedding_item(self, item: Dict[str, Any]) -> QueryEmbedding:
        sparse_raw = (
            item.get("sparse_embedding")
            or item.get("sparse")
            or item.get("sparse_vector")
            or {}
        )
        return QueryEmbedding(
            dense_vector=self._parse_dense_vector(item),
            sparse_vector=self._parse_sparse_vector(sparse_raw),
        )

    def _parse_dense_vector(self, item: Dict[str, Any]) -> List[float]:
        dense_vector = item.get("embedding")
        if not isinstance(dense_vector, list):
            raise ServiceError(
                error_code="embedding_invalid_payload",
                message="Embedding response missing dense vector.",
                status_code=502,
            )
        return [float(value) for value in dense_vector]

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

    def _build_sparse_embeddings_from_vllm(
        self,
        pooling_items: Sequence[Dict[str, Any]],
        tokenized_items: Sequence[Dict[str, Any]],
    ) -> List[SparseEmbedding]:
        sparse_embeddings = []
        for pooling_item, tokenized_item in zip(pooling_items, tokenized_items):
            sparse_embeddings.append(
                self._build_sparse_embedding_from_vllm(pooling_item, tokenized_item)
            )
        return sparse_embeddings

    def _build_sparse_embedding_from_vllm(
        self,
        pooling_item: Dict[str, Any],
        tokenized_item: Dict[str, Any],
    ) -> SparseEmbedding:
        token_scores = self._parse_pooling_token_scores(pooling_item)
        token_pairs = self._extract_token_pairs(tokenized_item)
        if len(token_scores) != len(token_pairs):
            token_pairs = [
                pair
                for pair in token_pairs
                if not self._is_special_token(pair["token_str"])
            ]
        if len(token_scores) != len(token_pairs):
            raise ServiceError(
                error_code="embedding_sparse_length_mismatch",
                message="Sparse pooling output and tokenizer output have different lengths.",
                status_code=502,
            )

        aggregated: Dict[int, float] = {}
        for index, token_pair in enumerate(token_pairs):
            token_score = token_scores[index]
            token_id = token_pair["token_id"]
            token_str = token_pair["token_str"]
            if token_score == 0 or self._is_special_token(token_str):
                continue
            aggregated[int(token_id)] = aggregated.get(int(token_id), 0.0) + float(token_score)

        ordered = sorted(aggregated.items())
        return SparseEmbedding(
            indices=[index for index, _ in ordered],
            values=[value for _, value in ordered],
        )

    def _parse_pooling_token_scores(self, item: Dict[str, Any]) -> List[float]:
        raw = item.get("data")
        if not isinstance(raw, list):
            raise ServiceError(
                error_code="embedding_sparse_invalid_payload",
                message="Sparse pooling response missing token scores.",
                status_code=502,
            )

        scores: List[float] = []
        for entry in raw:
            if isinstance(entry, (int, float)):
                scores.append(float(entry))
                continue
            if isinstance(entry, list) and len(entry) == 1 and isinstance(entry[0], (int, float)):
                scores.append(float(entry[0]))
                continue
            raise ServiceError(
                error_code="embedding_sparse_invalid_payload",
                message="Sparse pooling response contains unsupported token score shape.",
                status_code=502,
            )
        return scores

    def _item_contains_sparse(self, item: Dict[str, Any]) -> bool:
        return any(
            key in item and item.get(key) not in (None, {}, [], ())
            for key in ("sparse_embedding", "sparse", "sparse_vector")
        )

    def _extract_token_pairs(self, tokenized_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        token_ids = tokenized_item["tokens"]
        token_strs = tokenized_item.get("token_strs")
        pairs = []
        for index, token_id in enumerate(token_ids):
            token_str = None
            if isinstance(token_strs, list) and index < len(token_strs):
                token_str = token_strs[index]
            pairs.append({"token_id": int(token_id), "token_str": token_str})
        return pairs

    def _is_special_token(self, token_str: Any) -> bool:
        if not isinstance(token_str, str):
            return False
        normalized = token_str.strip()
        if normalized in {"<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"}:
            return True
        if normalized.startswith("<") and normalized.endswith(">"):
            return True
        return False

    def _build_openai_url(self, path: str) -> str:
        return "{0}{1}".format(self._openai_base_url, path)

    def _build_server_root_url(self, path: str) -> str:
        return "{0}{1}".format(self._server_root_url, path)

    def _derive_server_root_url(self, base_url: str) -> str:
        parsed = urlsplit(base_url.rstrip("/"))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
