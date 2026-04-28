import uuid
from datetime import datetime
from typing import List, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.config import Settings
from app.errors import ServiceError
from app.models.domain import QueryEmbedding, SearchCandidate, SourceCase
from app.utils.logger import get_logger


class QdrantService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger("qdrant_service")
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=settings.qdrant_https,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.http_timeout_seconds,
            check_compatibility=False,
        )

    async def close(self) -> None:
        await self._client.close()

    async def check_ready(self) -> None:
        await self._client.get_collections()

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._settings.qdrant_collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._settings.qdrant_collection,
                vectors_config={
                    self._settings.qdrant_dense_vector_name: models.VectorParams(
                        size=self._settings.qdrant_dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self._settings.qdrant_sparse_vector_name: models.SparseVectorParams()
                },
                on_disk_payload=True,
            )
        await self._ensure_payload_indexes()

    async def recreate_collection(self) -> None:
        exists = await self._client.collection_exists(self._settings.qdrant_collection)
        if exists:
            await self._client.delete_collection(self._settings.qdrant_collection)
        await self.ensure_collection()

    async def upsert_cases(
        self,
        cases: Sequence[SourceCase],
        embeddings: Sequence[QueryEmbedding],
    ) -> int:
        points = []
        for source_case, embedding in zip(cases, embeddings):
            points.append(
                models.PointStruct(
                    id=self._point_id_for_case(source_case.case_id),
                    vector={
                        self._settings.qdrant_dense_vector_name: embedding.dense_vector,
                        self._settings.qdrant_sparse_vector_name: models.SparseVector(
                            indices=embedding.sparse_vector.indices,
                            values=embedding.sparse_vector.values,
                        ),
                    },
                    payload=self._build_payload(source_case),
                )
            )
        await self._client.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )
        return len(points)

    async def search_dense(
        self,
        embedding: QueryEmbedding,
        query_filter: models.Filter,
        limit: int,
    ) -> List[SearchCandidate]:
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=embedding.dense_vector,
            using=self._settings.qdrant_dense_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = response.points
        return [self._point_to_candidate(point, score_field="dense_score") for point in points]

    async def search_sparse(
        self,
        embedding: QueryEmbedding,
        query_filter: models.Filter,
        limit: int,
    ) -> List[SearchCandidate]:
        if not embedding.sparse_vector.indices or not embedding.sparse_vector.values:
            self._logger.info("empty_sparse_query")
            return []
        sparse_vector = models.SparseVector(
            indices=embedding.sparse_vector.indices,
            values=embedding.sparse_vector.values,
        )
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            query=sparse_vector,
            using=self._settings.qdrant_sparse_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = response.points
        return [self._point_to_candidate(point, score_field="sparse_score") for point in points]

    async def fetch_filtered_candidates(
        self,
        query_filter: models.Filter,
        limit: int,
    ) -> List[SearchCandidate]:
        response = await self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = response[0]
        return [self._point_to_candidate(point, score_field="dense_score") for point in points]

    async def _ensure_payload_indexes(self) -> None:
        await self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="location",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="create_time",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
        await self._client.create_payload_index(
            collection_name=self._settings.qdrant_collection,
            field_name="updated_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )

    def _build_payload(self, source_case: SourceCase):
        return {
            "case_id": source_case.case_id,
            "location": source_case.location,
            "location_district": source_case.location_district,
            "reported_persons": source_case.reported_persons,
            "reporter": source_case.reporter,
            "create_time": source_case.create_time.isoformat(),
            "updated_at": source_case.updated_at.isoformat(),
            "description_text": source_case.description_text,
            "status": source_case.status,
            "extra": source_case.extra,
        }

    def _point_to_candidate(
        self,
        point: models.ScoredPoint | models.Record,
        score_field: str,
    ) -> SearchCandidate:
        payload = point.payload or {}
        extra_payload = payload.get("extra") or {}
        base_kwargs = {
            "case_id": str(payload.get("case_id", point.id)),
            "petition_id": extra_payload.get("petition_id"),
            "location": payload.get("location"),
            "location_district": payload.get("location_district"),
            "reported_persons": payload.get("reported_persons") or [],
            "reporter": payload.get("reporter"),
            "description_text": payload.get("description_text", ""),
            "create_time": self._parse_datetime(payload.get("create_time")),
            "updated_at": self._parse_datetime(payload.get("updated_at")),
        }
        base_kwargs[score_field] = float(getattr(point, "score", 0.0))
        return SearchCandidate(**base_kwargs)

    def _parse_datetime(self, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ServiceError(
            error_code="qdrant_invalid_payload",
            message="Qdrant payload is missing datetime values.",
            status_code=500,
        )

    def _point_id_for_case(self, case_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, case_id))
