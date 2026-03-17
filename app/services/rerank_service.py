from typing import Dict, List

from app.errors import ServiceError
from app.services.base_http import BaseHTTPService


class RerankService(BaseHTTPService):
    async def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, float]]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        response = await self._client.post("/rerank", json=payload)
        if response.is_error:
            raise ServiceError(
                error_code="rerank_failed",
                message="Rerank request failed: {0}".format(response.text[:300]),
                status_code=502,
                retryable=True,
            )
        body = response.json()
        results = body.get("results") or body.get("data") or []
        parsed = []
        for item in results:
            parsed.append(
                {
                    "index": int(item["index"]),
                    "score": float(item.get("relevance_score", item.get("score", 0.0))),
                }
            )
        return parsed
