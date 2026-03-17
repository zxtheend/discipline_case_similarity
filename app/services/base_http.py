from typing import Optional

import httpx

from app.errors import ServiceError


class BaseHTTPService:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        timeout_seconds: float,
    ) -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = "Bearer {0}".format(api_key)
        self.model_name = model_name
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def check_ready(self) -> None:
        endpoints = ("/models", "/health", "")
        last_error = None
        for endpoint in endpoints:
            try:
                response = await self._client.get(endpoint)
                if response.is_success:
                    return
                last_error = "{0} {1}".format(response.status_code, response.text[:200])
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = str(exc)
        raise ServiceError(
            error_code="dependency_unavailable",
            message="Service {0} is not ready: {1}".format(self.model_name, last_error),
            status_code=503,
            retryable=True,
        )
