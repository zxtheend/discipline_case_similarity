from typing import Any, Dict, List

import httpx

from app.errors import ServiceError
from app.services.base_http import BaseHTTPService


class LLMService(BaseHTTPService):
    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ServiceError(
                error_code="llm_timeout",
                message="LLM request timed out.",
                status_code=504,
                retryable=True,
                details={"error": str(exc)},
            ) from exc
        if response.is_error:
            raise ServiceError(
                error_code="llm_failed",
                message="LLM request failed: {0}".format(response.text[:300]),
                status_code=502,
                retryable=True,
            )
        body = response.json()
        choices: List[Dict[str, Any]] = body.get("choices") or []
        if not choices:
            raise ServiceError(
                error_code="llm_empty_response",
                message="LLM response contains no choices.",
                status_code=502,
                retryable=True,
            )
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
            if parts:
                return "".join(parts)
        raise ServiceError(
            error_code="llm_invalid_response",
            message="LLM response format is unsupported.",
            status_code=502,
        )
