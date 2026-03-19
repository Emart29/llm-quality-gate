"""LocalAI provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class LocalAIProvider(LLMProvider):
    """LocalAI provider (OpenAI-compatible local API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8080")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.model = self.model_name
        self.timeout = config.get("timeout", 120)

    def _get_api_key(self) -> str:
        """LocalAI may or may not require API key depending on setup."""
        import os
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env)
            return api_key.strip() if api_key else "local"
        return "local"

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )

    def _auth_headers(self) -> Dict[str, str]:
        if self.api_key and self.api_key != "local":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LocalAI API (OpenAI-compatible)."""
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }

            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            async with self._make_client() as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=self._auth_headers(),
                )
            response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                metadata={
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "created": data.get("created"),
                    "object": data.get("object")
                }
            )

        except httpx.TimeoutException:
            error_msg = "LocalAI request timed out"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"LocalAI API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
        except Exception as e:
            error_msg = f"LocalAI provider error: {type(e).__name__}: {e}" if str(e) else f"LocalAI provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)

    async def health_check(self) -> bool:
        """Check if LocalAI server is running."""
        try:
            async with self._make_client() as client:
                response = await client.get("/v1/models", headers=self._auth_headers())
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LocalAI health check failed: {e}")
            return False
