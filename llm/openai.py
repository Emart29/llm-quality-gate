"""OpenAI LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.timeout = config.get("timeout", 30)

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        self.ensure_enabled()

        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }

            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._make_headers(),
                timeout=self.timeout,
            ) as client:
                response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            choices = data.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                metadata={
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "request_id": response.headers.get("x-request-id")
                }
            )

        except httpx.TimeoutException:
            error_msg = "OpenAI request timed out"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"OpenAI API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except Exception as e:
            error_msg = f"OpenAI provider error: {type(e).__name__}: {e}" if str(e) else f"OpenAI provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
