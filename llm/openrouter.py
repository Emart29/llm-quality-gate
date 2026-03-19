"""OpenRouter LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider (OpenAI-compatible API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.model_name = config.get("model", "microsoft/wizardlm-2-8x22b")
        self.model = self.model_name
        self.site_url = config.get("site_url", "https://llm-quality-gate")
        self.app_name = config.get("app_name", "LLM Quality Gate")
        self.timeout = config.get("timeout", 60)

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenRouter API."""
        self.ensure_enabled()

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
                model=self.model_name,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                metadata={
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "model_used": data.get("model"),
                    "provider_used": data.get("provider", {}).get("name") if data.get("provider") else None
                }
            )

        except httpx.TimeoutException:
            error_msg = "OpenRouter request timed out"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"OpenRouter API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
        except Exception as e:
            error_msg = f"OpenRouter provider error: {type(e).__name__}: {e}" if str(e) else f"OpenRouter provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
