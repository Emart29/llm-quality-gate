"""Ollama local LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model", "llama2")
        self.model = self.model_name
        self.timeout = config.get("timeout", 120)

    def _get_api_key(self) -> str:
        """Ollama doesn't require API key for local usage."""
        return "local"

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama API."""
        try:
            if request.system_prompt:
                full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"
            else:
                full_prompt = request.prompt

            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }

            if request.stop_sequences:
                payload["options"]["stop"] = request.stop_sequences

            async with self._make_client() as client:
                response = await client.post("/api/generate", json=payload)
            response.raise_for_status()

            data = response.json()
            content = data.get("response", "")
            prompt_tokens = len(full_prompt.split())
            completion_tokens = len(content.split())

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                metadata={
                    "done": data.get("done", False),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count")
                }
            )

        except httpx.TimeoutException:
            error_msg = "Ollama request timed out"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Ollama API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
        except Exception as e:
            error_msg = f"Ollama provider error: {type(e).__name__}: {e}" if str(e) else f"Ollama provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)

    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            async with self._make_client() as client:
                response = await client.get("/api/tags")
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            return any(self.model_name in name for name in model_names)
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
