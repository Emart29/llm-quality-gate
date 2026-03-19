"""Groq LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq LLM provider using OpenAI-compatible API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.groq.com/openai/v1")
        self.timeout = config.get("timeout", 30)

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Groq API."""
        # Ensure provider is enabled before making API calls
        self.ensure_enabled()

        try:
            # Build messages array
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            # Prepare API request
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }

            if request.stop_sequences:
                payload["stop"] = request.stop_sequences

            # Create a fresh client per call to avoid event-loop cross-contamination
            # when called from worker threads (each thread runs its own asyncio.run())
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._make_headers(),
                timeout=self.timeout,
            ) as client:
                response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response content
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
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Groq API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.TimeoutException:
            error_msg = f"Groq request timed out after {self.timeout}s"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except Exception as e:
            error_msg = f"Groq provider error: {type(e).__name__}: {e}" if str(e) else f"Groq provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
    
