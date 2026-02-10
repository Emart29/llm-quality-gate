"""OpenRouter LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse
import logging

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider (OpenAI-compatible API)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.model_name = config.get("model", "microsoft/wizardlm-2-8x22b")
        self.site_url = config.get("site_url", "https://llm-quality-gate")
        self.app_name = config.get("app_name", "LLM Quality Gate")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name
            },
            timeout=config.get("timeout", 60)
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenRouter API."""
        try:
            # Build messages array (OpenAI-compatible format)
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Prepare API request
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False
            }
            
            if request.stop_sequences:
                payload["stop"] = request.stop_sequences
            
            # Make API call
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response content
            content = data["choices"][0]["message"]["content"]
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
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "model_used": data.get("model"),  # OpenRouter may use different model
                    "provider_used": data.get("provider", {}).get("name") if data.get("provider") else None
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"OpenRouter API error: {error_msg}")
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"OpenRouter provider error: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                error=error_msg
            )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()