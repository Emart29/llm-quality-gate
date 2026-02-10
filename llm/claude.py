"""Anthropic Claude LLM provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse
import logging

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=config.get("timeout", 30)
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Claude API."""
        try:
            # Build messages array (Claude format)
            messages = []
            
            # Claude handles system prompts differently
            system_prompt = request.system_prompt or ""
            
            messages.append({"role": "user", "content": request.prompt})
            
            # Prepare API request
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if request.stop_sequences:
                payload["stop_sequences"] = request.stop_sequences
            
            # Make API call
            response = await self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response content
            content = data["content"][0]["text"] if data["content"] else ""
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                },
                metadata={
                    "stop_reason": data.get("stop_reason"),
                    "request_id": response.headers.get("request-id")
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Claude API error: {error_msg}")
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Claude provider error: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model,
                error=error_msg
            )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()