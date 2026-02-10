"""LocalAI provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse
import logging

logger = logging.getLogger(__name__)


class LocalAIProvider(LLMProvider):
    """LocalAI provider (OpenAI-compatible local API)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8080")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"},
            timeout=config.get("timeout", 120)  # Local models can be slower
        )
    
    def _get_api_key(self) -> str:
        """LocalAI may or may not require API key depending on setup."""
        import os
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            return os.getenv(api_key_env, "local")
        return "local"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LocalAI API (OpenAI-compatible)."""
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
            
            # Add authorization header if API key is provided
            headers = {}
            if self.api_key and self.api_key != "local":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API call
            response = await self.client.post(
                "/v1/chat/completions", 
                json=payload,
                headers=headers
            )
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
                    "created": data.get("created"),
                    "object": data.get("object")
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"LocalAI API error: {error_msg}")
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"LocalAI provider error: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                error=error_msg
            )
    
    async def health_check(self) -> bool:
        """Check if LocalAI server is running."""
        try:
            # Check if server is running by listing models
            headers = {}
            if self.api_key and self.api_key != "local":
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            response = await self.client.get("/v1/models", headers=headers)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"LocalAI health check failed: {e}")
            return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()