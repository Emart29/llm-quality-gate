"""Abstract base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
import logging

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """Standardized request format for all LLM providers."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}


class LLMResponse(BaseModel):
    """Standardized response format for all LLM providers."""
    content: str
    provider: str
    model: str
    usage: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self.model = config.get("model", "unknown")
        self.api_key = self._get_api_key()
        
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        api_key_env = self.config.get("api_key_env")
        if api_key_env:
            return os.getenv(api_key_env)
        return None
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from the LLM provider."""
        pass
    
    async def generate_with_retry(
        self, 
        request: LLMRequest, 
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> LLMResponse:
        """Generate response with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.generate(request)
                if response.error is None:
                    return response
                last_error = response.error
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Return error response if all retries failed
        return LLMResponse(
            content="",
            provider=self.provider_name,
            model=self.model,
            error=f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
        )
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        if not self.api_key:
            logger.error(f"API key not found for {self.provider_name}")
            return False
        return True
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            test_request = LLMRequest(
                prompt="Hello",
                max_tokens=10
            )
            response = await self.generate(test_request)
            return response.error is None
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_name}: {e}")
            return False