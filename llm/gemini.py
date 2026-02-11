"""Google Gemini LLM provider implementation."""

import google.genai as genai
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "gemini-1.5-flash")
        self.client = None
        
        # Initialize client if enabled
        if self.enabled:
            self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Gemini client with proper API key."""
        # Only create client if we have an API key
        if not self.api_key:
            return
            
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini API."""
        # Ensure provider is enabled before making API calls
        self.ensure_enabled()
        
        try:
            if not self.client:
                return LLMResponse(
                    content="",
                    provider=self.provider_name,
                    model=self.model_name,
                    error="Gemini client not initialized"
                )
            
            # Build contents for the new API
            contents = []
            if request.system_prompt:
                contents.append({"role": "system", "parts": [{"text": request.system_prompt}]})
            contents.append({"role": "user", "parts": [{"text": request.prompt}]})
            
            # Configure generation parameters
            config = genai.types.GenerateContentConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )
            
            if request.stop_sequences:
                config.stop_sequences = request.stop_sequences
            
            # Generate response using the new API
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Extract content
            content = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # Extract metadata
            metadata = {}
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                metadata = {
                    "finish_reason": getattr(candidate, 'finish_reason', None),
                    "safety_ratings": []
                }
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    metadata["safety_ratings"] = [
                        f"{rating.category}:{rating.probability}" 
                        for rating in candidate.safety_ratings
                    ]
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage=usage,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Gemini provider error: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                error=error_msg
            )