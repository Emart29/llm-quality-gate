"""Google Gemini LLM provider implementation."""

import google.generativeai as genai
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse
import logging

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "gemini-pro")
        
        # Configure the API key
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini API."""
        try:
            if not self.model:
                return LLMResponse(
                    content="",
                    provider=self.provider_name,
                    model=self.model_name,
                    error="Gemini model not initialized"
                )
            
            # Build prompt (Gemini doesn't separate system/user like other APIs)
            full_prompt = ""
            if request.system_prompt:
                full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
            else:
                full_prompt = request.prompt
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )
            
            if request.stop_sequences:
                generation_config.stop_sequences = request.stop_sequences
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract content
            content = response.text if response.text else ""
            
            # Extract usage information if available
            usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage=usage,
                metadata={
                    "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                    "safety_ratings": [rating.category.name + ":" + rating.probability.name 
                                     for rating in response.candidates[0].safety_ratings] if response.candidates else []
                }
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