"""Hugging Face Inference API provider implementation."""

import httpx
from typing import Dict, Any
from .base import LLMProvider, LLMRequest, LLMResponse, ProviderNotConfiguredError
import logging

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://api-inference.huggingface.co"
        self.model_name = config.get("model", "microsoft/DialoGPT-medium")
        self.model = self.model_name
        self.timeout = config.get("timeout", 60)

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Hugging Face Inference API."""
        self.ensure_enabled()

        try:
            if request.system_prompt:
                input_text = f"System: {request.system_prompt}\nUser: {request.prompt}\nAssistant:"
            else:
                input_text = f"User: {request.prompt}\nAssistant:"

            payload = {
                "inputs": input_text,
                "parameters": {
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_tokens,
                    "return_full_text": False,
                    "do_sample": request.temperature > 0,
                }
            }

            if request.stop_sequences:
                payload["parameters"]["stop"] = request.stop_sequences

            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._make_headers(),
                timeout=self.timeout,
            ) as client:
                response = await client.post(f"/models/{self.model_name}", json=payload)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                if "generated_text" in data[0]:
                    content = data[0]["generated_text"].strip()
                else:
                    content = str(data[0]).strip()
            elif isinstance(data, dict) and "generated_text" in data:
                content = data["generated_text"].strip()
            else:
                content = str(data).strip()

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                usage={
                    "prompt_tokens": len(input_text.split()),
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(input_text.split()) + len(content.split())
                },
                metadata={
                    "model_status": response.headers.get("x-compute-type", "unknown")
                }
            )

        except httpx.TimeoutException:
            error_msg = "HuggingFace request timed out"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model, error=error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Hugging Face API error: {error_msg}")
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
        except Exception as e:
            error_msg = f"Hugging Face provider error: {type(e).__name__}: {e}" if str(e) else f"Hugging Face provider error: {type(e).__name__}"
            logger.error(error_msg)
            return LLMResponse(content="", provider=self.provider_name, model=self.model_name, error=error_msg)
