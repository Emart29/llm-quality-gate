"""Factory for creating LLM provider instances."""

from typing import Dict, Any, List, Optional
import yaml
import os
from .base import LLMProvider, LLMRequest, LLMResponse
from .groq import GroqProvider
from .openai import OpenAIProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .huggingface import HuggingFaceProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider
from .localai import LocalAIProvider
import logging

logger = logging.getLogger(__name__)


class BaseLLM:
    """Wrapper class to provide a compatible interface for the metrics system."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs
    ) -> LLMResponse:
        """Generate response using the provider."""
        # Convert messages format to LLMRequest
        system_prompt = None
        user_prompt = ""
        
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_prompt = message["content"]
        
        request = LLMRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Use asyncio to run the async generate method
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.provider.generate(request))


class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    PROVIDERS = {
        # Free/Open-source hosted
        "groq": GroqProvider,
        "huggingface": HuggingFaceProvider,
        "openrouter": OpenRouterProvider,
        
        # Proprietary APIs
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        
        # Local providers
        "ollama": OllamaProvider,
        "localai": LocalAIProvider,
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the factory with configuration."""
        self.config = self.load_config(config_path)
    
    @classmethod
    def load_config(cls, config_path: str = "config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def create_llm(self, provider_name: str, model_name: str) -> BaseLLM:
        """
        Create an LLM instance compatible with the metrics system.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            
        Returns:
            BaseLLM instance
        """
        provider = self.create_provider(provider_name, model_name)
        return BaseLLM(provider)
    
    def create_provider(
        self, 
        provider_name: str, 
        model_name: Optional[str] = None
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        if provider_name not in self.PROVIDERS:
            available = ", ".join(self.PROVIDERS.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
        
        provider_config = self.config["providers"].get(provider_name, {})
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")
        
        # Override model if specified
        if model_name:
            provider_config = provider_config.copy()
            provider_config["model"] = model_name
        
        # Merge global LLM config with provider-specific config
        global_config = self.config.get("llm", {})
        merged_config = {**global_config, **provider_config}
        
        provider_class = self.PROVIDERS[provider_name]
        return provider_class(merged_config)
    
    def create_default_provider(self) -> LLMProvider:
        """Create the default LLM provider."""
        default_provider = self.config.get("llm", {}).get("default_provider", "groq")
        return self.create_provider(default_provider)
    
    def create_provider_for_role(self, role: str) -> LLMProvider:
        """Create an LLM provider for a specific role (generator or judge)."""
        roles_config = self.config.get("roles", {})
        role_config = roles_config.get(role, {})
        
        if not role_config:
            logger.warning(f"No role configuration found for '{role}', using default provider")
            return self.create_default_provider()
        
        provider_name = role_config.get("provider")
        if not provider_name:
            logger.warning(f"No provider specified for role '{role}', using default")
            return self.create_default_provider()
        
        # Override model if specified in role config
        provider_config = self.config["providers"].get(provider_name, {}).copy()
        if "model" in role_config:
            provider_config["model"] = role_config["model"]
        
        # Merge global LLM config
        global_config = self.config.get("llm", {})
        merged_config = {**global_config, **provider_config}
        
        provider_class = self.PROVIDERS[provider_name]
        return provider_class(merged_config)
    
    def list_available_providers(self) -> Dict[str, bool]:
        """List available providers and their health status."""
        providers = {}
        for provider_name in self.PROVIDERS.keys():
            try:
                provider = self.create_provider(provider_name)
                providers[provider_name] = provider.validate_config()
            except Exception as e:
                logger.warning(f"Failed to create provider {provider_name}: {e}")
                providers[provider_name] = False
        
        return providers