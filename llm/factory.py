"""Factory for creating LLM provider instances."""

from typing import Dict, Any, List, Optional
import yaml
import os
from dotenv import load_dotenv
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

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.provider.generate(request)).result()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.provider.generate(request))


class LLMFactory:
    """Factory for creating LLM provider instances.

    Supports both instance-based and classmethod-based usage:
        # Instance-based (uses config.yaml)
        factory = LLMFactory()
        provider = factory.create_provider("groq")

        # Classmethod-based (pass config dict)
        config = LLMFactory.load_config()
        provider = LLMFactory.create_provider("groq", config=config)
    """

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
        # Load environment variables from .env file
        load_dotenv()
        self.config = self.load_config(config_path)

    @classmethod
    def load_config(cls, config_path: str = "config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # Load environment variables from .env file
        load_dotenv()
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise

    @classmethod
    def _resolve_config(cls, self_or_config=None, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Resolve config from either self (instance) or explicit config dict."""
        if config is not None:
            return config
        if isinstance(self_or_config, dict):
            return self_or_config
        if self_or_config is not None and hasattr(self_or_config, 'config'):
            return self_or_config.config
        raise ValueError("No configuration provided. Pass a config dict or use an instance.")

    def create_llm(self, provider_name: str, model_name: Optional[str] = None, config: Optional[Dict] = None) -> "BaseLLM":
        """Create an LLM instance compatible with the metrics system."""
        # Use instance config if no config provided
        resolved_config = config if config is not None else self.config
        provider = self.create_provider(provider_name, model_name=model_name, config=resolved_config)
        return BaseLLM(provider)

    def create_provider(
        self,
        provider_name: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance.

        Can be called as instance method (uses self.config) or classmethod (pass config).
        """
        # For instance calls, use self.config if no config provided
        if config is None:
            if hasattr(self, 'config'):
                resolved_config = self.config
            else:
                raise ValueError("Config must be provided when calling as classmethod")
        else:
            resolved_config = config

        if provider_name not in self.PROVIDERS:
            available = ", ".join(self.PROVIDERS.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

        provider_config = resolved_config.get("providers", {}).get(provider_name, {})
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")

        provider_config = provider_config.copy()
        if model_name:
            provider_config["model"] = model_name

        global_config = resolved_config.get("llm", {})
        merged_config = {**global_config, **provider_config}

        provider_class = self.PROVIDERS[provider_name]
        provider = provider_class(merged_config)
        
        # Check if provider is enabled (has valid API key)
        if not provider.enabled:
            api_key_env = merged_config.get("api_key_env", f"{provider_name.upper()}_API_KEY")
            raise ValueError(
                f"Provider '{provider_name}' is not configured. "
                f"Please set {api_key_env} in your environment or .env file."
            )
        
        return provider

    @classmethod
    def create_provider_classmethod(
        cls,
        provider_name: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance (classmethod version)."""
        if config is None:
            raise ValueError("Config must be provided when calling as classmethod")

        if provider_name not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

        provider_config = config.get("providers", {}).get(provider_name, {})
        if not provider_config:
            raise ValueError(f"No configuration found for provider: {provider_name}")

        provider_config = provider_config.copy()
        if model_name:
            provider_config["model"] = model_name

        global_config = config.get("llm", {})
        merged_config = {**global_config, **provider_config}

        provider_class = cls.PROVIDERS[provider_name]
        provider = provider_class(merged_config)
        
        # Check if provider is enabled (has valid API key)
        if not provider.enabled:
            api_key_env = merged_config.get("api_key_env", f"{provider_name.upper()}_API_KEY")
            raise ValueError(
                f"Provider '{provider_name}' is not configured. "
                f"Please set {api_key_env} in your environment or .env file."
            )
        
        return provider

    @classmethod
    def create_default_provider(cls, config: Optional[Dict[str, Any]] = None) -> LLMProvider:
        """Create the default LLM provider."""
        if config is None:
            raise ValueError("Config must be provided when calling as classmethod")
        default_provider = config.get("llm", {}).get("default_provider", "groq")
        return cls.create_provider_classmethod(default_provider, config=config)

    @classmethod
    def create_provider_for_role(cls, role: str, config: Optional[Dict[str, Any]] = None) -> LLMProvider:
        """Create an LLM provider for a specific role (generator or judge)."""
        if config is None:
            raise ValueError("Config must be provided when calling as classmethod")

        roles_config = config.get("roles", {})
        role_config = roles_config.get(role, {})

        if not role_config:
            logger.warning(f"No role configuration found for '{role}', using default provider")
            return cls.create_default_provider(config=config)

        provider_name = role_config.get("provider")
        if not provider_name:
            logger.warning(f"No provider specified for role '{role}', using default")
            return cls.create_default_provider(config=config)

        provider_config = config.get("providers", {}).get(provider_name, {}).copy()
        if "model" in role_config:
            provider_config["model"] = role_config["model"]

        global_config = config.get("llm", {})
        merged_config = {**global_config, **provider_config}

        provider_class = cls.PROVIDERS[provider_name]
        provider = provider_class(merged_config)
        
        # Check if provider is enabled (has valid API key)
        if not provider.enabled:
            api_key_env = merged_config.get("api_key_env", f"{provider_name.upper()}_API_KEY")
            raise ValueError(
                f"Provider '{provider_name}' is not configured. "
                f"Please set {api_key_env} in your environment or .env file."
            )
        
        return provider

    @classmethod
    def list_available_providers(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """List available providers and their configuration status."""
        if config is None:
            raise ValueError("Config must be provided when calling as classmethod")

        providers = {}
        for provider_name in cls.PROVIDERS.keys():
            try:
                provider = cls.create_provider_classmethod(provider_name, config=config)
                providers[provider_name] = provider.enabled
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {e}")
                providers[provider_name] = False

        return providers
    
    @classmethod
    def get_enabled_providers(cls, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get list of enabled provider names."""
        if config is None:
            raise ValueError("Config must be provided when calling as classmethod")
            
        enabled = []
        for provider_name in cls.PROVIDERS.keys():
            try:
                provider = cls.create_provider_classmethod(provider_name, config=config)
                if provider.enabled:
                    enabled.append(provider_name)
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {e}")
                
        return enabled
