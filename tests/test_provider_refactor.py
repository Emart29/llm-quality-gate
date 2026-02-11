"""Tests for the refactored provider system."""

import os
import pytest
from unittest.mock import patch, MagicMock
from llm.base import LLMProvider, ProviderNotConfiguredError
from llm.factory import LLMFactory
from llm.openai import OpenAIProvider
from llm.groq import GroqProvider


class TestProviderEnabled:
    """Test provider enabled/disabled functionality."""
    
    def test_provider_enabled_with_api_key(self):
        """Test provider is enabled when API key is present."""
        config = {
            "api_key_env": "TEST_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"}):
            provider = OpenAIProvider(config)
            assert provider.enabled is True
            assert provider.api_key == "test-key-123"
    
    def test_provider_disabled_without_api_key(self):
        """Test provider is disabled when API key is missing."""
        config = {
            "api_key_env": "MISSING_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(config)
            assert provider.enabled is False
            assert provider.api_key is None
    
    def test_provider_disabled_with_empty_api_key(self):
        """Test provider is disabled when API key is empty."""
        config = {
            "api_key_env": "EMPTY_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {"EMPTY_API_KEY": ""}):
            provider = OpenAIProvider(config)
            assert provider.enabled is False
    
    def test_provider_disabled_with_whitespace_api_key(self):
        """Test provider is disabled when API key is only whitespace."""
        config = {
            "api_key_env": "WHITESPACE_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {"WHITESPACE_API_KEY": "   "}):
            provider = OpenAIProvider(config)
            assert provider.enabled is False


class TestProviderErrorHandling:
    """Test provider error handling."""
    
    def test_ensure_enabled_raises_error_when_disabled(self):
        """Test ensure_enabled raises ProviderNotConfiguredError when disabled."""
        config = {
            "api_key_env": "MISSING_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(config)
            
            with pytest.raises(ProviderNotConfiguredError) as exc_info:
                provider.ensure_enabled()
            
            assert "openai" in str(exc_info.value).lower()
            assert "MISSING_API_KEY" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_raises_error_when_disabled(self):
        """Test generate method raises error when provider is disabled."""
        config = {
            "api_key_env": "MISSING_API_KEY",
            "model": "test-model"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(config)
            
            from llm.base import LLMRequest
            request = LLMRequest(prompt="test")
            
            with pytest.raises(ProviderNotConfiguredError):
                await provider.generate(request)


class TestProviderFactory:
    """Test provider factory with new enabled/disabled logic."""
    
    def test_factory_raises_error_for_disabled_provider(self):
        """Test factory raises error when trying to create disabled provider."""
        config = {
            "providers": {
                "openai": {
                    "api_key_env": "MISSING_OPENAI_KEY",
                    "model": "gpt-3.5-turbo"
                }
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMFactory.create_provider_classmethod("openai", config=config)
            
            assert "not configured" in str(exc_info.value)
            assert "MISSING_OPENAI_KEY" in str(exc_info.value)
    
    def test_factory_creates_enabled_provider(self):
        """Test factory successfully creates enabled provider."""
        config = {
            "providers": {
                "groq": {
                    "api_key_env": "GROQ_API_KEY",
                    "model": "llama-3.1-8b-instant"
                }
            }
        }
        
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            provider = LLMFactory.create_provider_classmethod("groq", config=config)
            assert provider.enabled is True
            assert isinstance(provider, GroqProvider)
    
    def test_get_enabled_providers_filters_correctly(self):
        """Test get_enabled_providers returns only enabled providers."""
        config = {
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-3.5-turbo"
                },
                "groq": {
                    "api_key_env": "GROQ_API_KEY", 
                    "model": "llama-3.1-8b-instant"
                },
                "claude": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-3-haiku-20240307"
                }
            }
        }
        
        # Only GROQ has API key
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            enabled = LLMFactory.get_enabled_providers(config=config)
            assert enabled == ["groq"]
    
    def test_list_available_providers_shows_enabled_status(self):
        """Test list_available_providers correctly shows enabled status."""
        config = {
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-3.5-turbo"
                },
                "groq": {
                    "api_key_env": "GROQ_API_KEY",
                    "model": "llama-3.1-8b-instant"
                }
            }
        }
        
        # Only GROQ has API key
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True):
            providers = LLMFactory.list_available_providers(config=config)
            assert providers["groq"] is True
            assert providers["openai"] is False


class TestNoEmptyBearerHeaders:
    """Test that empty Bearer headers are never created."""
    
    def test_openai_no_client_when_disabled(self):
        """Test OpenAI provider doesn't create HTTP client when disabled."""
        config = {
            "api_key_env": "MISSING_API_KEY",
            "model": "gpt-3.5-turbo"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(config)
            assert provider._client is None
    
    def test_groq_no_client_when_disabled(self):
        """Test Groq provider doesn't create HTTP client when disabled."""
        config = {
            "api_key_env": "MISSING_API_KEY", 
            "model": "llama-3.1-8b-instant"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = GroqProvider(config)
            assert provider._client is None
    
    def test_client_property_raises_error_when_disabled(self):
        """Test client property raises error when provider is disabled."""
        config = {
            "api_key_env": "MISSING_API_KEY",
            "model": "gpt-3.5-turbo"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(config)
            
            with pytest.raises(ProviderNotConfiguredError):
                _ = provider.client


class TestLocalProviders:
    """Test local providers (Ollama, LocalAI) are always enabled."""
    
    def test_ollama_always_enabled(self):
        """Test Ollama provider is always enabled."""
        from llm.ollama import OllamaProvider
        
        config = {"model": "llama2"}
        provider = OllamaProvider(config)
        assert provider.enabled is True
        assert provider.api_key == "local"
    
    def test_localai_enabled_without_api_key(self):
        """Test LocalAI provider is enabled even without API key."""
        from llm.localai import LocalAIProvider
        
        config = {"model": "gpt-3.5-turbo"}
        with patch.dict(os.environ, {}, clear=True):
            provider = LocalAIProvider(config)
            assert provider.enabled is True
            assert provider.api_key == "local"
    
    def test_localai_uses_api_key_when_provided(self):
        """Test LocalAI provider uses API key when provided."""
        from llm.localai import LocalAIProvider
        
        config = {
            "api_key_env": "LOCALAI_API_KEY",
            "model": "gpt-3.5-turbo"
        }
        
        with patch.dict(os.environ, {"LOCALAI_API_KEY": "test-local-key"}):
            provider = LocalAIProvider(config)
            assert provider.enabled is True
            assert provider.api_key == "test-local-key"