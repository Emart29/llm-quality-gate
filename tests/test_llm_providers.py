"""Tests for LLM provider implementations."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from llm.base import LLMRequest, LLMResponse
from llm.groq import GroqProvider
from llm.openai import OpenAIProvider
from llm.claude import ClaudeProvider
from llm.gemini import GeminiProvider
from llm.huggingface import HuggingFaceProvider
from llm.openrouter import OpenRouterProvider
from llm.ollama import OllamaProvider
from llm.localai import LocalAIProvider
from llm.factory import LLMFactory


class TestLLMProviders:
    """Test LLM provider implementations."""

    @pytest.fixture
    def mock_config(self):
        return {
            "model": "test-model",
            "api_key_env": "TEST_API_KEY",
            "timeout": 30
        }

    @pytest.fixture
    def sample_request(self):
        return LLMRequest(
            prompt="Hello, world!",
            system_prompt="You are a helpful assistant.",
            temperature=0.0,
            max_tokens=100
        )

    @patch.dict('os.environ', {'TEST_API_KEY': 'test-key-123'})
    def test_all_providers_init(self, mock_config):
        """Test all provider initializations."""
        providers = [
            GroqProvider, OpenAIProvider, ClaudeProvider,
            HuggingFaceProvider, OpenRouterProvider, OllamaProvider, LocalAIProvider
        ]

        for provider_class in providers:
            provider = provider_class(mock_config)
            expected_name = provider_class.__name__.replace("Provider", "").lower()
            assert provider.provider_name == expected_name
            assert provider.model == "test-model"

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TEST_API_KEY': 'test-key-123'})
    @patch('httpx.AsyncClient.post')
    async def test_openai_compatible_providers(self, mock_post, mock_config, sample_request):
        """Test OpenAI-compatible providers (Groq, OpenAI, OpenRouter, LocalAI)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello! How can I help you?"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        mock_response.headers = {"x-request-id": "req-123"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        providers = [GroqProvider, OpenAIProvider, OpenRouterProvider, LocalAIProvider]

        for provider_class in providers:
            provider = provider_class(mock_config)
            response = await provider.generate(sample_request)

            assert response.content == "Hello! How can I help you?"
            assert response.error is None
            assert response.usage["total_tokens"] == 18

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TEST_API_KEY': 'test-key-123'})
    @patch('httpx.AsyncClient.post')
    async def test_huggingface_provider(self, mock_post, mock_config, sample_request):
        """Test Hugging Face provider with different response format."""
        mock_response = Mock()
        mock_response.json.return_value = [{"generated_text": "This is a generated response"}]
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = HuggingFaceProvider(mock_config)
        response = await provider.generate(sample_request)

        assert response.content == "This is a generated response"
        assert response.error is None

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TEST_API_KEY': 'test-key-123'})
    @patch('google.generativeai.GenerativeModel')
    async def test_gemini_provider(self, mock_model_class, mock_config, sample_request):
        """Test Gemini provider."""
        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 8
        mock_usage.total_token_count = 18

        mock_response = Mock()
        mock_response.text = "Gemini generated response"
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].safety_ratings = []

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        provider = GeminiProvider(mock_config)
        response = await provider.generate(sample_request)

        assert response.content == "Gemini generated response"
        assert response.error is None

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'TEST_API_KEY': 'test-key-123'})
    @patch('httpx.AsyncClient.post')
    async def test_ollama_provider(self, mock_post, mock_config, sample_request):
        """Test Ollama provider."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Ollama generated response",
            "done": True,
            "total_duration": 1000000,
            "prompt_eval_count": 10,
            "eval_count": 8
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider(mock_config)
        response = await provider.generate(sample_request)

        assert response.content == "Ollama generated response"
        assert response.error is None
        assert response.metadata["done"] is True


class TestLLMFactory:
    """Test LLM factory with all providers."""

    @pytest.fixture
    def comprehensive_config(self):
        return {
            "llm": {
                "default_provider": "groq",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            "providers": {
                "groq": {"api_key_env": "GROQ_API_KEY", "model": "llama3-8b-8192"},
                "openai": {"api_key_env": "OPENAI_API_KEY", "model": "gpt-3.5-turbo"},
                "claude": {"api_key_env": "ANTHROPIC_API_KEY", "model": "claude-3-haiku"},
                "gemini": {"api_key_env": "GOOGLE_API_KEY", "model": "gemini-pro"},
                "huggingface": {"api_key_env": "HUGGINGFACE_API_KEY", "model": "microsoft/DialoGPT-medium"},
                "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "model": "microsoft/wizardlm-2-8x22b"},
                "ollama": {"model": "llama2", "base_url": "http://localhost:11434"},
                "localai": {"model": "gpt-3.5-turbo", "base_url": "http://localhost:8080"}
            },
            "roles": {
                "generator": {"provider": "groq", "model": "llama3-8b-8192"},
                "judge": {"provider": "openai", "model": "gpt-3.5-turbo"}
            }
        }

    def test_all_providers_available(self, comprehensive_config):
        """Test that all providers are available in factory."""
        expected_providers = {
            "groq", "openai", "claude", "gemini",
            "huggingface", "openrouter", "ollama", "localai"
        }

        assert set(LLMFactory.PROVIDERS.keys()) == expected_providers

    @patch.dict('os.environ', {
        'GROQ_API_KEY': 'groq-key', 'OPENAI_API_KEY': 'openai-key',
        'ANTHROPIC_API_KEY': 'claude-key', 'GOOGLE_API_KEY': 'gemini-key'
    })
    def test_create_all_providers(self, comprehensive_config):
        """Test creating all provider types."""
        for provider_name in LLMFactory.PROVIDERS.keys():
            try:
                provider = LLMFactory.create_provider(provider_name, config=comprehensive_config)
                assert provider.provider_name == provider_name
            except Exception as e:
                if provider_name not in ["ollama", "localai"]:
                    raise e

    @patch.dict('os.environ', {'GROQ_API_KEY': 'groq-key', 'OPENAI_API_KEY': 'openai-key'})
    def test_role_based_providers(self, comprehensive_config):
        """Test role-based provider creation."""
        generator = LLMFactory.create_provider_for_role("generator", config=comprehensive_config)
        judge = LLMFactory.create_provider_for_role("judge", config=comprehensive_config)

        assert generator.provider_name == "groq"
        assert judge.provider_name == "openai"
        assert generator.model == "llama3-8b-8192"
        assert judge.model == "gpt-3.5-turbo"

    def test_deterministic_default(self):
        """Test that default temperature is 0.0 for deterministic generation."""
        request = LLMRequest(prompt="test")
        assert request.temperature == 0.0
