"""Unit tests for LLM provider layer."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from llm_provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    ModelConfig,
    ModelRole,
    ProviderType,
)
from llm_provider.factory import ProviderFactory


class DummyProvider(BaseLLMProvider):
    """Minimal concrete provider for testing."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="ok", model=request.model, provider=self.provider_type)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "ok"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


@pytest.mark.unit
class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self) -> None:
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert config.provider.value == "openai"
        assert config.model_name == "gpt-4o-mini"
        assert config.api_key == "sk-test"
        assert config.extra_params == {}

    def test_model_config_defaults(self) -> None:
        config = ModelConfig(provider=ProviderType.OPENAI, model_name="gpt-4o-mini")
        assert config.role == ModelRole.GENERATOR
        assert config.base_url is None


@pytest.mark.unit
class TestLLMModels:
    """Test LLMRequest and LLMMessage models."""

    def test_llm_request_creation(self) -> None:
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")], model="gpt-4o-mini"
        )
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

    def test_message_with_system(self) -> None:
        msg = LLMMessage(role="system", content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


@pytest.mark.unit
class TestProviderFactoryAndBase:
    """Test BaseLLMProvider and ProviderFactory."""

    def test_provider_initialization(self) -> None:
        config = ModelConfig(
            provider=ProviderType.OPENAI, model_name="gpt-4o-mini", api_key="sk-test"
        )
        provider = DummyProvider(config)
        assert provider.config == config

    async def test_provider_health_check(self) -> None:
        config = ModelConfig(
            provider=ProviderType.OPENAI, model_name="gpt-4o-mini", api_key="sk-test"
        )
        provider = DummyProvider(config)
        assert await provider.health_check() is True

    async def test_factory_creates_ollama_providers(self) -> None:
        factory = ProviderFactory(LLMSettings())
        providers = factory.create_default_providers()
        assert "ollama-llama31" in providers
        assert "ollama-embed" in providers
        await factory.close_all()
