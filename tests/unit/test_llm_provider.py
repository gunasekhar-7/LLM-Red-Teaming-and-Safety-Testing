"""Unit tests for LLM provider layer."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_provider import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ProviderType,
    BaseLLMProvider,
)


class DummyProvider(BaseLLMProvider):
    """Minimal concrete provider for testing."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, request: LLMRequest) -> "AsyncIterator[str]":
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class TestModelConfig:
    """Test ModelConfig pydantic model."""

    def test_model_config_creation(self):
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert config.provider.value == "openai"
        assert config.model_name == "gpt-4o-mini"
        assert config.api_key == "sk-test"

    def test_model_config_defaults(self):
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
        )
        assert config.role is not None
        assert config.base_url is None


class TestLLMRequest:
    """Test LLMRequest model."""

    def test_llm_request_creation(self):
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")],
            model="gpt-4o-mini",
        )
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"


class TestBaseProvider:
    """Test BaseLLMProvider base class."""

    def test_provider_initialization(self):
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            api_key="sk-test",
        )
        provider = DummyProvider(config)
        assert provider is not None

    async def test_provider_health_check(self):
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            api_key="sk-test",
        )
        provider = DummyProvider(config)
        result = await provider.health_check()
        assert result is True


class TestLLMMessage:
    """Test LLMMessage model."""

    def test_message_creation(self):
        msg = LLMMessage(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"

    def test_message_with_system(self):
        msg = LLMMessage(role="system", content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"