"""Multi-provider LLM abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    CUSTOM = "custom"


class ModelRole(str, Enum):
    """Model role in the system."""
    GENERATOR = "generator"
    CLASSIFIER = "classifier"
    EMBEDDER = "embedder"
    CRITIC = "critic"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model."""
    provider: ProviderType
    model_name: str
    role: ModelRole = ModelRole.GENERATOR
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: float = 60.0
    extra_params: dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            object.__setattr__(self, "extra_params", {})


class LLMMessage(BaseModel):
    """Standardized message format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    """Request to an LLM provider."""
    messages: list[LLMMessage]
    model: str
    temperature: float = 0.0
    max_tokens: int = 2000
    stream: bool = False
    extra_params: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str
    model: str
    provider: ProviderType
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None
    latency_ms: float = 0.0
    raw_response: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    @property
    def provider_type(self) -> ProviderType:
        ...
    
    @property
    def config(self) -> ModelConfig:
        ...
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        ...
    
    async def health_check(self) -> bool:
        ...
    
    async def close(self) -> None:
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ModelConfig):
        self._config = config
    
    @property
    def config(self) -> ModelConfig:
        return self._config
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        pass
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass


class ProviderRegistry:
    """Registry for LLM providers."""
    
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._configs: dict[str, ModelConfig] = {}
    
    def register(self, name: str, provider: LLMProvider) -> None:
        self._providers[name] = provider
        self._configs[name] = provider.config
    
    def register_config(self, name: str, config: ModelConfig) -> None:
        self._configs[name] = config
    
    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)
    
    def get_config(self, name: str) -> ModelConfig | None:
        return self._configs.get(name)
    
    def list_providers(self) -> list[str]:
        return list(self._providers.keys())
    
    def list_configs(self) -> list[str]:
        return list(self._configs.keys())
    
    async def close_all(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


# Global registry instance
registry = ProviderRegistry()


class LLMSettings(BaseSettings):
    """Global LLM settings from environment."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # OpenAI
    openai_api_key: str | None = None
    openai_org_id: str | None = None
    openai_base_url: str | None = None
    
    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # LiteLLM
    litellm_master_key: str | None = None
    litellm_base_url: str | None = None
    
    # Defaults
    default_model: str = "gpt-4o-mini"
    default_provider: ProviderType = ProviderType.OPENAI
    request_timeout: float = 60.0
    max_retries: int = 3