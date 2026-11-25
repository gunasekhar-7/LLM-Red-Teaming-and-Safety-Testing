"""Factory for creating LLM providers from configuration."""

from __future__ import annotations

from typing import Any

from llm_provider import (
    ModelConfig,
    ProviderType,
    LLMProvider,
    registry,
    LLMSettings,
)
from llm_provider.openai import OpenAIProvider, OpenAICompatibleProvider
from llm_provider.anthropic import AnthropicProvider
from llm_provider.ollama import OllamaProvider, OllamaEmbeddingProvider


class ProviderFactory:
    """Factory for creating and registering LLM providers."""
    
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
    
    def create_provider(self, name: str, config: ModelConfig) -> LLMProvider:
        """Create a provider instance from config."""
        provider_map = {
            ProviderType.OPENAI: self._create_openai,
            ProviderType.ANTHROPIC: self._create_anthropic,
            ProviderType.OLLAMA: self._create_ollama,
            ProviderType.LITELLM: self._create_litellm,
            ProviderType.CUSTOM: self._create_custom,
        }
        
        creator = provider_map.get(config.provider)
        if not creator:
            raise ValueError(f"Unknown provider type: {config.provider}")
        
        provider = creator(config)
        registry.register(name, provider)
        return provider
    
    def _create_openai(self, config: ModelConfig) -> LLMProvider:
        if config.base_url and config.base_url != "https://api.openai.com/v1":
            return OpenAICompatibleProvider(config)
        return OpenAIProvider(config)
    
    def _create_anthropic(self, config: ModelConfig) -> LLMProvider:
        return AnthropicProvider(config)
    
    def _create_ollama(self, config: ModelConfig) -> LLMProvider:
        if config.role.value == "embedder":
            return OllamaEmbeddingProvider(config)
        return OllamaProvider(config)
    
    def _create_litellm(self, config: ModelConfig) -> LLMProvider:
        # LiteLLM uses OpenAI-compatible interface
        return OpenAICompatibleProvider(config)
    
    def _create_custom(self, config: ModelConfig) -> LLMProvider:
        raise NotImplementedError("Custom providers must be implemented separately")
    
    def create_from_dict(self, name: str, config_dict: dict[str, Any]) -> LLMProvider:
        """Create provider from dictionary config."""
        config = ModelConfig(**config_dict)
        return self.create_provider(name, config)
    
    def create_default_providers(self) -> dict[str, LLMProvider]:
        """Create default providers from settings."""
        providers = {}
        
        # OpenAI
        if self.settings.openai_api_key:
            config = ModelConfig(
                provider=ProviderType.OPENAI,
                model_name="gpt-4o-mini",
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            providers["openai-gpt4o-mini"] = self.create_provider("openai-gpt4o-mini", config)
            
            config = ModelConfig(
                provider=ProviderType.OPENAI,
                model_name="gpt-4o",
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            providers["openai-gpt4o"] = self.create_provider("openai-gpt4o", config)
        
        # Anthropic
        if self.settings.anthropic_api_key:
            config = ModelConfig(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-5-haiku-20241022",
                api_key=self.settings.anthropic_api_key,
                base_url=self.settings.anthropic_base_url,
            )
            providers["anthropic-haiku"] = self.create_provider("anthropic-haiku", config)
            
            config = ModelConfig(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                api_key=self.settings.anthropic_api_key,
                base_url=self.settings.anthropic_base_url,
            )
            providers["anthropic-sonnet"] = self.create_provider("anthropic-sonnet", config)
        
        # Ollama (always available locally)
        config = ModelConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama3.1:8b",
            base_url=self.settings.ollama_base_url,
        )
        providers["ollama-llama31"] = self.create_provider("ollama-llama31", config)
        
        config = ModelConfig(
            provider=ProviderType.OLLAMA,
            model_name="nomic-embed-text",
            role="embedder",
            base_url=self.settings.ollama_base_url,
        )
        providers["ollama-embed"] = self.create_provider("ollama-embed", config)
        
        return providers
    
    async def close_all(self) -> None:
        await registry.close_all()