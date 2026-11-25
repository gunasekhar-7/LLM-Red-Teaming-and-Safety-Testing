"""Anthropic provider implementation."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from llm_provider import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ProviderType,
    BaseLLMProvider,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC
    
    def _convert_messages(self, messages: list[LLMMessage]) -> tuple[str | None, list[dict]]:
        """Convert standardized messages to Anthropic format."""
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return system, anthropic_messages
    
    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        
        system, messages = self._convert_messages(request.messages)
        
        response = await self._client.messages.create(
            model=request.model,
            messages=messages,
            system=system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            **request.extra_params,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        if request.stream:
            return LLMResponse(
                content="",
                model=request.model,
                provider=self.provider_type,
                latency_ms=latency_ms,
                raw_response=response,
            )
        
        content = "".join(block.text for block in response.content if block.type == "text")
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        
        return LLMResponse(
            content=content,
            model=request.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            raw_response=response,
        )
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        system, messages = self._convert_messages(request.messages)
        
        stream = await self._client.messages.create(
            model=request.model,
            messages=messages,
            system=system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            **request.extra_params,
        )
        
        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text
    
    async def health_check(self) -> bool:
        try:
            # Anthropic doesn't have a models list endpoint, use a minimal request
            await self._client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        await self._client.close()