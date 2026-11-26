"""Anthropic provider implementation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from llm_provider import (
    BaseLLMProvider,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ProviderType,
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def _convert_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[MessageParam]]:
        """Convert standardized messages to Anthropic format."""
        system: str | None = None
        anthropic_messages: list[MessageParam] = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                role = "assistant" if msg.role == "assistant" else "user"
                anthropic_messages.append(
                    cast(MessageParam, {"role": role, "content": msg.content})
                )

        return system, anthropic_messages

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        system, messages = self._convert_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
            **request.extra_params,
        }
        if system is not None:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = "".join(block.text for block in response.content if isinstance(block, TextBlock))
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

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            **request.extra_params,
        }
        if system is not None:
            kwargs["system"] = system

        stream = await self._client.messages.create(**kwargs)
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                text = getattr(chunk.delta, "text", None)
                if isinstance(text, str):
                    yield text

    async def health_check(self) -> bool:
        try:
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
