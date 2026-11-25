"""OpenAI provider implementation."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        response = await self._client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            **request.extra_params,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        if request.stream:
            # Streaming handled separately
            return LLMResponse(
                content="",
                model=request.model,
                provider=self.provider_type,
                latency_ms=latency_ms,
                raw_response=response,
            )
        
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else {}
        
        return LLMResponse(
            content=choice.message.content or "",
            model=request.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            raw_response=response,
        )
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        stream = await self._client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            **request.extra_params,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        await self._client.close()


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic OpenAI-compatible provider (for local models via vLLM, etc.)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url or "http://localhost:8000/v1",
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else None,
        )
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            **request.extra_params,
        }
        
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        if request.stream:
            return LLMResponse(
                content="",
                model=request.model,
                provider=self.provider_type,
                latency_ms=latency_ms,
                raw_response=data,
            )
        
        choice = data["choices"][0]
        usage = data.get("usage", {})
        
        return LLMResponse(
            content=choice["message"]["content"],
            model=request.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            **request.extra_params,
        }
        
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json
                    chunk = json.loads(data)
                    if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        await self._client.aclose()