"""Ollama provider implementation for local models."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
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


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        base_url = config.base_url or "http://localhost:11434"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=config.timeout,
        )
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA
    
    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Convert standardized messages to Ollama format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        
        messages = self._convert_messages(request.messages)
        
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "stream": request.stream,
            **request.extra_params,
        }
        
        response = await self._client.post("/api/chat", json=payload)
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
        
        content = data.get("message", {}).get("content", "")
        
        return LLMResponse(
            content=content,
            model=request.model,
            provider=self.provider_type,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            },
            finish_reason=data.get("done_reason"),
            latency_ms=latency_ms,
            raw_response=data,
        )
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        messages = self._convert_messages(request.messages)
        
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "stream": True,
            **request.extra_params,
        }
        
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json
                data = json.loads(line)
                if data.get("message", {}).get("content"):
                    yield data["message"]["content"]
                if data.get("done"):
                    break
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        await self._client.aclose()


class OllamaEmbeddingProvider(BaseLLMProvider):
    """Ollama embedding provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        base_url = config.base_url or "http://localhost:11434"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=config.timeout,
        )
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            payload = {
                "model": self.config.model_name,
                "prompt": text,
            }
            response = await self._client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings.append(data.get("embedding", []))
        return embeddings
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("Use OllamaProvider for completions")
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        raise NotImplementedError("Use OllamaProvider for streaming")
    
    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        await self._client.aclose()