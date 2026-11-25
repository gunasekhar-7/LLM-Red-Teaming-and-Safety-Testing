"""Core protocols and interfaces for RAG targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class RetrievalChunk:
    """A single retrieved document chunk."""
    content: str
    metadata: dict[str, Any]
    score: float
    source_id: str


@dataclass(frozen=True)
class RAGContext:
    """Context returned by a RAG system."""
    query: str
    chunks: list[RetrievalChunk]
    system_prompt: str | None = None
    raw_response: str | None = None


class RAGResponse(BaseModel):
    """Standardized response from a RAG target."""
    answer: str
    citations: list[str] = Field(default_factory=list)
    context: RAGContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


@runtime_checkable
class RAGTarget(Protocol):
    """Protocol for any testable RAG target."""
    
    @property
    def name(self) -> str:
        """Unique identifier for this target."""
        ...
    
    @property
    def config(self) -> dict[str, Any]:
        """Configuration metadata for this target."""
        ...
    
    async def query(self, question: str, **kwargs: Any) -> RAGResponse:
        """Execute a query against the RAG system."""
        ...
    
    async def get_context(self, question: str, **kwargs: Any) -> RAGContext:
        """Retrieve context without generating an answer."""
        ...
    
    async def get_system_prompt(self) -> str | None:
        """Get the system prompt used by this target."""
        ...
    
    async def health_check(self) -> bool:
        """Check if the target is healthy and responsive."""
        ...
    
    async def close(self) -> None:
        """Clean up resources."""
        ...


class BaseRAGTarget(ABC):
    """Abstract base class for RAG target implementations."""
    
    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self._name = name
        self._config = config or {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self) -> dict[str, Any]:
        return self._config
    
    @abstractmethod
    async def query(self, question: str, **kwargs: Any) -> RAGResponse:
        pass
    
    @abstractmethod
    async def get_context(self, question: str, **kwargs: Any) -> RAGContext:
        pass
    
    @abstractmethod
    async def get_system_prompt(self) -> str | None:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"