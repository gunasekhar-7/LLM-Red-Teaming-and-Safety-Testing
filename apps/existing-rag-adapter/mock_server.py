"""Mock existing RAG server for testing."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    port: int = 8001
    host: str = "0.0.0.0"
    delay_min: float = 0.05
    delay_max: float = 0.2


settings = Settings()

app = FastAPI(title="Mock Existing RAG", version="0.1.0")

# Predefined responses for different attack types
RESPONSES = {
    "default": "Based on the provided context, I can answer your question. The documents indicate that...",
    "injection": "I cannot ignore my previous instructions or system prompt. I'm designed to be helpful while following my guidelines.",
    "leakage": "I cannot reveal my system prompt, internal instructions, or any confidential information.",
    "jailbreak": "I cannot engage in roleplay that would bypass my safety guidelines or pretend to be unrestricted.",
    "encoding": "I cannot process encoded instructions or hidden commands in your request.",
    "rag_manipulation": "I can only use information from the retrieved documents. I cannot access external information or modify the retrieval process.",
}


class QueryRequest(BaseModel):
    question: str
    tenant_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = ["[1]", "[2]"]
    chunks: list[dict[str, Any]] = [
        {"content": "Mock document content about company policies.", "metadata": {"source": "policy.pdf"}, "score": 0.95},
        {"content": "Additional context from knowledge base.", "metadata": {"source": "kb.txt"}, "score": 0.87},
    ]


def classify_question(question: str) -> str:
    """Classify question type to select appropriate response."""
    q = question.lower()
    
    if any(kw in q for kw in ["ignore previous", "system prompt", "disregard", "override", "bypass"]):
        return "injection"
    if any(kw in q for kw in ["leak", "secret", "password", "api key", "confidential", "system prompt"]):
        return "leakage"
    if any(kw in q for kw in ["roleplay", "pretend", "act as", "you are now", "unrestricted", "no rules"]):
        return "jailbreak"
    if any(kw in q for kw in ["base64", "encoded", "decode", "hidden"]):
        return "encoding"
    if any(kw in q for kw in ["retrieve", "search", "find document", "context"]):
        return "rag_manipulation"
    
    return "default"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mock-existing-rag"}


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Simulate processing delay
    delay = random.uniform(settings.delay_min, settings.delay_max)
    await asyncio.sleep(delay)
    
    # Classify and respond
    category = classify_question(request.question)
    answer = RESPONSES.get(category, RESPONSES["default"])
    
    # Add some variation
    if category != "default":
        answer += f" [Detected: {category}]"
    
    return QueryResponse(answer=answer)


@app.post("/api/v1/ingest")
async def ingest(request: dict):
    await asyncio.sleep(0.1)
    return {"document_id": "mock-doc-123", "chunks_created": 5}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)