"""Main FastAPI application for reference RAG."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from guards import create_default_pipelines
from reference_rag.config import settings
from reference_rag.database import close_db, init_db
from reference_rag.hardened_rag_service import HardenedRAGService
from reference_rag.routes import router as api_router
from reference_rag.routes import set_rag_service

REQUEST_COUNT = Counter("rag_requests_total", "Total requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(latency)

        return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    settings.validate_security_settings()
    await init_db()
    service = await create_rag_service()
    set_rag_service(service)

    try:
        yield
    finally:
        set_rag_service(None)
        await close_db()


async def create_rag_service() -> HardenedRAGService:
    """Create hardened RAG service with configured LLM, embeddings, and safety guards."""
    from pydantic import SecretStr

    llm_secret = SecretStr(settings.llm_api_key) if settings.llm_api_key else None
    embed_secret = SecretStr(settings.embedding_api_key) if settings.embedding_api_key else None

    llm: Any
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        openai_kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "api_key": llm_secret,
            "base_url": settings.llm_base_url,
        }
        llm = ChatOpenAI(**openai_kwargs)
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "api_key": llm_secret,
            "base_url": settings.llm_base_url,
        }
        llm = ChatAnthropic(**anthropic_kwargs)
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    embeddings: Any
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=embed_secret,
            base_url=settings.embedding_base_url,
        )
    elif settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or "http://localhost:11434",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    pipelines = create_default_pipelines()
    return HardenedRAGService(
        llm=llm,
        embeddings=embeddings,
        system_prompt=settings.system_prompt,
        input_guard=pipelines["input"],
        retrieval_guard=pipelines["retrieval"],
        output_guard=pipelines["output"],
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

allow_wildcard = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not allow_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.include_router(api_router, prefix="/api/v1")


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "disabled",
    }


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "reference_rag.main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
    )
