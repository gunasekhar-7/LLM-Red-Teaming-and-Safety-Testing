"""Main FastAPI application for reference RAG."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

from reference_rag.config import settings
from reference_rag.database import close_db, init_db
from reference_rag.hardened_rag_service import HardenedRAGService
from reference_rag.routes import router as api_router
from reference_rag.auth import get_current_tenant
from guards import create_default_pipelines


# Prometheus metrics
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start = asyncio.get_event_loop().time()
        response = await call_next(request)
        latency = asyncio.get_event_loop().time() - start
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(latency)
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    
    # Initialize RAG service with LLM and embeddings
    global rag_service
    rag_service = await create_rag_service()
    
    yield
    
    # Shutdown
    await close_db()


async def create_rag_service() -> HardenedRAGService:
    """Create hardened RAG service with configured LLM, embeddings, and safety guards."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models import ChatOllama
    
    # Create LLM
    if settings.llm_provider == "openai":
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    elif settings.llm_provider == "anthropic":
        llm = ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    elif settings.llm_provider == "ollama":
        llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    
    # Create embeddings
    if settings.embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
    elif settings.embedding_provider == "ollama":
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
    
    # Create guard pipelines
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
app.add_middleware(MetricsMiddleware)

# Routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "disabled",
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "reference_rag.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )