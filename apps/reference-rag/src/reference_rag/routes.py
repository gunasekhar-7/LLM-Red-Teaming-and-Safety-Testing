"""API routes for reference RAG."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sqlalchemy import select

from reference_rag.auth import require_tenant
from reference_rag.config import settings
from reference_rag.database import Document, Tenant, async_session_factory
from reference_rag.hardened_rag_service import HardenedRAGService

router = APIRouter()

_rag_service: HardenedRAGService | None = None


def set_rag_service(service: HardenedRAGService | None) -> None:
    """Register the active HardenedRAGService instance."""
    global _rag_service
    _rag_service = service


def get_rag_service() -> HardenedRAGService:
    if _rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG service not initialized"
        )
    return _rag_service


class QueryRequest(BaseModel):
    """Query request."""

    question: str = Field(..., min_length=1, max_length=10000)
    tenant_id: str | None = None


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    guard_results: dict[str, list[dict[str, Any]]] | None = None


class IngestRequest(BaseModel):
    """Document ingestion request."""

    content: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    source: str = Field(..., min_length=1, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None


class IngestResponse(BaseModel):
    """Ingestion response."""

    document_id: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: bool
    rag_service: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    db_healthy = False
    try:
        async with async_session_factory() as session:
            await session.execute(select(1))
        db_healthy = True
    except Exception:
        pass

    rag_healthy = False
    if _rag_service is not None:
        rag_healthy = await _rag_service.health_check()

    return HealthResponse(
        status="healthy" if db_healthy and rag_healthy else "degraded",
        version=settings.app_version,
        database=db_healthy,
        rag_service=rag_healthy,
    )


@router.get("/guard-status")
async def guard_status(
    _tenant: Tenant = Depends(require_tenant), svc: HardenedRAGService = Depends(get_rag_service)
) -> dict[str, Any]:
    """Get status of all safety guard pipelines."""
    return await svc.get_guard_status()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    tenant: Tenant = Depends(require_tenant),
    svc: HardenedRAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Query the hardened RAG system."""
    tenant_id = request.tenant_id or tenant.id
    if tenant_id != tenant.id and not getattr(tenant, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to tenant")

    result = await svc.query(request.question, tenant_id)
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        chunks=result["chunks"],
        blocked=result.get("blocked", False),
        block_reason=result.get("block_reason"),
        guard_results=result.get("guard_results"),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    tenant: Tenant = Depends(require_tenant),
    svc: HardenedRAGService = Depends(get_rag_service),
) -> IngestResponse:
    """Ingest a document into the RAG system."""
    tenant_id = request.tenant_id or tenant.id
    if tenant_id != tenant.id and not getattr(tenant, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to tenant")

    document_id = await svc.ingest(
        content=request.content,
        tenant_id=tenant_id,
        title=request.title,
        source=request.source,
        metadata=request.metadata,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_text(request.content)

    return IngestResponse(document_id=document_id, chunks_created=len(chunks))


@router.get("/documents")
async def list_documents(
    tenant: Tenant = Depends(require_tenant), limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """List documents for the current tenant."""
    async with async_session_factory() as session:
        stmt = (
            select(Document)
            .where(Document.tenant_id == tenant.id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        documents = result.scalars().all()

        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "source": doc.source,
                "metadata": doc.doc_metadata,
                "created_at": doc.created_at.isoformat(),
            }
            for doc in documents
        ]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str, tenant: Tenant = Depends(require_tenant)
) -> dict[str, str]:
    """Delete a document."""
    async with async_session_factory() as session:
        stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant.id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        await session.delete(doc)
        await session.commit()

    return {"status": "deleted"}
