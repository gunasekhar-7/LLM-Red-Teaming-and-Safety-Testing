"""Database models and vector store for reference RAG."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from reference_rag.config import settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Document model."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_documents_tenant_created", "tenant_id", "created_at"),)


class DocumentChunk(Base):
    """Document chunk with vector embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.pgvector_dimension), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
    )


class Tenant(Base):
    """Tenant model for multi-tenancy."""

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_settings: Mapped[dict[str, Any]] = mapped_column("settings", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)
    is_admin: Mapped[bool] = mapped_column(default=False)


engine = create_async_engine(settings.database_url, echo=settings.debug, pool_pre_ping=True)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Dependency for getting DB session."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
