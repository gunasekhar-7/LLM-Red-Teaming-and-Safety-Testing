"""RAG service for retrieval and generation."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reference_rag.config import settings
from reference_rag.database import Document, DocumentChunk, Tenant, async_session_factory


class PGVectorEmbeddings(Embeddings):
    """LangChain embeddings wrapper for pgvector."""
    
    def __init__(self, embedding_model: BaseChatModel | Embeddings):
        self.embedding_model = embedding_model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text)
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedding_model.aembed_documents(texts)
    
    async def aembed_query(self, text: str) -> list[float]:
        return await self.embedding_model.aembed_query(text)


class PGVectorRetriever:
    """Custom retriever using pgvector with tenant isolation."""
    
    def __init__(
        self,
        session_factory,
        embeddings: Embeddings,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        self.session_factory = session_factory
        self.embeddings = embeddings
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        max_tokens: int = 3000,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        # Embed query
        query_embedding = await self.embeddings.aembed_query(query)
        
        async with self.session_factory() as session:
            # Use pgvector cosine similarity
            stmt = select(
                DocumentChunk.id,
                DocumentChunk.content,
                DocumentChunk.metadata,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("similarity"),
            ).where(
                DocumentChunk.tenant_id == tenant_id,
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)) >= self.similarity_threshold,
            ).order_by(
                (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).desc()
            ).limit(self.top_k)
            
            result = await session.execute(stmt)
            rows = result.all()
            
            chunks = []
            total_tokens = 0
            
            for i, row in enumerate(rows):
                # Rough token count
                chunk_tokens = len(row.content) // 4
                if total_tokens + chunk_tokens > max_tokens:
                    break
                
                chunks.append({
                    "id": str(row.id),
                    "content": row.content,
                    "metadata": row.metadata or {},
                    "document_id": str(row.document_id),
                    "chunk_index": row.chunk_index,
                    "score": float(row.similarity),
                    "index": i,
                })
                total_tokens += chunk_tokens
            
            return chunks


class RAGService:
    """Main RAG service combining retrieval and generation."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.system_prompt = system_prompt or settings.system_prompt
        self.retriever = PGVectorRetriever(
            async_session_factory,
            embeddings,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    
    async def query(
        self,
        question: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Execute a RAG query."""
        # Retrieve context
        chunks = await self.retriever.retrieve(
            question,
            tenant_id,
            max_tokens=settings.max_context_tokens,
        )
        
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "citations": [],
                "chunks": [],
            }
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[{i+1}] {chunk['content']}")
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({
            "context": context,
            "question": question,
        })
        
        # Extract citations (simple approach)
        citations = [f"[{i+1}]" for i in range(len(chunks))]
        
        return {
            "answer": answer,
            "citations": citations,
            "chunks": chunks,
        }
    
    async def ingest(
        self,
        content: str,
        tenant_id: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Ingest a document into the vector store."""
        document_id = uuid.uuid4()
        
        # Split into chunks
        texts = self.text_splitter.split_text(content)
        
        # Embed chunks
        embeddings = await self.embeddings.aembed_documents(texts)
        
        async with self.session_factory() as session:
            # Create document
            doc = Document(
                id=document_id,
                tenant_id=tenant_id,
                title=title,
                source=source,
                metadata=metadata or {},
            )
            session.add(doc)
            
            # Create chunks
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                chunk = DocumentChunk(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    content=text,
                    embedding=embedding,
                    chunk_index=i,
                    metadata={"title": title, "source": source},
                )
                session.add(chunk)
            
            await session.commit()
        
        return str(document_id)
    
    async def health_check(self) -> bool:
        """Check if service is healthy."""
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception:
            return False


def format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format chunks for inclusion in prompt."""
    return "\n\n".join(
        f"[{i+1}] {chunk['content']}"
        for i, chunk in enumerate(chunks)
    )