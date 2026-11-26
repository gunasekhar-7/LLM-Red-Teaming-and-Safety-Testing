"""Hardened RAG service with safety guards."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

from guards import GuardAction, GuardPipeline, create_default_pipelines
from reference_rag.config import settings
from reference_rag.database import Document, DocumentChunk, async_session_factory
from reference_rag.rag_service import PGVectorRetriever, format_chunks_for_prompt


class HardenedRAGService:
    """Hardened RAG service with input, retrieval, and output guardrails."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        system_prompt: str | None = None,
        input_guard: GuardPipeline | None = None,
        retrieval_guard: GuardPipeline | None = None,
        output_guard: GuardPipeline | None = None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.system_prompt = system_prompt or settings.system_prompt
        self.session_factory = async_session_factory
        self.retriever = PGVectorRetriever(
            async_session_factory,
            embeddings,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

        if input_guard is None or retrieval_guard is None or output_guard is None:
            pipelines = create_default_pipelines()
            self.input_guard = input_guard or pipelines["input"]
            self.retrieval_guard = retrieval_guard or pipelines["retrieval"]
            self.output_guard = output_guard or pipelines["output"]
        else:
            self.input_guard = input_guard
            self.retrieval_guard = retrieval_guard
            self.output_guard = output_guard

    def _build_guard_context(
        self, question: str, tenant_id: str, chunks: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Build context dictionary for guard evaluation."""
        context: dict[str, Any] = {"question": question, "tenant_id": tenant_id}
        if chunks is not None:
            context["chunks"] = chunks
        return context

    async def query(self, question: str, tenant_id: str) -> dict[str, Any]:
        """Execute a hardened RAG query with guardrails."""
        guard_context = self._build_guard_context(question, tenant_id)

        # === INPUT GUARD ===
        sanitized_question, input_results = await self.input_guard.run(
            question, guard_context, stop_on_block=True
        )

        for result in input_results:
            if result.action == GuardAction.BLOCK:
                return {
                    "answer": f"Request blocked: {result.message}",
                    "citations": [],
                    "chunks": [],
                    "blocked": True,
                    "block_reason": result.message,
                    "guard_results": {
                        "input": [r.to_dict() for r in input_results],
                        "retrieval": [],
                        "output": [],
                    },
                }

        if sanitized_question != question:
            question = sanitized_question
            guard_context["question"] = sanitized_question

        # === RETRIEVAL ===
        chunks = await self.retriever.retrieve(
            question, tenant_id, max_tokens=settings.max_context_tokens
        )

        if not chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "citations": [],
                "chunks": [],
                "blocked": False,
                "guard_results": {
                    "input": [r.to_dict() for r in input_results],
                    "retrieval": [],
                    "output": [],
                },
            }

        guard_context["chunks"] = chunks

        # === RETRIEVAL GUARD ===
        _, retrieval_results = await self.retrieval_guard.run(
            json.dumps(chunks), guard_context, stop_on_block=True
        )

        for result in retrieval_results:
            if result.action == GuardAction.BLOCK:
                return {
                    "answer": f"Retrieval blocked: {result.message}",
                    "citations": [],
                    "chunks": [],
                    "blocked": True,
                    "block_reason": result.message,
                    "guard_results": {
                        "input": [r.to_dict() for r in input_results],
                        "retrieval": [r.to_dict() for r in retrieval_results],
                        "output": [],
                    },
                }

        chunks = guard_context.get("chunks", chunks)

        # Build context
        context = format_chunks_for_prompt(chunks)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context, "question": question})
        citations = [f"[{i + 1}]" for i in range(len(chunks))]

        # === OUTPUT GUARD ===
        output_context = {**guard_context, "answer": answer}
        sanitized_answer, output_results = await self.output_guard.run(
            answer, output_context, stop_on_block=True
        )

        for result in output_results:
            if result.action == GuardAction.BLOCK:
                return {
                    "answer": f"Response blocked: {result.message}",
                    "citations": [],
                    "chunks": chunks,
                    "blocked": True,
                    "block_reason": result.message,
                    "guard_results": {
                        "input": [r.to_dict() for r in input_results],
                        "retrieval": [r.to_dict() for r in retrieval_results],
                        "output": [r.to_dict() for r in output_results],
                    },
                }

        final_answer = sanitized_answer if sanitized_answer != answer else answer
        return {
            "answer": final_answer,
            "citations": citations,
            "chunks": chunks,
            "blocked": False,
            "guard_results": {
                "input": [r.to_dict() for r in input_results],
                "retrieval": [r.to_dict() for r in retrieval_results],
                "output": [r.to_dict() for r in output_results],
            },
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
        texts = self.text_splitter.split_text(content)
        embeddings = await self.embeddings.aembed_documents(texts)

        async with self.session_factory() as session:
            doc = Document(
                id=document_id,
                tenant_id=tenant_id,
                title=title,
                source=source,
                doc_metadata=metadata or {},
            )
            session.add(doc)

            for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
                chunk = DocumentChunk(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    content=text,
                    embedding=embedding,
                    chunk_index=i,
                    chunk_metadata={"title": title, "source": source},
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

    async def get_guard_status(self) -> dict[str, Any]:
        """Get status of all guard pipelines."""
        return {
            "input_guard": {
                "enabled": True,
                "guards": [g.name for g in self.input_guard.guards if g.enabled],
            },
            "retrieval_guard": {
                "enabled": True,
                "guards": [g.name for g in self.retrieval_guard.guards if g.enabled],
            },
            "output_guard": {
                "enabled": True,
                "guards": [g.name for g in self.output_guard.guards if g.enabled],
            },
        }
