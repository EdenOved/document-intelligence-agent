from __future__ import annotations

from uuid import UUID

from langchain.tools import tool
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.llm import StructuredLLMService
from app.retrieval.pipeline import RetrievalService


def build_tools(session_factory: async_sessionmaker):
    retrieval_service = RetrievalService()
    llm_service = StructuredLLMService()

    @tool
    async def search_documents(
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ):
        """Search documents for the most relevant chunks matching a user query."""
        async with session_factory() as db:
            parsed_ids = [UUID(doc_id) for doc_id in document_ids] if document_ids else None
            result = await retrieval_service.search(db, query, parsed_ids, top_k, rerank_top_n)
            return result.model_dump(mode="json")

    @tool
    async def summarize_document(
        query: str,
        document_ids: list[str],
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ):
        """Summarize one or more documents using retrieved, source-grounded evidence."""
        async with session_factory() as db:
            parsed_ids = [UUID(doc_id) for doc_id in document_ids]
            search_result = await retrieval_service.search(db, query, parsed_ids, top_k, rerank_top_n)
            result = await llm_service.summarize(query, parsed_ids, search_result.hits)
            return result.model_dump(mode="json")

    @tool
    async def extract_action_items(
        query: str,
        document_ids: list[str],
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ):
        """Extract action items, next steps, owners, and deadlines from documents."""
        async with session_factory() as db:
            parsed_ids = [UUID(doc_id) for doc_id in document_ids]
            search_result = await retrieval_service.search(db, query, parsed_ids, top_k, rerank_top_n)
            result = await llm_service.extract_action_items(query, parsed_ids, search_result.hits)
            return result.model_dump(mode="json")

    @tool
    async def compare_documents(
        query: str,
        document_ids: list[str],
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ):
        """Compare multiple documents and return similarities, differences, and a conclusion."""
        async with session_factory() as db:
            parsed_ids = [UUID(doc_id) for doc_id in document_ids]
            search_result = await retrieval_service.search_for_comparison(
                db,
                query,
                parsed_ids,
                top_k,
                rerank_top_n,
            )
            result = await llm_service.compare(query, parsed_ids, search_result.hits)
            return result.model_dump(mode="json")

    return [search_documents, summarize_document, extract_action_items, compare_documents]