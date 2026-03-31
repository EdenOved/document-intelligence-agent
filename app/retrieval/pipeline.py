from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.rerank import CohereReranker
from app.schemas import Citation, SearchHit, SearchResult

settings = get_settings()


class RetrievalService:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.reranker = CohereReranker() if settings.cohere_api_key else None

    async def search(
        self,
        db: AsyncSession,
        query: str,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> SearchResult:
        top_k = top_k or settings.default_top_k
        rerank_top_n = rerank_top_n or settings.default_rerank_top_n
        query_embedding = await self.embedding_service.embed_query(query)

        distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        statement: Select[tuple[DocumentChunk, Document, float]] = (
            select(DocumentChunk, Document, distance)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "completed")
        )
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))

        statement = statement.order_by(distance).limit(top_k)
        rows = (await db.execute(statement)).all()
        return await self._rows_to_result(
            query=query,
            rows=rows,
            rerank_top_n=rerank_top_n,
        )

    async def search_for_comparison(
        self,
        db: AsyncSession,
        query: str,
        document_ids: list[UUID],
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> SearchResult:
        if not document_ids:
            return SearchResult(query=query, total_hits=0, hits=[])

        total_top_k = top_k or settings.default_top_k
        per_document_top_k = max(1, total_top_k // max(1, len(document_ids)))
        per_document_rerank = rerank_top_n or settings.default_rerank_top_n

        combined_hits: list[SearchHit] = []
        for document_id in document_ids:
            partial = await self.search(
                db=db,
                query=query,
                document_ids=[document_id],
                top_k=per_document_top_k,
                rerank_top_n=min(per_document_rerank, per_document_top_k),
            )
            combined_hits.extend(partial.hits)

        reindexed_hits = self._reindex_hits(combined_hits)
        return SearchResult(query=query, total_hits=len(reindexed_hits), hits=reindexed_hits)

    async def _rows_to_result(
        self,
        query: str,
        rows: list[tuple[DocumentChunk, Document, float]],
        rerank_top_n: int,
    ) -> SearchResult:
        if not rows:
            return SearchResult(query=query, total_hits=0, hits=[])

        provisional_hits: list[SearchHit] = []
        for index, (chunk, document, distance) in enumerate(rows, start=1):
            citation = Citation(
                source_id=f"S{index}",
                document_id=document.id,
                filename=document.filename,
                chunk_id=chunk.id,
                page_number=chunk.page_number,
                excerpt=chunk.content[:300],
            )
            provisional_hits.append(
                SearchHit(
                    document_id=document.id,
                    filename=document.filename,
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=self._distance_to_score(distance),
                    page_number=chunk.page_number,
                    metadata=chunk.metadata_json,
                    citation=citation,
                )
            )

        if self.reranker is None:
            return SearchResult(query=query, total_hits=len(provisional_hits), hits=provisional_hits)

        reranked = await self.reranker.rerank(
            query=query,
            documents=[hit.content for hit in provisional_hits],
            top_n=min(rerank_top_n, len(provisional_hits)),
        )

        final_hits: list[SearchHit] = []
        for new_index, item in enumerate(reranked, start=1):
            hit = provisional_hits[item["index"]]
            hit.rerank_score = float(item.get("relevance_score", 0.0))
            hit.citation.source_id = f"S{new_index}"
            final_hits.append(hit)

        return SearchResult(query=query, total_hits=len(final_hits), hits=final_hits)

    @staticmethod
    def _distance_to_score(distance: float | None) -> float:
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distance)))

    @staticmethod
    def _reindex_hits(hits: Iterable[SearchHit]) -> list[SearchHit]:
        reindexed: list[SearchHit] = []
        for index, hit in enumerate(hits, start=1):
            hit.citation.source_id = f"S{index}"
            reindexed.append(hit)
        return reindexed