from __future__ import annotations

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.ingestion.chunking import ChunkingService
from app.ingestion.parsers import parse_upload_file
from app.retrieval.embeddings import EmbeddingService
from app.schemas import DocumentUploadResponse


class IngestionService:
    def __init__(self) -> None:
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()

    async def ingest(self, file: UploadFile, db: AsyncSession) -> DocumentUploadResponse:
        parsed_document = await parse_upload_file(file)
        chunks = self.chunking_service.chunk_document(parsed_document)

        document = Document(
            filename=parsed_document.filename,
            source_type=parsed_document.source_type,
            status="processing",
            metadata_json=parsed_document.metadata,
        )
        db.add(document)
        await db.flush()

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.embed_texts(chunk_texts) if chunk_texts else []

        for index, chunk in enumerate(chunks):
            chunk_metadata = dict(chunk.metadata or {})
            chunk_metadata["chunk_index"] = chunk.chunk_index

            db.add(
                DocumentChunk(
                    document_id=document.id,
                    content=chunk.text,
                    page_number=chunk.page_number,
                    metadata_json=chunk_metadata,
                    embedding=embeddings[index] if index < len(embeddings) else None,
                )
            )

        document.status = "completed"
        await db.commit()

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            chunks_created=len(chunks),
            status="completed",
        )