from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.schemas import ChunkPayload, ParsedDocument

settings = get_settings()


class ChunkingService:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

    def chunk_document(self, document: ParsedDocument) -> list[ChunkPayload]:
        chunks: list[ChunkPayload] = []
        running_chunk_index = 0
        for page in document.pages:
            split_texts = self.splitter.split_text(page.text)
            for local_index, text in enumerate(split_texts):
                chunks.append(
                    ChunkPayload(
                        page_number=page.page_number,
                        chunk_index=running_chunk_index,
                        text=text,
                        metadata={
                            "source_type": document.source_type,
                            "page_number": page.page_number,
                            "local_chunk_index": local_index,
                        },
                    )
                )
                running_chunk_index += 1
        return chunks
