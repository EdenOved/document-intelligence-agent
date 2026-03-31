from app.ingestion.chunking import ChunkingService
from app.schemas import ParsedDocument, ParsedPage


def test_chunking_preserves_page_numbers():
    document = ParsedDocument(
        filename="sample.txt",
        source_type="txt",
        pages=[ParsedPage(page_number=1, text="hello world " * 200)],
    )
    service = ChunkingService(chunk_size=100, chunk_overlap=10)
    chunks = service.chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert chunks[0].chunk_index == 0
