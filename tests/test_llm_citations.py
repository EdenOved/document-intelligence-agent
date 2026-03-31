from uuid import uuid4

from app.llm import StructuredLLMService
from app.schemas import Citation, SearchHit


def _make_hit(source_id: str, content: str) -> SearchHit:
    return SearchHit(
        document_id=uuid4(),
        filename="notes.txt",
        chunk_id=uuid4(),
        content=content,
        score=0.83,
        page_number=1,
        metadata={},
        citation=Citation(
            source_id=source_id,
            document_id=uuid4(),
            filename="notes.txt",
            chunk_id=uuid4(),
            page_number=1,
            excerpt=content[:100],
        ),
    )


def test_resolve_citations_normalizes_embedded_source_ids():
    hits = [
        _make_hit("S1", "Build the ingestion pipeline first"),
        _make_hit("S2", "Dana to prepare architecture document by 2026-04-05"),
    ]

    citations = StructuredLLMService._resolve_citations(["[S1]", "see source s2"], hits)

    assert [citation.source_id for citation in citations] == ["S1", "S2"]


def test_infer_citations_uses_text_overlap_when_model_returns_none():
    service = StructuredLLMService.__new__(StructuredLLMService)
    hits = [
        _make_hit("S1", "Build the ingestion pipeline first and use PostgreSQL with pgvector"),
        _make_hit("S2", "Dana to prepare architecture document by 2026-04-05"),
    ]

    citations = service._infer_citations_from_texts(
        ["Prepare architecture document", "Dana", "2026-04-05"],
        hits,
        max_citations=2,
    )

    assert citations
    assert citations[0].source_id == "S2"