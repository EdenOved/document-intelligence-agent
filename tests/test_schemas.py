from uuid import uuid4

from app.schemas import Citation, SummaryResponse


def test_summary_response_schema():
    citation = Citation(
        source_id="S1",
        document_id=uuid4(),
        filename="a.pdf",
        chunk_id=uuid4(),
        page_number=1,
        excerpt="Important evidence",
    )
    response = SummaryResponse(
        document_ids=[uuid4()],
        summary="A short summary",
        key_points=["One", "Two"],
        citations=[citation],
    )

    assert response.summary == "A short summary"
    assert response.citations[0].source_id == "S1"
