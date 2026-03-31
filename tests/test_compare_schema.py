from uuid import uuid4

from app.schemas import Citation, ComparisonFinding, DocumentComparisonResponse


def _citation(source_id: str) -> Citation:
    return Citation(
        source_id=source_id,
        document_id=uuid4(),
        filename="doc.txt",
        chunk_id=uuid4(),
        page_number=1,
        excerpt="evidence",
    )


def test_comparison_response_supports_overview_and_conclusion_citations():
    response = DocumentComparisonResponse(
        document_ids=[uuid4(), uuid4()],
        overview="Both documents discuss project planning.",
        overview_citations=[_citation("S1")],
        similarities=[
            ComparisonFinding(
                topic="Planning",
                summary="Both mention planning.",
                citations=[_citation("S1")],
            )
        ],
        differences=[
            ComparisonFinding(
                topic="Scope",
                summary="One focuses on backend, the other on operations.",
                citations=[_citation("S2")],
            )
        ],
        conclusion="They overlap on process but differ in scope.",
        conclusion_citations=[_citation("S2")],
    )

    assert response.overview_citations[0].source_id == "S1"
    assert response.conclusion_citations[0].source_id == "S2"