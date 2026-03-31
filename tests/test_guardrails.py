from uuid import uuid4

from app.agent.orchestrator import AgentOrchestrator
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


def test_has_actionable_evidence_detects_real_action_section():
    service = StructuredLLMService.__new__(StructuredLLMService)
    hits = [
        _make_hit(
            "S1",
            "Action items:\n- Dana to prepare architecture document by 2026-04-05\n- Amir to validate Docker environment by 2026-04-02",
        )
    ]
    assert service._has_actionable_evidence(hits) is True


def test_has_actionable_evidence_rejects_negative_noise_text():
    service = StructuredLLMService.__new__(StructuredLLMService)
    hits = [
        _make_hit(
            "S1",
            "ZEBRA OMEGA 914\nNo real owner, no real deadlines, no action plan.",
        )
    ]
    assert service._has_actionable_evidence(hits) is False


def test_strip_inline_source_ids_removes_bracketed_ids():
    assert StructuredLLMService._strip_inline_source_ids("Summary text [S1, S2]") == "Summary text"


def test_agent_empty_result_detection_for_summary_requires_citations():
    result = {
        "summary": "No information is available.",
        "key_points": [],
        "citations": [],
    }
    assert AgentOrchestrator._is_empty_result("summarize_document", result) is True


def test_agent_empty_result_detection_for_compare_requires_grounding():
    result = {
        "overview": "Some comparison",
        "overview_citations": [],
        "similarities": [],
        "differences": [],
        "conclusion": "Some conclusion",
        "conclusion_citations": [],
    }
    assert AgentOrchestrator._is_empty_result("compare_documents", result) is True