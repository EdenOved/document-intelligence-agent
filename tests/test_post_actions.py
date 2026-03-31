from uuid import uuid4

from app.agent.orchestrator import AgentOrchestrator
from app.schemas import PostActionResult


def test_summary_result_without_citations_is_considered_weak():
    result = {
        "document_ids": [str(uuid4())],
        "summary": "Some summary",
        "key_points": ["A"],
        "citations": [],
    }
    assert AgentOrchestrator._is_empty_result("summarize_document", result) is True


def test_action_items_result_without_items_is_considered_weak():
    result = {
        "document_ids": [str(uuid4())],
        "action_items": [],
    }
    assert AgentOrchestrator._is_empty_result("extract_action_items", result) is True


def test_compare_result_without_grounding_is_considered_weak():
    result = {
        "document_ids": [str(uuid4()), str(uuid4())],
        "overview": "Overview",
        "overview_citations": [],
        "similarities": [],
        "differences": [],
        "conclusion": "Conclusion",
        "conclusion_citations": [],
    }
    assert AgentOrchestrator._is_empty_result("compare_documents", result) is True


def test_post_action_result_schema():
    result = PostActionResult(
        action="send_summary_to_slack",
        status="dry_run",
        message="ok",
        payload={"x": 1},
    )
    assert result.action == "send_summary_to_slack"