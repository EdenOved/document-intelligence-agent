from app.evals.runner import deterministic_checks


def test_summary_deterministic_checks_pass():
    payload = {
        "summary": "This is a summary",
        "citations": [{"source_id": "S1"}],
    }
    result = deterministic_checks("summary", payload)
    assert result["passed"] is True


def test_action_items_deterministic_checks_fail_on_empty():
    payload = {
        "action_items": [],
    }
    result = deterministic_checks("action_items", payload)
    assert result["passed"] is False


def test_comparison_deterministic_checks_require_grounding():
    payload = {
        "overview": "Overview",
        "overview_citations": [],
        "conclusion": "Conclusion",
        "conclusion_citations": [],
    }
    result = deterministic_checks("comparison", payload)
    assert result["passed"] is False


def test_approval_flow_checks_pass():
    payload = {
        "status": "awaiting_approval",
        "post_action_results": [
            {
                "status": "pending_approval",
            }
        ],
    }
    result = deterministic_checks("approval_flow", payload)
    assert result["passed"] is True