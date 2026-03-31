from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.agent.orchestrator import AgentOrchestrator
from app.db.models import Document, PendingApproval, ToolCall
from app.db.session import AsyncSessionLocal
from app.evals.judge import LLMJudge
from app.llm import StructuredLLMService
from app.retrieval.pipeline import RetrievalService

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
REPORT_PATH = BASE_DIR / "eval_report.json"


@dataclass
class EvalCase:
    id: str
    task_type: str
    query: str
    filenames: list[str]
    expected_behavior: list[str]
    post_actions: list[str] | None = None


def load_dataset() -> list[EvalCase]:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            task_type=item["task_type"],
            query=item["query"],
            filenames=item["filenames"],
            expected_behavior=item["expected_behavior"],
            post_actions=item.get("post_actions"),
        )
        for item in raw
    ]


async def resolve_document_ids(filenames: list[str]) -> list[UUID]:
    async with AsyncSessionLocal() as db:
        stmt = select(Document.id, Document.filename).where(Document.filename.in_(filenames))
        rows = (await db.execute(stmt)).all()

    mapping = {filename: doc_id for doc_id, filename in rows}
    missing = [name for name in filenames if name not in mapping]
    if missing:
        raise RuntimeError(f"Missing documents for evals: {missing}")

    return [mapping[name] for name in filenames]


def deterministic_checks(task_type: str, output_payload: dict) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    if task_type == "summary":
        checks["has_summary"] = bool(output_payload.get("summary"))
        checks["has_citations"] = len(output_payload.get("citations", [])) > 0
        checks["passed"] = checks["has_summary"] and checks["has_citations"]
        return checks

    if task_type == "action_items":
        items = output_payload.get("action_items", [])
        checks["has_action_items"] = len(items) > 0
        checks["all_items_grounded"] = all(item.get("citations") for item in items) if items else False
        checks["passed"] = checks["has_action_items"] and checks["all_items_grounded"]
        return checks

    if task_type == "comparison":
        checks["has_overview"] = bool(output_payload.get("overview"))
        checks["has_overview_citations"] = len(output_payload.get("overview_citations", [])) > 0
        checks["has_conclusion"] = bool(output_payload.get("conclusion"))
        checks["has_conclusion_citations"] = len(output_payload.get("conclusion_citations", [])) > 0
        checks["passed"] = (
            checks["has_overview"]
            and checks["has_overview_citations"]
            and checks["has_conclusion"]
            and checks["has_conclusion_citations"]
        )
        return checks

    if task_type == "approval_flow":
        checks["awaiting_approval"] = output_payload.get("status") == "awaiting_approval"
        results = output_payload.get("post_action_results", [])
        checks["has_pending_approval_result"] = bool(results) and results[0].get("status") == "pending_approval"
        checks["passed"] = checks["awaiting_approval"] and checks["has_pending_approval_result"]
        return checks

    checks["passed"] = False
    return checks


async def run_summary_case(case: EvalCase) -> dict[str, Any]:
    document_ids = await resolve_document_ids(case.filenames)

    async with AsyncSessionLocal() as db:
        retrieval = RetrievalService()
        llm = StructuredLLMService()
        search_result = await retrieval.search(
            db=db,
            query=case.query,
            document_ids=document_ids,
            top_k=5,
            rerank_top_n=3,
        )
        result = await llm.summarize(case.query, document_ids, search_result.hits)

    return result.model_dump(mode="json")


async def run_action_items_case(case: EvalCase) -> dict[str, Any]:
    document_ids = await resolve_document_ids(case.filenames)

    async with AsyncSessionLocal() as db:
        retrieval = RetrievalService()
        llm = StructuredLLMService()
        search_result = await retrieval.search(
            db=db,
            query=case.query,
            document_ids=document_ids,
            top_k=5,
            rerank_top_n=3,
        )
        result = await llm.extract_action_items(case.query, document_ids, search_result.hits)

    return result.model_dump(mode="json")


async def run_comparison_case(case: EvalCase) -> dict[str, Any]:
    document_ids = await resolve_document_ids(case.filenames)

    async with AsyncSessionLocal() as db:
        retrieval = RetrievalService()
        llm = StructuredLLMService()
        search_result = await retrieval.search_for_comparison(
            db=db,
            query=case.query,
            document_ids=document_ids,
            top_k=8,
            rerank_top_n=4,
        )
        result = await llm.compare(case.query, document_ids, search_result.hits)

    return result.model_dump(mode="json")


async def run_approval_case(case: EvalCase) -> dict[str, Any]:
    document_ids = await resolve_document_ids(case.filenames)

    async with AsyncSessionLocal() as db:
        orchestrator = AgentOrchestrator(AsyncSessionLocal)
        result = await orchestrator.run(
            query=case.query,
            db=db,
            document_ids=document_ids,
            post_actions=case.post_actions or [],
            dry_run=False,
            require_approval=True,
        )

        pending_stmt = select(PendingApproval.id, PendingApproval.status).order_by(PendingApproval.created_at.desc())
        pending_rows = (await db.execute(pending_stmt)).all()

        tool_stmt = select(ToolCall.tool_name, ToolCall.success).order_by(ToolCall.created_at.desc())
        tool_rows = (await db.execute(tool_stmt)).all()

    payload = result.model_dump(mode="json")
    payload["pending_approvals_snapshot"] = [
        {"approval_id": str(row[0]), "status": row[1]} for row in pending_rows[:5]
    ]
    payload["tool_calls_snapshot"] = [
        {"tool_name": row[0], "success": row[1]} for row in tool_rows[:10]
    ]
    return payload


async def run_case(case: EvalCase, judge: LLMJudge) -> dict[str, Any]:
    if case.task_type == "summary":
        output_payload = await run_summary_case(case)
    elif case.task_type == "action_items":
        output_payload = await run_action_items_case(case)
    elif case.task_type == "comparison":
        output_payload = await run_comparison_case(case)
    elif case.task_type == "approval_flow":
        output_payload = await run_approval_case(case)
    else:
        raise RuntimeError(f"Unsupported eval task type: {case.task_type}")

    checks = deterministic_checks(case.task_type, output_payload)

    judge_result = None
    if case.task_type in {"summary", "action_items", "comparison"} and checks["passed"]:
        judge_result = await judge.judge(
            task_type=case.task_type,
            query=case.query,
            expected_behavior=case.expected_behavior,
            output_payload=output_payload,
        )

    final_pass = checks["passed"]
    if judge_result is not None:
        final_pass = (
            final_pass
            and judge_result.passed
            and judge_result.relevance_score >= 4
            and judge_result.groundedness_score >= 4
            and judge_result.completeness_score >= 4
        )

    return {
        "id": case.id,
        "task_type": case.task_type,
        "query": case.query,
        "filenames": case.filenames,
        "deterministic_checks": checks,
        "judge_result": judge_result.model_dump(mode="json") if judge_result else None,
        "passed": final_pass,
        "output_payload": output_payload,
    }


async def main() -> None:
    dataset = load_dataset()
    judge = LLMJudge()

    results = []
    for case in dataset:
        result = await run_case(case, judge)
        results.append(result)

    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item["passed"]),
        "failed_cases": sum(1 for item in results if not item["passed"]),
    }

    report = {
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation report written to: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())