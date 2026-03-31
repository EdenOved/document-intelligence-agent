from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.tools import build_tools
from app.config import get_settings
from app.db.models import AgentRun, ToolCall
from app.integrations.monday import MondayService
from app.integrations.slack import SlackService
from app.schemas import (
    ActionItemsResponse,
    AgentRunResponse,
    PostActionName,
    PostActionResult,
    SummaryResponse,
)
from app.services.approval_service import ApprovalService

settings = get_settings()


class AgentOrchestrator:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agent orchestration")
        self.session_factory = session_factory
        self.tools = build_tools(session_factory)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model = ChatOpenAI(
            model=settings.openai_reasoning_model,
            api_key=settings.openai_api_key,
            temperature=0,
            use_responses_api=True,
        ).bind_tools(self.tools)

    async def run(
        self,
        query: str,
        db: AsyncSession,
        document_ids: list[UUID] | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        post_actions: list[PostActionName] | None = None,
        dry_run: bool = True,
        require_approval: bool = False,
    ) -> AgentRunResponse:
        run = AgentRun(user_query=query, status="started")
        db.add(run)
        await db.flush()

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "query": query,
                        "document_ids": [str(doc_id) for doc_id in document_ids] if document_ids else None,
                        "top_k": top_k,
                        "rerank_top_n": rerank_top_n,
                    }
                )
            ),
        ]

        post_action_results: list[PostActionResult] = []

        try:
            for iteration in range(1, settings.max_agent_iterations + 1):
                decision = await self.model.ainvoke(messages)
                if not decision.tool_calls:
                    return await self._fail_run(
                        db=db,
                        run=run,
                        iteration=iteration,
                        message="Model returned no tool call",
                    )

                tool_call = decision.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = dict(tool_call.get("args", {}))
                tool = self.tools_by_name[tool_name]

                if document_ids and "document_ids" not in tool_args and tool_name != "search_documents":
                    tool_args["document_ids"] = [str(doc_id) for doc_id in document_ids]
                if top_k is not None and "top_k" not in tool_args:
                    tool_args["top_k"] = top_k
                if rerank_top_n is not None and "rerank_top_n" not in tool_args:
                    tool_args["rerank_top_n"] = rerank_top_n

                db_tool_call = ToolCall(
                    run_id=run.id,
                    tool_name=tool_name,
                    arguments=tool_args,
                    success=False,
                )
                db.add(db_tool_call)
                await db.flush()

                try:
                    result = await tool.ainvoke(tool_args)
                except Exception as exc:
                    db_tool_call.error_message = str(exc)
                    run.selected_tool = tool_name
                    run.tool_arguments = tool_args
                    run.iterations = iteration
                    return await self._fail_run(
                        db=db,
                        run=run,
                        iteration=iteration,
                        message=f"Tool {tool_name} failed: {exc}",
                    )

                db_tool_call.output_payload = result
                run.selected_tool = tool_name
                run.tool_arguments = tool_args
                run.iterations = iteration
                run.final_response = result

                is_useful = not self._is_empty_result(tool_name, result)
                db_tool_call.success = is_useful

                if is_useful:
                    response_type = self._tool_to_response_type(tool_name)

                    post_action_results, awaiting_approval = await self._execute_post_actions(
                        db=db,
                        run_id=run.id,
                        response_type=response_type,
                        result=result,
                        post_actions=post_actions or [],
                        dry_run=dry_run,
                        require_approval=require_approval,
                    )

                    run.status = "awaiting_approval" if awaiting_approval else "completed"
                    await db.commit()

                    return self._build_response(
                        run_id=run.id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        iteration=iteration,
                        result=result,
                        created_at=run.created_at,
                        post_action_results=post_action_results,
                        status=run.status,
                    )

                db_tool_call.error_message = "Tool returned an empty or weak result"

                if iteration >= settings.max_agent_iterations:
                    return await self._fail_run(
                        db=db,
                        run=run,
                        iteration=iteration,
                        message=f"Tool {tool_name} returned an empty or weak result",
                    )

                messages.append(decision)
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "feedback": "The previous tool result was empty or too weak. Retry with a better query or narrower scope.",
                                "previous_result": result,
                            }
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )

        except RuntimeError:
            raise
        except Exception as exc:
            return await self._fail_run(
                db=db,
                run=run,
                iteration=run.iterations or 0,
                message=f"Unexpected orchestration error: {exc}",
            )

        return await self._fail_run(
            db=db,
            run=run,
            iteration=run.iterations or 0,
            message="Max iterations reached without a useful result",
        )

    async def _execute_post_actions(
        self,
        db: AsyncSession,
        run_id: UUID,
        response_type: str,
        result: dict[str, Any],
        post_actions: list[PostActionName],
        dry_run: bool,
        require_approval: bool,
    ) -> tuple[list[PostActionResult], bool]:
        results: list[PostActionResult] = []
        awaiting_approval = False
        approval_service = ApprovalService()

        for action in post_actions:
            if action == "send_summary_to_slack":
                if response_type != "summary":
                    results.append(
                        PostActionResult(
                            action=action,
                            status="skipped",
                            message="send_summary_to_slack requires a summary response",
                            payload={},
                        )
                    )
                    continue

                summary_result = SummaryResponse.model_validate(result)

                if require_approval:
                    preview_payload = {
                        "summary_response": summary_result.model_dump(mode="json"),
                    }
                    approval = await approval_service.create_pending_approval(
                        db=db,
                        run_id=run_id,
                        action=action,
                        preview_payload=preview_payload,
                    )
                    awaiting_approval = True
                    results.append(
                        PostActionResult(
                            action=action,
                            status="pending_approval",
                            message="Approval required before sending summary to Slack",
                            payload=approval.model_dump(mode="json"),
                        )
                    )
                    continue

                db_tool_call = ToolCall(
                    run_id=run_id,
                    tool_name=action,
                    arguments={"dry_run": dry_run},
                    success=False,
                )
                db.add(db_tool_call)
                await db.flush()

                try:
                    slack_service = SlackService()
                    payload = await slack_service.send_summary(summary_result, dry_run=dry_run)
                    db_tool_call.output_payload = payload
                    db_tool_call.success = payload["status"] in {"dry_run", "sent"}
                    results.append(
                        PostActionResult(
                            action=action,
                            status="dry_run" if dry_run else "success",
                            message=payload["message"],
                            payload=payload,
                        )
                    )
                except Exception as exc:
                    db_tool_call.error_message = str(exc)
                    results.append(
                        PostActionResult(
                            action=action,
                            status="failed",
                            message=str(exc),
                            payload={},
                        )
                    )
                continue

            if action == "create_monday_items":
                if response_type != "action_items":
                    results.append(
                        PostActionResult(
                            action=action,
                            status="skipped",
                            message="create_monday_items requires an action_items response",
                            payload={},
                        )
                    )
                    continue

                action_items_result = ActionItemsResponse.model_validate(result)

                if require_approval:
                    preview_payload = {
                        "action_items_response": action_items_result.model_dump(mode="json"),
                    }
                    approval = await approval_service.create_pending_approval(
                        db=db,
                        run_id=run_id,
                        action=action,
                        preview_payload=preview_payload,
                    )
                    awaiting_approval = True
                    results.append(
                        PostActionResult(
                            action=action,
                            status="pending_approval",
                            message="Approval required before creating Monday items",
                            payload=approval.model_dump(mode="json"),
                        )
                    )
                    continue

                db_tool_call = ToolCall(
                    run_id=run_id,
                    tool_name=action,
                    arguments={"dry_run": dry_run},
                    success=False,
                )
                db.add(db_tool_call)
                await db.flush()

                try:
                    monday_service = MondayService()
                    payload = await monday_service.create_items(action_items_result, dry_run=dry_run)
                    db_tool_call.output_payload = payload
                    db_tool_call.success = payload["status"] in {"dry_run", "created"}
                    results.append(
                        PostActionResult(
                            action=action,
                            status="dry_run" if dry_run else "success",
                            message=payload["message"],
                            payload=payload,
                        )
                    )
                except Exception as exc:
                    db_tool_call.error_message = str(exc)
                    results.append(
                        PostActionResult(
                            action=action,
                            status="failed",
                            message=str(exc),
                            payload={},
                        )
                    )
                continue

            results.append(
                PostActionResult(
                    action=action,
                    status="failed",
                    message=f"Unsupported post action: {action}",
                    payload={},
                )
            )

        return results, awaiting_approval

    @staticmethod
    def _tool_to_response_type(tool_name: str) -> str:
        mapping = {
            "search_documents": "search",
            "summarize_document": "summary",
            "extract_action_items": "action_items",
            "compare_documents": "comparison",
        }
        return mapping[tool_name]

    @staticmethod
    def _is_empty_result(tool_name: str, result: dict[str, Any]) -> bool:
        if tool_name == "search_documents":
            return result.get("total_hits", 0) == 0

        if tool_name == "summarize_document":
            return (not result.get("summary")) or len(result.get("citations", [])) == 0

        if tool_name == "extract_action_items":
            items = result.get("action_items", [])
            if len(items) == 0:
                return True
            return not any(item.get("citations") for item in items)

        if tool_name == "compare_documents":
            return (
                not result.get("overview")
                or len(result.get("overview_citations", [])) == 0
                or not result.get("conclusion")
                or len(result.get("conclusion_citations", [])) == 0
            )

        return False

    @staticmethod
    def _build_response(
        run_id: UUID,
        tool_name: str,
        tool_args: dict[str, Any],
        iteration: int,
        result: dict[str, Any],
        created_at: datetime | None,
        post_action_results: list[PostActionResult],
        status: str,
    ) -> AgentRunResponse:
        response_type_map = {
            "search_documents": "search",
            "summarize_document": "summary",
            "extract_action_items": "action_items",
            "compare_documents": "comparison",
        }
        return AgentRunResponse(
            run_id=run_id,
            status=status,
            selected_tool=tool_name,
            tool_arguments=tool_args,
            iterations=iteration,
            response_type=response_type_map[tool_name],
            response=result,
            post_action_results=post_action_results,
            created_at=created_at or datetime.now(timezone.utc),
        )

    async def _fail_run(
        self,
        db: AsyncSession,
        run: AgentRun,
        iteration: int,
        message: str,
    ) -> AgentRunResponse:
        run.status = "failed"
        run.iterations = iteration
        run.error_message = message
        await db.commit()
        raise RuntimeError(message)