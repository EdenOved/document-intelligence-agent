from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PendingApproval, ToolCall
from app.integrations.monday import MondayService
from app.integrations.slack import SlackService
from app.schemas import ApprovalDecisionResponse, ApprovalPreview, PostActionName, SummaryResponse, ActionItemsResponse


class ApprovalService:
    async def create_pending_approval(
        self,
        db: AsyncSession,
        run_id: UUID,
        action: PostActionName,
        preview_payload: dict[str, Any],
    ) -> ApprovalPreview:
        approval = PendingApproval(
            run_id=run_id,
            action_name=action,
            status="pending",
            preview_payload=preview_payload,
        )
        db.add(approval)
        await db.flush()

        return ApprovalPreview(
            approval_id=approval.id,
            run_id=approval.run_id,
            action=action,
            status="pending",
            preview_payload=approval.preview_payload,
            edited_payload=None,
            created_at=approval.created_at,
            reviewed_at=None,
        )

    async def decide(
        self,
        db: AsyncSession,
        approval_id: UUID,
        decision: str,
        edited_payload: dict[str, Any] | None = None,
    ) -> ApprovalDecisionResponse:
        approval = await db.scalar(
            select(PendingApproval).where(PendingApproval.id == approval_id)
        )
        if not approval:
            raise RuntimeError(f"Pending approval {approval_id} was not found")

        if approval.status != "pending":
            raise RuntimeError(f"Pending approval {approval_id} is already {approval.status}")

        if decision == "reject":
            approval.status = "rejected"
            approval.reviewed_at = datetime.now(timezone.utc)
            await db.commit()
            return ApprovalDecisionResponse(
                approval_id=approval.id,
                action=approval.action_name,
                status="rejected",
                message="Post action was rejected by reviewer",
                execution_payload={},
            )

        if decision not in {"approve", "edit_and_approve"}:
            raise RuntimeError(f"Unsupported approval decision: {decision}")

        if decision == "edit_and_approve":
            if not edited_payload:
                raise RuntimeError("edited_payload is required for edit_and_approve")
            approval.edited_payload = edited_payload

        payload_to_execute = approval.edited_payload or approval.preview_payload
        execution_payload = await self._execute_action(approval.action_name, payload_to_execute)

        approval.status = "executed"
        approval.execution_payload = execution_payload
        approval.reviewed_at = datetime.now(timezone.utc)

        tool_call = ToolCall(
            run_id=approval.run_id,
            tool_name=approval.action_name,
            arguments={"approval_id": str(approval.id), "approved": True},
            output_payload=execution_payload,
            success=True,
        )
        db.add(tool_call)

        await db.commit()

        return ApprovalDecisionResponse(
            approval_id=approval.id,
            action=approval.action_name,
            status="executed",
            message="Post action approved and executed successfully",
            execution_payload=execution_payload,
        )

    async def _execute_action(
        self,
        action_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if action_name == "send_summary_to_slack":
            summary = SummaryResponse.model_validate(payload["summary_response"])
            slack_service = SlackService()
            return await slack_service.send_summary(summary, dry_run=False)

        if action_name == "create_monday_items":
            action_items = ActionItemsResponse.model_validate(payload["action_items_response"])
            monday_service = MondayService()
            return await monday_service.create_items(action_items, dry_run=False)

        raise RuntimeError(f"Unsupported approval action: {action_name}")