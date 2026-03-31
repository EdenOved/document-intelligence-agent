from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings
from app.schemas import SummaryResponse

settings = get_settings()


class SlackService:
    def __init__(self) -> None:
        self.webhook_url = settings.slack_webhook_url

    async def send_summary(
        self,
        result: SummaryResponse,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        if not result.summary:
            raise RuntimeError("Cannot send empty summary to Slack")

        if not result.citations:
            raise RuntimeError("Cannot send Slack summary without citations")

        payload = self._build_payload(result)

        if dry_run:
            return {
                "status": "dry_run",
                "message": "Slack message was not sent because dry_run=true",
                "payload": payload,
            }

        if not self.webhook_url:
            raise RuntimeError("SLACK_WEBHOOK_URL is required for Slack delivery")

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()

        return {
            "status": "sent",
            "message": "Summary sent to Slack successfully",
            "payload": payload,
        }

    def _build_payload(self, result: SummaryResponse) -> dict[str, Any]:
        key_points_text = "\n".join(f"• {point}" for point in result.key_points[:8]) or "No key points."
        sources_text = "\n".join(
            f"• {citation.filename} (page {citation.page_number or 1}, {citation.source_id})"
            for citation in result.citations[:5]
        )

        summary_text = result.summary.strip()
        if len(summary_text) > 2500:
            summary_text = summary_text[:2497] + "..."

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Document Summary"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Summary*\n{summary_text}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Key points*\n{key_points_text}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Sources*\n{sources_text}"},
            },
        ]

        return {
            "text": f"Document summary: {summary_text}",
            "blocks": blocks,
        }