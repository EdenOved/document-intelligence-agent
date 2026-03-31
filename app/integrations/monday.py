from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings
from app.schemas import ActionItemsResponse

settings = get_settings()


class MondayService:
    def __init__(self) -> None:
        self.api_token = settings.monday_api_token
        self.api_version = settings.monday_api_version
        self.board_id = settings.monday_board_id
        self.group_id = settings.monday_group_id

    async def create_items(
        self,
        result: ActionItemsResponse,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        if not result.action_items:
            raise RuntimeError("Cannot create Monday items from empty action items response")

        valid_items = [item for item in result.action_items if item.citations]
        if not valid_items:
            raise RuntimeError("Cannot create Monday items without grounded action items")

        prepared_items = [self._build_item_payload(item) for item in valid_items]

        if dry_run:
            return {
                "status": "dry_run",
                "message": "Monday items were not created because dry_run=true",
                "items": prepared_items,
            }

        if not self.api_token:
            raise RuntimeError("MONDAY_API_TOKEN is required for Monday integration")
        if not self.board_id:
            raise RuntimeError("MONDAY_BOARD_ID is required for Monday integration")

        created: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for prepared in prepared_items:
                mutation = """
                mutation CreateItem($boardId: ID!, $groupId: String, $itemName: String!, $columnValues: JSON) {
                  create_item(board_id: $boardId, group_id: $groupId, item_name: $itemName, column_values: $columnValues) {
                    id
                    name
                  }
                }
                """

                variables = {
                    "boardId": self.board_id,
                    "groupId": self.group_id,
                    "itemName": prepared["item_name"],
                    "columnValues": json.dumps(prepared["column_values"]),
                }

                response = await client.post(
                    "https://api.monday.com/v2",
                    headers={
                        "Authorization": self.api_token,
                        "Content-Type": "application/json",
                        "API-Version": self.api_version,
                    },
                    json={"query": mutation, "variables": variables},
                )
                response.raise_for_status()
                payload = response.json()

                if payload.get("errors"):
                    raise RuntimeError(f"Monday API returned errors: {payload['errors']}")

                created.append(payload["data"]["create_item"])

        return {
            "status": "created",
            "message": "Monday items created successfully",
            "items": created,
        }

    def _build_item_payload(self, item) -> dict[str, Any]:
        column_values: dict[str, Any] = {}

        if settings.monday_owner_column_id and item.owner:
            column_values[settings.monday_owner_column_id] = item.owner

        if settings.monday_due_date_column_id and item.due_date:
            column_values[settings.monday_due_date_column_id] = {"date": item.due_date}

        if settings.monday_priority_column_id:
            column_values[settings.monday_priority_column_id] = item.priority

        if settings.monday_source_column_id and item.citations:
            citation = item.citations[0]
            column_values[settings.monday_source_column_id] = (
                f"{citation.filename} | {citation.source_id} | page {citation.page_number or 1}"
            )

        return {
            "item_name": item.title,
            "column_values": column_values,
        }