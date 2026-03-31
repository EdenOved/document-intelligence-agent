from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

settings = get_settings()


class CohereReranker:
    def __init__(self) -> None:
        if not settings.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is required for reranking")
        self.base_url = "https://api.cohere.com/v2/rerank"

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        payload = {
            "model": settings.cohere_rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {settings.cohere_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        return body.get("results", [])
