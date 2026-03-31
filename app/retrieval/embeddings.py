from __future__ import annotations

from collections.abc import Sequence

from openai import AsyncOpenAI

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding generation")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])
        return vectors[0]
