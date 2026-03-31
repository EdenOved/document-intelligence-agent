from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_debug: bool = True
    app_name: str = "document-intelligence-agent"

    database_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/document_intelligence"

    openai_api_key: str | None = None
    openai_reasoning_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"

    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    chunk_size: int = 1200
    chunk_overlap: int = 200

    default_top_k: int = 5
    default_rerank_top_n: int = 3
    max_agent_iterations: int = 2

    # Slack
    slack_webhook_url: str | None = None
    slack_default_channel_name: str | None = None

    # Monday
    monday_api_token: str | None = None
    monday_api_version: str = "2026-01"
    monday_board_id: str | None = None
    monday_group_id: str | None = None

    monday_owner_column_id: str | None = None
    monday_due_date_column_id: str | None = None
    monday_priority_column_id: str | None = None
    monday_source_column_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()