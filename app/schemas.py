from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str = Field(description="Short source reference used during prompting, e.g. S1")
    document_id: UUID
    filename: str
    chunk_id: UUID
    page_number: int | None = None
    excerpt: str


class SearchHit(BaseModel):
    document_id: UUID
    filename: str
    chunk_id: UUID
    content: str
    score: float
    rerank_score: float | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citation: Citation


class SearchResult(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchHit]


class SummaryResponse(BaseModel):
    document_ids: list[UUID]
    summary: str
    key_points: list[str]
    citations: list[Citation]


class ActionItem(BaseModel):
    title: str
    owner: str | None = None
    due_date: str | None = None
    priority: Literal["low", "medium", "high"] = "medium"
    rationale: str
    citations: list[Citation]


class ActionItemsResponse(BaseModel):
    document_ids: list[UUID]
    action_items: list[ActionItem]


class ComparisonFinding(BaseModel):
    topic: str
    summary: str
    citations: list[Citation]


class DocumentComparisonResponse(BaseModel):
    document_ids: list[UUID]
    overview: str
    overview_citations: list[Citation] = Field(default_factory=list)
    similarities: list[ComparisonFinding]
    differences: list[ComparisonFinding]
    conclusion: str
    conclusion_citations: list[Citation] = Field(default_factory=list)


PostActionName = Literal["send_summary_to_slack", "create_monday_items"]
ApprovalStatus = Literal["pending", "approved", "rejected", "executed"]


class PostActionResult(BaseModel):
    action: PostActionName
    status: Literal["dry_run", "success", "skipped", "failed", "pending_approval"]
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    run_id: UUID
    status: Literal["completed", "failed", "awaiting_approval"]
    selected_tool: str
    tool_arguments: dict[str, Any]
    iterations: int
    response_type: Literal["search", "summary", "action_items", "comparison"]
    response: SearchResult | SummaryResponse | ActionItemsResponse | DocumentComparisonResponse
    post_action_results: list[PostActionResult] = Field(default_factory=list)
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    chunks_created: int
    status: Literal["completed", "failed"]


class ParsedPage(BaseModel):
    page_number: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    filename: str
    source_type: Literal["pdf", "docx", "txt"]
    pages: list[ParsedPage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkPayload(BaseModel):
    page_number: int | None = None
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentQueryRequest(BaseModel):
    query: str
    document_ids: list[UUID] | None = None
    top_k: int | None = None
    rerank_top_n: int | None = None
    post_actions: list[PostActionName] = Field(default_factory=list)
    dry_run: bool = True
    require_approval: bool = False


class SearchRequest(BaseModel):
    query: str
    document_ids: list[UUID] | None = None
    top_k: int | None = None
    rerank_top_n: int | None = None


class SummaryRequest(BaseModel):
    query: str
    document_ids: list[UUID]
    top_k: int | None = None
    rerank_top_n: int | None = None


class ActionItemsRequest(BaseModel):
    query: str
    document_ids: list[UUID]
    top_k: int | None = None
    rerank_top_n: int | None = None


class CompareRequest(BaseModel):
    query: str
    document_ids: list[UUID]
    top_k: int | None = None
    rerank_top_n: int | None = None


class ApprovalPreview(BaseModel):
    approval_id: UUID
    run_id: UUID
    action: PostActionName
    status: ApprovalStatus
    preview_payload: dict[str, Any]
    edited_payload: dict[str, Any] | None = None
    created_at: datetime
    reviewed_at: datetime | None = None


class ApprovalDecisionRequest(BaseModel):
    approval_id: UUID
    decision: Literal["approve", "reject", "edit_and_approve"]
    edited_payload: dict[str, Any] | None = None


class ApprovalDecisionResponse(BaseModel):
    approval_id: UUID
    action: PostActionName
    status: Literal["approved", "rejected", "executed", "failed"]
    message: str
    execution_payload: dict[str, Any] = Field(default_factory=dict)