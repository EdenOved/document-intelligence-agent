from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.orchestrator import AgentOrchestrator
from app.db.session import AsyncSessionLocal, get_db
from app.ingestion.pipeline import IngestionService
from app.retrieval.pipeline import RetrievalService
from app.schemas import (
    ActionItemsRequest,
    ActionItemsResponse,
    AgentQueryRequest,
    AgentRunResponse,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    CompareRequest,
    DocumentComparisonResponse,
    DocumentUploadResponse,
    SearchRequest,
    SearchResult,
    SummaryRequest,
    SummaryResponse,
)
from app.services.approval_service import ApprovalService

router = APIRouter()


def _service_unavailable(exc: RuntimeError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    try:
        service = IngestionService()
        return await service.ingest(file, db)
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/search", response_model=SearchResult)
async def search_documents(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResult:
    try:
        service = RetrievalService()
        return await service.search(
            db=db,
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_documents(
    request: SummaryRequest,
    db: AsyncSession = Depends(get_db),
) -> SummaryResponse:
    try:
        retrieval = RetrievalService()
        from app.llm import StructuredLLMService

        search_result = await retrieval.search(
            db=db,
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
        service = StructuredLLMService()
        return await service.summarize(request.query, request.document_ids, search_result.hits)
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/action-items", response_model=ActionItemsResponse)
async def extract_action_items(
    request: ActionItemsRequest,
    db: AsyncSession = Depends(get_db),
) -> ActionItemsResponse:
    try:
        retrieval = RetrievalService()
        from app.llm import StructuredLLMService

        search_result = await retrieval.search(
            db=db,
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
        service = StructuredLLMService()
        return await service.extract_action_items(request.query, request.document_ids, search_result.hits)
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/compare", response_model=DocumentComparisonResponse)
async def compare_documents(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
) -> DocumentComparisonResponse:
    try:
        retrieval = RetrievalService()
        from app.llm import StructuredLLMService

        search_result = await retrieval.search_for_comparison(
            db=db,
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
        )
        service = StructuredLLMService()
        return await service.compare(request.query, request.document_ids, search_result.hits)
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentRunResponse:
    try:
        orchestrator = AgentOrchestrator(AsyncSessionLocal)
        return await orchestrator.run(
            query=request.query,
            db=db,
            document_ids=request.document_ids,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
            post_actions=request.post_actions,
            dry_run=request.dry_run,
            require_approval=request.require_approval,
        )
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc


@router.post("/post-actions/decision", response_model=ApprovalDecisionResponse)
async def post_action_decision(
    request: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> ApprovalDecisionResponse:
    try:
        service = ApprovalService()
        return await service.decide(
            db=db,
            approval_id=request.approval_id,
            decision=request.decision,
            edited_payload=request.edited_payload,
        )
    except RuntimeError as exc:
        raise _service_unavailable(exc) from exc