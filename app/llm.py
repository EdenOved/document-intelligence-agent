from __future__ import annotations

import re
from collections import Counter
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.agent.prompts import (
    ACTION_ITEMS_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from app.config import get_settings
from app.schemas import (
    ActionItem,
    ActionItemsResponse,
    Citation,
    ComparisonFinding,
    DocumentComparisonResponse,
    SearchHit,
    SummaryResponse,
)

settings = get_settings()

_SOURCE_ID_PATTERN = re.compile(r"S\d+", re.IGNORECASE)
_INLINE_SOURCE_PATTERN = re.compile(r"\s*\[?\bS\d+(?:\s*,\s*S\d+)*\b\]?\s*", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_ACTION_SECTION_PATTERN = re.compile(r"\b(action items?|next steps?|todo|tasks?)\b", re.IGNORECASE)
_ACTION_LINE_PATTERN = re.compile(
    r"(?im)^\s*[-*•]\s+.+(?:\bto\b|\bby\s+\d{4}-\d{2}-\d{2}\b|\bowner\b|\bdeadline\b).*$"
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "with", "will", "using", "use", "document", "documents",
    "project", "summary", "compare", "comparison", "action", "items",
    "item", "main",
}

_GENERIC_SIMILARITY_TOKENS = {
    "format", "document", "documents", "text", "written", "note",
    "notes", "reference", "summary", "structure", "structured",
    "meeting", "meetings", "deadline", "deadlines",
}


class SummaryDraft(BaseModel):
    summary: str
    key_points: list[str]
    citation_ids: list[str] = Field(default_factory=list)


class ActionItemDraft(BaseModel):
    title: str
    owner: str | None = None
    due_date: str | None = None
    priority: str = "medium"
    rationale: str
    citation_ids: list[str] = Field(default_factory=list)


class ActionItemsDraft(BaseModel):
    action_items: list[ActionItemDraft]


class ComparisonFindingDraft(BaseModel):
    topic: str
    summary: str
    citation_ids: list[str] = Field(default_factory=list)


class ComparisonDraft(BaseModel):
    overview: str
    overview_citation_ids: list[str] = Field(default_factory=list)
    similarities: list[ComparisonFindingDraft]
    differences: list[ComparisonFindingDraft]
    conclusion: str
    conclusion_citation_ids: list[str] = Field(default_factory=list)


class StructuredLLMService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for reasoning calls")

        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "langchain_openai is not installed in the current Python environment. "
                "Run the app/tests inside Docker or install project dependencies locally."
            ) from exc

        self.model = ChatOpenAI(
            model=settings.openai_reasoning_model,
            api_key=settings.openai_api_key,
            temperature=0,
            use_responses_api=True,
        )

    async def summarize(
        self,
        query: str,
        document_ids: list[UUID],
        hits: list[SearchHit],
    ) -> SummaryResponse:
        if not hits:
            return SummaryResponse(
                document_ids=document_ids,
                summary="",
                key_points=[],
                citations=[],
            )

        structured = self.model.with_structured_output(SummaryDraft, method="json_schema")
        draft = await structured.ainvoke(self._build_messages(SUMMARY_SYSTEM_PROMPT, query, hits))

        summary = self._strip_inline_source_ids(draft.summary)
        key_points = [self._strip_inline_source_ids(point) for point in draft.key_points]

        citations = self._resolve_citations(draft.citation_ids, hits)
        if not citations:
            citations = self._infer_citations_from_texts([summary, *key_points], hits)

        return SummaryResponse(
            document_ids=document_ids,
            summary=summary,
            key_points=key_points,
            citations=citations,
        )

    async def extract_action_items(
        self,
        query: str,
        document_ids: list[UUID],
        hits: list[SearchHit],
    ) -> ActionItemsResponse:
        if not hits:
            return ActionItemsResponse(document_ids=document_ids, action_items=[])

        if not self._has_actionable_evidence(hits):
            return ActionItemsResponse(document_ids=document_ids, action_items=[])

        structured = self.model.with_structured_output(ActionItemsDraft, method="json_schema")
        draft = await structured.ainvoke(self._build_messages(ACTION_ITEMS_SYSTEM_PROMPT, query, hits))

        action_items: list[ActionItem] = []
        for item in draft.action_items:
            title = self._strip_inline_source_ids(item.title).strip()
            rationale = self._strip_inline_source_ids(item.rationale).strip()
            owner = self._strip_inline_source_ids(item.owner).strip() if item.owner else None

            if not title:
                continue

            citations = self._resolve_citations(item.citation_ids, hits)
            if not citations:
                citations = self._infer_citations_from_texts(
                    [title, owner or "", item.due_date or "", rationale],
                    hits,
                )

            if not citations:
                continue

            action_items.append(
                ActionItem(
                    title=title,
                    owner=owner,
                    due_date=item.due_date,
                    priority=item.priority if item.priority in {"low", "medium", "high"} else "medium",
                    rationale=rationale,
                    citations=citations,
                )
            )

        return ActionItemsResponse(document_ids=document_ids, action_items=action_items)

    async def compare(
        self,
        query: str,
        document_ids: list[UUID],
        hits: list[SearchHit],
    ) -> DocumentComparisonResponse:
        if not hits:
            return DocumentComparisonResponse(
                document_ids=document_ids,
                overview="",
                overview_citations=[],
                similarities=[],
                differences=[],
                conclusion="",
                conclusion_citations=[],
            )

        structured = self.model.with_structured_output(ComparisonDraft, method="json_schema")
        draft = await structured.ainvoke(self._build_comparison_messages(query, hits))

        overview = self._strip_inline_source_ids(draft.overview)
        conclusion = self._strip_inline_source_ids(draft.conclusion)

        overview_citations = self._resolve_citations(draft.overview_citation_ids, hits)
        if not overview_citations:
            overview_citations = self._infer_citations_from_texts([overview], hits)

        conclusion_citations = self._resolve_citations(draft.conclusion_citation_ids, hits)
        if not conclusion_citations:
            conclusion_citations = self._infer_citations_from_texts([conclusion], hits)

        similarities = [
            self._to_finding(item, hits)
            for item in draft.similarities
            if self._is_meaningful_similarity(item)
        ]

        if not similarities:
            similarities = self._build_similarity_fallback(hits)

        differences = [self._to_finding(item, hits) for item in draft.differences]

        return DocumentComparisonResponse(
            document_ids=document_ids,
            overview=overview,
            overview_citations=overview_citations,
            similarities=similarities,
            differences=differences,
            conclusion=conclusion,
            conclusion_citations=conclusion_citations,
        )

    @staticmethod
    def _build_messages(system_prompt: str, query: str, hits: list[SearchHit]) -> list[dict[str, Any]]:
        sources = []
        for hit in hits:
            sources.append(
                f"source_id={hit.citation.source_id} filename={hit.filename} "
                f"page={hit.page_number} chunk_id={hit.chunk_id}\n{hit.content}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User request: {query}\n\nSources:\n\n" + "\n\n".join(sources),
            },
        ]

    def _build_comparison_messages(self, query: str, hits: list[SearchHit]) -> list[dict[str, Any]]:
        grouped: dict[UUID, list[SearchHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.document_id, []).append(hit)

        sections: list[str] = []
        for doc_hits in grouped.values():
            header = f"Document {doc_hits[0].filename} ({doc_hits[0].document_id})"
            body = "\n\n".join(
                f"source_id={hit.citation.source_id} page={hit.page_number} chunk_id={hit.chunk_id}\n{hit.content}"
                for hit in doc_hits
            )
            sections.append(f"{header}\n{body}")

        return [
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User request: {query}\n\nCompare the following document groups fairly:\n\n"
                + "\n\n".join(sections),
            },
        ]

    @staticmethod
    def _resolve_citations(source_ids: list[str], hits: list[SearchHit]) -> list[Citation]:
        index = {hit.citation.source_id.upper(): hit.citation for hit in hits}
        citations: list[Citation] = []
        for source_id in source_ids:
            for normalized_id in StructuredLLMService._normalize_source_ids(source_id):
                citation = index.get(normalized_id)
                if citation and citation not in citations:
                    citations.append(citation)
        return citations

    def _to_finding(self, draft: ComparisonFindingDraft, hits: list[SearchHit]) -> ComparisonFinding:
        topic = self._strip_inline_source_ids(draft.topic)
        summary = self._strip_inline_source_ids(draft.summary)

        citations = self._resolve_citations(draft.citation_ids, hits)
        if not citations:
            citations = self._infer_citations_from_texts([topic, summary], hits)

        return ComparisonFinding(topic=topic, summary=summary, citations=citations)

    @staticmethod
    def _normalize_source_ids(raw_value: str) -> list[str]:
        return [match.upper() for match in _SOURCE_ID_PATTERN.findall(raw_value or "")]

    @staticmethod
    def _strip_inline_source_ids(text: str | None) -> str:
        if not text:
            return ""
        cleaned = _INLINE_SOURCE_PATTERN.sub(" ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _infer_citations_from_texts(
        self,
        texts: list[str],
        hits: list[SearchHit],
        max_citations: int = 2,
    ) -> list[Citation]:
        combined = " ".join(text for text in texts if text).strip()
        if not combined:
            return [hit.citation for hit in hits[:max_citations]]

        scored_hits: list[tuple[float, Citation]] = []
        combined_tokens = self._tokenize(combined)
        for hit in hits:
            hit_tokens = self._tokenize(hit.content)
            score = self._token_overlap_score(combined_tokens, hit_tokens)
            if score > 0:
                scored_hits.append((score, hit.citation))

        if not scored_hits:
            return [hit.citation for hit in hits[:max_citations]]

        scored_hits.sort(key=lambda item: item[0], reverse=True)
        citations: list[Citation] = []
        for _, citation in scored_hits[:max_citations]:
            if citation not in citations:
                citations.append(citation)
        return citations

    @staticmethod
    def _tokenize(text: str) -> Counter[str]:
        tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
        filtered = [
            token
            for token in tokens
            if token not in _STOPWORDS and len(token) > 1
        ]
        return Counter(filtered)

    @staticmethod
    def _token_overlap_score(query_tokens: Counter[str], hit_tokens: Counter[str]) -> float:
        if not query_tokens or not hit_tokens:
            return 0.0
        overlap = set(query_tokens) & set(hit_tokens)
        if not overlap:
            return 0.0
        score = 0.0
        for token in overlap:
            score += min(query_tokens[token], hit_tokens[token])
        return score / max(1, len(query_tokens))

    @staticmethod
    def _has_actionable_evidence(hits: list[SearchHit]) -> bool:
        for hit in hits:
            text = hit.content or ""
            if _ACTION_SECTION_PATTERN.search(text):
                return True
            if _ACTION_LINE_PATTERN.search(text):
                return True
        return False

    def _is_meaningful_similarity(self, item: ComparisonFindingDraft) -> bool:
        combined = f"{item.topic} {item.summary}"
        tokens = set(self._tokenize(combined))
        if not tokens:
            return False
        non_generic_tokens = {
            token for token in tokens
            if token not in _GENERIC_SIMILARITY_TOKENS and not token.isdigit()
        }
        return len(non_generic_tokens) >= 2

    def _build_similarity_fallback(self, hits: list[SearchHit]) -> list[ComparisonFinding]:
        grouped: dict[UUID, list[SearchHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.document_id, []).append(hit)

        if len(grouped) < 2:
            return []

        token_sets: list[set[str]] = []
        citations: list[Citation] = []

        for doc_hits in grouped.values():
            combined_text = " ".join(hit.content for hit in doc_hits)
            tokens = {
                token
                for token in self._tokenize(combined_text)
                if token not in _GENERIC_SIMILARITY_TOKENS and not token.isdigit()
            }
            token_sets.append(tokens)
            citations.append(doc_hits[0].citation)

        shared_tokens = set.intersection(*token_sets) if token_sets else set()
        shared_tokens = {
            token for token in shared_tokens
            if token not in _GENERIC_SIMILARITY_TOKENS and len(token) > 2
        }

        if len(shared_tokens) < 2:
            return []

        pretty = [self._pretty_token(token) for token in sorted(shared_tokens)[:4]]
        summary = "Both documents share recurring themes/entities: " + ", ".join(pretty) + "."

        return [
            ComparisonFinding(
                topic="Shared themes",
                summary=summary,
                citations=citations[:2],
            )
        ]

    @staticmethod
    def _pretty_token(token: str) -> str:
        mapping = {
            "postgresql": "PostgreSQL",
            "pgvector": "pgvector",
            "dana": "Dana",
            "amir": "Amir",
            "alpha": "Alpha",
            "slack": "Slack",
            "docker": "Docker",
        }
        return mapping.get(token, token)