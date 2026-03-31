# document-intelligence-agent

A document-centric AI agent that ingests PDF, DOCX, and TXT files and turns them into searchable, actionable knowledge.

## Stack

- Python + FastAPI
- Pydantic for validation
- PostgreSQL + pgvector
- OpenAI `text-embedding-3-large` for embeddings
- Cohere Rerank 4 for stage-2 reranking
- GPT-4o for structured reasoning
- LangChain Core tools + custom ReAct-style orchestrator
- LangSmith-ready tracing

## What is implemented here

- Upload and parse PDF / DOCX / TXT
- Chunk documents with recursive splitting + overlap
- Persist documents, chunks, runs, and tool calls
- Generate embeddings and store them in pgvector
- Semantic search + optional Cohere reranking
- Tool-based agent that selects one of four tools:
  - `search_documents`
  - `summarize_document`
  - `extract_action_items`
  - `compare_documents`
- Source-grounded outputs with citations
- Retry loop when tool results are empty

## Build order

1. Contracts: settings, schemas, DB models
2. Ingestion: parsers -> chunking -> persistence
3. Retrieval: embeddings -> pgvector search -> rerank
4. Structured generation: summary / action-items / compare
5. Agent orchestration: tool choice + retry loop + run persistence
6. API routes
7. Tests

## Local run

```bash
cp .env.example .env
# fill API keys

docker compose up --build
```

Then open:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Suggested next improvements

- Add Alembic migrations
- Add richer parser metadata (headings, sections, page spans)
- Add query rewriting and document-grading in the retry loop
- Add evaluation datasets and LangSmith experiments
- Add authentication and file storage
