# Document Intelligence Agent

A document-centric AI agent that ingests unstructured files and turns them into searchable, actionable knowledge.

The system supports semantic document search, grounded summarization, action-item extraction, document comparison, and controlled workflow actions such as sending summaries to Slack and creating tasks in Monday.com.

## What this project does

This project is built to solve a common problem in real-world teams: important information lives inside unstructured documents, but it is difficult to search, summarize, compare, and operationalize reliably.

The system processes uploaded documents, stores chunk embeddings in PostgreSQL with pgvector, retrieves relevant context with semantic search and reranking, and uses GPT-4o to produce structured, source-grounded outputs.

It also includes an agent orchestration layer that selects the right analysis tool for the user’s request and supports human-in-the-loop approval before external write actions.

## Core capabilities

- Upload and process `PDF`, `DOCX`, and `TXT` documents
- Parse and chunk documents with overlap
- Store embeddings in PostgreSQL using `pgvector`
- Perform semantic retrieval with OpenAI embeddings
- Improve retrieval quality with Cohere reranking
- Support tool-based agent orchestration with LangChain
- Return structured outputs validated with Pydantic
- Ground responses with source-linked citations
- Persist agent runs, tool calls, and approval steps for traceability
- Send summaries to Slack
- Create tasks in Monday.com
- Require approval before executing external write actions

## Architecture overview

The system is built in two main stages.

### 1. Offline document processing
Documents are uploaded and processed before runtime:
- file parsing
- text normalization
- chunking with overlap
- embedding generation
- storage in PostgreSQL with pgvector

### 2. Runtime agent workflow
At runtime:
- the user sends a request
- the agent selects the best analysis tool
- the selected tool performs retrieval and generation
- the orchestrator validates the result
- if requested, a post-action is prepared
- if approval is required, execution pauses until approve or reject
- only then is the external action executed

## Tech stack

- **Language:** Python
- **API framework:** FastAPI
- **Validation / schemas:** Pydantic
- **LLM orchestration:** LangChain
- **Reasoning model:** GPT-4o
- **Embedding model:** OpenAI `text-embedding-3-large`
- **Reranking:** Cohere Rerank 4
- **Database:** PostgreSQL
- **Vector extension:** pgvector
- **Async database layer:** SQLAlchemy Async
- **External integrations:** Slack Incoming Webhooks, Monday.com GraphQL API
- **Containerization:** Docker, Docker Compose

## Project structure

```text
document-intelligence-agent/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── llm.py
│   ├── routes.py
│   │
│   ├── db/
│   │   ├── session.py
│   │   └── models.py
│   │
│   ├── ingestion/
│   │   ├── parsers.py
│   │   ├── chunking.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── rerank.py
│   │   └── pipeline.py
│   │
│   ├── agent/
│   │   ├── orchestrator.py
│   │   ├── tools.py
│   │   └── prompts.py
│   │
│   ├── integrations/
│   │   ├── slack.py
│   │   └── monday.py
│   │
│   └── services/
│       └── approval_service.py
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

## Main analysis tools

The agent selects one of the following analysis tools:
- `search_documents`
- `summarize_document`
- `extract_action_items`
- `compare_documents`

These tools are used for document understanding only.

External actions such as Slack or Monday are handled as controlled post-actions after a validated final result exists.

## Human-in-the-loop flow

For external write actions, the system supports approval before execution.

Supported flow:
1. run document analysis
2. validate that the result is usable and grounded
3. create a pending approval request
4. approve, reject, or edit-and-approve
5. execute the external action only after approval

This is used for:
- `send_summary_to_slack`
- `create_monday_items`

## Data model

The project persists the following entities:
- `documents`
- `document_chunks`
- `agent_runs`
- `tool_calls`
- `pending_approvals`

This provides:
- traceability
- debugging
- auditability
- observability into agent behavior

## API endpoints

### Health
- `GET /health`

### Document ingestion
- `POST /documents/upload`

### Direct capability endpoints
- `POST /search`
- `POST /summarize`
- `POST /action-items`
- `POST /compare`

### Agent endpoint
- `POST /agent/run`

### Approval endpoint
- `POST /post-actions/decision`

## Example workflows

### 1. Grounded document summary
- upload a document
- call `/agent/run` with a summary request
- receive a structured summary with citations

### 2. Extract action items
- upload meeting notes
- call `/agent/run` or `/action-items`
- receive structured action items with owners, due dates, and citations

### 3. Compare documents
- upload two project documents
- call `/compare` or `/agent/run`
- receive overview, similarities, differences, and conclusion with grounding

### 4. Slack delivery with approval
- request a summary with `post_actions=["send_summary_to_slack"]`
- set `require_approval=true`
- receive `pending_approval`
- approve through `/post-actions/decision`
- send summary to Slack

### 5. Monday task creation with approval
- request action extraction with `post_actions=["create_monday_items"]`
- set `require_approval=true`
- receive `pending_approval`
- approve through `/post-actions/decision`
- create tasks in Monday.com

## How to run locally

### 1. Clone the repository
```bash
git clone https://github.com/EdenOved/document-intelligence-agent.git
cd document-intelligence-agent
```

### 2. Create your environment file
Copy `.env.example` to `.env` and fill in your real values.

Minimum required for core functionality:
- `OPENAI_API_KEY`
- `DATABASE_URL`

Optional but recommended:
- `COHERE_API_KEY` for reranking
- `SLACK_WEBHOOK_URL` for Slack integration
- `MONDAY_API_TOKEN` and `MONDAY_BOARD_ID` for Monday integration

### 3. Start the system
```bash
docker compose up -d --build
```

### 4. Open the API docs
```text
http://localhost:8000/docs
```

## Testing

Run tests inside the API container:

```bash
docker compose exec api python -m pytest -q
```

The validation flow used during development covered:
- ingestion
- search
- summarization
- action-item extraction
- comparison
- failure handling
- post-action dry runs
- approval flow
- Slack execution
- Monday execution

## Guardrails

The project includes several guardrails to avoid weak or unsafe execution:
- empty or weak analysis results are rejected
- summaries require citations
- action items require grounded evidence
- compare outputs require grounded overview and conclusion
- post-actions only run when the response type matches the action
- external actions support `dry_run`
- approval can be required before write actions

## Observability and traceability

The project stores:
- selected tool
- tool arguments
- tool outputs
- success or failure state
- error messages
- approval status
- final response

This makes it possible to inspect the full agent workflow rather than only the final answer.

## Current status

This project is a strong MVP for a document intelligence and workflow automation system.

The core runtime flow is implemented and validated:
- ingestion
- retrieval
- reranking
- grounded generation
- agent orchestration
- post-actions
- approval flow

TXT-based end-to-end flows are the most validated during testing.

PDF and DOCX are supported in the ingestion pipeline, and parsing quality for more complex real-world files remains an area for continued improvement.


## Why this project matters

This project goes beyond basic chat-with-documents demos.

It combines:
- semantic retrieval
- tool-based agent orchestration
- structured outputs
- source grounding
- workflow automation
- human approval before external actions

That makes it much closer to a real AI system used in business workflows than a simple RAG prototype.
