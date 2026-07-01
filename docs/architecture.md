# Architecture

Enterprise Support AI Copilot is the current stable backend baseline for enterprise knowledge retrieval, controlled ticket creation, and AgentOps audit flows.

This document describes the runtime paths, module boundaries, and request flows that are stable today. It intentionally documents the current implementation rather than future refactor ideas.

## System Overview

Main product path:

```text
Enterprise documents / uploaded documents
-> Document Backend lifecycle
-> Chroma retrieval
-> RAG answer with sources
-> Ticket Agent preview
-> Human confirmation
-> Ticket creation
-> AgentOps audit and metrics
```

Current stable capabilities:

```text
Enterprise RAG Core
Document Backend
Ticket CRUD
Ticket Agent preview / confirm
AgentOps audit + read APIs
Retrieval Logs / Metrics
Docker Compose local runtime
Smoke scripts
```

## Repository Layout

```text
enterprise-support-ai-copilot-api/
|-- main.py
|-- database.py
|-- routers/
|-- schemas/
|-- services/
|-- models/
|-- rag_runtime/
|-- experiments/
|   |-- docs/
|   |-- evals/
|   `-- rag_local/
|-- scripts/
|-- tests/
|-- docker-compose.yml
`-- README.md
```

Layer responsibilities:

| Layer | Main Location | Responsibility |
|---|---|---|
| API Layer | `routers/` | HTTP request handling and response shaping |
| Schema Layer | `schemas/` | Request and response models |
| Service Layer | `services/` | Business workflow orchestration |
| Persistence | `models/`, `database.py` | SQLModel tables and SQLite session setup |
| RAG Runtime | `rag_runtime/` | Chroma retrieval, RAG assembly, and LLM calls |
| Compatibility Layer | `experiments/rag_local/` | Thin wrappers for legacy imports and `python -m` entry points |
| Evaluation | `experiments/evals/` | Retrieval evals and metric reporting |
| Scripts | `scripts/` | Smoke and local verification scripts |

## Runtime Decisions

### Official Runtime Path

`rag_runtime/` is now the official runtime package.

`experiments/rag_local/` is still kept in the repository as a compatibility layer so that:

- old imports continue to resolve
- historical `python -m experiments.rag_local.*` commands continue to run
- existing evals and notes do not break all at once

### Active Service Imports

The active runtime chain now imports `rag_runtime.*` from:

- `services/rag_service.py`
- `services/document_service.py`
- `services/ticket_agent_service.py`

### Legacy Todo Compatibility

Legacy Todo and AI Todo endpoints remain available for compatibility and historical coverage, but they are no longer the primary project story.

Retained legacy surface:

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`
- `tests/test_todos.py`

### Database Filename

The default SQLite path remains `data/todos.db` for baseline stability. The name is historical and does not change the current product positioning.

## Core Request Flows

### RAG Search / Ask

```text
POST /rag/search or /rag/ask
-> routers/rag.py
-> services/rag_service.py
-> rag_runtime/*
-> Chroma retrieval
-> answer / sources
-> retrieval logs
```

### Ticket Agent

```text
POST /agent/ticket/preview
-> search_kb tool call
-> classify_ticket tool call
-> ticket draft
-> approval_request.pending

POST /agent/ticket/confirm
-> approval validation
-> create_ticket tool call
-> real ticket creation
-> AgentOps status update
```

### Document Backend

```text
POST /documents/upload
-> document record + file write

POST /documents/{document_id}/index
-> text split
-> embedding
-> Chroma write

DELETE /documents/{document_id}
-> record update
-> chunk cleanup
-> embedding cleanup
```

## Boundaries For This Baseline

Intentionally unchanged in this baseline freeze:

- HTTP paths, request bodies, response bodies, and status codes
- business logic behavior
- default database filename
- Docker runtime behavior
- `experiments/evals/*` import paths in the first migration round

Intentionally changed in this Phase C migration:

- official runtime package moves to `rag_runtime/`
- `experiments/rag_local/` becomes a compatibility wrapper layer
- active services stop importing `experiments.rag_local.*`
- runtime docs recommend `python -m rag_runtime.*`

## Related Docs

- [project_summary.md](project_summary.md): current baseline summary
- [agent_workflow.md](agent_workflow.md): ticket agent flow
- [security.md](security.md): current security boundary
- `docs/*_report.md`: historical phase reports retained for context
