# AGENTS.md

## Project Context

This repository is being upgraded from a FastAPI Todo + RAG learning project into **Enterprise Support AI Copilot**.

The current `main` branch includes:

```text
Enterprise RAG Core
Ticket CRUD
Ticket Agent preview / confirm
AgentOps audit records
AgentOps read APIs
approval reject / cancel APIs
AgentOps metrics summary API

```

## Current Focus

The current focus is **Ticket Agent v1 cleanup and AgentOps hardening**.

The main goals are:

```text
1. Keep the RAG retrieval baseline stable.
2. Keep Ticket CRUD stable.
3. Harden Ticket Agent preview / confirm behavior.
4. Ensure AgentOps records are correct and auditable.
5. Keep project documentation aligned with implementation.
```

## Development Rules

* Keep changes small and reviewable.
* Run focused tests before committing.
* Prefer service / schema / router separation.
* Keep tenant isolation in mind for all ticket, RAG, and AgentOps features.
* Agent actions must stay controlled by preview / confirm flow.
* Do not let the agent create real tickets without explicit confirm.
* Record important agent actions through AgentOps records when relevant.

## Important Commands

### Core RAG tests

```bash
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
```

### Ticket and AgentOps tests

```bash
pytest tests/test_tickets.py
pytest tests/test_agent_ops_service.py
pytest tests/test_agent_ops_api.py
pytest tests/test_ticket_agent_service.py
pytest tests/test_agent_ticket_api.py
```


### Todo legacy tests

```bash
pytest tests/test_todos.py
```

### Retrieval eval

```bash
python -m experiments.evals.eval_retrieval
python -m experiments.evals.eval_chroma_retrieval
```

## Historical Baselines (not current suite totals)

### RAG eval

```text
JSON learning eval:
Total: 15
hit@1: 0.93
hit@3: 1.00
mrr@3: 0.97

Chroma enterprise eval:
Total: 30
hit@1: 0.97
hit@3: 1.00
mrr@3: 0.98
```

```text
tests/test_query_chroma.py         4 passed
tests/test_rag_api.py              8 passed
tests/test_rag_service.py          2 passed
tests/test_todos.py                4 passed
tests/test_tickets.py              9 passed
tests/test_agent_ops_service.py    10 passed
tests/test_agent_ops_api.py        7 passed
tests/test_ticket_agent_service.py 9 passed
tests/test_agent_ticket_api.py     6 passed
```


## Current Architecture Notes

The current agent flow is:

```text
User message
↓
/agent/ticket/preview
↓
RAG search
↓
TicketDraft
↓
ApprovalRequest
↓
/agent/ticket/confirm
↓
ToolCall
↓
Ticket creation
↓
AgentOps status update
↓
AgentOps read APIs
↓
Approval reject / cancel APIs
↓
AgentOps metrics summary API

```

The current system resolves tenant and user context from Demo JWT Bearer tokens:

```text
CurrentUser.user_id
CurrentUser.tenant_id
CurrentUser.role
```

Document, RAG, Ticket and AgentOps paths use this server-side context; do not trust
client-supplied tenant IDs. Legacy Todo paths are tenant-scoped too. Pre-migration
Todo rows with unknown ownership remain stored but hidden until ownership is verified.
Demo credentials/JWT defaults are not production IAM or database-level isolation.
Hybrid is integrated as explicit opt-in; Dense remains the repository default.
Refusal candidates remain experimental; do not rewrite frozen negative results.
