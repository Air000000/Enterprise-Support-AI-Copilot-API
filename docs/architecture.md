# Architecture

Enterprise Support AI Copilot 当前稳定基线的系统结构说明。

本文档只描述当前 `main` 基线下已经稳定的运行方式、模块边界和关键链路，不包含未来的目录重构计划。

## System Overview

当前系统定位为企业内部支持场景下的 AI Copilot 后端，覆盖以下主链路：

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

当前稳定能力：

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

## High-Level Layers

```text
Client / Swagger / Smoke Scripts
-> FastAPI Routers
-> Pydantic Schemas
-> Service Layer
-> SQLModel Models / SQLite
-> Chroma Vector Store
-> DashScope-compatible Embedding / LLM APIs
```

模块视图：

```text
enterprise-support-ai-copilot-api/
├── main.py
├── routers/
├── schemas/
├── services/
├── models/
├── docs/
├── scripts/
├── tests/
├── experiments/
│   ├── docs/
│   ├── evals/
│   └── rag_local/
├── database.py
├── docker-compose.yml
└── README.md
```

分层职责：

| Layer | Main Location | Responsibility |
|---|---|---|
| API Layer | `routers/` | 接收 HTTP 请求、参数校验、组织 response |
| Schema Layer | `schemas/` | request / response 数据结构 |
| Service Layer | `services/` | 编排业务流程与状态流转 |
| Persistence | `models/`, `database.py` | SQLModel 表结构与 SQLite 会话 |
| RAG Runtime | `experiments/rag_local/` | Chroma 检索、RAG 组装、LLM 调用 |
| Evaluation | `experiments/evals/` | retrieval eval 与指标统计 |
| Scripts | `scripts/` | smoke 验证脚本 |

## Key Runtime Decisions

### RAG runtime stays in `experiments/rag_local/`
本轮基线冻结明确保留当前运行路径，不迁移到 `rag_runtime/`。

原因：

- 当前路径已经被 services、tests、scripts、evals 和文档命令使用
- 该路径虽然带有实验目录历史，但不影响当前稳定运行
- 基线冻结阶段优先保证低风险收口，不做路径级重构

### Todo remains as Legacy compatibility
Todo 与旧 AI Todo 能力继续保留，但不作为当前主能力展示。

保留项包括：

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`
- `tests/test_todos.py`

### Default database filename stays unchanged
默认数据库文件名继续使用 `data/todos.db`。

这是历史命名遗留，不代表当前项目仍是 Todo API。本轮不改它，避免带来数据迁移、Docker volume 和文档命令变化。

## Core Request Flows

### RAG Search / Ask

```text
POST /rag/search or /rag/ask
-> routers/rag.py
-> services/rag_service.py
-> experiments/rag_local/*
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

## Current Boundary Notes

当前基线明确不做：

- HTTP 接口改名
- request / response shape 调整
- 业务规则重写
- `experiments/rag_local` 目录迁移
- 默认数据库文件名调整

当前基线允许做：

- 项目命名收口
- README / docs 收口
- OpenAPI title 和 `/about` 文案更新
- mock tenant / user 常量集中管理

## Related Docs

- [project_summary.md](project_summary.md): 当前基线总结
- [agent_workflow.md](agent_workflow.md): Ticket Agent 流程
- [security.md](security.md): 当前安全边界
- `docs/*_report.md`: 历史阶段性记录
