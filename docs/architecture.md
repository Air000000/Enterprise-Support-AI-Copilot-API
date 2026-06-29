# Architecture

Enterprise Support AI Copilot 架构说明。

本文档说明当前 `main` 分支的系统分层、运行组件、数据存储、核心请求链路、审计链路、Docker 本地运行布局，以及当前 MVP 与未来生产化架构之间的边界。

当前系统定位：

```text
Enterprise Support AI Copilot
=
Enterprise RAG Core
+ Document Backend
+ Ticket CRUD
+ Ticket Agent preview / confirm
+ AgentOps Audit
+ Retrieval Logs / Metrics
+ Docker Compose local runtime
+ Smoke Scripts
```

---

## 1. System Overview

本项目模拟企业内部支持场景。用户可以围绕 IT、HR、财务、行政、安全等内部制度提出问题，系统通过 RAG 检索企业知识库并返回带 sources 的回答。

当用户问题更适合进入支持流程时，Ticket Agent 会生成工单预览，而不是直接创建工单。只有用户确认后，系统才会创建真实 ticket。整个过程中，AgentOps 会记录 agent run、tool call、approval request 和 metrics summary。

高层链路：

```text
Enterprise documents / Uploaded documents
↓
Document Backend lifecycle
↓
Text split
↓
Embedding
↓
Chroma vector store
↓
RAG search / ask
↓
Ticket Agent preview
↓
Human approval
↓
Ticket creation
↓
AgentOps audit
↓
Metrics summary
```

---

## 2. High-Level Architecture

当前系统采用典型 FastAPI 分层架构：

```text
Client / Swagger / Smoke Scripts
↓
FastAPI Routers
↓
Pydantic Schemas
↓
Service Layer
↓
SQLModel Models / SQLite
↓
Chroma Vector Store
↓
DashScope-compatible Embedding / LLM APIs
```

模块视图：

```text
fastapi-todo-api/
├── main.py
├── routers/
│   ├── rag.py
│   ├── documents.py
│   ├── tickets.py
│   ├── agent_ticket.py
│   └── agent_ops.py
├── schemas/
│   ├── rag.py
│   ├── document.py
│   ├── ticket.py
│   ├── agent_ticket.py
│   └── agent_ops.py
├── services/
│   ├── rag_service.py
│   ├── document_service.py
│   ├── ticket_service.py
│   ├── ticket_agent_service.py
│   └── agent_ops_service.py
├── models/
│   ├── document.py
│   ├── ticket.py
│   └── agent_ops.py
├── experiments/
│   ├── docs/
│   ├── rag_local/
│   └── evals/
├── scripts/
│   ├── smoke_agentops_flow.py
│   └── smoke_document_backend_flow.py
└── tests/
```

分层职责：

| 层级 | 主要文件 | 职责 |
|---|---|---|
| API Layer | `routers/` | 接收 HTTP 请求，执行 request validation，调用 service，返回 response |
| Schema Layer | `schemas/` | 定义 request / response 数据结构 |
| Service Layer | `services/` | 实现业务流程、状态变更、跨模块编排 |
| Persistence Layer | `models/`, `database.py` | 定义 SQLModel 表结构与数据库会话 |
| RAG Runtime | `experiments/rag_local/` | 执行文档加载、切分、embedding、Chroma 查询、LLM 调用 |
| Evaluation | `experiments/evals/` | 运行 retrieval eval，计算 hit@1 / hit@3 / mrr@3 |
| Scripts | `scripts/` | 端到端 smoke 验收脚本 |
| Tests | `tests/` | pytest focused tests，隔离外部依赖 |

---

## 3. Runtime Components

当前运行时组件包括：

```text
FastAPI application
SQLite database
Local document storage
Chroma vector store
DashScope-compatible embedding API
DashScope-compatible LLM API
Smoke scripts
```

### 3.1 FastAPI Application

FastAPI 负责提供 HTTP API：

```text
/rag/search
/rag/ask
/documents/*
/tickets/*
/agent/ticket/*
/agent-ops/*
/health
```

### 3.2 SQLite

SQLite 用于保存业务数据和审计数据：

```text
tickets
agent_runs
tool_calls
approval_requests
documents
document_chunks
retrieval_logs
```

当前 SQLite 仅用于本地开发和 MVP demo。生产化时应替换为 PostgreSQL，并使用 Alembic 管理 schema migration。

### 3.3 Local Document Storage

Document Backend 上传的原始 md/txt 文件会保存到本地 storage 目录。数据库中的 `documents.source_path` 记录其路径。

### 3.4 Chroma Vector Store

Chroma 保存企业文档和上传文档的 chunk embeddings。RAG 检索时通过 tenant/category metadata filter 查询 Chroma。

### 3.5 DashScope-compatible APIs

Embedding 与 LLM 调用通过 OpenAI-compatible SDK 接入 DashScope / Bailian。

典型环境变量：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
EMBEDDING_MODEL=text-embedding-v4
```

---

## 4. Data Stores

当前系统有三类主要数据存储。

### 4.1 SQLite Business and Audit Data

SQLite 保存结构化业务数据：

```text
tickets:
  工单业务对象

agent_runs:
  Ticket Agent 一次运行记录

tool_calls:
  search_kb / classify_ticket / create_ticket 工具调用审计记录

approval_requests:
  preview 阶段生成的人工确认请求

documents:
  上传文档 metadata、状态、分类和 source_path

document_chunks:
  上传文档切分后的 chunk metadata 与 embedding_id

retrieval_logs:
  RAG 检索日志、sources、no-context query 和 failure 记录
```

### 4.2 Local File Storage

本地文件存储保存上传的原始文档：

```text
storage/documents/
docker_storage/
```

具体使用哪个目录取决于运行方式和环境变量。

### 4.3 Chroma Vector Store

Chroma 保存向量数据：

```text
enterprise baseline docs embeddings
uploaded document embeddings
chunk metadata
tenant_id
category
document_id
chunk_id
source_path
```

Document Backend 删除文档时，需要同步删除 Chroma 中对应 embeddings，避免 RAG 继续召回已删除文档。

---

## 5. Request Flow: RAG Search

Endpoint:

```http
POST /rag/search
```

链路：

```text
Client
↓
routers/rag.py::rag_search()
↓
schemas/rag.py request validation
↓
services/rag_service.py::search_documents()
↓
mock tenant context: tenant_demo
↓
experiments/rag_local/query_chroma.py::search_chroma()
↓
Chroma metadata filter:
  tenant_id = tenant_demo
  category = request.category
↓
return structured search results
```

响应包含：

```text
document_id
chunk_id
title
source_path
chunk_index
distance
preview
tenant_id
category
```

`/rag/search` 主要用于查看底层检索结果，不生成最终自然语言回答。

---

## 6. Request Flow: RAG Ask

Endpoint:

```http
POST /rag/ask
```

链路：

```text
Client
↓
routers/rag.py::rag_ask()
↓
schemas/rag.py request validation
↓
services/rag_service.py::answer_question()
↓
Chroma retrieval
↓
context construction
↓
LLM answer generation
↓
structured sources
↓
retrieval logs / metrics
↓
return answer + sources
```

当检索不到足够相关内容时，系统返回拒答：

```text
我在已提供资料中没有找到足够依据。
```

RAG Ask 体现完整 RAG 行为：

```text
retrieve
↓
ground answer in retrieved context
↓
return answer
↓
return sources
↓
record retrieval status
```

---

## 7. Request Flow: Document Upload / Index / Delete

Document Backend 提供文档生命周期管理。

### 7.1 Upload

Endpoint:

```http
POST /documents/upload
```

链路：

```text
Client uploads md/txt file
↓
routers/documents.py::upload_document()
↓
schemas/document.py response schema
↓
services/document_service.py::create_document_from_bytes()
↓
write file to storage
↓
insert documents row
↓
documents.status = uploaded
↓
chunk_count = 0
```

上传成功只表示文档已进入后端系统，还没有进入向量库，因此不会立即被 RAG 召回。

### 7.2 Index

Endpoint:

```http
POST /documents/{document_id}/index
```

链路：

```text
Client triggers index
↓
routers/documents.py::index_document()
↓
services/document_service.py::index_document()
↓
read documents.source_path
↓
split text into chunks
↓
call embedding API
↓
write embeddings to Chroma
↓
insert document_chunks rows
↓
documents.status = indexed
↓
chunk_count > 0
```

索引成功后，上传文档可以被 `/rag/search` 和 `/rag/ask` 检索到。

### 7.3 Delete

Endpoint:

```http
DELETE /documents/{document_id}
```

链路：

```text
Client deletes document
↓
routers/documents.py::delete_document()
↓
services/document_service.py::delete_document()
↓
read document_chunks.embedding_id
↓
delete embeddings from Chroma
↓
delete / clear document_chunks
↓
documents.status = deleted
↓
RAG search no longer returns this document
```

删除闭环的核心要求：

```text
业务数据库状态改变
+
Chroma embeddings 被清理
+
RAG 不再召回已删除文档
```

---

## 8. Request Flow: Ticket CRUD

Ticket CRUD 是 Agent 最终创建业务对象的落点。

典型 API：

```http
POST /tickets
GET  /tickets
GET  /tickets/{ticket_id}
PATCH /tickets/{ticket_id}
```

链路：

```text
Client
↓
routers/tickets.py
↓
schemas/ticket.py
↓
services/ticket_service.py
↓
models/ticket.py
↓
SQLite tickets table
```

Ticket CRUD 可以被用户直接调用，也可以由 Ticket Agent 在 confirm 阶段通过 `create_ticket` 工具调用间接使用。

---

## 9. Request Flow: Ticket Agent Preview / Confirm

Ticket Agent 使用两阶段执行模型：

```text
preview = 建议阶段，不执行状态变更
confirm = 人工确认后，执行真实 ticket 创建
```

### 9.1 Preview Flow

Endpoint:

```http
POST /agent/ticket/preview
```

链路：

```text
Client message
↓
routers/agent_ticket.py::preview_ticket()
↓
services/ticket_agent_service.py::preview_ticket()
↓
create agent_run
↓
tool_call: search_kb
↓
RAG retrieval
↓
tool_call: classify_ticket
↓
generate TicketDraft
↓
create approval_request.pending
↓
return:
  agent_run_id
  approval_request_id
  should_create_ticket
  reason
  draft
  sources
```

Preview 阶段不会创建真实 ticket。

### 9.2 Confirm Flow

Endpoint:

```http
POST /agent/ticket/confirm
```

链路：

```text
Client submits agent_run_id + approval_request_id + draft
↓
routers/agent_ticket.py::confirm_ticket()
↓
services/ticket_agent_service.py::confirm_ticket()
↓
load approval_request
↓
validate ownership:
  approval_request.agent_run_id == request.agent_run_id
↓
validate status:
  approval_request.status == pending
↓
validate draft consistency:
  request.draft == approval_request.draft_json
↓
mark approval_request approved
↓
use server-side approval_request.draft_json
↓
tool_call: create_ticket
↓
create ticket
↓
mark agent_run completed
↓
return created ticket
```

Confirm 阶段的关键安全原则：

```text
客户端提交的 draft 只用于一致性校验；
真实创建 ticket 的 payload 来自服务端保存的 approval_request.draft_json。
```

---

## 10. AgentOps Audit Flow

AgentOps 负责记录 Agent 执行过程。

核心表：

```text
agent_runs
tool_calls
approval_requests
```

一次完整 preview / confirm 流程会产生：

```text
agent_runs:
  1 条

tool_calls:
  search_kb
  classify_ticket
  create_ticket

approval_requests:
  1 条 pending → approved
```

查询 API：

```http
GET /agent-ops/runs
GET /agent-ops/runs/{agent_run_id}
GET /agent-ops/runs/{agent_run_id}/tool-calls
GET /agent-ops/runs/{agent_run_id}/approval-requests
GET /agent-ops/metrics/summary
```

审批状态 API：

```http
POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

AgentOps 的作用：

```text
解释 Agent 做了什么
解释 Agent 为什么建议创建工单
记录用户是否批准
记录最终是否创建 ticket
记录失败工具调用和 error_type
提供 metrics summary
```

---

## 11. Retrieval Metrics Flow

Retrieval Logs / Metrics 用于观察 RAG 检索行为。

当前能力包括：

```text
retrieval log detail
summary metrics
sources metrics
no-context queries
failures
```

典型用途：

```text
查看哪些 query 没有命中文档
查看哪些 sources 被频繁召回
查看失败检索
查看 RAG answer 的检索状态
为后续 dashboard 提供数据基础
```

与 AgentOps 的区别：

```text
Retrieval metrics:
  关注 RAG 检索质量和检索行为

AgentOps metrics:
  关注 Agent run、tool call、approval request 和工具调用结果
```

---

## 12. Smoke Scripts Flow

当前项目提供两个 smoke script：

```text
scripts/smoke_agentops_flow.py
scripts/smoke_document_backend_flow.py
```

### 12.1 AgentOps Smoke Script

验证链路：

```text
health
↓
agent ticket preview
↓
search_kb / classify_ticket tool_calls
↓
pending approval_request
↓
confirm
↓
create_ticket tool_call
↓
metrics summary
```

预期输出：

```text
Smoke test passed.
Validated: preview -> search_kb/classify_ticket -> approval -> confirm -> create_ticket -> metrics
```

### 12.2 Document Backend Smoke Script

验证链路：

```text
health
↓
upload document
↓
index document
↓
rag/search returns uploaded document
↓
delete document
↓
rag/search no longer returns deleted document
```

预期输出：

```text
Smoke test passed.
Validated: health -> upload -> index -> search hit -> delete -> search miss
```

Smoke scripts 调用真实运行中的 API 服务，因此适合本地验收和 demo 前检查，不放入默认 GitHub Actions CI。

---

## 13. Docker Runtime Layout

Docker Compose 用于本地 demo、功能验证和交付复现。

当前 Compose 设计：

```text
host: 127.0.0.1:8000
↓
container: api service
↓
uvicorn main:app --host 0.0.0.0 --port 8000
```

运行数据目录：

| 本地目录 | 容器内路径 | 用途 |
|---|---|---|
| `docker_data/` | `/app/data` | SQLite 数据库 |
| `docker_storage/` | `/app/storage` | 上传文档 |
| `docker_chroma_db/` | `/app/experiments/chroma_db` | Chroma 向量库 |

设计目的：

```text
Docker demo 数据与本地开发数据隔离
端口只绑定 127.0.0.1，避免默认暴露到局域网或公网
.env 通过 env_file 注入，不提交到 GitHub
```

当前 Docker Compose 是本地运行版，不是生产部署版。

---

## 14. Synchronous vs Future Async Boundaries

当前 MVP 中，大部分链路是同步执行的。

### 14.1 当前同步链路

```text
/documents/{document_id}/index:
  同步读取文件、切分、embedding、写入 Chroma

/agent/ticket/preview:
  同步检索、分类、生成 draft

/agent/ticket/confirm:
  同步校验 approval、创建 ticket、记录 tool_call

/rag/ask:
  同步检索、构造上下文、调用 LLM、返回 answer
```

### 14.2 后续应异步化的链路

生产化时建议异步化：

```text
Document indexing
Large file parsing
PDF / Word parsing
Embedding batch jobs
Index retry and compensation
Long-running Agent workflows
Metrics aggregation
Dashboard refresh
```

未来形态：

```text
API request
↓
create job
↓
queue
↓
worker
↓
update job status
↓
persist logs
↓
client polls / subscribes to result
```

---

## 15. MVP Boundaries

当前项目是 MVP / demo 系统，不是生产系统。

当前已实现：

```text
RAG search / ask
Document upload / index / delete lifecycle
Ticket CRUD
Ticket Agent preview / confirm
Approval ownership validation
Pending approval validation
Draft consistency check
Tool call audit
AgentOps metrics summary
Retrieval logs / metrics
Docker Compose local run
Smoke scripts
Focused pytest suites
```

当前未实现或仅为 mock：

```text
real authentication
real authorization
real tenant context
real user context
production upload security
virus scanning
PII / sensitive content detection
rate limit
cost budget
async indexing queue
PostgreSQL
Alembic migration
production deployment
dashboard
frontend approval UI
```

---

## 16. Future Production Architecture

未来生产化架构建议：

```text
Client / Frontend
↓
API Gateway / Reverse Proxy / TLS
↓
Authentication / Authorization
↓
FastAPI App
↓
PostgreSQL
↓
Object Storage
↓
Vector Store
↓
Async Job Queue
↓
Worker Pool
↓
Embedding / LLM Provider
↓
Monitoring / Logging / Alerting
```

推荐演进方向：

```text
1. SQLite → PostgreSQL
2. Local storage → Object storage
3. Sync document indexing → Async indexing jobs
4. Mock tenant → authenticated tenant context
5. Mock user → authenticated user context
6. Local-only Docker Compose → production deployment manifests
7. Manual metrics API → dashboard + alerting
8. Basic .env secrets → secret manager
9. No rate limit → per-user / per-tenant quota
10. Basic tests + smoke scripts → staged CI/CD
```

---

## 17. Architecture Summary

当前系统可以概括为：

```text
Document Backend 管理知识库文档生命周期。
RAG Core 负责基于 Chroma 的企业知识检索和问答。
Ticket CRUD 提供业务对象落点。
Ticket Agent 负责把用户问题转成受控工单流程。
preview / confirm 保证状态变更前经过人工确认。
AgentOps 记录 Agent 运行、工具调用、审批状态和 metrics。
Retrieval Metrics 记录 RAG 检索质量和失败情况。
Smoke Scripts 验证真实 API 链路能跑通。
Docker Compose 提供本地可复现运行环境。
```

这个架构的核心原则是：

```text
先检索，再建议；
先 preview，再 confirm；
先人工确认，再执行状态变更；
所有关键工具调用都进入审计记录；
上传文档必须经过索引才进入 RAG；
删除文档必须同步清理向量库。
```

---
