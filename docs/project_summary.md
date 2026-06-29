# Project Summary

Enterprise Support AI Copilot 项目总结。

本文档总结当前 `main` 分支已经完成的能力、解决的问题、核心业务链路、技术架构、数据与审计设计、验证方式、Docker 复现方式、当前限制和后续计划。

---

## 1. 项目定位

本项目是一个面向企业内部支持场景的 AI Copilot 后端系统。

项目最初由 FastAPI Todo / AI Todo API 演进而来，当前已经扩展为包含以下能力的企业支持后端：

```text
Enterprise RAG Core
Document Backend
Ticket CRUD
Ticket Agent preview / confirm
AgentOps audit
Retrieval Logs / Metrics
Docker Compose local runtime
Smoke Scripts
```

项目核心目标不是单纯回答问题，而是打通从知识检索到业务执行的受控链路：

```text
企业内部知识检索
↓
基于 sources 的回答
↓
识别是否需要创建工单
↓
生成工单 preview
↓
人工确认
↓
创建真实 ticket
↓
记录 AgentOps 审计轨迹
```

---

## 2. 解决的问题

企业内部支持系统通常面临三类问题：

```text
1. 员工不知道内部制度和流程在哪里
2. 简单问答无法进入真实支持流程
3. AI Agent 执行动作缺少审批和审计
```

本项目分别通过以下模块解决：

| 问题 | 当前解决方式 |
|---|---|
| 企业知识分散 | 使用 Enterprise RAG Core 对 IT / HR / finance / admin / security 文档进行检索 |
| 回答缺少依据 | `/rag/ask` 返回 answer + structured sources |
| 上传文档无法进入知识库 | Document Backend 提供 upload / index / delete 生命周期 |
| AI 直接执行动作风险高 | Ticket Agent 使用 preview / confirm 两阶段流程 |
| 状态变更缺少人工确认 | create_ticket 必须经过 approval_request |
| Agent 行为不可追踪 | AgentOps 记录 agent_runs / tool_calls / approval_requests |
| 检索效果不可评估 | Retrieval eval 和 Retrieval Logs / Metrics 记录检索质量 |
| demo 难以复现 | Docker Compose 和 smoke scripts 提供本地复现方式 |

---

## 3. 当前完成能力

当前 `main` 分支已经完成 Enterprise Support AI Copilot MVP。

### 3.1 Enterprise RAG Core

已完成：

```text
企业内部支持文档集
文档加载与切块
embedding 批处理
Chroma 向量库
tenant/category metadata filter
/rag/search
/rag/ask
answer + structured sources
no-context fallback
retrieval eval
hit@1 / hit@3 / mrr@3
category breakdown
```

当前企业文档集覆盖：

```text
it
hr
finance
admin
security
```

### 3.2 Document Backend

已完成：

```text
POST /documents/upload
GET /documents
GET /documents/{document_id}
POST /documents/{document_id}/index
DELETE /documents/{document_id}
documents table
document_chunks table
upload → index → search → delete lifecycle
```

Document Backend 的关键能力是：上传文档不是孤立文件，而是可以进入 RAG 知识库生命周期。

核心闭环：

```text
upload document
↓
documents.status = uploaded
↓
manual index
↓
document_chunks created
↓
Chroma embeddings created
↓
/rag/search can retrieve uploaded document
↓
delete document
↓
Chroma embeddings removed
↓
/rag/search no longer returns deleted document
```

### 3.3 Ticket CRUD

已完成：

```text
create ticket
list tickets
get ticket
update ticket
```

Ticket CRUD 是 Ticket Agent 最终执行业务动作的落点。

### 3.4 Ticket Agent

已完成：

```text
POST /agent/ticket/preview
POST /agent/ticket/confirm
TicketDraft
TicketAgentSource
approval ownership validation
pending approval validation
draft payload consistency check
server-side approval draft execution
```

Ticket Agent 的关键原则是：

```text
Agent can suggest, but cannot directly execute state-changing actions.
```

### 3.5 AgentOps

已完成：

```text
agent_runs
tool_calls
approval_requests
search_kb tool audit
classify_ticket tool audit
create_ticket tool audit
approval reject / cancel APIs
decision_reason
tool_call error_type
AgentOps metrics summary API
```

AgentOps 让系统可以回答：

```text
Agent 做了什么？
Agent 查了哪些知识？
Agent 为什么建议创建工单？
用户是否批准？
最终是否创建了 ticket？
哪个工具调用失败了？
```

### 3.6 Retrieval Logs / Metrics

已完成：

```text
retrieval log detail
retrieval summary
sources metrics
no-context queries
failure metrics
```

Retrieval Metrics 让 RAG 从“能回答”进一步变成“能观测”。

### 3.7 Docker Compose

已完成本地安全版 Docker Compose：

```text
服务只绑定 127.0.0.1:8000
使用 docker_data/
使用 docker_storage/
使用 docker_chroma_db/
通过 .env 注入本地配置
```

该配置适合本地 demo、功能验证和交付复现，不是生产部署版。

### 3.8 Smoke Scripts

已完成：

```text
scripts/smoke_agentops_flow.py
scripts/smoke_document_backend_flow.py
```

两个 smoke script 分别验证：

```text
AgentOps smoke:
preview -> search_kb/classify_ticket -> approval -> confirm -> create_ticket -> metrics

Document Backend smoke:
health -> upload -> index -> search hit -> delete -> search miss
```

---

## 4. 核心业务链路

当前项目最重要的端到端链路是：

```text
企业内部文档 / 上传文档
↓
Document Backend 登记与生命周期管理
↓
文档切分
↓
embedding
↓
Chroma vector store
↓
tenant/category 过滤检索
↓
RAG answer with sources
↓
Ticket Agent preview
↓
approval_request.pending
↓
human approval
↓
Ticket Agent confirm
↓
create_ticket
↓
tool_calls / agent_runs / approval_requests
↓
AgentOps metrics summary
```

这条链路体现了项目的核心设计：

```text
知识检索不是终点；
受控业务执行才是后端 Copilot 的关键价值。
```

---

## 5. 技术架构

当前项目采用 FastAPI 分层架构：

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

主要技术栈：

| 类型 | 技术 |
|---|---|
| Web 框架 | FastAPI |
| 数据校验 | Pydantic |
| ORM / 数据库 | SQLModel + SQLite |
| Vector store | Chroma |
| LLM / Embedding API | OpenAI-compatible SDK + DashScope / Bailian |
| 测试 | pytest + FastAPI TestClient |
| 本地复现 | Docker / Docker Compose |
| 语言 | Python |

---

## 6. 数据与审计设计

当前系统的数据分为四类。

### 6.1 业务数据

```text
tickets
```

用于保存真实工单。

### 6.2 文档数据

```text
documents
document_chunks
```

用于管理上传文档的 metadata、状态、chunk 和 embedding_id。

### 6.3 AgentOps 审计数据

```text
agent_runs
tool_calls
approval_requests
```

用于记录 Agent 运行、工具调用和人工审批。

### 6.4 Retrieval 观测数据

```text
retrieval logs
retrieval sources
no-context queries
failures
```

用于分析 RAG 检索行为和失败情况。

---

## 7. 安全边界

当前 MVP 已实现的关键安全边界包括：

```text
1. Ticket Agent preview 阶段不创建真实 ticket
2. create_ticket 必须经过 confirm
3. approval_request 必须属于当前 agent_run
4. approval_request.status 必须是 pending
5. confirm draft 必须与服务端保存的 approval_request.draft_json 一致
6. 创建 ticket 使用服务端 draft_json，不直接信任客户端 draft
7. search_kb / classify_ticket / create_ticket 均进入 tool_calls 审计
8. 删除文档时同步清理 Chroma embeddings
9. Docker Compose 默认只绑定 127.0.0.1
10. .env 不应提交到 GitHub
```

当前仍属于 MVP，不具备生产级安全能力。主要原因：

```text
tenant_id / user_id 仍是 mock context
尚未接入 authentication / authorization
Document Backend 缺少生产级上传安全控制
RAG sources 尚未做真实权限过滤
/index 和 /rag/ask 缺少 rate limit 与成本控制
SQLite 不适合作为生产数据库
Docker Compose 当前是本地运行版
```

详细安全边界见：

```text
docs/security.md
```

---

## 8. 测试与验证

当前项目通过三类方式验证。

### 8.1 Focused pytest

pytest 覆盖：

```text
Todo
RAG API
RAG service
Document models
Document service
Document API
Ticket CRUD
AgentOps service
AgentOps API
Ticket Agent service
Ticket Agent API
```

RAG 相关测试使用 monkeypatch 隔离真实 Chroma、embedding 和 LLM 调用，因此适合放入 CI。

### 8.2 Retrieval Eval

Enterprise RAG eval 覆盖企业内部文档检索质量。

当前企业 eval 指标：

```text
Total: 30
hit@1: 0.97
hit@3: 1.00
mrr@3: 0.98
```

### 8.3 Smoke Scripts

Smoke scripts 调用真实运行中的 API 服务：

```text
scripts/smoke_agentops_flow.py
scripts/smoke_document_backend_flow.py
```

它们验证 pytest 之外的真实链路：

```text
FastAPI
SQLite
Chroma
embedding
Document lifecycle
Ticket Agent
AgentOps metrics
```

Smoke scripts 不放入默认 GitHub Actions CI，因为它们可能触发真实 embedding / LLM API 调用。

---

## 9. Docker Compose 本地复现

当前项目支持 Docker Compose 本地运行：

```powershell
docker compose up --build
```

服务地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

运行数据目录：

```text
docker_data/
docker_storage/
docker_chroma_db/
```

这些目录用于隔离 Docker demo 数据，避免污染本地开发数据。

Docker Compose 运行后，可以在宿主机执行 smoke scripts：

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

---

## 10. 当前限制

当前项目的主要限制：

```text
1. tenant_id / user_id 仍使用 mock context
2. 尚未接入真实登录、认证和授权
3. 尚未实现真实前端审批界面
4. AgentOps API 暂未区分管理员权限
5. Document Backend 只支持 md/txt
6. Document Backend 缺少文件大小限制、病毒扫描、敏感信息检测
7. Document indexing 仍为同步执行
8. 缺少 indexing job logs 和异步任务队列
9. 缺少文档版本管理
10. 缺少 PDF / Word 等复杂文档解析
11. 缺少 per-user / per-tenant rate limit
12. 缺少 embedding / LLM 成本统计
13. SQLite 仅用于本地开发和 demo
14. 尚未接入 Alembic migration
15. Docker Compose 当前是本地运行版，不是生产部署版
```

---

## 11. 后续计划

短期后续：

```text
1. 保持 README / architecture / security / demo_script / project_summary 同步
2. 补充最终项目收尾说明
3. 整理更简洁的项目展示材料
```

中期增强：

```text
1. indexing job logs
2. 文档版本管理
3. PDF / Word 等复杂文档解析
4. 真实 tenant / user auth context
5. 真实前端审批界面
6. AgentOps dashboard / 时间窗口筛选
7. Docker Compose 最终部署版
```

生产化方向：

```text
1. PostgreSQL 替换 SQLite
2. Alembic migration
3. Authentication / Authorization
4. Object storage
5. Async job queue
6. Rate limit / cost control
7. Secret manager
8. Monitoring / logging / alerting
9. Production deployment manifests
```

---

## 12. 项目完成度总结

当前项目已经从早期 Todo API 演进为一个完整的企业支持 AI Copilot 后端 MVP。

已完成的关键闭环包括：

```text
RAG:
企业文档 → embedding → Chroma → /rag/search → /rag/ask → sources

Document Backend:
upload → index → RAG search hit → delete → RAG search miss

Ticket Agent:
preview → approval_request.pending → confirm → create_ticket

AgentOps:
agent_run → tool_calls → approval_request → metrics summary

Validation:
pytest focused tests → retrieval eval → smoke scripts

Runtime:
local uvicorn → Docker Compose local runtime
```

当前项目的核心价值可以概括为：

```text
它不是一个只会回答问题的 RAG demo，
而是一个把企业知识库、文档生命周期、受控工单执行和 AgentOps 审计串起来的后端系统。
```

当前项目仍是 MVP，不是生产系统。但作为后端工程作品，它已经具备：

```text
清晰的业务场景
可解释的系统架构
可验证的端到端链路
可审计的 Agent 执行记录
可复现的本地 Docker 运行方式
明确的安全边界和生产化路线
```

---
