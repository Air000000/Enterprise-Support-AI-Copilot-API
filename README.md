# Enterprise Support AI Copilot

[![Tests](https://github.com/Air000000/fastapi-todo-api/actions/workflows/tests.yml/badge.svg?branch=learn-rag)](https://github.com/Air000000/fastapi-todo-api/actions/workflows/tests.yml)

企业内部知识库与工单 AgentOps 后端系统。

本项目最初由 FastAPI Todo / AI Todo API 演进而来，当前 `learn-rag` 分支已经升级为面向企业内部支持场景的 AI Copilot 后端项目。

当前系统已经完成 Enterprise RAG Core、Ticket CRUD、Ticket Agent preview / confirm，以及 AgentOps 审计记录的 MVP 闭环。项目重点从单纯知识库问答扩展为“知识检索 → 工单预览 → 人工确认 → 工单创建 → AgentOps 审计”的企业支持 Copilot 后端。

当前版本定位：

```text
Enterprise Support AI Copilot
├── Enterprise RAG Core
│   ├── 企业内部支持文档集
│   ├── 文档加载与切块
│   ├── embedding 批处理
│   ├── Chroma 向量库
│   ├── tenant/category metadata filter
│   ├── /rag/search
│   ├── /rag/ask
│   ├── answer + structured sources
│   ├── retrieval eval
│   ├── hit@1 / hit@3 / mrr@3
│   └── category breakdown
├── Ticket CRUD
│   ├── create ticket
│   ├── list ticket
│   ├── get ticket
│   └── update ticket
├── Ticket Agent
│   ├── /agent/ticket/preview
│   ├── /agent/ticket/confirm
│   ├── TicketDraft
│   ├── TicketAgentSource
│   ├── approval ownership validation
│   ├── pending approval validation
│   └── draft payload consistency check
└── AgentOps
    ├── agent_runs
    ├── approval_requests
    ├── tool_calls
    ├── search_kb tool audit
    ├── classify_ticket tool audit
    ├── create_ticket tool audit
    ├── approval reject / cancel APIs
    └── AgentOps metrics summary API
```

---

## 1. 项目定位

本项目模拟企业内部支持场景。员工可以围绕 IT、HR、财务、行政、安全等内部制度和流程提出问题，系统基于企业知识库进行检索，并返回带来源依据的回答。

当前版本已扩展到工单处理场景：当用户问题更适合进入支持流程时，Ticket Agent 会生成工单预览，并通过人工确认后创建真实工单。系统同时记录 Agent 运行过程、工具调用输入输出、审批状态和基础运行指标。

核心链路如下：

```text
企业内部文档
↓
文档加载与切块
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
human approval
↓
create ticket
↓
tool_calls / agent_runs / approval_requests
↓
AgentOps metrics summary
```

当前项目由三层组成：

| 层级    | 模块                         | 说明                                                             |
| ----- | -------------------------- | -------------------------------------------------------------- |
| 知识访问层 | Enterprise RAG Core        | 负责企业文档检索、引用来源返回、无依据拒答和 retrieval eval                          |
| 业务执行层 | Ticket CRUD / Ticket Agent | 负责工单创建、查询、更新，以及 preview / confirm 两阶段流程                        |
| 审计观测层 | AgentOps                   | 负责记录 agent_runs、tool_calls、approval_requests 和 metrics summary |

---

## 2. 当前阶段

当前阶段：Enterprise Support AI Copilot MVP

已完成能力：

| 模块                       | 当前状态                                                 |
| ------------------------ | ---------------------------------------------------- |
| FastAPI 基础后端             | 已完成                                                  |
| Todo CRUD                | 已保留，作为早期基础后端能力                                       |
| AI Todo / LLM 调用         | 已保留，作为早期 LLM API 调用练习                                |
| RAG API                  | 已完成 `/rag/search`、`/rag/ask`                         |
| 企业文档集                    | 已完成 10 份企业内部支持文档                                     |
| 文档分类                     | 已支持 `it`、`hr`、`finance`、`admin`、`security`           |
| 文档 metadata              | 已支持 `tenant_id`、`category`                           |
| Chunk metadata           | 已支持 `tenant_id`、`category`                           |
| Chunk 策略                 | 已针对企业长文档调优                                           |
| Embedding                | 已支持分批请求                                              |
| Vector store             | 使用 Chroma 持久化向量库                                     |
| Metadata filter          | 已支持 tenant/category 过滤                               |
| API category filter      | 已支持 request.category + mock tenant context           |
| Sources                  | RAG response 返回结构化来源信息                               |
| Retrieval eval           | 已支持 hit@1、hit@3、mrr@3、avg latency、category breakdown |
| Ticket CRUD              | 已完成 create / list / get / update                     |
| Ticket Agent             | 已完成 preview / confirm 两阶段流程                          |
| Human approval           | 已通过 approval_requests 表记录                            |
| Tool calls audit         | 已记录 search_kb / classify_ticket / create_ticket      |
| AgentOps records         | 已支持 agent_runs / approval_requests / tool_calls      |
| AgentOps 查询 API          | 已完成 MVP                                              |
| Approval reject / cancel | 已完成 MVP                                              |
| AgentOps metrics summary | 已完成 MVP                                              |
| 测试                       | RAG / Ticket / AgentOps / Todo focused tests 均已覆盖    |

后续开发重点：

| 模块 | 状态 |
|---|---|
| AgentRun latency / retrieval summary | 已完成 MVP，后续补充时间窗口统计 |
| Retrieval logs / metrics | 已完成 MVP，已支持日志明细、summary、sources、no-context queries 和 failures |
| ToolCall 错误记录 | 已完成 MVP，后续细化 error_type |
| AgentOps dashboard / 时间窗口筛选 | 后续增强 |
| Document upload API | 待开发 |
| Docker Compose 最终部署版 | 待整理 |
| 真实 tenant / user auth context | 待开发 |
| 真实前端审批界面 | 待开发 |

---

## 3. AgentOps 当前能力

当前 Ticket Agent 的完整执行链路会产生三类记录：

```text
agent_runs
tool_calls
approval_requests
```

一次成功的 preview / confirm 流程会产生：

```text
1 条 agent_run
3 条 tool_call:
  - search_kb
  - classify_ticket
  - create_ticket
1 条 approval_request
```

### search_kb

`search_kb` 记录知识库检索动作，包括：

```text
query
top_k
tenant_id
category
results_count
document_ids
top_distance
```

### classify_ticket

`classify_ticket` 记录工单判断动作，包括：

```text
message
requested_category
sources_count
source_categories
source_document_ids
should_create_ticket
category
priority
reason
```

### create_ticket

`create_ticket` 记录真实工单创建动作，包括：

```text
title
description
category
priority
ticket_id
ticket status
```

### AgentOps 审计增强

当前版本的 AgentOps 不只记录 Agent Run、Tool Call 和 Approval Request 的基础状态，还补充了更细的审计字段：

* `approval_requests.decision_reason`：记录审批通过、拒绝或取消的原因。
* `tool_calls.error_type`：记录工具调用失败类型，例如 `search_kb_failed`、`classify_ticket_failed`、`create_ticket_failed`。
* `metrics summary.tool_call_error_types`：按失败类型统计 tool call，便于观察 Agent 执行链路中最常见的失败来源。

这使系统从“能记录执行结果”进一步升级为“能解释失败原因，并支持后续运维分析”。

---

## 4. Ticket Agent 安全边界

Ticket Agent 使用 preview / confirm 两阶段流程。

Preview 阶段只生成工单草稿和审批请求，不创建真实工单：

```text
POST /agent/ticket/preview
↓
agent_run
↓
search_kb tool_call
↓
classify_ticket tool_call
↓
ticket draft
↓
approval_request.pending
```

Confirm 阶段在创建真实工单前会执行三类校验：

```text
1. approval_request 必须属于当前 agent_run
2. approval_request.status 必须是 pending
3. confirm draft 必须与服务端保存的 approval_request.draft_json 一致
```

校验通过后，系统使用服务端保存的 draft 创建真实工单，并记录 `create_ticket` tool_call：

```text
POST /agent/ticket/confirm
↓
approval validation
↓
draft consistency check
↓
approval_request.approved
↓
create_ticket tool_call
↓
ticket created
```

该设计用于避免重复确认、拒绝后执行、跨 agent_run 使用 approval_request，以及 confirm 阶段 payload 被修改的问题。

---

## 5. AgentOps API

当前 AgentOps 查询接口如下：

```http
GET  /agent-ops/runs
GET  /agent-ops/runs/{agent_run_id}
GET  /agent-ops/runs/{agent_run_id}/tool-calls
GET  /agent-ops/runs/{agent_run_id}/approval-requests
GET  /agent-ops/metrics/summary

POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

`/agent-ops/metrics/summary` 示例：

```json
{
  "total_agent_runs": 1,
  "running_agent_runs": 0,
  "completed_agent_runs": 1,
  "failed_agent_runs": 0,
  "cancelled_agent_runs": 0,
  "total_tool_calls": 3,
  "pending_tool_calls": 0,
  "successful_tool_calls": 3,
  "failed_tool_calls": 0,
  "total_approval_requests": 1,
  "pending_approval_requests": 0,
  "approved_approval_requests": 1,
  "rejected_approval_requests": 0,
  "cancelled_approval_requests": 0
}
```

---


## 6. 技术栈

| 类型                | 技术                                        |
| ------------------- | ------------------------------------------- |
| Web 框架            | FastAPI                                     |
| 数据校验            | Pydantic                                    |
| ORM / 数据库        | SQLModel + SQLite                           |
| LLM / Embedding API | OpenAI-compatible SDK + DashScope / Bailian |
| Vector store        | Chroma                                      |
| 测试                | pytest + FastAPI TestClient                 |
| 部署基础            | Docker                                      |
| 语言                | Python                                      |

------

## 7. 当前项目结构

```text
fastapi-todo-api/
├── main.py
├── database.py
├── models/
│   ├── ticket.py
│   └── agent_ops.py
├── routers/
│   ├── rag.py
│   ├── tickets.py
│   ├── agent_ticket.py
│   └── agent_ops.py
├── schemas/
│   ├── rag.py
│   ├── ticket.py
│   ├── agent_ticket.py
│   └── agent_ops.py
├── services/
│   ├── rag_service.py
│   ├── ticket_service.py
│   ├── ticket_agent_service.py
│   └── agent_ops_service.py
├── experiments/
│   ├── docs/
│   │   ├── admin/
│   │   ├── finance/
│   │   ├── hr/
│   │   ├── it/
│   │   └── security/
│   ├── index/
│   ├── rag_local/
│   └── evals/
├── docs/
│   ├── agent_workflow.md
│   ├── demo_script.md
│   ├── security.md
│   ├── rag_v0.1_report.md
│   ├── rag_core_v1_report.md
│   ├── ticket_crud_mvp_report.md
│   ├── ticket_agent_mvp_report.md
│   └── retrieval_metrics.md
├── tests/
│   ├── test_todos.py
│   ├── test_query_chroma.py
│   ├── test_rag_api.py
│   ├── test_rag_service.py
│   ├── test_tickets.py
│   ├── test_agent_ops_service.py
│   ├── test_agent_ops_api.py
│   ├── test_ticket_agent_service.py
│   └── test_agent_ticket_api.py
├── data/
├── Dockerfile
├── requirements.txt
├── .env.example
├── AGENTS.md
└── README.md
```

------

## 8. 文档索引

| 文档                                                                   | 说明                                                                                                   |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------|
| [`docs/agent_workflow.md`](docs/agent_workflow.md)                   | Ticket Agent preview / confirm 流程、tool_calls 链路、AgentOps 查询方式                                   |
| [`docs/demo_script.md`](docs/demo_script.md)                         | Swagger / API 手动演示脚本，用于展示 RAG、Ticket Agent、approval 和 metrics                                |
| [`docs/security.md`](docs/security.md)                               | Ticket Agent 安全边界，包括 approval ownership、pending 校验、draft consistency check 和 tool call audit   |
| [`docs/rag_core_v1_report.md`](docs/rag_core_v1_report.md)           | Enterprise RAG Core 阶段报告                                                                              |
| [`docs/ticket_crud_mvp_report.md`](docs/ticket_crud_mvp_report.md)   | Ticket CRUD MVP 阶段报告                                                                                  |
| [`docs/ticket_agent_mvp_report.md`](docs/ticket_agent_mvp_report.md) | Ticket Agent MVP 阶段报告                                                                                 |
| [`docs/retrieval_metrics.md`](docs/retrieval_metrics.md)             | Retrieval Logs / Metrics 阶段报告，说明 RAG 检索日志、summary、sources、no-context queries 和 failures 指标 |

------

## 9. 环境变量

在项目根目录创建 `.env` 文件，并参考 `.env.example` 配置本地运行参数：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
```

------

## 10. 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn main:app --reload
```

打开 Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

------

## 11. AgentOps Smoke Test

本项目提供一个端到端 smoke script，用于验证 Ticket Agent 与 AgentOps 的完整 API 链路。

该脚本会依次执行：

```text
GET  /health
POST /agent/ticket/preview
GET  /agent-ops/runs/{agent_run_id}/tool-calls
GET  /agent-ops/runs/{agent_run_id}/approval-requests
POST /agent/ticket/confirm
GET  /agent-ops/runs/{agent_run_id}/tool-calls
GET  /agent-ops/metrics/summary
```
运行前先启动服务：
uvicorn main:app --reload

另开一个终端执行：
python scripts/smoke_agentops_flow.py

预期输出：
Smoke test passed.
Validated: preview -> search_kb/classify_ticket -> approval -> confirm -> create_ticket -> metrics

如需指定 API 地址：
API_BASE_URL=http://127.0.0.1:8000 python scripts/smoke_agentops_flow.py

Windows PowerShell 里最后一条可以写成：

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/smoke_agentops_flow.py
```
------

## 12. 构建 Chroma 向量索引

当前企业文档位于：

```text
experiments/docs/
```

构建 Chroma index：

```bash
python -m experiments.rag_local.build_chroma_index
```

当前企业文档集构建结果：

```text
Loaded documents: 10
Generated chunks: 40
Generated embeddings: 40
Collection count: 40
```

------

## 13. RAG 检索命令示例

### IT 类问题

```bash
python -m experiments.rag_local.query_chroma "VPN 连不上应该先检查什么？" --top-k 3 --tenant-id tenant_demo --category it
```

### HR 类问题

```bash
python -m experiments.rag_local.query_chroma "请假需要在系统里提交吗？" --top-k 3 --tenant-id tenant_demo --category hr
```

### 财务类问题

```bash
python -m experiments.rag_local.query_chroma "差旅报销需要哪些材料？" --top-k 3 --tenant-id tenant_demo --category finance
```

### 行政类问题

```bash
python -m experiments.rag_local.query_chroma "会议室临时不用需要释放吗？" --top-k 3 --tenant-id tenant_demo --category admin
```

### 安全类问题

```bash
python -m experiments.rag_local.query_chroma "外包人员可以共用账号吗？" --top-k 3 --tenant-id tenant_demo --category security
```

------

## 14. RAG API

### POST `/rag/search`

功能：基于 Chroma 向量库执行 top-k 检索，支持 category filter。

请求示例：

```json
{
  "query": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "category": "it"
}
```

字段说明：

| 字段       | 含义                                                         |
| ---------- | ------------------------------------------------------------ |
| `query`    | 用户问题                                                     |
| `top_k`    | 返回的检索结果数量                                           |
| `category` | 可选分类过滤，例如 `it`、`hr`、`finance`、`admin`、`security` |

当前 API 层只暴露 `category` filter。`tenant_id` 暂时由系统内部 mock tenant context 提供：

```text
tenant_demo
```

后续接入登录系统后，`tenant_id` 将从当前用户身份中获取。

响应字段：

| 字段          | 含义                 |
| ------------- | -------------------- |
| `document_id` | 来源文档 ID          |
| `chunk_id`    | 命中的 chunk ID      |
| `title`       | 文档标题             |
| `source_path` | 文档路径             |
| `chunk_index` | chunk 在文档中的顺序 |
| `distance`    | 向量距离             |
| `preview`     | chunk 内容预览       |
| `tenant_id`   | 来源租户             |
| `category`    | 来源分类             |

------

### POST `/rag/ask`

功能：执行完整 RAG 流程，包括 Chroma 检索、上下文构造、LLM 回答生成和 sources 返回。

请求示例：

```json
{
  "question": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "max_distance": 0.9,
  "category": "it"
}
```

响应字段：

| 字段               | 含义                      |
| ------------------ | ------------------------- |
| `answer`           | 基于检索上下文生成的回答  |
| `retrieval_status` | 检索状态                  |
| `top_distance`     | top1 检索结果距离         |
| `sources`          | 回答引用的来源 chunk 列表 |

当检索不到足够相关内容时，系统返回拒答：

```text
我在已提供资料中没有找到足够依据。
```

------

## 15. 当前 API 分层

```text
POST /rag/search
→ routers/rag.py::rag_search()
→ services/rag_service.py::search_documents()
→ experiments/rag_local/query_chroma.py::search_chroma()

POST /rag/ask
→ routers/rag.py::rag_ask()
→ services/rag_service.py::answer_question()
→ experiments/rag_local/query_rag_chroma.py::ask_rag()
```

各层职责：

| 层                        | 职责                                            |
| ------------------------- | ----------------------------------------------- |
| `routers/rag.py`          | 处理 HTTP 请求、调用 service、组装 API response |
| `schemas/rag.py`          | 定义 request / response schema                  |
| `services/rag_service.py` | 提供 RAG 业务入口                               |
| `experiments/rag_local/`  | 执行底层 RAG、Chroma、embedding、LLM 逻辑       |
| `tests/`                  | 验证 API 层和 service 层行为                    |

------

## 16. 测试

运行测试：

```bash
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
pytest tests/test_todos.py
pytest tests/test_tickets.py
pytest tests/test_agent_ops_service.py
pytest tests/test_agent_ops_api.py
pytest tests/test_ticket_agent_service.py
pytest tests/test_agent_ticket_api.py
```

当前结果：

```text
RAG / Ticket / AgentOps / Todo focused test suites are passing.
```

测试覆盖：

| 文件 | 覆盖内容 |
| ---- | -------- |
| `tests/test_todos.py` | Todo 基础 CRUD 和请求校验 |
| `tests/test_query_chroma.py` | Chroma metadata filter |
| `tests/test_rag_api.py` | RAG API happy path、validation、service error |
| `tests/test_rag_service.py` | RAG service 层参数透传和下游调用 |
| `tests/test_tickets.py` | Ticket CRUD |
| `tests/test_agent_ops_service.py` | AgentRun / ToolCall / ApprovalRequest service + metrics summary |
| `tests/test_agent_ops_api.py` | AgentOps read API + approval reject / cancel API + metrics summary API |
| `tests/test_ticket_agent_service.py` | Ticket Agent preview / confirm / AgentOps 轨迹 |
| `tests/test_agent_ticket_api.py` | Ticket Agent API request / response / validation |

RAG 测试使用 `monkeypatch` 隔离真实 Chroma、embedding 和 LLM 调用，因此不会消耗 token。

------

## 17. 当前版本记录

| Version                         | Documents | Categories                           | Chunks | Vector Store  | Eval |
| ------------------------------- | --------- | ------------------------------------ | ------ | ------------- | ---- |
| RAG v0.1 learning-doc baseline  | 5         | general                              | 5      | JSON / Chroma | 15 条 learning eval |
| Enterprise RAG Core             | 10        | it / hr / finance / admin / security | 40     | Chroma        | 30 条 enterprise eval |
| Ticket Agent MVP                | 10        | it / hr / finance / admin / security | 40     | Chroma + SQLite | preview / confirm / AgentOps tests |

旧 learning-doc baseline 的 eval 结果：

| Retriever         | hit@1 | hit@3 | top1 miss cases | failed cases |
| ----------------- | ----- | ----- | --------------- | ------------ |
| JSON cosine index | 0.93  | 1.00  | 1               | 0            |
| Chroma            | 0.93  | 1.00  | 1               | 0            |

旧 eval 基于 FastAPI、Docker、Embedding、RAG、SQLModel 学习文档，主要用于记录早期 RAG 学习阶段的 baseline。当前企业文档集会使用新的 enterprise RAG eval 进行评测。

企业 Chroma eval 当前结果：

| Metric | Result |
| ------ | ------ |
| Total | 30 |
| hit@1 | 0.97 |
| hit@3 | 1.00 |
| mrr@3 | 0.98 |

当前唯一主要 top1 miss 是 finance 场景中 `doc_travel_reimbursement` 与 `doc_invoice_rules` 的边界问题，属于业务文档语义重叠，不是检索链路故障。
------


## 18. 下一步计划

短期下一步：

```text
AgentOps demo and presentation cleanup
```

计划包括：
1. 端到端 API smoke test
2. README / demo script 补充完整演示流程
3. AgentOps dashboard / 时间窗口筛选
4. ToolCall error_type 细化
5. 真实 tenant / user auth context
6. Docker Compose 最终部署版

中期计划：
1. retrieval logs 后端化
2. documents / chunks 后端化
3. document upload API
4. 真实 tenant / user auth context
5. AgentOps dashboard
6. Docker Compose 最终部署版

最终目标：
企业内部知识库 RAG + 受控 Ticket Agent + AgentOps 审计与评测
------

## 19. 项目说明

仓库名保留为 `fastapi-todo-api`，因为项目最初从 FastAPI Todo API 开始演进。当前 `learn-rag` 分支是 `Enterprise Support AI Copilot` 的开发分支。

当前 `learn-rag` 分支已经完成 Enterprise RAG Core v1、Ticket CRUD MVP、Ticket Agent MVP、AgentOps read API、approval reject / cancel API 和 AgentOps metrics summary API。后续可以在该分支继续推进 AgentRun latency / retrieval summary、ToolCall 错误类型细化、真实 tenant / user auth context、真实前端审批界面、AgentOps dashboard / 时间窗口筛选和 Docker Compose 最终部署版。

