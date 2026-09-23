# 系统架构

本文档描述 Enterprise Support AI Copilot 当前已经稳定存在的运行路径、模块边界和请求流程。重点记录**现状**，不把未来重构设想写成已实现能力。

## 1. 系统主链路

```text
企业知识文档 / 上传文档
→ 文档生命周期管理
→ Chroma 检索
→ RAG 回答 + 来源
→ 工单 Agent 预览
→ 人工确认
→ 创建真实工单
→ AgentOps 审计与指标
```

当前稳定能力：

- 企业 RAG 核心
- 文档后端
- Ticket CRUD
- 工单 Agent 预览 / 确认
- AgentOps 审计与查询 API
- Retrieval Log / Metrics
- Docker Compose 本地运行
- 冒烟测试脚本

## 2. 仓库结构

```text
enterprise-support-ai-copilot-api/
├── main.py
├── database.py
├── routers/
├── schemas/
├── services/
├── models/
├── rag_runtime/
├── experiments/
│   ├── docs/
│   ├── evals/
│   └── rag_local/
├── scripts/
├── tests/
├── docker-compose.yml
└── README.md
```

| 层 | 主要目录 | 职责 |
| --- | --- | --- |
| API 层 | `routers/` | HTTP 请求处理、认证依赖、状态码与响应组装 |
| Schema 层 | `schemas/` | 请求 / 响应数据模型 |
| Service 层 | `services/` | 业务编排与状态流转 |
| 持久化层 | `models/`、`database.py` | SQLModel 表与 SQLite Session |
| RAG 运行时 | `rag_runtime/` | Chroma 检索、Context 组装、LLM 调用 |
| 兼容层 | `experiments/rag_local/` | 兼容历史 import 与 `python -m` 入口 |
| 评测层 | `experiments/evals/` | TechQA 检索、生成、拒答与失败归因 |
| 脚本 | `scripts/` | 冒烟测试与本地验证 |

## 3. 当前运行时边界

### 正式 RAG 运行时

`rag_runtime/` 是当前正式运行时包。

`experiments/rag_local/` 继续保留，仅作为历史兼容层：

- 旧 import 仍能解析；
- 历史 `python -m experiments.rag_local.*` 命令仍可运行；
- 已有实验与笔记不会因一次性迁移全部失效。

当前业务服务直接引用 `rag_runtime.*`：

- `services/rag_service.py`
- `services/document_service.py`
- `services/ticket_agent_service.py`

### 历史 Todo 兼容

Todo / AI Todo API 仍保留用于兼容和历史测试，但不再是项目主线：

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`

### SQLite 文件名

默认数据库仍为：

```text
data/todos.db
```

文件名来自早期项目历史，不代表当前产品定位。

## 4. RAG 请求链路

### /rag/search

```text
POST /rag/search
→ routers/rag.py
→ services/rag_service.py
→ rag_runtime/query_chroma.py
→ Query Embedding
→ Chroma query
→ Search Results
→ RetrievalLog
```

### /rag/ask

```text
POST /rag/ask
→ routers/rag.py
→ services/rag_service.py
→ rag_runtime/query_rag_chroma.py
→ Dense Retrieval
→ relevance gate
→ evidence Context
→ LLM answer / fixed refusal
→ sources
→ RetrievalLog
```

当前在线 serving 仍是 Dense Chroma。BM25、RRF、Hybrid、`qwen3-rerank` 与直接 Top14 目前属于离线冻结工程候选。

## 5. 文档生命周期

```text
POST /documents/upload
→ documents 表记录
→ 文件写入 storage
→ status = uploaded

POST /documents/{document_id}/index
→ split_text
→ embedding
→ Chroma.add
→ DocumentChunk
→ status = indexed

DELETE /documents/{document_id}
→ 删除 relational chunks
→ 删除 Chroma embeddings
→ Document status = deleted
```

上传不会自动进入检索；只有显式 index 后才可被 RAG 查询。

SQLite 与 Chroma 是两个独立存储，目前没有跨存储分布式事务，因此异常中断时存在状态不一致风险。

## 6. 工单 Agent

```text
POST /agent/ticket/preview
→ 创建 AgentRun
→ search_kb ToolCall
→ 知识库检索
→ classify_ticket ToolCall
→ 规则化工单判断
→ TicketDraft
→ ApprovalRequest.pending
→ 返回 preview
```

Confirm：

```text
POST /agent/ticket/confirm
→ 校验 ApprovalRequest 属于当前 AgentRun
→ 校验 status == pending
→ 校验 request.draft == server draft_json
→ Approval approved
→ create_ticket ToolCall
→ 创建真实 Ticket
→ 更新 AgentRun
```

`classify_ticket` 当前是确定性规则，不描述为自主 LLM planning。

## 7. AgentOps

主要实体：

```text
agent_runs
├── tool_calls
└── approval_requests

retrieval_logs
```

AgentOps 用于回答：

- 一次 Agent 运行发生了什么；
- 哪些工具被调用；
- 哪一步失败；
- 是否有待审批动作；
- 审批最终状态；
- RAG 是否命中、拒答或失败；
- 单次 Run 的完整 trace。

这些记录由应用显式写入，不是自动采集，也不是模型 Chain-of-Thought。

## 8. 认证与租户范围

当前使用 Demo JWT：

```text
Bearer Token
→ CurrentUser
   ├─ user_id
   ├─ tenant_id
   └─ role
```

用于：

- RAG tenant filter；
- Document tenant scope；
- Ticket / Approval tenant scope；
- AgentOps tenant scope；
- support / admin 角色限制。

它验证的是应用层认证上下文传播，不描述为完整生产级 IAM 或数据库级多租户隔离。

## 9. 当前明确边界

当前不声明：

- 在线已经切换到 Hybrid Retrieval；
- Ticket Agent 是自主 ReAct / Function Calling Agent；
- Demo JWT 是生产级 IAM；
- Docker Compose 是生产部署；
- approval pending 检查提供并发 exactly-once 保证；
- SQLite + Chroma 具备跨存储事务；
- Flat Top14 是通用最优 Context。

## 10. 相关文档

- [项目 README](../README.md)
- [工单 Agent 工作流](agent_workflow.md)
- [安全边界](security.md)
- [TechQA 评测总览](../experiments/evals/README.md)
- [RAG 冻结架构](../experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md)
