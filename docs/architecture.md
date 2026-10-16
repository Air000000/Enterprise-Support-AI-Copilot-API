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

### /rag/search（默认 Dense）

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

默认 serving、`/rag/ask` 和 Ticket Agent 仍是 Dense Chroma。`/rag/search` 增加默认关闭的 `retrieval_mode=hybrid` 路径：

```text
CurrentUser tenant + category
→ SQL indexed/current-version DocumentChunk allowlist
→ Dense100 (Chroma ids + metadata filter) + BM25100 (same SQL scope)
→ RRF(k=60) Top100
→ SQL visibility/content recheck
→ qwen3-rerank → SQL recheck → Top14 prefix
→ typed scores + RetrievalLog
```

`rag_runtime/hybrid_search.py` 复用现有分词、BM25、RRF 和重排纯组件，不运行评测或加载评测数据。
BM25 按请求重建、最多 5000 条可见 chunk；无额外持久索引或缓存失效系统。
无技术词的中文查询跳过 BM25。重排失败返回脱敏 503，由调用方显式选 Dense 回退。
本路径不生成答案、不套用 Dense 拒答阈值。开启、外发说明和分数契约见 [README](../README.md#31-可选-hybrid-搜索)。
SQL 复查缩小并发变更窗口，但不提供跨 SQL/Chroma 的原子快照。

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
→ 数据库条件更新：同一租户的 pending 审批原子转换为 approved
→ create_ticket ToolCall
→ 创建真实 Ticket
→ 更新 AgentRun
```

`classify_ticket` 当前是确定性规则，不描述为自主 LLM planning。

confirm / reject / cancel 共用审批条件更新；并发决策只有一个成功，终态不能覆盖或重新打开。
该保证仅限审批至多一次消费，工单与审计仍分别写入，不提供跨事务恰好一次或崩溃恢复保证。

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

- 默认线上检索或问答/工单已经切换到 Hybrid Retrieval（目前仅默认关闭的可选搜索）；
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
