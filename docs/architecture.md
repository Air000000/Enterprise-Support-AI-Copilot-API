# 系统架构

本文档描述 Enterprise Support AI Copilot 当前已经稳定存在的运行路径、模块边界和请求流程。重点记录**现状**，不把未来重构设想写成已实现能力。

## 1. 系统主链路

```text
企业知识文档 / 上传文档
→ 文档生命周期管理
→ Hybrid 检索（Chroma Dense + BM25 + RRF + rerank）
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
| RAG 运行时 | `rag_runtime/` | Dense / Hybrid 检索、证据门控、Context 组装、LLM 调用 |
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

### /rag/search（显式 Dense 回退）

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

### /rag/ask（显式 Dense 回退）

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

默认搜索、问答和 Ticket preview 使用 `retrieval_mode=hybrid`；问答默认 `source_scope_experimental`。显式 Dense 回退流程如上。两个 Hybrid 开关默认开启，已有配置中的显式 `false` 仍生效。默认检索路径：

```text
CurrentUser tenant + category
→ SQL indexed/current-version DocumentChunk allowlist
→ Dense100 (Chroma tenant/category + managed-source metadata filter, then SQL eligibility check)
  + BM25100 (same SQL scope)
→ RRF(k=60) Top100
→ SQL visibility/content recheck
→ qwen3-rerank → SQL recheck → Top14 prefix
→ typed scores + RetrievalLog
```

`rag_runtime/hybrid_search.py` 复用现有分词、BM25、RRF 和重排纯组件，不运行评测或加载评测数据。
BM25 使用按租户 / 分类 / 完整 SQL chunk 内容键控的进程内 LRU 缓存；默认最多 5000 条可见 chunk，无额外持久索引。
无技术词的中文查询跳过 BM25。重排失败返回脱敏 503，由调用方显式选 Dense 回退。
搜索本身不生成答案、不套用 Dense 拒答阈值。配置、外发说明和分数契约见 [README](../README.md#31-默认-hybrid-搜索)。
SQL 复查缩小并发变更窗口，但不提供跨 SQL/Chroma 的原子快照。

`/rag/ask` 默认使用以下来源级证据门控路径，省略模式 / 策略即可；`max_distance` 必须省略或为 null。显式旧策略保留原契约：

```text
Hybrid Top14 prefix
→ current SQL visibility/content recheck
→ source_scope classifier → strict source-ID / requirement / coverage validation
→ INSUFFICIENT: fixed refusal (no generation)
→ SUFFICIENT: full-context generation → supporting-source ID validation
→ current SQL recheck → response + typed scores/gate audit
```

实现位于 `rag_runtime/hybrid_ask.py`，复用冻结的纯提示词/校验函数，不运行评测或改写冻结源码。
仅按策略选择已有消息构建函数，检索、解析、生成与可见性复查共用；响应及各类审计保留原策略名，不自动切换版本。v5.3.1 接线通过不代表真实模型质量已通过验收。
显式 `v5_3_1_contract_experimental` 另在 v5.3.1 分类提示末尾重申输出契约，保持严格校验；分类仍读取完整 Top14 前缀，生成改为仅使用门控批准来源的完整 chunk，保留原编号与顺序，响应 sources 不裁剪。它不修改冻结策略构建器、默认模式或历史请求，既不修补模型输出，也不重试。该上下文选择可能影响条件覆盖，离线接线检查不能证明真实语义质量或可靠性改善；原冻结评测运行器不用于该新契约。
无上下文为 `no_context`；gate 不足为 `refused_insufficient_evidence`；生成器拒答为 `refused_by_generator`。
格式、截断、非法引用、提供方或可见性错误均返回脱敏 503、审计 `failed`，不会当作正确拒答或自动切换 Dense。
分类使用强制专用 Phase B 配置，生成复用通用 chat 配置；两者无自动重试，完整上下文属于新增外发。
Qwen3.5 Plus 生成与分类均显式关闭思考模式；生成仍为 1024 输出 tokens、30 秒超时。
失败审计仅增加分类/生成/未细分服务阶段与有界异常类型链，不记录异常正文；验收脚本另记响应读取阶段、已知 usage 和结束原因，不记录隐藏思考。
来源级分类器只输出来源 ID，程序投影完整原文并验证；`evidence_unit=full_source_chunk` 不代表精确句段选择或语义支持。生成只接收批准来源的完整正文。版本覆盖审计仍有已知语义错误；切换默认不改写历史负结果。详细配置、输出上限与限制见 [证据门控问答说明](../README.md#32-默认-hybrid-问答与证据门控)。
它不是 v5.3 / v5.3.1 正式晋级、生产可用性承诺或 Ticket Agent 逻辑替换。

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

`preview` 默认 `retrieval_mode=hybrid`，复用 `services.rag_service.search_documents`，受搜索开关控制，默认 Top14；无需 ask 开关，显式 Dense 使用 Top3。来源距离可为空，独立保留重排/融合分数与通道排名；Hybrid 的 `search_kb` 输出及运行摘要不把距离冒充最终分数。Hybrid 失败脱敏为 503，运行/工具均记失败，不继续产生审批或暗中切回 Dense。

此处只共享检索；工单建议仍由原规则生成，不运行 v5.x 证据分类或答案生成。`confirm` 不重检索，保留 pending、租户、草稿匹配及条件消费约束，只有显式确认才创建工单。

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

- 仓库默认切换等于某个运行进程已更新、公网已部署或已证实质量改善；
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
