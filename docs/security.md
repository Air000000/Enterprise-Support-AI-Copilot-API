# 安全设计

Enterprise Support AI Copilot 安全设计说明。

本文档记录当前 MVP 已实现的安全边界、仍然存在的风险，以及后续生产化需要补齐的控制项。

当前项目覆盖：

```text
Enterprise RAG Core
Document Backend
Ticket Agent 预览 / 确认
AgentOps audit
Retrieval Logs / Metrics
Docker Compose local runtime
Smoke Scripts
```

本文档重点说明：

```text
1. Ticket Agent 的 预览 / 确认 两阶段执行边界
2. approval_request 归属 / pending / 草稿一致性 校验
3. Document Backend 的上传、索引、删除安全边界
4. RAG tenant / category filter 的 MVP 边界
5. AgentOps 审计数据的能力与限制
6. Docker Compose 本地运行边界
7. API key / .env 管理边界
8. 公网暴露、限流、成本控制和生产化风险
```

---

## 1. 适用范围

当前安全设计覆盖以下模块：

```text
Ticket Agent
AgentOps
Ticket creation workflow
Approval request workflow
Tool call audit
Document Backend
RAG retrieval
Docker Compose local runtime
Smoke Scripts
```

相关 API：

```http
POST   /rag/search
POST   /rag/ask

POST   /documents/upload
GET    /documents
GET    /documents/{document_id}
POST   /documents/{document_id}/index
DELETE /documents/{document_id}

POST   /agent/ticket/preview
POST   /agent/ticket/confirm

GET    /agent-ops/runs
GET    /agent-ops/runs/{agent_run_id}
GET    /agent-ops/runs/{agent_run_id}/tool-calls
GET    /agent-ops/runs/{agent_run_id}/approval-requests
GET    /agent-ops/metrics/summary

POST   /agent-ops/approval-requests/{approval_request_id}/reject
POST   /agent-ops/approval-requests/{approval_request_id}/cancel
```

当前版本使用 mock tenant / user context。后续接入真实认证系统后，`tenant_id` 和 `user_id` 必须从认证上下文中获取，而不是由客户端直接传入。

---

## 2. 当前 MVP 安全目标

当前 MVP 的安全目标是：

```text
1. Agent 不直接执行状态变更动作
2. 创建真实 ticket 前必须经过人工确认
3. 确认阶段不能使用不可信客户端 payload 创建 ticket
4. approval_request 不能跨 agent_run 使用
5. rejected / cancelled / approved approval_request 不能再次执行
6. 每次工具调用都要留下结构化审计记录
7. 工具调用失败需要进入 AgentOps 记录
8. Document Backend 删除文档时同步清理 Chroma embeddings
9. Docker Compose 本地运行默认只绑定 127.0.0.1
10. API key 通过 .env 管理，不提交到 GitHub
```

其中，`create_ticket` 属于状态变更动作，因此不能在 预览阶段直接执行。

---

## 3. 信任边界

当前系统中的信任边界如下：

| 数据来源 | 信任级别 | 当前处理方式 |
|---|---|---|
| 客户端请求 | 不可信 | 需要 Pydantic Schema 校验 和业务校验 |
| 预览响应中的草稿 | 不可信副本 | 仅供用户查看和 confirm 提交 |
| approval_request.draft_json | 服务端可信记录 | 确认阶段创建 ticket 的依据 |
| agent_run_id | 不完全可信 | 必须和 approval_request 绑定校验 |
| approval_request_id | 不完全可信 | 必须校验 tenant_id、agent_run_id 和 status |
| 上传文件 | 不可信 | 当前仅限制 md/txt，后续需要大小限制、内容扫描和权限校验 |
| RAG 查询 | 不可信 | 当前执行 Schema 校验 和 category filter，后续需要权限上下文 |
| RAG 检索来源 | 辅助依据 | 进入 response 和 tool_call audit |
| Chroma embeddings | 派生数据 | 删除文档时需要同步删除对应 embeddings |
| 工具调用记录 | 审计记录 | 用于追踪工具输入、输出、状态和错误 |
| `.env` | 本地敏感配置 | 不应提交到 GitHub，不应公开粘贴 |
| Docker 运行数据 | 本地运行数据 | 存放于 docker_data / docker_storage / docker_chroma_db，不应提交 |

核心原则：

```text
客户端提交的 draft 只能用于一致性校验；
真正创建 ticket 的 payload 必须来自服务端保存的 approval_request.draft_json。

上传文档在进入 RAG 前必须经过后端登记、索引和 metadata 绑定；
删除文档时必须同时清理业务数据库记录和向量库 embeddings。
```

---

## 4. 工单 Agent 预览 / 确认两阶段执行

Ticket Agent 使用两阶段流程：

```text
预览阶段：生成建议，不执行状态变更
确认阶段：人工确认后，执行真实 ticket 创建
```

### 4.1 预览阶段

接口：

```http
POST /agent/ticket/preview
```

预览阶段会：

```text
1. 创建 agent_run
2. 执行 search_kb
3. 执行 classify_ticket
4. 生成 ticket draft
5. 创建 approval_request.pending
6. 返回 preview response
```

预览阶段不会创建真实 ticket。

预览阶段产生的数据：

```text
agent_runs: 1 条
tool_calls: search_kb + classify_ticket
approval_requests: 1 条 pending
```

### 4.2 确认阶段

接口：

```http
POST /agent/ticket/confirm
```

确认阶段会：

```text
1. 读取 approval_request
2. 校验 approval_request 是否属于当前 agent_run
3. 校验 approval_request 是否仍然是 pending
4. 从 approval_request.draft_json 读取服务端保存的 draft
5. 校验 request.draft 是否与服务端 draft 一致
6. 将 approval_request 更新为 approved
7. 使用服务端 draft 创建真实 ticket
8. 记录 create_ticket tool_call
9. 更新 agent_run 状态
```

确认阶段产生的数据：

```text
ticket: 1 条
tool_calls: create_ticket
approval_requests: status 从 pending 变为 approved
```

---

## 5. 审批归属校验

确认阶段必须校验：

```text
approval_request.agent_run_id == request.agent_run_id
```

该校验用于防止 approval_request 被跨 Agent run 使用。

### 5.1 风险场景

没有 归属校验 时，可能出现：

```text
用户 A 发起 preview，得到 approval_request_id = 10
用户 B 或另一个请求拿到该 approval_request_id
攻击者把它和另一个 agent_run_id 组合后提交 confirm
系统错误地执行不属于当前 agent_run 的审批请求
```

### 5.2 当前处理方式

当 `approval_request.agent_run_id` 与 `request.agent_run_id` 不一致时，系统拒绝 confirm：

```text
Approval request does not belong to agent run
```

并且不执行：

```text
approval_request approved
create_ticket tool_call
ticket creation
agent_run completed update
```

---

## 6. 待审批状态校验

确认阶段必须校验：

```text
approval_request.status == "pending"
```

只有 pending 状态可以继续执行 confirm。

以下状态不能再次执行：

```text
approved
rejected
cancelled
```

### 6.1 防护目标

该校验用于防止：

```text
1. 已经 approved 的 approval_request 被重复 confirm
2. 已经 rejected 的 approval_request 被绕过后再次执行
3. 已经 cancelled 的 approval_request 被恢复执行
4. 同一个 approval_request 导致重复创建 ticket
```

### 6.2 当前处理方式

当 approval_request 不是 pending 时，系统拒绝 confirm：

```text
Approval request is not pending
```

拒绝后不会创建真实 ticket，也不会执行 `create_ticket`。

---

## 7. 草稿一致性校验

预览阶段会把生成的 ticket draft 保存到：

```text
approval_request.draft_json
```

确认阶段会校验：

```text
request.draft == approval_request.draft_json
```

### 7.1 风险场景

没有 草稿一致性校验 时，可能出现：

```text
1. 用户在 预览阶段看到的是低风险 ticket draft
2. 客户端在 确认阶段修改 title / description / category / priority
3. 后端直接使用客户端提交的 draft 创建 ticket
4. 系统创建了用户未真正确认过的内容
```

例如：

```text
preview draft priority = medium
confirm draft priority = urgent
```

或者：

```text
preview draft category = it
confirm draft category = security
```

这类变更会改变业务处理路径，因此必须被拒绝。

### 7.2 当前处理方式

确认阶段会从服务端读取 `approval_request.draft_json`，并与 `request.draft` 做一致性校验。

当二者不一致时，系统拒绝 confirm：

```text
Confirm draft does not match approval draft
```

拒绝后不会执行：

```text
approval_request approved
create_ticket tool_call
ticket creation
```

---

## 8. 服务端审批草稿

当前系统创建真实 ticket 时，使用的是：

```text
approval_request.draft_json
```

而不是直接使用：

```text
request.draft
```

这意味着：

```text
客户端 draft 只参与一致性校验；
服务端 draft 才是 create_ticket 的执行 payload。
```

### 8.1 设计原因

`request.draft` 来自客户端，不属于可信输入。

即使客户端提交的 draft 通过了 Pydantic Schema 校验，也只能证明字段格式合法，不能证明它就是 预览阶段用户看到并确认的内容。

因此，确认阶段的执行 payload 必须来自服务端保存的 approval draft。

---

## 9. 文档后端安全边界

Document Backend 当前支持：

```http
POST   /documents/upload
GET    /documents
GET    /documents/{document_id}
POST   /documents/{document_id}/index
DELETE /documents/{document_id}
```

当前文档生命周期：

```text
upload
↓
documents.status = uploaded
↓
manual index
↓
documents.status = indexed
↓
RAG search can retrieve uploaded document
↓
delete
↓
documents.status = deleted
↓
Chroma embeddings are removed
↓
RAG search no longer returns deleted document
```

### 9.1 已实现边界

当前 MVP 已实现：

```text
1. 上传文档先进入 documents 表，不会自动进入向量库
2. 只有调用 /documents/{document_id}/index 后才会写入 Chroma
3. 文档 chunk metadata 绑定 tenant_id、category、document_id
4. 删除文档时会清理 document_chunks
5. 删除文档时会删除 Chroma 中对应 embeddings
6. 默认 GET /documents/{document_id} 不返回 deleted 文档
7. /rag/search 删除后不应继续召回该文档
```

这些边界用于保证 Document Backend 不是孤立文件上传 API，而是接入了 RAG 知识库生命周期。

### 9.2 当前 MVP 限制

当前 Document Backend 仍然不适合生产使用，原因包括：

```text
1. 尚未接入真实用户认证
2. tenant_id 仍来自 模拟上下文
3. 尚未实现真实 tenant 级文档权限校验
4. 尚未实现上传文件大小限制
5. 尚未实现病毒扫描或恶意内容检测
6. 尚未实现敏感信息检测或脱敏
7. 尚未实现文档版本管理
8. 尚未实现异步索引任务队列
9. 尚未实现索引任务失败重试和 job logs
10. 尚未实现 PDF / Word 等复杂文档解析安全策略
```

### 9.3 Upload Risk

上传接口接收用户文件，因此上传内容必须视为不可信输入。

后续生产化前需要补齐：

```text
1. 文件大小限制
2. 文件类型 allowlist
3. MIME type 校验
4. 文件名规范化
5. 存储路径隔离
6. 恶意文件扫描
7. 文档内容敏感信息检测
8. 每个 tenant 的存储配额
```

### 9.4 Indexing Risk

`POST /documents/{document_id}/index` 会触发：

```text
读取文件
切分文本
调用 embedding API
写入 Chroma
写入 document_chunks
```

因此存在以下风险：

```text
1. 大文件导致 embedding 成本过高
2. 大量重复索引导致 API 费用增加
3. 恶意用户反复触发 index 造成资源消耗
4. embedding API 失败导致部分索引状态不一致
5. 并发索引同一文档可能导致重复 chunk 或重复 embeddings
```

当前 MVP 适合本地 demo 和功能验证。生产化前需要引入：

```text
1. index job 队列
2. indexing 状态机
3. 幂等索引设计
4. per-user / per-tenant 限流
5. token / embedding 成本统计
6. 失败重试和补偿逻辑
7. 操作日志和管理员可见的 job logs
```

### 9.5 Delete Risk

删除接口必须保证：

```text
业务数据库中的 document_chunks 被清理
Chroma 中对应 embeddings 被删除
后续 RAG 不再召回已删除文档
```

当前 MVP 已覆盖删除后 检索未命中 的 冒烟 验证。

后续生产化前还需要考虑：

```text
1. 软删除和硬删除策略
2. 删除操作权限控制
3. 删除审计记录
4. 删除失败时的补偿任务
5. 多版本文档删除时的版本级清理
```

---

## 10. RAG / 租户 / 类别边界

当前 RAG API：

```http
POST /rag/search
POST /rag/ask
```

当前系统支持：

```text
tenant_id metadata filter
category metadata filter
structured sources
retrieval_status
no-context fallback
retrieval logs / metrics
```

### 10.1 当前实现边界

当前 API 层只暴露 `category` filter。`tenant_id` 暂时由系统内部 模拟租户上下文 提供：

```text
tenant_demo
```

这意味着当前版本可以展示 tenant metadata filter 的工程形态，但不能视为真实多租户权限隔离。

### 10.2 风险说明

如果没有真实认证和授权，生产环境可能出现：

```text
1. 用户访问不属于自己 tenant 的文档
2. 用户通过 category 枚举探测其他知识分类
3. 上传文档被错误地写入公共检索空间
4. RAG sources 泄露文档路径、文档标题或敏感 metadata
5. 未授权用户消耗 embedding / LLM API 成本
```

### 10.3 后续生产化要求

生产化前必须补齐：

```text
1. authentication
2. authorization
3. tenant_id 从认证上下文获取
4. user_id 从认证上下文获取
5. RAG search 按 tenant_id 强制过滤
6. Document Backend 按 tenant_id 强制隔离
7. AgentOps 查询按 tenant 或管理员权限隔离
8. sources 字段做权限过滤
```

---

## 11. Tool Call Audit

当前 Ticket Agent 会记录三个工具调用：

```text
search_kb
classify_ticket
create_ticket
```

### 11.1 search_kb

`search_kb` 记录知识库检索动作。

典型输入：

```json
{
  "query": "VPN 连不上，重启客户端也没用",
  "top_k": 3,
  "tenant_id": "tenant_demo",
  "category": "it"
}
```

典型输出：

```json
{
  "results_count": 1,
  "document_ids": ["doc_vpn_guide"],
  "top_distance": 0.52
}
```

审计价值：

```text
记录 Agent 查了什么知识库内容
记录检索分类和租户边界
记录召回文档 ID 和 top distance
```

### 11.2 classify_ticket

`classify_ticket` 记录工单判断动作。

典型输入：

```json
{
  "message": "VPN 连不上，重启客户端也没用",
  "requested_category": "it",
  "sources_count": 1,
  "source_categories": ["it"],
  "source_document_ids": ["doc_vpn_guide"]
}
```

典型输出：

```json
{
  "should_create_ticket": true,
  "category": "it",
  "priority": "high",
  "reason": "用户描述包含故障、异常、申请或权限类信号，建议生成工单草稿，用户确认后创建工单。"
}
```

审计价值：

```text
记录 Agent 为什么建议创建工单
记录分类和优先级判断结果
记录判断时可见的 sources 信息
```

### 11.3 create_ticket

`create_ticket` 记录真实 ticket 创建动作。

该工具只允许在 确认阶段执行。

典型输入：

```json
{
  "title": "VPN 连不上，重启客户端也没用",
  "description": "用户问题：VPN 连不上，重启客户端也没用",
  "category": "it",
  "priority": "high"
}
```

典型输出：

```json
{
  "ticket_id": 1,
  "status": "open"
}
```

审计价值：

```text
记录系统最终创建了哪个 ticket
记录 create_ticket 的输入 payload
记录 ticket 创建结果
```

---

## 12. 失败记录

工具调用失败时，系统应更新对应 `tool_call`：

```text
status = failed
error_type = <machine-readable failure type>
error_message = <exception message>
```

并根据失败位置更新 `agent_run`：

```text
status = failed
result_summary = <failure summary>
```

当前重点失败路径：

| 失败点 | 预期记录 |
|---|---|
| search_kb failed | search_kb tool_call failed, agent_run failed |
| classify_ticket failed | classify_ticket tool_call failed, agent_run failed |
| create_ticket failed | create_ticket tool_call failed, agent_run failed |
| document index failed | document status / error should be visible to caller |
| embedding API failed | error should not be silently swallowed |

失败记录的目标是让错误进入结构化数据，而不是只存在于控制台日志。

---

## 13. Approval Reject / Cancel

当前系统支持：

```http
POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

### 13.1 Reject

Reject 表示用户明确拒绝该操作。

状态变化：

```text
pending → rejected
```

Rejected approval_request 不允许再次 confirm。

### 13.2 Cancel

Cancel 表示操作被取消。

状态变化：

```text
pending → cancelled
```

Cancelled approval_request 不允许再次 confirm。

### 13.3 Approval Decision Reason

审批请求会记录 `decision_reason`，用于说明审批结果背后的原因。

审批通过、拒绝或取消后，系统可以记录：

```text
审批状态
操作人 / mock user
决策原因
决策时间
```

这避免了审批记录只包含状态、缺少上下文的问题，也方便后续审计和复盘。

---

## 14. AgentOps Metrics Summary

AgentOps metrics summary 用于查看当前系统运行统计。

接口：

```http
GET /agent-ops/metrics/summary
```

示例：

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

该接口可用于快速检查：

```text
Agent run 是否完成
tool_call 是否全部成功
approval_request 是否仍有 pending
是否存在 failed tool_call
失败 tool_call 的 error_type 分布
```

### 14.1 当前限制

当前 AgentOps metrics 适合 MVP 观察，不等价于生产级监控系统。后续需要补充：

```text
1. 时间窗口筛选
2. tenant 维度筛选
3. user 维度筛选
4. endpoint latency 指标
5. embedding / LLM 成本指标
6. dashboard
7. alerting
```

---

## 15. Docker Compose Local Runtime Boundary

当前 Docker Compose 设计目标是本地运行、demo 和交付复现，不是生产部署。

当前配置的安全边界：

```text
1. 服务端口绑定 127.0.0.1:8000
2. API 只在本机可访问
3. 使用 docker_data / docker_storage / docker_chroma_db 独立运行目录
4. 通过 env_file: .env 注入本地环境变量
5. .env 不应提交到 GitHub
```

### 15.1 Public Exposure Risk

当前项目不能直接暴露到公网。

原因包括：

```text
1. 尚未接入真实 authentication
2. 尚未接入真实 authorization
3. tenant_id / user_id 仍是 模拟上下文
4. Document upload 缺少生产级文件安全控制
5. RAG / ask / index 可能消耗真实模型 API 成本
6. AgentOps API 暂未做管理员权限隔离
7. SQLite 不适合作为公网生产数据库
8. 缺少 限流、WAF、TLS、reverse proxy 和审计告警
```

如果需要部署到共享环境或公网，必须先补齐生产化安全控制。

### 15.2 Docker Runtime Data

Docker Compose 使用独立目录：

```text
docker_data/
docker_storage/
docker_chroma_db/
```

这些目录包含运行数据，不应提交：

```text
SQLite 数据库
上传文档
Chroma embeddings
本地 demo 产生的临时数据
```

这些目录应加入 `.gitignore`。

---

## 16. API Key and `.env` Boundary

当前项目通过 `.env` 管理本地模型 API 配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
EMBEDDING_MODEL=text-embedding-v4
```

### 16.1 安全要求

```text
1. .env 不应提交到 GitHub
2. .env 不应被 Dockerfile COPY 进镜像
3. .dockerignore 应包含 .env
4. docker compose config 会展开 .env，不能把完整输出公开粘贴
5. API key 泄露后应及时轮换
6. 本地 shell_env 和 .env 应保持一致，避免 Docker 与本机行为不一致
```

### 16.2 环境来源风险

本地运行时可能存在多个配置来源：

```text
PowerShell / shell environment
.env file
Docker Compose env_file
IDE / terminal injected environment
```

推荐规则：

```text
.env 是项目本地运行的唯一配置源。
Docker、本地 uvicorn、冒烟 scripts 都应以 .env 为准。
不要长期依赖 shell_env 里偶然存在的 key。
```

---

## 17. Rate Limit and Cost Control Boundary

当前项目会在以下场景调用真实模型或 embedding API：

```text
RAG ask
Document index
Chroma index build
Smoke scripts
可能的 Agent preview 检索链路
```

当前 MVP 尚未实现生产级 限流 和成本控制，因此存在：

```text
1. 高频请求导致 API 费用增加
2. 大文档索引导致 embedding 成本增加
3. 恶意用户反复调用 /documents/{document_id}/index
4. 恶意用户反复调用 /rag/ask
5. 冒烟 scripts 在错误环境中反复运行导致额外成本
```

生产化前需要补齐：

```text
1. per-user 限流
2. per-tenant 限流
3. daily budget limit
4. embedding token / cost accounting
5. LLM token / cost accounting
6. index job quota
7. request timeout
8. retry policy with backoff
9. admin-visible cost dashboard
```

---

## 18. Smoke Scripts Boundary

当前项目提供：

```text
scripts/冒烟_agentops_flow.py
scripts/冒烟_document_backend_flow.py
```

Smoke scripts 的定位：

```text
调用真实运行中的 API 服务
验证 FastAPI、SQLite、Chroma、embedding 和业务链路能否串起来
用于本地验收、Docker Compose 验收和 demo 前检查
```

Smoke scripts 与 pytest 的区别：

```text
pytest:
验证 model / service / API 的单元或集成行为，通常使用 monkeypatch 隔离外部依赖。

冒烟 scripts:
调用真实运行中的 API 服务，可能触发真实 embedding / LLM 调用。
```

因此：

```text
1. 冒烟 scripts 不进入默认 GitHub Actions CI
2. 冒烟 scripts 运行前需要有效 .env
3. Document Backend 冒烟 会触发真实 embedding
4. 运行失败时需要清理临时上传文档
5. 不应在无成本控制的公网环境中开放给任意用户触发
```

---

## 19. Current Production Limitations

当前 MVP 不能直接用于生产环境，主要限制如下：

```text
1. tenant_id / user_id 仍使用 模拟上下文
2. 尚未接入真实 authentication
3. 尚未接入真实 authorization
4. AgentOps API 暂未区分管理员权限
5. Document Backend 缺少生产级上传安全控制
6. Document Backend 缺少文件大小限制、病毒扫描和敏感内容检测
7. RAG sources 暂未做生产级权限过滤
8. /documents/{document_id}/index 缺少 限流 和成本控制
9. /rag/ask 缺少 限流 和成本控制
10. SQLite 仅用于本地开发和 MVP 演示
11. 数据库 schema 变更尚未接入 Alembic migration
12. Docker Compose 当前是本地运行版，不是生产部署版
13. 缺少 TLS、reverse proxy、WAF、日志脱敏和告警
14. 缺少 request_id / trace_id 贯穿全链路
15. 缺少 PII / sensitive data policy
```

---

## 20. Production Hardening Checklist

后续生产化建议按以下顺序推进：

### 20.1 Identity and Access Control

```text
1. 接入 authentication
2. 接入 authorization
3. 从认证上下文获取 tenant_id
4. 从认证上下文获取 user_id
5. AgentOps API 增加管理员权限边界
6. Document Backend 增加文档 owner / tenant 权限校验
```

### 20.2 Document Security

```text
1. 文件大小限制
2. 文件类型 allowlist
3. MIME type 校验
4. 文件名规范化
5. 病毒扫描
6. 敏感内容检测
7. 文档版本管理
8. index job logs
9. 异步 indexing queue
10. 删除补偿任务
```

### 20.3 RAG and Model Usage Security

```text
1. RAG search 强制 tenant filter
2. sources 字段权限过滤
3. prompt injection 检测
4. no-context 拒答策略继续强化
5. per-user / per-tenant 限流
6. embedding / LLM 成本统计
7. daily budget limit
8. timeout / retry / circuit breaker
```

### 20.4 AgentOps and Audit

```text
1. request_id / trace_id
2. 时间窗口 metrics
3. tenant / user 维度 metrics
4. dashboard
5. audit log retention policy
6. error_type taxonomy
7. alerting
8. 日志脱敏
```

### 20.5 Infrastructure

```text
1. PostgreSQL 替换 SQLite
2. Alembic migration
3. 生产版 Docker Compose 或 Kubernetes manifests
4. TLS
5. reverse proxy
6. WAF
7. secret manager
8. backup / restore
9. monitoring / logging stack
```

---

## 21. Security Summary

当前系统的安全边界可以概括为：

```text
Agent can suggest, but cannot directly execute state-changing actions.
Ticket creation requires approval.
Approval must belong to the same agent_run.
Approval must still be pending.
Confirm draft must match the server-side approval draft.
Ticket creation uses 服务端草稿_json, not untrusted client payload.
Every major tool action is recorded in tool_calls.
Document upload does not automatically enter RAG until indexed.
Document deletion removes corresponding Chroma embeddings.
RAG currently demonstrates tenant/category filtering, but tenant_id is still mock.
Docker Compose is local-only and binds 127.0.0.1.
.env is local secret configuration and must not be committed or publicly pasted.
Smoke scripts are manual validation tools, not default CI jobs.
The current project is an MVP and demo system, not production-ready.
```

---
