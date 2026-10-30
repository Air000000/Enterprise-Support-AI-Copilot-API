# 工单 Agent 工作流

## 1. 模块定位

Ticket Agent 是 Enterprise Support AI Copilot 中的业务执行层。

它不负责直接回答所有问题，而是把企业内部支持请求分成两类：

1. 可以通过知识库回答的问题；
2. 需要创建工单并交给人工支持人员处理的问题。

本模块的核心目标不是让 Agent 自动执行所有动作，而是实现一个可控、可审计、可解释的工单创建闭环：

```text
用户问题
↓
知识库检索
↓
工单判断与分类
↓
生成 工单预览
↓
人工确认
↓
创建真实 工单
↓
记录 agent_runs / tool_calls / approval_requests / 指标
```

---

## 2. 设计原则

Ticket Agent 遵循以下原则：

```text
LLM / Agent 只能建议；
高风险动作必须人工确认；
后端必须校验审批状态和执行 payload；
所有工具调用必须记录；
失败必须进入审计记录，而不是只出现在控制台日志里。
```

因此，系统没有让 Agent 直接创建工单，而是拆成两个阶段：

```text
预览阶段：生成建议，不执行动作
确认阶段：用户确认后，才创建真实工单
```

这样可以避免 Agent 误调用工具、用户重复确认、拒绝后仍执行、客户端篡改工单内容等风险。

---

## 3. 相关 API

### 工单 Agent API

```http
POST /agent/ticket/preview
POST /agent/ticket/confirm
```

### AgentOps API

```http
GET  /agent-ops/runs
GET  /agent-ops/runs/{agent_run_id}
GET  /agent-ops/runs/{agent_run_id}/trace
GET  /agent-ops/runs/{agent_run_id}/tool-calls
GET  /agent-ops/runs/{agent_run_id}/approval-requests
GET  /agent-ops/metrics/summary

POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

`/agent-ops` 接口要求 support/admin 角色，查询限定 JWT 所属租户；`/runs/{agent_run_id}/trace` 聚合 run、tool_calls 与 approval_requests。

当前项目使用 `/agent-ops` 作为运维观测接口前缀，用于查看 Agent 运行记录、工具调用记录、审批记录和指标汇总。

`preview` 默认使用冻结 Hybrid Top14，`RAG_HYBRID_SEARCH_ENABLED` 默认开启；显式传 `retrieval_mode="dense"` 可回退到 Dense Top3。该模式只改变检索，不调用问答证据门控或生成模型，不修改工单建议规则。来源 `distance` 可为空，重排分数及排名单独返回；工具输入/输出与运行摘要记录实际模式。Hybrid 关闭或失败时返回脱敏 503，审计失败，不生成审批或自动回退 Dense。`confirm` 请求体、租户隔离、pending 和草稿校验保持不变，不重跑检索。

---

## 4. 核心数据表职责

### agent_runs

`agent_runs` 记录一次 Agent 任务的整体状态。

它回答的问题是：

```text
这次 Agent 任务是谁发起的？
输入是什么？
当前状态是什么？
执行结果是什么？
耗时是多少？
检索摘要是什么？
```

典型字段：

```text
id
tenant_id
user_id
agent_name
input_message
category
status
result_summary
latency_ms
retrieval_summary_json
created_at
updated_at
```

---

### tool_calls

`tool_calls` 记录 Agent 在运行过程中调用过的工具。

它回答的问题是：

```text
Agent 调用了哪个工具？
工具输入是什么？
工具输出是什么？
调用成功还是失败？
失败原因是什么？
```

当前 Ticket Agent 使用三个工具：

```text
search_kb
classify_ticket
create_ticket
```

---

### approval_requests

`approval_requests` 记录需要人工确认的高风险动作。

它回答的问题是：

```text
Agent 建议执行什么动作？
这个动作是否已经被确认？
是谁确认的？
确认前的 草稿 payload 是什么？
```

`create_ticket` 属于会修改业务状态的动作，因此必须先生成 `approval_request`，再等待用户确认。

---

## 5. 预览阶段流程

### 接口

```http
POST /agent/ticket/preview
```

### 示例请求

```json
{
  "message": "VPN 连不上，重启客户端也没用",
  "category": "it"
}
```

### 执行流程

```text
1. 创建 agent_run，状态为 running
2. 将用户传入的 category 转换为 RAG 类别过滤
3. 创建 search_kb 工具调用，状态为 pending
4. 调用知识库检索
5. 检索成功后，将 search_kb 工具调用 更新为 success
6. 将检索结果转换为 来源
7. 创建 classify_ticket 工具调用，状态为 pending
8. 根据用户问题和 来源 判断是否需要创建工单
9. 推断 工单 category
10. 推断 工单 priority
11. 生成 reason
12. 将 classify_ticket 工具调用 更新为 success
13. 如果不需要创建工单，更新 agent_run 为 completed，并返回 无需建单响应
14. 如果需要创建工单，生成 工单草稿
15. 创建 approval_request，状态为 pending
16. 更新 agent_run 为 completed
17. 返回 工单预览
```

### 预览阶段数据流

```text
TicketAgentPreviewRequest
↓
agent_runs
↓
search_kb 工具调用
↓
RAG 检索
↓
来源
↓
classify_ticket 工具调用
↓
TicketDraft
↓
approval_requests.pending
↓
TicketAgentPreviewResponse
```

---

## 6. search_kb 工具调用

`search_kb` 是对底层 RAG 检索的业务语义抽象。

底层默认使用 Chroma Dense + BM25 + RRF + rerank，但 AgentOps 中记录的工具名是 `search_kb`，而不是 `search_chroma`。这样做的好处是：以后底层从 Chroma 换成 Milvus、pgvector 或 Elasticsearch，Agent 工具语义不需要变化。

以下为默认 Hybrid 的示意审计值，实际 ID 和分数以响应为准；`distance=null` 可以来自 BM25 独有命中，重排分数不是概率。

### 工具输入 JSON 示例

```json
{
  "query": "VPN 连不上，重启客户端也没用",
  "top_k": 14,
  "tenant_id": "tenant_demo",
  "category": "it",
  "retrieval_mode": "hybrid"
}
```

### 工具输出 JSON 示例

```json
{
  "results_count": 1,
  "document_ids": ["doc_vpn_guide"],
  "top_distance": null,
  "retrieval_mode": "hybrid",
  "score_type": "rerank_relevance",
  "larger_is_better": true,
  "rerank_scores": [0.92]
}
```

### 失败处理

如果知识库检索失败，系统会：

```text
1. 将 search_kb 工具调用 更新为 failed
2. 记录 error_message
3. 将 agent_run 更新为 failed
4. 重新抛出异常
```

这样失败不会只出现在控制台，而是会进入 AgentOps 审计记录。

---

## 7. classify_ticket 工具调用

`classify_ticket` 记录工单判断和分类过程。

它是确定性规则决策步骤，不调用 LLM 分类。它基于用户问题和 RAG 来源 判断：

```text
是否需要创建工单
工单 category 是什么
优先级 priority 是什么
为什么这样判断
```

### 工具输入 JSON 示例

```json
{
  "message": "VPN 连不上，重启客户端也没用",
  "requested_category": "it",
  "sources_count": 1,
  "source_categories": ["it"],
  "source_document_ids": ["doc_vpn_guide"]
}
```

### 工具输出 JSON 示例

```json
{
  "should_create_ticket": true,
  "category": "it",
  "priority": "high",
  "reason": "用户描述包含故障、异常、申请或权限类信号，建议生成工单草稿，用户确认后创建工单。"
}
```

### 设计意义

`classify_ticket` 让系统可以解释：

```text
为什么这次请求需要创建工单？
为什么分类为 it？
为什么优先级是 high？
判断依据是什么？
```

这样 预览阶段不只是返回一个 草稿，而是留下了决策轨迹。

---

## 8. 预览响应示例

```json
{
  "agent_run_id": 1,
  "approval_request_id": 10,
  "retrieval_mode": "hybrid",
  "should_create_ticket": true,
  "reason": "用户描述包含故障、异常、申请或权限类信号，建议生成工单草稿，用户确认后创建工单。",
  "draft": {
    "title": "VPN 连不上，重启客户端也没用",
    "description": "用户问题：VPN 连不上，重启客户端也没用\n\n系统根据用户描述生成工单草稿，建议支持人员进一步确认和处理。\n\n相关知识库来源：\n1. vpn_guide (doc_vpn_guide, doc_vpn_guide_chunk_1)",
    "category": "it",
    "priority": "high"
  },
  "sources": [
    {
      "document_id": "doc_vpn_guide",
      "chunk_id": "doc_vpn_guide_chunk_1",
      "title": "vpn_guide",
      "source_path": "storage/documents/tenant_demo/doc_vpn_guide/v1/vpn_guide.md",
      "distance": null,
      "rerank_score": 0.92,
      "fusion_score": 0.016393,
      "dense_rank": null,
      "bm25_rank": 1,
      "preview": "VPN 故障排查说明。",
      "category": "it"
    }
  ]
}
```

---

## 9. 确认阶段流程

### 接口

```http
POST /agent/ticket/confirm
```

### 示例请求

```json
{
  "agent_run_id": 1,
  "approval_request_id": 10,
  "draft": {
    "title": "VPN 连不上，重启客户端也没用",
    "description": "用户问题：VPN 连不上，重启客户端也没用\n\n系统根据用户描述生成工单草稿，建议支持人员进一步确认和处理。\n\n相关知识库来源：\n1. vpn_guide (doc_vpn_guide, doc_vpn_guide_chunk_1)",
    "category": "it",
    "priority": "high"
  }
}
```

### 执行流程

```text
1. 根据 approval_request_id 和 tenant_id 读取 approval_request
2. 校验 approval_request.agent_run_id 是否等于 request.agent_run_id
3. 校验 approval_request.status 是否仍然是 pending
4. 从 approval_request.draft_json 中还原服务端保存的 TicketDraft
5. 校验 request.draft 是否和服务端保存的 TicketDraft 一致
6. 以审批 ID、tenant_id 和旧状态 pending 条件更新为 approved
7. 使用服务端保存的 approval_draft 构造 TicketCreate
8. 创建 create_ticket 工具调用，状态为 pending
9. 调用 工单服务 创建真实工单
10. 创建成功后，将 create_ticket 工具调用 更新为 success
11. 更新 agent_run 为 completed
12. 返回创建后的 工单
```

---

confirm / reject / cancel 共用 pending 条件更新，只有一个审批决策能成功，终态不能覆盖。
这保证审批至多一次消费；工单和审计分别写入，不提供跨事务恰好一次或崩溃恢复保证。

## 10. 确认阶段安全校验

### 10.1 approval_request 必须属于当前 agent_run

系统会检查：

```text
approval_request.agent_run_id == request.agent_run_id
```

如果不一致，返回错误：

```text
Approval request does not belong to agent run
```

这个校验可以防止用户拿其他 Agent run 的审批请求来执行当前动作。

---

### 10.2 approval_request 必须仍然是 pending

系统会检查：

```text
approval_request.status == "pending"
```

如果状态已经是：

```text
approved
rejected
cancelled
```

则不能再次执行。

这个校验可以防止：

```text
已经拒绝的审批被重新执行
已经取消的审批被重新执行
已经批准的审批被重复执行
```

---

### 10.3 确认 草稿 必须和服务端保存的 draft_json 一致

Preview 阶段会把 工单草稿 保存到：

```text
approval_request.draft_json
```

Confirm 阶段会重新读取这个服务端保存的 草稿，并和请求体里的 `draft` 做一致性校验。

如果不一致，返回错误：

```text
Confirm draft does not match approval draft
```

这个校验可以防止：

```text
用户在 预览阶段看到的是 A；
确认阶段客户端偷偷提交 B；
后端错误地创建 B。
```

真正用于创建工单的数据是服务端保存的 `approval_draft`，而不是客户端临时提交的 草稿。

---

## 11. create_ticket 工具调用

`create_ticket` 是真正修改业务状态的工具调用。

它只会在 确认阶段执行，并且必须满足：

```text
approval_request 属于当前 agent_run
approval_request.status == pending
确认 草稿 未被篡改
```

### 工具输入 JSON 示例

```json
{
  "title": "VPN 连不上，重启客户端也没用",
  "description": "用户问题：VPN 连不上，重启客户端也没用\n\n系统根据用户描述生成工单草稿，建议支持人员进一步确认和处理。\n\n相关知识库来源：\n1. vpn_guide (doc_vpn_guide, doc_vpn_guide_chunk_1)",
  "category": "it",
  "priority": "high"
}
```

### 工具输出 JSON 示例

```json
{
  "ticket_id": 100,
  "status": "open"
}
```

---

## 12. 完整工具调用链路

一次成功的工单创建会产生三条 工具调用：

```text
search_kb
classify_ticket
create_ticket
```

它们分别回答不同问题：

| tool_name       | 阶段      | 作用                 |
| --------------- | ------- | ------------------ |
| search_kb       | 预览 | 记录 Agent 查了什么知识库   |
| classify_ticket | 预览 | 记录 Agent 为什么建议创建工单 |
| create_ticket   | 确认 | 记录用户确认后系统真正执行了什么   |

完整链路：

```text
用户问题
↓
agent_run
↓
工具调用: search_kb
↓
工具调用: classify_ticket
↓
approval_request
↓
用户确认
↓
工具调用: create_ticket
↓
工单
↓
指标 summary
```

---

## 13. AgentOps 查询示例

### 查看 Agent run 列表

```http
GET /agent-ops/runs
```

### 查看某次 Agent run

```http
GET /agent-ops/runs/{agent_run_id}
```

### 查看某次 Agent run 的工具调用

```http
GET /agent-ops/runs/{agent_run_id}/tool-calls
```

预期可以看到：

```text
search_kb
classify_ticket
create_ticket
```

### 查看某次 Agent run 的审批请求

```http
GET /agent-ops/runs/{agent_run_id}/approval-requests
```

### 查看 AgentOps 汇总指标

```http
GET /agent-ops/metrics/summary
```

示例结果：

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

## 14. 工具调用 失败类型统计

在 AgentOps 中，`tool_calls` 表不仅记录每次工具调用的输入、输出、状态和错误信息，还通过 `error_type` 字段记录失败类别。

当前约定的失败类型包括：

* `search_kb_failed`：知识库检索失败。
* `classify_ticket_failed`：工单分类或草稿生成失败。
* `create_ticket_failed`：确认后创建工单失败。

`/agent-ops/metrics/summary` 会返回 `tool_call_error_types`，用于按失败类型聚合统计。例如：

```json
{
  "failed_tool_calls": 1,
  "tool_call_error_types": {
    "create_ticket_failed": 1
  }
}
```

这个设计让 AgentOps 不只具备审计能力，也具备基础的故障归因能力。

---

## 15. 手动 Demo 流程

### 步骤 1： Preview

```http
POST /agent/ticket/preview
```

请求：

```json
{
  "message": "VPN 连不上，重启客户端也没用",
  "category": "it"
}
```

记录返回中的：

```text
agent_run_id
approval_request_id
草稿
```

---

### 步骤 2： 查看 预览 工具调用

```http
GET /agent-ops/runs/{agent_run_id}/tool-calls
```

预期结果：

```text
search_kb: success
classify_ticket: success
```

---

### 步骤 3： 查看 approval request

```http
GET /agent-ops/runs/{agent_run_id}/approval-requests
```

预期结果：

```text
status = pending
```

---

### 步骤 4： Confirm

```http
POST /agent/ticket/confirm
```

请求体使用 预览 返回的 草稿，不能修改 草稿 内容。

---

### 步骤 5： 再次查看工具调用

```http
GET /agent-ops/runs/{agent_run_id}/tool-calls
```

预期结果：

```text
search_kb: success
classify_ticket: success
create_ticket: success
```

---

### 步骤 6： 查看 指标

```http
GET /agent-ops/metrics/summary
```

预期结果：

```text
total_agent_runs = 1
completed_agent_runs = 1
total_tool_calls = 3
successful_tool_calls = 3
total_approval_requests = 1
approved_approval_requests = 1
```

---

## 16. 失败场景与防护

### 场景 1：RAG 检索失败

处理方式：

```text
search_kb 工具调用 → failed
agent_run → failed
error_message 记录异常原因
```

---

### 场景 2：approval_request 不属于当前 agent_run

处理方式：

```text
拒绝 确认
不更新 approval_request
不创建 create_ticket 工具调用
不创建真实 工单
```

---

### 场景 3：approval_request 已经 rejected / cancelled / approved

处理方式：

```text
拒绝 确认
不允许重复执行
不创建真实 工单
```

---

### 场景 4：确认 草稿 被篡改

处理方式：

```text
比较 request.draft 和 approval_request.draft_json
如果不一致，拒绝 确认
不创建真实 工单
```

---

## 17. 当前设计取舍

### 为什么不用全自动 Agent 直接创建工单？

因为创建工单是业务状态变更动作。即使风险不如转账、删库等操作高，也仍然应该经过用户确认。

本项目选择 预览 / 确认 两阶段，是为了体现企业系统中常见的 人工参与审批 设计。

---

### 为什么 search_kb 和 classify_ticket 都记录为 工具调用？

因为 Agent 的可解释性不仅来自最终结果，也来自中间过程。

`search_kb` 解释：

```text
Agent 基于哪些知识库内容做判断？
```

`classify_ticket` 解释：

```text
Agent 为什么认为需要创建工单？
为什么 category 是这个？
为什么 priority 是这个？
```

---

### 为什么 确认阶段使用服务端保存的 draft_json？

因为客户端请求不可信。

如果 确认阶段直接使用客户端传入的 草稿，就可能出现：

```text
用户看到的是 A；
客户端提交的是 B；
系统创建了 B。
```

因此，真正用于创建工单的数据必须来自 预览阶段服务端保存的 `approval_request.draft_json`。
