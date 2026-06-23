# Security Design

Enterprise Support AI Copilot 安全设计说明。

本文档记录当前 Ticket Agent 与 AgentOps MVP 中已经实现的关键安全边界，重点覆盖：

```text
preview / confirm 两阶段执行
approval_request ownership validation
pending approval validation
draft payload consistency check
server-side approval draft
tool_calls audit
AgentOps metrics
```

---

## 1. Scope

当前安全设计覆盖以下模块：

```text
Ticket Agent
AgentOps
Ticket creation workflow
Approval request workflow
Tool call audit
```

相关 API：

```http
POST /agent/ticket/preview
POST /agent/ticket/confirm

GET  /agent-ops/runs
GET  /agent-ops/runs/{agent_run_id}
GET  /agent-ops/runs/{agent_run_id}/tool-calls
GET  /agent-ops/runs/{agent_run_id}/approval-requests
GET  /agent-ops/metrics/summary

POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

当前版本使用 mock tenant / user context。后续接入真实认证系统后，`tenant_id` 和 `user_id` 应从认证上下文中获取，而不是由客户端直接传入。

---

## 2. Security Goals

Ticket Agent 的安全目标是：

```text
1. Agent 不直接执行状态变更动作
2. 创建真实 ticket 前必须经过人工确认
3. confirm 阶段不能使用不可信客户端 payload 创建 ticket
4. approval_request 不能跨 agent_run 使用
5. rejected / cancelled / approved approval_request 不能再次执行
6. 每次工具调用都要留下结构化审计记录
7. 工具调用失败需要进入 AgentOps 记录
```

其中，`create_ticket` 属于状态变更动作，因此不能在 preview 阶段直接执行。

---

## 3. Trust Boundaries

当前系统中的信任边界如下：

| 数据来源                        | 信任级别    | 处理方式                                 |
| --------------------------- | ------- | ------------------------------------ |
| client request              | 不可信     | 需要 schema validation 和业务校验           |
| preview draft in response   | 不可信副本   | 仅供用户查看和 confirm 提交                   |
| approval_request.draft_json | 服务端可信记录 | confirm 阶段创建 ticket 的依据              |
| agent_run_id                | 不完全可信   | 必须和 approval_request 绑定校验            |
| approval_request_id         | 不完全可信   | 必须校验 tenant_id、agent_run_id 和 status |
| RAG retrieved sources       | 辅助依据    | 进入 response 和 tool_call audit        |
| tool_call records           | 审计记录    | 用于追踪工具输入、输出、状态和错误                    |

核心原则：

```text
客户端提交的 draft 只能用于一致性校验；
真正创建 ticket 的 payload 必须来自服务端保存的 approval_request.draft_json。
```

---

## 4. Preview / Confirm Two-Stage Execution

Ticket Agent 使用两阶段流程：

```text
preview 阶段：生成建议，不执行状态变更
confirm 阶段：人工确认后，执行真实 ticket 创建
```

### 4.1 Preview Stage

Endpoint:

```http
POST /agent/ticket/preview
```

Preview 阶段会：

```text
1. 创建 agent_run
2. 执行 search_kb
3. 执行 classify_ticket
4. 生成 ticket draft
5. 创建 approval_request.pending
6. 返回 preview response
```

Preview 阶段不会创建真实 ticket。

Preview 阶段产生的数据：

```text
agent_runs: 1 条
tool_calls: search_kb + classify_ticket
approval_requests: 1 条 pending
```

---

### 4.2 Confirm Stage

Endpoint:

```http
POST /agent/ticket/confirm
```

Confirm 阶段会：

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

Confirm 阶段产生的数据：

```text
ticket: 1 条
tool_calls: create_ticket
approval_requests: status 从 pending 变为 approved
```

---

## 5. Approval Ownership Validation

Confirm 阶段必须校验：

```text
approval_request.agent_run_id == request.agent_run_id
```

该校验用于防止 approval_request 被跨 Agent run 使用。

### 5.1 风险场景

没有 ownership validation 时，可能出现：

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

## 6. Pending Approval Validation

Confirm 阶段必须校验：

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

## 7. Draft Payload Consistency Check

Preview 阶段会把生成的 ticket draft 保存到：

```text
approval_request.draft_json
```

Confirm 阶段会校验：

```text
request.draft == approval_request.draft_json
```

### 7.1 风险场景

没有 draft consistency check 时，可能出现：

```text
1. 用户在 preview 阶段看到的是低风险 ticket draft
2. 客户端在 confirm 阶段修改 title / description / category / priority
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

Confirm 阶段会从服务端读取 `approval_request.draft_json`，并与 `request.draft` 做一致性校验。

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

## 8. Server-Side Approval Draft

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

即使客户端提交的 draft 通过了 Pydantic schema validation，也只能证明字段格式合法，不能证明它就是 preview 阶段用户看到并确认的内容。

因此，confirm 阶段的执行 payload 必须来自服务端保存的 approval draft。

---

## 9. Tool Call Audit

当前 Ticket Agent 会记录三个工具调用：

```text
search_kb
classify_ticket
create_ticket
```

### 9.1 search_kb

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

---

### 9.2 classify_ticket

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

---

### 9.3 create_ticket

`create_ticket` 记录真实 ticket 创建动作。

该工具只允许在 confirm 阶段执行。

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

## 10. Failure Recording

工具调用失败时，系统应更新对应 `tool_call`：

```text
status = failed
error_message = <exception message>
```

并根据失败位置更新 `agent_run`：

```text
status = failed
result_summary = <failure summary>
```

当前重点失败路径：

| Failure point          | Expected record                                    |
| ---------------------- | -------------------------------------------------- |
| search_kb failed       | search_kb tool_call failed, agent_run failed       |
| classify_ticket failed | classify_ticket tool_call failed, agent_run failed |
| create_ticket failed   | create_ticket tool_call failed, agent_run failed   |

失败记录的目标是让错误进入结构化 AgentOps 数据，而不是只存在于控制台日志。

---

## 11. Approval Reject / Cancel

当前系统支持：

```http
POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
```

### 11.1 Reject

Reject 表示用户明确拒绝该操作。

状态变化：

```text
pending → rejected
```

Rejected approval_request 不允许再次 confirm。

---

### 11.2 Cancel

Cancel 表示操作被取消。

状态变化：

```text
pending → cancelled
```

Cancelled approval_request 不允许再次 confirm。

---

## 12. Approval Decision Reason

审批请求现在会记录 `decision_reason`，用于说明审批结果背后的原因。

当用户拒绝或取消审批请求时，系统会保存：

* 审批状态：`rejected` 或 `cancelled`
* 操作人：`approved_by`
* 决策原因：`decision_reason`
* 决策时间：`decided_at`

这避免了审批记录只包含状态、缺少上下文的问题，也方便后续审计和复盘。

## Tool Call Error Type

工具调用失败时，系统会记录两个层级的信息：

* `error_type`：机器可聚合的失败类型。
* `error_message`：具体错误信息。

这种分层设计可以避免只依赖自由文本错误信息，也能支持 metrics summary 对失败类型进行聚合统计。


## 13. Metrics Summary

AgentOps metrics summary 用于查看当前系统运行统计。

Endpoint:

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
```

---

## 14. Current Limitations

当前 MVP 仍有以下限制：

```text
1. tenant_id / user_id 仍使用 mock context
2. 尚未接入真实认证和授权系统
3. AgentOps API 暂未区分管理员权限
4. approval_request reject / cancel 尚未记录操作人和原因
5. tool_call error_type 仍待细化
6. SQLite 仅用于本地开发和 MVP 演示
7. 数据库 schema 变更尚未接入 Alembic migration
```

---

## 15. Planned Improvements

后续安全增强方向：

```text
1. 接入真实 authentication / authorization
2. 从认证上下文获取 tenant_id 和 user_id
3. 为 AgentOps API 添加管理员权限边界
4. approval_request 增加 rejected_by / cancelled_by / reason
5. tool_calls 增加 error_type
6. 增加 request_id / trace_id
7. 接入 Alembic 管理数据库迁移
8. 将 SQLite 替换为 PostgreSQL
9. 对高风险工具增加 allowlist
10. 对工具输入增加更严格的 schema validation
```

---

## 16. Security Summary

当前 Ticket Agent 的安全边界可以概括为：

```text
Agent can suggest, but cannot directly execute state-changing actions.
Ticket creation requires approval.
Approval must belong to the same agent_run.
Approval must still be pending.
Confirm draft must match the server-side approval draft.
Ticket creation uses server-side draft_json, not untrusted client payload.
Every major tool action is recorded in tool_calls.
AgentOps APIs expose runs, tool calls, approvals, and metrics for inspection.
```

---
