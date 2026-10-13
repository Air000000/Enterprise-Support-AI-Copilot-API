# Ticket Agent MVP 阶段报告

本文保留早期 mock-context MVP 的阶段快照，不是当前能力清单。当前已使用 Demo JWT 传递操作人和租户，
并通过数据库条件更新防止并发审批重复消费；精确实现与边界以 [README](../README.md) 和代码为准。

## 1. 阶段目标

本阶段目标是在 RAG Core、Ticket CRUD 和 AgentOps 基础上，实现一个受控 Ticket Agent MVP。

Agent 不直接自动创建工单，而是采用 preview / confirm 两阶段流程：

```text
用户问题
↓
RAG 检索企业知识库
↓
生成 ticket preview
↓
创建 agent_run 运行记录
↓
必要时创建 approval_request
↓
用户确认
↓
记录 tool_call
↓
创建真实 ticket
↓
更新 AgentOps 状态
```

该设计强调三个原则：

```text
1. 不自动执行高影响动作
2. 用户确认后才创建真实工单
3. Agent 每一步都可追踪、可审计
```

---

## 2. 当前实现范围

当前已实现：

```text
POST /agent/ticket/preview
POST /agent/ticket/confirm

Ticket CRUD
TicketDraft
TicketAgentSource
TicketAgentPreviewResponse
TicketAgentConfirmResponse

agent_runs
approval_requests
tool_calls

preview 阶段创建 agent_run
preview 阶段根据规则决定是否建议创建工单
preview 阶段在 should_create_ticket=True 时创建 approval_request
confirm 阶段校验 approval_request 属于当前 agent_run
confirm 阶段更新 approval_request 为 approved
confirm 阶段创建 tool_call
confirm 阶段调用 ticket_service.create_ticket()
confirm 成功后更新 tool_call 为 success
confirm 失败后更新 tool_call 为 failed
confirm 成功或失败后更新 agent_run 状态
mock tenant/user context
API 层测试
Service 层测试
AgentOps service 测试
AgentOps read API
GET /agent-ops/runs
GET /agent-ops/runs/{agent_run_id}
GET /agent-ops/runs/{agent_run_id}/tool-calls
GET /agent-ops/runs/{agent_run_id}/approval-requests
GET /agent-ops/metrics/summary
POST /agent-ops/approval-requests/{approval_request_id}/reject
POST /agent-ops/approval-requests/{approval_request_id}/cancel
AgentOps API 测试

```

---

## 3. 核心 API

| Method | Path                                               | 作用                         |
| ------ | -------------------------------------------------- | -------------------------- |
| POST   | `/agent/ticket/preview`                            | 根据用户问题生成工单草稿和审批请求          |
| POST   | `/agent/ticket/confirm`                            | 用户确认后创建真实工单                |
| POST   | `/agent-ops/approval-requests/{approval_request_id}/reject` | 拒绝审批请求，不创建 ticket 或 tool_call |
| POST   | `/agent-ops/approval-requests/{approval_request_id}/cancel` | 取消审批请求，不创建 ticket 或 tool_call |
| GET    | `/agent-ops/runs`                                  | 查询当前 tenant 下的 AgentRun 列表 |
| GET    | `/agent-ops/runs/{agent_run_id}`                   | 查询单次 AgentRun 详情           |
| GET    | `/agent-ops/runs/{agent_run_id}/tool-calls`        | 查询单次 AgentRun 下的工具调用记录     |
| GET    | `/agent-ops/runs/{agent_run_id}/approval-requests` | 查询单次 AgentRun 下的审批请求记录     |
| GET    | `/agent-ops/metrics/summary`                       | 查询当前 tenant 下的 AgentOps 汇总指标 |


---

## 4. Preview 流程

```text
TicketAgentPreviewRequest
↓
preview_ticket()
↓
create_agent_run()
↓
normalize_rag_category()
↓
search_chroma()
↓
TicketAgentSource
↓
should_create_ticket()
↓
TicketDraft
↓
create_approval_request()  # only when should_create_ticket=True
↓
update_agent_run()
↓
TicketAgentPreviewResponse
```

Preview 阶段不会创建真实 ticket。

当 `should_create_ticket=False` 时：

```text
不创建 approval_request
不创建 ticket
agent_run 标记为 completed
返回 sources 和 reason
```

当 `should_create_ticket=True` 时：

```text
生成 TicketDraft
创建 approval_request
agent_run 标记为 completed
返回 draft、sources、approval_request_id
```

---

## 5. Confirm 流程

```text
TicketAgentConfirmRequest
↓
confirm_ticket()
↓
get_approval_request()
↓
validate approval_request.agent_run_id == request.agent_run_id
↓
update_approval_request(status="approved")
↓
create_tool_call(status="pending")
↓
TicketCreate
↓
ticket_service.create_ticket()
↓
update_tool_call(status="success" or "failed")
↓
update_agent_run(status="completed" or "failed")
↓
TicketAgentConfirmResponse
```

Confirm 阶段负责真正执行创建工单动作。

当前 confirm 阶段已经包含 ownership hardening：

```text
approval_request 必须属于同一个 agent_run。
如果 approval_request.agent_run_id 与 request.agent_run_id 不一致，则返回 400，且不会执行 approve、tool_call 或 create_ticket。
```

---

## 6. AgentOps 审计模型

当前 AgentOps 包含三类记录：

```text
AgentRun
ToolCall
ApprovalRequest
```

### AgentRun

用于记录一次 agent 运行。

主要字段：

```text
tenant_id
user_id
agent_name
input_message
category
status
result_summary
created_at
updated_at
```

### ApprovalRequest

用于记录需要人工确认的动作。

主要字段：

```text
agent_run_id
tenant_id
approval_type
status
draft_json
approved_by
created_at
decided_at
```

### ToolCall

用于记录 agent 执行的工具调用。

主要字段：

```text
agent_run_id
tenant_id
tool_name
tool_input_json
tool_output_json
status
error_message
created_at
finished_at
```

---

## 7. AgentOps 查询、审批决策与指标汇总 API

当前已经新增 AgentOps 只读查询 API，用于查看 Ticket Agent 的执行轨迹。

第一版只提供 read API，不提供 create / update API。AgentOps 记录由 workflow 自动写入，查询 API 只负责读取。

当前支持：

```text
GET /agent-ops/runs
GET /agent-ops/runs/{agent_run_id}
GET /agent-ops/runs/{agent_run_id}/tool-calls
GET /agent-ops/runs/{agent_run_id}/approval-requests
GET /agent-ops/metrics/summary
```
查询能力：

1. 查询 agent run 列表，可按 status / agent_name 过滤。
2. 查询单次 agent run 详情。
3. 查询单次 agent run 关联的 tool_calls。
4. 查询单次 agent run 关联的 approval_requests。

审批决策能力：

1. reject：将 approval_request 更新为 rejected。
2. cancel：将 approval_request 更新为 cancelled。
3. reject / cancel 不会创建 ticket。
4. reject / cancel 不会创建 tool_call。

当前设计中，真实 ticket 创建仍然只发生在 `/agent/ticket/confirm` 流程中。

指标汇总能力：

```text
1. total_agent_runs：当前 tenant 下的 agent run 总数。
2. running / completed / failed / cancelled agent runs：按运行状态统计。
3. total_tool_calls：当前 tenant 下的 tool call 总数。
4. pending / success / failed tool calls：按工具调用状态统计。
5. total_approval_requests：当前 tenant 下的 approval request 总数。
6. pending / approved / rejected / cancelled approval requests：按审批状态统计。
```

当前 metrics summary 是基础计数型汇总，暂不包含时间窗口、latency、token usage 或 retrieval distance 聚合。


设计原则：

Agent workflow 负责写入 AgentOps records。
AgentOps read API 负责查看执行轨迹。
外部调用方不能直接伪造 AgentOps 写入记录。

当前仍使用 mock tenant context：

tenant_id = tenant_demo

后续接入真实认证后，应从 current user / current tenant 中解析 tenant_id。


## 8. 规则型 Agent 设计

当前第一版使用规则判断是否建议创建工单。

支持类关键词包括：

```text
无法
不能
失败
报错
异常
打不开
连接不上
连不上
收不到
锁定
忘记密码
退回
损坏
丢失
过期
开通
审批
权限
```

紧急类关键词包括：

```text
紧急
生产系统
全员
大面积
客户现场
客户阻塞
无法办公
数据泄露
安全事件
```

低优先级咨询类关键词包括：

```text
咨询
了解
请问
怎么
如何
是否
```

当前规则：

```text
无知识库来源：建议创建工单
包含故障、异常、权限、开通等支持类信号：建议创建工单
普通知识咨询：暂不建议创建工单
生产系统、大面积、数据泄露、安全事件等问题：priority=urgent
普通故障类问题：priority=high
咨询类问题：priority=low
默认情况：priority=medium
```

---

## 9. 已发现并修复的问题

### 9.1 子串误判问题

原始规则中，`客户` 会误命中 `客户端`：

```text
"客户" in "客户端" == True
```

这会导致普通 VPN 客户端问题被误判为 urgent。

修复方式：

```text
删除过于宽泛的 urgent 关键词
使用更具体的短语，如 客户现场、客户阻塞
增加 service 测试防止回归
```

### 9.2 咨询问题误判问题

`请问请假怎么申请？` 原先可能因为包含 `申请` 被判定为需要创建工单。

修复方式：

```text
从支持类关键词中移除过宽的 申请
保留更明确的故障、异常、权限、开通类信号
增加知识咨询测试用例
```

### 9.3 approval_request 与 agent_run 错配问题

Confirm 阶段现在会校验：

```text
approval_request.agent_run_id == request.agent_run_id
```

如果不匹配，则返回 400，并且不会执行后续副作用：

```text
不会 approve approval_request
不会 create tool_call
不会 create ticket
```

---

## 10. 测试覆盖

当前测试覆盖：

| 文件                                   | 覆盖内容                                          |
| ------------------------------------ | --------------------------------------------- |
| `tests/test_tickets.py`              | Ticket CRUD                                   |
| `tests/test_agent_ops_service.py`    | AgentRun / ToolCall / ApprovalRequest service |
| `tests/test_ticket_agent_service.py` | Agent service 规则、preview、confirm、AgentOps 轨迹  |
| `tests/test_agent_ticket_api.py`     | Agent API request / response / validation     |
| `tests/test_rag_api.py`              | RAG API                                       |
| `tests/test_rag_service.py`          | RAG service                                   |
| `tests/test_query_chroma.py`         | Chroma metadata filter                        |
| `tests/test_todos.py`                | Todo CRUD                                     |
| `tests/test_agent_ops_api.py` | AgentOps read API request / response / tenant context|

当前 Ticket / AgentOps 相关测试：

```powershell
pytest tests/test_tickets.py
pytest tests/test_agent_ops_service.py
pytest tests/test_agent_ops_api.py
pytest tests/test_ticket_agent_service.py
pytest tests/test_agent_ticket_api.py
```

当前通过情况：

```text
tests/test_tickets.py              9 passed
tests/test_agent_ops_service.py    10 passed
tests/test_agent_ops_api.py        7 passed
tests/test_ticket_agent_service.py 9 passed
tests/test_agent_ticket_api.py     6 passed
```

---

## 11. 当前局限

当前 Ticket Agent MVP 仍是规则型 workflow，暂未引入复杂 LLM planner。

当前局限：

```text
1. 当前 Ticket Agent 规则仍然比较简单。
2. 没有复杂多轮确认。
3. 没有真实权限系统，只使用 mock tenant/user。
4. AgentOps 已支持基础 metrics summary 聚合统计，但尚未支持 dashboard、时间窗口筛选、latency、token usage 或 retrieval distance 聚合。
5. 当前 approval confirm / reject / cancel 都已具备后端 API，但尚未接入真实前端审批界面。
6. 当前 approval decision 仍使用 mock user context，尚未接入真实审批人身份。
7. 当前 agent 规则依赖关键词，仍可能出现边界误判。
```


---

## 12. 下一步

建议进入 Ticket Agent v1 cleanup：

```text
1. 在 AgentRun 中记录 latency 和 retrieval summary。
2. 在 ToolCall 中记录更完整的错误类型。
3. 增加端到端 API smoke test。
4. 后续接入真实 tenant / user auth context。
5. 后续接入真实前端审批界面。
6. 后续支持 AgentOps dashboard 和时间窗口筛选。
```


---

## 13. 阶段结论

Ticket Agent MVP 已经完成受控 preview / confirm 闭环。

当前系统已经具备：

RAG 知识检索
Ticket CRUD
Ticket preview
人工确认
真实 ticket 创建
agent_runs 运行记录
approval_requests 审批记录
tool_calls 工具调用审计
AgentOps read API
基础 tenant 隔离
service 和 API 测试

该阶段已经可以作为后续 Enterprise Support AI Copilot 的 action workflow 基线。
