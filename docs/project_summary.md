# Project Summary

Enterprise Support AI Copilot 当前稳定基线总结。

本文档总结当前 `main` 分支已经固定下来的能力、验证方式、结构边界和后续不在本轮范围内的事项。

## 项目定位

这是一个面向企业内部支持场景的 AI Copilot 后端，核心目标不是单纯问答，而是打通一条受控的支持链路：

```text
知识检索
-> 带来源回答
-> 判断是否需要创建工单
-> 生成 ticket preview
-> 人工确认
-> 创建真实 ticket
-> 记录 AgentOps 审计轨迹
```

项目最初由 FastAPI Todo / AI Todo API 演进而来。当前基线已经收口为企业支持后端，Todo 相关能力仅作为 Legacy compatibility 保留。

## 当前已完成能力

```text
Enterprise RAG Core
Document Backend
Ticket CRUD
Ticket Agent preview / confirm
AgentOps audit + read APIs
Approval reject / cancel APIs
Retrieval Logs / Metrics
Docker Compose local runtime
Smoke scripts
```

补充说明：

- RAG 支持 tenant/category metadata filter
- Ticket Agent 采用 preview / confirm 两阶段控制真实工单创建
- AgentOps 支持 run / tool call / approval request 查询与汇总
- Document Backend 支持上传、索引、删除闭环

## Legacy Compatibility

以下能力保留，但不再作为当前项目主能力展示：

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`
- `tests/test_todos.py`

保留原因：

- 兼容既有测试和历史演进说明
- 冻结基线阶段避免不必要删除
- 保留项目从学习型 Todo API 演进到企业支持后端的轨迹

## 当前结构与边界

当前稳定结构仍然包含：

```text
routers/
schemas/
services/
models/
experiments/rag_local/
experiments/evals/
docs/
scripts/
tests/
```

边界说明：

- `experiments/rag_local/` 本轮不迁移
- 默认数据库文件名 `data/todos.db` 本轮不改
- 不改 HTTP 接口和业务逻辑
- 允许做命名收口和 mock context 常量收口

## 验证方式

推荐 focused tests：

```text
tests/test_query_chroma.py
tests/test_rag_api.py
tests/test_rag_service.py
tests/test_document_models.py
tests/test_document_service.py
tests/test_document_api.py
tests/test_todos.py
tests/test_tickets.py
tests/test_agent_ops_service.py
tests/test_agent_ops_api.py
tests/test_ticket_agent_service.py
tests/test_agent_ticket_api.py
```

推荐 smoke：

```text
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

## 当前文档入口

- [README.md](../README.md): 唯一入口文档
- [architecture.md](architecture.md): 当前系统结构说明
- [agent_workflow.md](agent_workflow.md): Ticket Agent 流程
- [security.md](security.md): 当前安全边界
- `docs/*_report.md`: 历史阶段性记录

## 后续但不在本轮范围内

以下事项明确不纳入本轮基线冻结收口：

- `experiments/rag_local` 到 `rag_runtime` 的迁移
- 默认数据库文件名调整
- 真实 tenant / user auth context 接入
- 前端审批界面
- AgentOps dashboard 与时间窗口筛选

## 推荐项目命名

- 对外展示名：`Enterprise Support AI Copilot`
- 中文名：`企业内部支持 AI Copilot`
- 仓库名：`enterprise-support-ai-copilot-api`
