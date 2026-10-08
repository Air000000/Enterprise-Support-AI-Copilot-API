# Project Summary

Enterprise Support AI Copilot 当前稳定基线总结。

更新：2026-10-08。本文区分当前 API 实现、离线冻结候选与尚未完成的质量验证。

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
Demo JWT authentication
TechQA offline evaluation / refusal gate demo
```

补充说明：

- RAG 支持 tenant/category metadata filter
- Ticket Agent 采用 preview / confirm 两阶段控制真实工单创建
- AgentOps 支持 run / tool call / approval request 查询与汇总
- Document Backend 支持上传、索引、删除闭环
- Demo JWT 提供认证上下文与应用层租户范围；不是生产级 IAM
- 线上仍为 Dense Chroma；Hybrid / rerank / Top14 与 v5.2 gate 均未接入线上

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
rag_runtime/
experiments/evals/
docs/
scripts/
tests/
```

边界说明：

- `rag_runtime/` 已存在，`experiments/` 保留离线研究与历史入口
- 默认数据库文件名 `data/todos.db` 保留历史兼容，不影响对外项目定位
- 历史学习集小样本指标不与 TechQA 主评测指标混用
- 未通过实验保留 FAIL / NO_GO，不因工程冻结或文档更新改写结论

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

Smoke 需要运行中的服务与认证，可能产生真实调用费用。
无需密钥的离线机制演示：`python -m scripts.demo_refusal_coverage`。

## 当前文档入口

- [README.md](../README.md): 唯一入口文档
- [architecture.md](architecture.md): 当前系统结构说明
- [agent_workflow.md](agent_workflow.md): Ticket Agent 流程
- [security.md](security.md): 当前边界提示及早期 MVP 安全记录
- [demo_script.md](demo_script.md#21-离线拒答-gate-演示): 零调用 gate 演示与已知限制
- `docs/*_report.md`: 历史阶段性记录

## 尚未完成的能力

以下能力尚未完成：

- 完整 Hybrid / rerank / Top14 线上集成与独立整链验收
- 拒答语义质量独立确认与最终生成验收
- 生产级认证、授权、密钥管理及数据库级租户隔离
- 并发恰好一次建单保证
- 前端审批界面
- AgentOps dashboard 与时间窗口筛选

## 推荐项目命名

- 对外展示名：`Enterprise Support AI Copilot`
- 中文名：`企业内部支持 AI Copilot`
- 仓库名：`enterprise-support-ai-copilot-api`

## 技术结果与边界

问答和工单是可组合的 API 路径，不是已经自动编排好的全链自主 Agent。
`classify_ticket` 使用规则；preview 不建单，confirm 校验审批归属、pending 和
草稿一致性后使用服务端审批草稿执行，但不保证并发恰好一次副作用。

离线 E1 Dense + rerank 的冻结 DEV Recall@5 为 64.4% → 72.5%；此收益不代表
Hybrid + Top14 整链已验证或已上线。Top14 在 54 条 TRAIN 证据样本上与 Top20
命中相同（答案证据 35/54、有用证据 43/54），是开发阶段预算选择，不是通用最优。

v5.1 对照 v4，与 AI proxy 一致性为 16/20 → 17/20：不足题误放行减少，但新增误拒，
费用约 2.06 倍。标签非独立，不能据此推断泛化或人工金标准确率。
v5.2 只有合成自检，真实引用仍可能被错误解释为语义支持，尚无真实模型质量确认。

数据依据：[E1 冻结 DEV](../experiments/evals/reports/e1_rerank/comparison.md)、
[检索/上下文冻结](../experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md)、
[v5.1 配对结果](../experiments/evals/reports/refusal_evidence_sufficiency/v5_1_paired_result.md)。
R4 C1 FAIL、G1/G2 NO_GO 保持原结论。
