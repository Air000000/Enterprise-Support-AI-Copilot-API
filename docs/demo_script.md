# Demo Script

本地求职演示主线（2026-10-30）：

```text
启动 API → Demo JWT / 租户上下文 → 上传并显式索引虚构文档
→ 默认 Hybrid 检索 → 带来源回答 / 证据不足拒答
→ Ticket Agent Preview → 人工 Confirm → 演示库 Ticket
→ AgentOps Trace / Metrics
```

问答和工单是可组合的 API，本演示由操作员依次调用，不是自主 Agent 自动接管业务。
本轮仅核对文档；没有执行真实提供方调用、建单或 Docker 构建。

## 1. 准备隔离环境与启动 API

使用**全新独立 clone**，不要复用正在工作的目录、数据库、上传目录或 Chroma。
例如在自选演示父目录运行下面命令，目标目录必须不存在：

```powershell
git clone --branch main https://github.com/Air000000/Enterprise-Support-AI-Copilot-API.git copilot-local-demo
Set-Location copilot-local-demo
```

安装依赖、填写未提交配置、迁移和启动步骤统一遵循
[README Quick Start](../README.md#9-quick-start)，环境变量以
[.env.example](../.env.example)和 README 为准。只在这个新 clone 中准备自己的 `.env`，
不要复制或覆盖日常服务的 `.env`，不要打印密钥、完整配置或认证响应。

配置检查点：

| 用途 | 配置 / 约束 |
| --- | --- |
| embedding 与生成 | `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`、`EMBEDDING_MODEL`、`DASHSCOPE_MODEL`（生成可由 `CHAT_MODEL` 覆盖） |
| Hybrid 搜索与 preview | 专用 `DASHSCOPE_RERANK_API_KEY` / `DASHSCOPE_RERANK_BASE_URL`；`RAG_HYBRID_SEARCH_ENABLED=true` |
| 默认 source_scope 问答 | 另需专用 `DASHSCOPE_PHASE_B_API_KEY` / `DASHSCOPE_PHASE_B_BASE_URL`；`RAG_HYBRID_ASK_ENABLED=true` |
| 演示数据库 | 在此新 clone 的配置中使用 `DATABASE_URL=sqlite:///data/demo.db`，迁移前创建 `data/`；不要删除或迁移旧业务库 |
| 上传与向量目录 | 此新 clone 的 `storage/documents` 与 `experiments/chroma_db`，不要链接或挂载已有数据 |
| 日志 | `SQL_ECHO=false`，避免演示控制台显示完整业务 SQL 参数 |

rerank 与 classifier 不会回退到通用密钥；核对实际区域及工作空间地址。
占位值不能用于真实问答。两个 Hybrid 开关默认开启，显式关闭后请求会失败，不会自动改成 Dense。
先核对当前终端是否已有 `DATABASE_URL`、存储目录或模型配置覆盖；迁移和启动使用同一环境。
Chroma 路径相对此新 clone 的工作目录确定，不通过虚构的 Chroma 环境变量切换。

按 README 完成依赖安装及 `alembic upgrade head` 后，在新 clone 根目录启动：

```powershell
python -m uvicorn main:app --reload --env-file .env
```

打开 [Swagger UI](http://127.0.0.1:8000/docs)，调用 `GET /health`，预期 `{"status":"ok"}`。
若端口被占用，为新演示选择空闲端口并相应更新地址，不停止现有服务。

真实提供方演示会产生费用：索引发送虚构文档到 embedding；搜索及 preview 发送查询、
最多 100 条本租户候选正文到 rerank；问答另发送上下文到分类与生成服务。
仅在另行明确批准本次外发与费用后手动执行后续真实调用；本文件不是模型调用授权。
错误即停，检查脱敏错误及审计，不自动重试，不尝试现有业务文档。

Docker 方式复用 [README Docker Compose](../README.md#docker-compose)。普通 Compose
会复用 `docker_data/`、`docker_storage/`、`docker_chroma_db/`，只能在同样的全新 clone
且确认这些目录没有业务数据时用于演示；本轮不修改 Compose 配置。
零调用的 Compose 验收覆盖配置使用假模型，不能用于非空真实问答。
仓库有历史构建与容器验收记录，本轮没有在当前环境重新构建验收，不能宣称已完成。

## 2. 认证与租户上下文

在 Swagger 调用 `POST /auth/token`，本地默认演示账户：

```json
{"username":"support","password":"support"}
```

把返回的 `access_token` 仅填入 Swagger 的 **Authorize**（HTTP Bearer 值），
不截图、打印或提交 token。之后各接口携带同一认证；token 过期重新登录。
若配置了 `AUTH_DEMO_USERS_JSON`，使用对应的本地 support/admin 演示账户。

调用 `GET /auth/me`，默认应看到 `tenant_id=tenant_demo`、`user_id=support_demo`、
`role=support`。Documents 与 AgentOps 要求 support/admin，employee 会被拒绝。
租户来自 JWT，不在请求体中填写 `tenant_id`；Demo JWT 不是生产 IAM。

## 3. 上传并索引虚构知识文档

本地新建 `atlas_vpn_demo.md`，内容仅用以下虚构说明，不放真实业务数据：

```markdown
# Atlas VPN 1.0 demo support note

Scope: Atlas VPN version 1.0, fictional local demonstration only.
For error 809 on Atlas VPN 1.0:
1. Check that the device has an internet connection.
2. Confirm that the Atlas VPN profile uses gateway vpn.example.invalid.
3. Restart the Atlas VPN client and try again.
If error 809 persists, contact support with the error code and client version.
This note does not cover version 2.0 or certificate renewal procedures.
```

通过 Swagger `POST /documents/upload` 选择文件，`category=it`。
记录响应的 `id` 为 `DOCUMENT_ID`；此时 `status=uploaded`、`chunk_count=0`。

**上传不会自动进入检索。** 再调用 `POST /documents/{document_id}/index`，
填入 `DOCUMENT_ID`；这是一次真实 embedding 操作。随后 `GET /documents/{document_id}`
确认 `status=indexed`、`chunk_count>0`，才进入下面的检索和问答。
默认 Hybrid 仅看当前租户、当前版本且已索引的受管文档；
旧静态 Chroma 或 TechQA 离线归档不会自动进入此范围，不运行旧静态建库脚本。

## 4. 默认 Hybrid 检索

调用 `POST /rag/search`：

```json
{"query":"Atlas VPN 1.0 error 809 troubleshooting","category":"it"}
```

省略模式和 `top_k`，即 Hybrid、默认 14（最多 14，实际命中可少于 14）。
链路固定为 Dense100 + BM25100 → RRF(k=60) Top100 → qwen3-rerank → Top14。

检查 `retrieval_mode=hybrid` 和 `results` 中的 `document_id=DOCUMENT_ID`、`title`、
`chunk_id`、`source_path`、`preview`、`category`；独立分数为 `rerank_score`、
`fusion_score`、`dense_rank`、`bm25_rank`，`distance` 可为 null。
重排分数不是概率，也不能套用 Dense 的 0.9 距离阈值。
搜索响应列表叫 `results`，问答与 preview 才使用 `sources`。

## 5. 默认带来源回答与拒答

调用 `POST /rag/ask`，不要传 `max_distance`：

```json
{"question":"How should I troubleshoot Atlas VPN 1.0 error 809?","category":"it"}
```

核对 `retrieval_mode=hybrid`、`refusal_policy=source_scope_experimental`、
`retrieval_status`、`answer`、`sources` 和 `evidence_gate`；Hybrid 的 `top_distance=null`。
模型放行时生成只读取批准来源的完整 chunk，响应来源及引用可追踪。
逐项对照文档检查步骤、条件和引用，不把有引用当作答案必然正确。

再用同一接口展示证据缺口：

```json
{"question":"Give the certificate renewal procedure for Atlas VPN version 2.0.","category":"it"}
```

虚构文档明确不覆盖该版本和流程，预期为 `refused_insufficient_evidence`；
没有充分证据时不生成答案。若出现 `refused_by_generator`，表示生成器拒答，应按实际状态讲解。
如果模型误放行或审计版本错误，保留并说明已知局限，不重试到得到满意结果。

需要确定性的空范围示例，可在仅有上述 it 文档的新库中改为 `category=hr`：
结果应为 `no_context`，且空范围不调用模型。这和非空证据不足是不同分支。
格式、引用或提供方失败是脱敏 503、审计 `failed`，不计为正确拒答。

显式 Dense 回退的合法示例（可选，不必额外调用）：

```json
{"question":"How should I troubleshoot Atlas VPN 1.0 error 809?","category":"it","retrieval_mode":"dense","top_k":3,"max_distance":0.9}
```

Dense 默认 Top3/.9，不使用 source_scope；不能只在默认 Hybrid 请求里添加距离阈值。
旧策略实验与当前固定 8 类场景验收的区别见 [评测总览](../experiments/evals/README.md)。

## 6. Ticket Agent Preview

先用 `GET /tickets` 记录演示租户当前工单列表，再调用 `POST /agent/ticket/preview`：

```json
{"message":"Atlas VPN 1.0 连不上，error 809，重启客户端后仍失败，需要支持处理。","category":"it"}
```

省略模式即默认 Hybrid Top14；这里只共享检索，不调用问答拒答 gate 或生成模型。
`classify_ticket` 是规则判断，不是自主 LLM 规划。此例包含故障关键词，预期
`should_create_ticket=true`、`approval_request_id` 非空、`draft` 和 `sources` 返回。

保存实际 `agent_run_id`、`approval_request_id` 及**完整未修改的 `draft`**。
Preview 创建 run、`search_kb` / `classify_ticket` 审计和 pending 审批，**不会建单**。
再次查看 `GET /tickets`，应没有新增工单。

## 7. 人工 Confirm 后创建演示 Ticket

操作员先阅读草稿，再手动调用 `POST /agent/ticket/confirm`。
填写 preview 返回的实际两个 ID，并原样复制 `draft`；不要把占位内容直接发送：

```text
{
  "agent_run_id": <preview.agent_run_id>,
  "approval_request_id": <preview.approval_request_id>,
  "draft": <preview.draft 原样 JSON 对象>
}
```

**只有这次显式 Confirm 才执行 Agent 建单**，写入的是新 clone 的隔离演示库。
成功返回 HTTP 201，读取响应 `ticket.id`，再用 `GET /tickets/{ticket_id}` 核对。
审批变为 approved，新增 `create_ticket` tool call，run 为 completed。
Confirm 不重跑检索或模型；它校验租户、审批归属、pending 与服务端冻结草稿一致性，
实际建单使用服务端 `draft_json`。重复确认或草稿篡改会被拒绝。
审批至多一次消费不等于跨事务恰好一次建单或崩溃恢复保证。

可选分支：另起 preview，只对其 pending 审批执行
`POST /agent-ops/approval-requests/{approval_request_id}/reject` 或 `/cancel`，
请求体为 `{"reason":"本地演示，人工决定不执行"}`。终态不能再次 confirm，且不建单。
不要拒绝用于上面主线确认的审批；本轮整理不实际执行这些写操作。

## 8. AgentOps Trace / Metrics

继续使用同一 support/admin token：

| API | 检查点 |
| --- | --- |
| `GET /agent-ops/runs/{agent_run_id}/trace` | `agent_run`、`tool_calls`、`approval_requests` 聚合轨迹 |
| `GET /agent-ops/runs/{agent_run_id}/tool-calls` | `search_kb`、`classify_ticket`、`create_ticket` 三条成功记录 |
| `GET /agent-ops/runs/{agent_run_id}/approval-requests` | 本次审批为 approved |
| `GET /agent-ops/metrics/summary` | run、tool call、approval 状态计数及 `tool_call_error_types` |
| `GET /agent-ops/retrieval-logs` | 之前独立 search / ask 的模式、策略、结果或失败审计 |
| `GET /agent-ops/metrics/retrieval` | 检索状态与延迟聚合；不作为业务质量分数 |

若新库只执行一次 preview + confirm，Agent 指标应为 1 run、3 tool calls、1 approved 审批。
额外 preview、失败或 reject/cancel 会改变计数。preview 的 `search_kb` 由工具记录和 run 摘要追踪，
不把它当成另一次独立 `/rag/search` 审计。所有查询限定 JWT 租户，记录不是模型思维链。

## 9. 可选文档删除闭环

只在问答与工单主线完成后，对此次 `DOCUMENT_ID` 调用 `DELETE /documents/{document_id}`。
核对 `status=deleted`、`deleted_embeddings`，再查详情应 404；再次搜索不应出现该 ID。
这只操作本次虚构文档，不批量清理，不删除数据库、分支或 worktree。
SQL 与 Chroma 没有跨存储事务，失败时应如实说明，不能宣称原子删除保证。

## 10. 真实提供方演示与离线测试

上面手动 API 演示使用真实提供方，可能失败且会计费；不是总体质量评测。
本轮不执行它，也不运行付费 eval、DEV 数据读取或会自动 confirm 的在线 smoke。

不使用模型凭据的已有回归：

```powershell
python -m pytest tests/test_hybrid_search.py -k "ticket_hybrid_document_to_decision or nonempty_hybrid_over_real_uvicorn_tcp"
```

这些测试使用临时 SQLite / Chroma、真实回环 HTTP / JWT，以及模型替身；只在临时库中
验证显式确认、拒绝、取消和隔离机制。通过不代表真实模型语义质量、Docker 构建或生产部署已验收。
无服务、无密钥的机械 gate 演示见下方第 21 节；它是历史 v5.2 候选，不是当前 source_scope。

## 11. 历史演示评测记录

早期企业内部文档学习集记录为 Total 30、hit@1 0.97、hit@3 1.00、mrr@3 0.98；
保留历史数值，不作为当前 Hybrid 或 TechQA 质量结论，也不在现场重新运行该评测。
TechQA 冻结历史、失败案例和延期事项统一见 [评测总览](../experiments/evals/README.md)。

面试展示完成标准：能解释从索引到来源回答、从草稿到人工确认建单的边界，并在 Trace 中
找到检索、规则判断、审批和写入记录；模型错误与未通过的质量结论同样要如实展示。

---

## 21. 离线拒答 gate 演示

这是独立于上面的真实 API smoke 的 **v5.2 离线候选演示**。只用虚构 Atlas
资料和手写 JSON 响应，复用真实输入构建、行引用还原、schema 校验和决策推导。
不需要 `.env`、模型密钥、FastAPI、数据库、Chroma 或第三方依赖；不读取 TRAIN / DEV
资料，也不调用模型、生成回答或修改线上拒答策略。

从仓库根目录运行（Python 3.11+，不要使用 `python -O`，自检使用 `assert`）：

```bash
python -m scripts.demo_refusal_coverage
```

脚本逐例输出问题、完整编号来源、手写响应和真实 gate 结果，断言失败会退出非零。
预期五个机械结果为：

| 顺序 | 展示点 | 结果 |
| --- | --- | --- |
| 1 | 一般可能原因，不强求具体事故诊断 | `SUFFICIENT` |
| 2 | 模拟响应声明“最新版”证据缺失 | `INSUFFICIENT` |
| 3 | 模拟响应声明具体事故适用性未确定 | `INSUFFICIENT` |
| 4 | 缺少必填 applicability 审计 | `VALIDATION_ERROR`，停止/审计，不算正确拒答 |
| 5 | 真实历史引用被故意标成最新版 `SUPPORTED` | 仍为 `SUFFICIENT`，已知错误放行 |

结尾 `5/5` 只表示复现了预设的机械行为，**不是五题全答对**。
第 5 例说明：引用原文真实，无法证明相关性、适用性或时间覆盖正确；错误的
`SUPPORTED` / `NOT_REQUIRED` 仍可通过。本节演示自身不调用模型，不能用它证明修复了真实错误。

已冻结的 v4 / v5.1 小样本配对实验为 20 条 TRAIN、40 次调用，
与 AI proxy 标签的一致性分别为 16/20 和 17/20，有改善也有退化；标签不独立，不能
当成人工金标或泛化证明。完整成本、延迟及剩余分歧见
[v5.1 冻结结果](../experiments/evals/reports/refusal_evidence_sufficiency/v5_1_paired_result.md)。
后续 [v5.2 控制题实验](../experiments/evals/reports/refusal_evidence_sufficiency/v5_2_control_result.md)
已有真实模型调用，但这不会把本节手写响应变成模型质量评测。
最新 [v5.2.1 / v5.3 开发回归](../experiments/evals/reports/refusal_evidence_sufficiency/v5_3_regression_result.md)
在同一批 30 条自然 TRAIN 上新跑两组，与二元 AI proxy 标签一致 22/25 → 23/25；
修正 2 次误拒、新增 1 次，同一个不足案例仍误放行（1/4）。另 5 条有歧义不计入二元一致率。
这是已见题开发结果，不是独立验收、生成答案准确率或线上效果，不能用演示的 `5/5` 代替它。
本节不验证线上集成或真实模型端到端质量；实际 API / 文档 / 工单闭环使用前面的
隔离手动演示，运行前需要另行批准真实外发与费用。
