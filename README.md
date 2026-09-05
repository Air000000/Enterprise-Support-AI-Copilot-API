# Enterprise Support AI Copilot

[![Tests](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml)

面向**企业技术支持（Technical Support）**场景的 **evaluation-driven RAG + Controlled Ticket Agent** 后端。

项目的数据与评测主线长期以 **TechQA** 为核心：使用完整 Technote 技术支持语料建立检索、生成、拒答与失败归因闭环；应用侧保留通用 Document Backend、Dense Chroma RAG、受控建单与 AgentOps，使系统既能回答“效果是否真的变好”，也能回答“真实业务写操作是否可控、可追踪”。

> **当前定位：** TechQA 是主技术支持语料与长期主评测基准，不再把它视为迁移到另一套主数据集之前的临时 Phase。未来若增加 multi-source / conflict / agentic stress 测试，只作为补充评测，不替换现有 TechQA 主线。

---

## 30 秒看项目

```text
                         TechQA
        28,481 Technotes / 610 retrieval queries
         910 generation & abstention QA records
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
       Primary Data Backbone     Offline Evaluation
                                 Dense / Rerank / Hybrid
                                 Evidence-level Audit
                                 Generation / Abstention
                │                     │
                └──────────┬──────────┘
                           ▼
                    Failure Diagnosis
                           │
                           ▼
                     System Iteration

Application Runtime
────────────────────────────────────────────────────────────
Document Lifecycle
        │
        ▼
Dense Chroma Retrieval ──► Answer + Sources / Refusal
        │
        ▼
Ticket Agent Preview
  ├─ search_kb
  ├─ classify_ticket
  └─ approval_request.pending
        │
        ▼
Human Confirm
  ├─ run ownership check
  ├─ pending-status check
  └─ server-side draft integrity check
        │
        ▼
create_ticket ──► Real Business Side Effect
        │
        └────────► AgentOps Trace
                   ├─ Agent Run
                   ├─ Tool Call
                   ├─ Approval
                   └─ Retrieval Log / Metrics
```

### 当前核心能力

| 能力 | 当前实现 |
| --- | --- |
| Primary Technical Support Data | TechQA 28,481 Technotes、610 条 answerable retrieval queries、910 条 generation/abstention QA |
| Controlled Ticket Agent | `search_kb` / `classify_ticket` / `create_ticket`，preview-confirm + Human-in-the-loop |
| RAG Runtime | Chroma Dense Retrieval、tenant/category filter、sources、低相关拒答 |
| RAG Evaluation | Frozen TRAIN / DEV、Document Recall@K / MRR、generation / abstention harness |
| Rerank / Hybrid Research | Dense Top-100 + `qwen3-rerank` 正式 held-out 对照；BM25 / RRF / Hybrid 为离线受控实验 |
| Failure Diagnosis | candidate coverage、chunk crowding、evidence-level audit、route-selection gate |
| AgentOps | Agent Run / Tool Call / Approval / Retrieval Trace 与聚合指标 |
| Engineering | Alembic、Pytest、Ruff、GitHub Actions、Docker Compose、Smoke |

---

# 1. Primary Technical Support Data：TechQA

TechQA 不是单独外挂的评测数据集，而是当前项目的数据与评测主线。

## Retrieval corpus

- **28,481** 篇 Technote 技术支持文档；
- **610** 条 answerable retrieval queries；
- 每条 query 恰好 1 个 relevant document；
- qrels 为 document-level；
- 实际 retriever 返回 chunk，因此正式 IR 评测会先保留原始 chunk ranking，再按 `document_id` 首次出现位置 collapse 成 document ranking。

## Generation / abstention set

- **610** 条 answerable；
- **300** 条 impossible；
- 共 **910** 条 QA records。

这使同一 technical-support domain 可以连续支撑：

```text
Retrieval
  ↓
Rerank / Hybrid comparison
  ↓
Evidence quality diagnosis
  ↓
Generation correctness / faithfulness
  ↓
Abstention / hallucination evaluation
```

完整数据版本、SHA256、split 和评测契约见：

- [experiments/evals/README.md](experiments/evals/README.md)
- [experiments/evals/datasets/techqa/manifest.json](experiments/evals/datasets/techqa/manifest.json)

---

# 2. Controlled Ticket Agent

Ticket Agent 的核心目标不是让模型直接修改业务状态，而是将预览 / 审批阶段与真实写操作分离。

```text
User Request
   │
   ▼
search_kb
   │
   ▼
classify_ticket
   │
   ├─ no ticket needed ──► return decision
   │
   └─ ticket needed
          │
          ▼
      Ticket Draft
          │
          ▼
approval_request.pending
          │
          ▼
   Human Confirm
          │
          ▼
     create_ticket
```

当前使用三个业务工具语义：

```text
search_kb
classify_ticket
create_ticket
```

其中 `classify_ticket` 当前是可解释的规则化决策步骤，不把它包装成自主 LLM planning。

## Preview / Confirm

### Preview

`POST /agent/ticket/preview`

Preview 阶段会：

1. 创建 `agent_run`；
2. 执行并记录 `search_kb`；
3. 根据用户请求与 RAG sources 执行并记录 `classify_ticket`；
4. 若需要建单，生成 ticket draft；
5. 将 draft 持久化到 `approval_request.draft_json`，状态保持 `pending`；
6. 返回 preview，不产生真实工单写操作。

### Confirm

`POST /agent/ticket/confirm`

只有以下条件全部成立时才创建真实工单：

```text
approval_request.agent_run_id == request.agent_run_id
approval_request.status == "pending"
request.draft == server-side approval_request.draft_json
```

真正用于创建工单的是服务端持久化的 approval draft，而不是客户端临时传入的数据。

这组校验用于拒绝：

- 跨 Agent Run 使用其他审批请求；
- rejected / cancelled / already-approved 等非 `pending` 审批再次确认；
- Preview 后由客户端篡改 draft payload。

> 当前流程不被描述为并发场景下的 exactly-once side-effect guarantee。

更多实现细节见 [docs/agent_workflow.md](docs/agent_workflow.md)。

---

# 3. Enterprise RAG Runtime

当前在线 / API serving 路径保持为 **Dense Chroma Retrieval**。

主要能力：

- `/rag/search`
- `/rag/ask`
- 文档 chunk 检索
- `tenant_id` / `category` metadata filter
- 结构化 sources 返回
- 无上下文与低相关拒答
- retrieval logging

问答路径仅根据检索 Context 生成答案，并返回对应 sources。当前低相关拒答使用 Dense Top-1 distance 作为工程信号。

> **在线 / 离线边界：** BM25 / RRF / Hybrid / `qwen3-rerank` 当前用于 `experiments/evals/` 的离线评测与受控对照，不把它们描述成线上 serving 已切换到 Hybrid Retrieval。

---

# 4. RAG Evaluation：Frozen TechQA Benchmark

`experiments/evals/` 是正式离线评测入口。

## Split contract

| Split | Answerable | Impossible | 用途 |
| --- | ---: | ---: | --- |
| TRAIN | 450 | 150 | development / failure analysis / parameter selection |
| DEV | 160 | 150 | frozen held-out comparison |

正式 E0 / E1 对照冻结后，不使用 individual DEV failure 反向调参。

## Dense → Rerank held-out result

正式 DEV retrieval 结果：

| Method | Document Recall@5 | Document Recall@20 | MRR@10 |
| --- | ---: | ---: | ---: |
| Dense baseline | 0.643750 | 0.818750 | 0.518931 |
| Dense Top-100 + `qwen3-rerank` | **0.725000** | **0.843750** | **0.560841** |

即：

- Recall@5：**64.4% → 72.5%（+8.1pp）**；
- Recall@20：81.9% → 84.4%；
- MRR@10：**0.519 → 0.561**。

结果文件：

- [experiments/evals/reports/e1_rerank/comparison.md](experiments/evals/reports/e1_rerank/comparison.md)

这里强调的是**同一冻结 Benchmark 上的 held-out improvement**，不跨不同数据集比较孤立绝对分数。

---

# 5. Failure Diagnosis：从指标到 Evidence

项目没有把“增加更多检索组件”直接等同于“系统一定更好”，而是通过受控实验与 gate 决定路线去留。

## BM25 / RRF / Hybrid

离线实验覆盖：

- Dense Retrieval
- BM25
- Dense + BM25 / RRF Hybrid
- Dense / Hybrid candidate pool + `qwen3-rerank`

C1 Hybrid + Rerank 的 TRAIN aggregate metrics 有改善，但**没有达到预注册 early-rank MRR gate**：

```text
Recall@20 gate: PASS
MRR@10 gate: FAIL
Overall C1 decision: FAIL
```

因此停止继续付费优化，而不是把小幅上涨包装成成功路线。

相关报告：

- [experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md](experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md)
- [experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md](experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md)

## Evidence-level audit

Document Recall 可能掩盖一个更细的失败模式：

> **命中了正确文档，不等于真正包含答案的 evidence chunk 已经进入高位 context。**

因此建立人工 evidence audit：

- 60 条已标注 TRAIN queries；
- 54 条进入正式 evidence evaluation；
- 187 个 candidate chunk labels；
- evidence 区分为 weak / useful / answer-bearing；
- 计算 AnswerEvidenceHit、Evidence MRR 与 GoldDocHitButEvidenceMiss 等指标。

相关 artifacts：

- [experiments/evals/reports/r1_evidence_audit/evidence_metrics.json](experiments/evals/reports/r1_evidence_audit/evidence_metrics.json)
- [experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl](experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl)

---

# 6. Generation / Abstention Harness

基于 retrieval 与 evidence failure analysis，Generation Eval Harness 当前使用：

```text
Dense Top-100
   ↓
qwen3-rerank
   ↓
Top-3 rerank anchors
   + Dense Top-1 rescue anchor
   ↓
per unique anchor document:
forward sibling expansion (max 3)
   ↓
deduplicate
   ↓
max 16 context chunks
```

当前 context policy：

```text
document_aware_forward_expansion_v1
```

实现：

- [experiments/evals/eval_techqa_generation.py](experiments/evals/eval_techqa_generation.py)

Harness 已包含：

- correctness；
- faithfulness；
- abstention accuracy；
- hallucination rate；
- end-to-end latency；
- frozen run identity / manifest / checkpoint。

> **结果边界：** 当前不声称该 context policy 已带来新的正式 generation uplift；在冻结评测完成前，只将其作为基于 failure diagnosis 实现的工程 intervention。

---

# 7. Document Lifecycle

Document Backend 提供从知识入库到下架的显式生命周期：

```text
Upload
  ↓
Document Record
  ↓
Explicit Index
  ↓
Chunk + Embedding
  ↓
RAG Retrieval
  ↓
Delete
  ├─ relational chunks removed
  └─ Chroma embeddings removed
```

主要 API：

```http
POST   /documents/upload
GET    /documents
GET    /documents/{document_id}
POST   /documents/{document_id}/index
DELETE /documents/{document_id}
```

当前支持 `md` / `txt` 上传，并由认证上下文中的 tenant 约束文档访问范围。

---

# 8. AgentOps / Observability

AgentOps 将关键执行信息持久化，而不是只写控制台日志。

主要实体：

```text
agent_runs
  ├─ tool_calls
  └─ approval_requests

retrieval_logs
```

主要查询能力：

```http
GET /agent-ops/runs
GET /agent-ops/runs/{agent_run_id}
GET /agent-ops/runs/{agent_run_id}/trace
GET /agent-ops/tool-calls
GET /agent-ops/approval-requests
GET /agent-ops/retrieval-logs
GET /agent-ops/metrics/summary
GET /agent-ops/metrics/retrieval
GET /agent-ops/metrics/retrieval/sources
GET /agent-ops/metrics/retrieval/no-context-queries
GET /agent-ops/metrics/retrieval/failures
```

支持按 tenant 查看：

- Agent Run 状态；
- Tool Call 成功 / 失败与 error type；
- Approval 状态；
- Retrieval no-context / refused / failed；
- Retrieval source distribution；
- 单次 Run Trace。

---

# 9. Authentication / Tenant Scope

项目包含用于工程验证的 Demo JWT Auth：

- Bearer token；
- `user_id`；
- `tenant_id`；
- `role`；
- `support` / `admin` 角色检查；
- tenant-scoped Document / AgentOps / RAG 访问。

> 这里验证的是认证上下文和 tenant scope 在应用链路中的传递，不把它描述为完整生产级 IAM / RBAC 或数据库级多租户隔离方案。

安全边界见 [docs/security.md](docs/security.md)。

---

# 10. Project Structure

```text
enterprise-support-ai-copilot-api/
├── main.py
├── auth.py
├── database.py
├── rag_runtime/
├── routers/
├── schemas/
├── services/
├── models/
├── experiments/
│   ├── evals/
│   ├── docs/
│   └── rag_local/
├── docs/
├── scripts/
├── tests/
├── migrations/
├── docker-compose.yml
├── Dockerfile
└── README.md
```

其中：

- `rag_runtime/`：正式在线 RAG runtime；
- `experiments/evals/`：TechQA 主评测、受控实验与 artifacts；
- `experiments/rag_local/`：早期兼容入口；
- Todo / AI Todo 路径保留为历史兼容，不作为当前项目定位。

---

# 11. Quick Start

## Environment

参考 `.env.example`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
DATABASE_URL=sqlite:///data/todos.db
SQL_ECHO=true
DOCUMENT_STORAGE_ROOT=storage/documents
```

## Install & migrate

```bash
pip install -r requirements.txt
alembic upgrade head
```

## Run API

```bash
uvicorn main:app --reload
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

## Docker Compose

```bash
docker compose up --build
```

Docker Compose 用于本地可复现运行与核心链路验证，不作为生产部署能力声明。

---

# 12. Tests / CI

完整本地测试：

```bash
python -m pytest -q
```

静态检查：

```bash
ruff check .
python -m compileall -q .
```

Smoke：

```bash
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

GitHub Actions `test` job 执行 Python 3.11 setup、dependency install、`compileall`、Ruff 和核心 focused tests。

Workflow：

- [.github/workflows/tests.yml](.github/workflows/tests.yml)

---

# 13. Documentation

推荐阅读顺序：

1. [README.md](README.md) — 项目定位与能力总览；
2. [experiments/evals/README.md](experiments/evals/README.md) — TechQA 长期主评测与实验契约；
3. [docs/architecture.md](docs/architecture.md) — 系统结构与边界；
4. [docs/agent_workflow.md](docs/agent_workflow.md) — Ticket Agent preview / confirm；
5. [docs/security.md](docs/security.md) — 当前认证与权限边界；
6. `experiments/evals/reports/` — retrieval / hybrid / evidence / generation artifacts。

`docs/*_report.md` 与 `docs/superpowers/` 中保留历史阶段报告、设计与实验计划，用于追溯项目演进；历史 roadmap 不自动代表当前产品方向。

---

# 14. Current Scope / Non-Claims

当前 README、代码和简历保持以下边界：

- TechQA 是长期主技术支持语料与主评测基准，不再计划迁移到另一套 primary corpus；
- 不把离线 BM25 / RRF / Hybrid 实验写成线上 Hybrid Serving；
- 不把规则化 Ticket 分类写成自主 ReAct / autonomous planning；
- 不把 Demo JWT + tenant scope 写成完整生产级 IAM / multi-tenant isolation；
- 不把 Docker Compose 写成生产部署；
- 不把 approval `pending` 校验写成并发 exactly-once guarantee；
- 不在正式 frozen generation evaluation 完成前声称 document-aware context expansion 带来生成质量提升；
- 不声称当前系统已具备完整 multi-source / conflict-resolution / autonomous Agentic RAG 能力。

项目当前关注的是：

> **在同一 Technical Support domain 上建立“可信 Benchmark → 失败归因 → 工程 intervention → 再评测”的连续主线，同时把 RAG 与真实业务 side effect 放进可控、可审计的 Agent 工作流。**

---

## Project Name

- 对外展示名：**Enterprise Support AI Copilot**
- 中文定位：**企业技术支持 RAG + Controlled Ticket Agent**
- Repository：`Enterprise-Support-AI-Copilot-API`
