# Enterprise Support AI Copilot

[![Tests](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml)

面向企业 IT 支持场景的 **RAG + Controlled Ticket Agent** 后端，重点解决四类问题：

- 企业知识如何完成入库、检索、来源追踪与删除下架；
- Agent 如何在真实业务写操作前保持可控，并通过 Human-in-the-loop 审批约束 side effect；
- RAG 效果如何通过冻结 Benchmark、held-out comparison 和 failure analysis 进行可信评测；
- Agent 运行、工具调用、审批与检索过程如何进入可查询、可审计的 AgentOps 轨迹。

当前项目已从早期 FastAPI Todo / RAG 学习代码演进为 Enterprise Support AI Copilot。公开主线聚焦 **Controlled Agent、RAG Evaluation、Failure Diagnosis、AgentOps / Engineering**，Legacy Todo 能力仅作为历史兼容保留。

---

## 30 秒看项目

```text
Enterprise Support AI Copilot

Knowledge / Document Lifecycle
        │
        ▼
Dense Chroma Retrieval ──► Answer + Sources / Low-relevance Refusal
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

Offline Evaluation
TechQA ──► Dense Baseline ──► Rerank / Hybrid Experiments
                         └──► Evidence-level Audit
                         └──► Generation Evaluation Harness
```

### 当前核心能力

| 能力 | 当前实现 |
| --- | --- |
| Controlled Ticket Agent | `search_kb` / `classify_ticket` / `create_ticket`，preview-confirm + Human-in-the-loop |
| RAG Runtime | Chroma Dense Retrieval、tenant/category filter、sources、低相关拒答 |
| RAG Evaluation | TechQA 28,481 文档、610 条可回答检索问题、冻结 TRAIN / DEV、Recall@K / MRR |
| Rerank / Hybrid Research | Dense Top-100 + `qwen3-rerank` 正式对照；BM25 / RRF / Hybrid 作为离线受控实验 |
| Failure Diagnosis | candidate coverage、chunk crowding、evidence-level audit、route-selection gate |
| AgentOps | Agent Run / Tool Call / Approval / Retrieval Trace 与聚合指标 |
| Access Control | Demo JWT、role check、tenant-scoped access |
| Engineering | Alembic、Pytest、Ruff、GitHub Actions、Docker Compose、Smoke |

---

# 1. Controlled Ticket Agent

Ticket Agent 的目标不是让模型直接修改业务状态，而是把“建议”和“真实写操作”拆开。

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

当前 Ticket Agent 使用三个业务工具语义：

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

Confirm 只有在以下条件全部成立时才允许创建真实工单：

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

> 边界说明：当前流程能拒绝非 `pending` 的再次确认，但不将其表述为并发场景下的 exactly-once side-effect guarantee。

更多实现细节见 [docs/agent_workflow.md](docs/agent_workflow.md)。

---

# 2. Enterprise RAG Runtime

当前在线 / API serving 路径保持为 **Dense Chroma Retrieval**。

主要能力：

- `/rag/search`
- `/rag/ask`
- 文档 chunk 检索
- `tenant_id` / `category` metadata filter
- 结构化 sources 返回
- 无上下文与低相关拒答
- retrieval logging

问答路径只允许基于检索 Context 生成答案，并返回对应 sources。

当前低相关拒答使用 Dense Top-1 distance 作为信号；该阈值是工程策略，不被描述为通用最优阈值。

> **重要边界：** BM25 / RRF / Hybrid / `qwen3-rerank` 当前主要用于 `experiments/evals/` 的离线评测与对照实验，不把它们描述成线上 serving 已切换到 Hybrid Retrieval。

---

# 3. Document Lifecycle

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

# 4. RAG Evaluation：TechQA

`experiments/evals/` 是当前正式离线评测入口。

## Frozen Data Contract

TechQA retrieval benchmark：

- **28,481** 篇 Technote 文档；
- **610** 条 answerable retrieval queries；
- 每条 query 恰好 1 个 relevant document；
- qrels 是 document-level，而运行时 retrieval unit 是 chunk；
- 正式评测先保留原始 chunk ranking，再按 `document_id` 首次出现位置 collapse 成 document ranking。

Split contract：

| Split | Answerable | Impossible | 用途 |
| --- | ---: | ---: | --- |
| TRAIN | 450 | 150 | development / failure analysis / parameter selection |
| DEV | 160 | 150 | frozen held-out comparison |

正式 E0 / E1 对照冻结后，不使用 individual DEV failure 反向调参。

完整数据版本、SHA256、split 与评测契约见：

- [experiments/evals/README.md](experiments/evals/README.md)
- [experiments/evals/datasets/techqa/manifest.json](experiments/evals/datasets/techqa/manifest.json)

## Dense → Rerank Held-out Result

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

这里强调的是同一冻结 Benchmark 上的 held-out improvement，而不是跨不同数据集比较绝对 Recall 数值。

---

# 5. Failure Diagnosis：不是继续堆组件

项目后续没有把“增加 Hybrid”直接等同于“系统一定更好”，而是采用受控实验和 route-selection gate 决定方案是否值得继续。

## BM25 / RRF / Hybrid

离线实验比较过：

- Dense Retrieval
- BM25
- Dense + BM25 / RRF Hybrid
- Dense / Hybrid candidate pool + `qwen3-rerank`

C1 Hybrid + Rerank 在 TRAIN aggregate metrics 上有改善，但 **没有达到预注册的 early-rank MRR gate**：

```text
Recall@20 gate: PASS
MRR@10 gate: FAIL
Overall C1 decision: FAIL
```

因此没有把“指标有一点上涨”包装成成功路线，也没有继续投入后续付费优化。

相关报告：

- [experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md](experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md)
- [experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md](experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md)

## Evidence-level Audit

Document Recall 仍可能掩盖一个重要问题：

> **命中了正确文档，不等于真正包含答案的 evidence chunk 已经进入高位 context。**

因此又建立了一层人工 evidence audit：

- 60 条已标注 TRAIN queries；
- 54 条进入正式 evidence evaluation；
- 187 个 candidate chunk labels；
- 将 evidence 区分为 weak / useful / answer-bearing；
- 单独计算 AnswerEvidenceHit、Evidence MRR 与 GoldDocHitButEvidenceMiss 等指标。

这层 audit 用于定位 candidate coverage、fusion compression 与 answer-bearing evidence ranking 的问题，不替代官方 TechQA document-level benchmark。

相关 artifacts：

- [experiments/evals/reports/r1_evidence_audit/evidence_metrics.json](experiments/evals/reports/r1_evidence_audit/evidence_metrics.json)
- [experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl](experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl)

---

# 6. Document-aware Context Expansion

基于前述 failure analysis，Generation Eval Harness 中实现了：

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

当前策略标识：

```text
document_aware_forward_expansion_v1
```

对应实现：

- [experiments/evals/eval_techqa_generation.py](experiments/evals/eval_techqa_generation.py)

Generation dataset 当前包含：

- 610 answerable；
- 300 impossible；
- 共 910 条 QA records。

评测 Harness 已包含 correctness、faithfulness、abstention / hallucination 等生成侧指标与 frozen run identity。

> **重要边界：** 当前 README 不声称该 context policy 已带来新的正式 generation uplift；在冻结评测完成前，只把它作为基于 failure diagnosis 实现的工程 intervention。

---

# 7. AgentOps / Observability

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

可以按 tenant 查询：

- Agent Run 状态；
- Tool Call 成功 / 失败与 error type；
- Approval 状态；
- Retrieval no-context / refused / failed；
- Retrieval source distribution；
- 单次 Run Trace。

---

# 8. Authentication / Tenant Scope

项目包含一个用于当前工程验证的 Demo JWT Auth：

- Bearer token；
- `user_id`；
- `tenant_id`；
- `role`；
- `support` / `admin` 等角色检查；
- tenant-scoped Document / AgentOps / RAG 访问。

> **边界说明：** 这里的目标是验证认证上下文和 tenant scope 在应用链路中的传递，不把它描述为完整生产级 IAM / RBAC 或数据库级多租户隔离方案。

安全边界见 [docs/security.md](docs/security.md)。

---

# 9. Project Structure

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

说明：

- `rag_runtime/` 是正式 RAG runtime 路径；
- `experiments/evals/` 保存 TechQA Benchmark、受控实验和评测 artifacts；
- `experiments/rag_local/` 作为早期兼容入口保留；
- `data/todos.db` 是历史文件名，未为展示目的强行迁移；
- Todo / AI Todo 能力仍保留，但不再作为项目核心展示。

---

# 10. Quick Start

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

## Install

```bash
pip install -r requirements.txt
```

## Database Migration

```bash
alembic upgrade head
```

如果旧本地 SQLite 数据库没有 Alembic 版本标记，可以在确认不需要保留开发数据后重建数据库；如需保留已有数据，请先备份并根据当前 schema 状态处理 migration / stamp。

## Run API

```bash
uvicorn main:app --reload
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

# 11. Docker

Build：

```bash
docker build -t enterprise-support-ai-copilot-api .
```

Run：

```bash
docker run -p 8000:8000 enterprise-support-ai-copilot-api
```

Compose：

```bash
docker compose up --build
```

Docker Compose 在本项目中用于本地可复现运行与核心链路验证，不作为生产部署能力声明。

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

GitHub Actions `test` job 当前执行：

1. Python 3.11 setup；
2. dependency install；
3. `compileall`；
4. `ruff check .`；
5. RAG / Document / Ticket / AgentOps / Smoke focused tests。

Workflow：

- [.github/workflows/tests.yml](.github/workflows/tests.yml)

## Smoke

```bash
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

---

# 13. Documentation

推荐阅读顺序：

1. [README.md](README.md) — 项目入口与当前能力；
2. [docs/architecture.md](docs/architecture.md) — 系统结构与边界；
3. [docs/agent_workflow.md](docs/agent_workflow.md) — Ticket Agent preview / confirm；
4. [experiments/evals/README.md](experiments/evals/README.md) — TechQA Eval Contract；
5. [docs/security.md](docs/security.md) — 当前认证与权限边界；
6. `experiments/evals/reports/` — retrieval / hybrid / evidence / generation 实验 artifacts。

`docs/*_report.md` 与 `docs/superpowers/` 中保留历史阶段报告、设计与实验计划，用于追溯项目演进过程。

---

# 14. Legacy Compatibility

本项目最初由 FastAPI Todo / AI Todo API 演进而来，以下能力继续保留用于兼容已有测试和展示代码演进，但不再作为当前核心能力：

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`
- `tests/test_todos.py`

保留这些代码的原因不是把 Todo 功能继续包装成项目亮点，而是避免为了展示而无必要地破坏稳定历史路径。

---

# 15. Current Scope / Non-Claims

为了让 README、代码和简历保持一致，当前项目明确不做以下过度 Claim：

- 不把离线 BM25 / RRF / Hybrid 实验写成线上 Hybrid Serving；
- 不把规则化 Ticket 分类写成自主 ReAct / autonomous planning；
- 不把 Demo JWT + tenant scope 写成完整生产级 IAM / multi-tenant isolation；
- 不把 Docker Compose 写成生产部署；
- 不把 approval `pending` 校验写成并发 exactly-once guarantee；
- 不在正式 frozen generation evaluation 完成前声称 document-aware context expansion 带来生成质量提升。

项目当前更关注：

> **把一个 RAG + Agent Demo 做成可控、可审计、可评测，并能通过失败归因决定下一步工程迭代的应用系统。**

---

## Project Name

- 对外展示名：**Enterprise Support AI Copilot**
- 中文定位：企业 IT 支持 RAG 工单 Agent
- Repository：`Enterprise-Support-AI-Copilot-API`
