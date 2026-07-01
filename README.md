# Enterprise Support AI Copilot

[![Tests](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml)

企业内部支持 AI Copilot 后端，面向企业知识检索、受控工单创建和 AgentOps 审计场景。

本项目最初由 FastAPI Todo / AI Todo API 演进而来，当前 `main` 分支已经固定为 Enterprise Support AI Copilot 的稳定基线。当前基线强调三件事：

- 企业知识库检索与带来源回答
- 预览 / 确认式 Ticket Agent 流程
- AgentOps 审计、审批与指标查询

## 当前基线

```text
Enterprise Support AI Copilot
├── Enterprise RAG Core
├── Document Backend
├── Ticket CRUD
├── Ticket Agent preview / confirm
├── AgentOps audit + read APIs
├── Retrieval Logs / Metrics
├── Docker Compose local runtime
└── Smoke scripts
```

当前版本目标是把“知识检索 -> 工单预览 -> 人工确认 -> 工单创建 -> AgentOps 审计”这条链路稳定下来，而不是继续扩展目录结构或重写运行时模块。

## 主能力

### 1. Enterprise RAG Core
- `/rag/search`
- `/rag/ask`
- 结构化 sources 返回
- tenant/category metadata filter
- retrieval eval: `hit@1` / `hit@3` / `mrr@3`

### 2. Document Backend
- `/documents/upload`
- `/documents`
- `/documents/{document_id}`
- `/documents/{document_id}/index`
- `DELETE /documents/{document_id}`
- 支持 `md` / `txt` 上传、索引、删除闭环

### 3. Ticket + Ticket Agent
- `/tickets` create / list / get / update
- `/agent/ticket/preview`
- `/agent/ticket/confirm`
- preview / confirm 两阶段控制真实工单创建

### 4. AgentOps
- `agent_runs`
- `tool_calls`
- `approval_requests`
- `/agent-ops/runs`
- `/agent-ops/metrics/summary`
- approval reject / cancel APIs

## Legacy Compatibility

以下能力仍然保留，用于兼容早期学习阶段和已有测试，但不再作为当前主能力展示：

- `/todos`
- `/chat`
- `/ai/chat`
- `/ai/extract-tasks`
- `/ai/create-todos`
- `tests/test_todos.py`

保留原因：

- 兼容现有测试与历史演进说明
- 避免在基线冻结阶段删除已稳定代码
- 作为早期项目演进的可追溯记录

## 当前结构说明

```text
enterprise-support-ai-copilot-api/
├── main.py
├── database.py
├── routers/
├── schemas/
├── services/
├── models/
├── docs/
├── scripts/
├── tests/
├── experiments/
│   ├── evals/
│   ├── docs/
│   └── rag_local/
├── docker-compose.yml
├── Dockerfile
└── README.md
```

说明：

- `rag_runtime/` 现在是正式运行时路径，`experiments/rag_local/` 保留为兼容层。`python -m experiments.rag_local.*` 旧命令仍可用，但新的运行与说明统一推荐 `python -m rag_runtime.*`。
- `data/todos.db` 的命名属于历史遗留，本轮不改，避免引入数据迁移和 Docker volume 变化。
- Todo 相关接口与测试继续保留，但按 Legacy 处理。

## 环境变量

参考 `.env.example`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
DATABASE_URL=sqlite:///data/todos.db
SQL_ECHO=true
DOCUMENT_STORAGE_ROOT=storage/documents
```

## 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn main:app --reload
```

打开 Swagger：

```text
http://127.0.0.1:8000/docs
```

## Docker 运行

构建镜像：

```bash
docker build -t enterprise-support-ai-copilot-api .
```

启动容器：

```bash
docker run -p 8000:8000 enterprise-support-ai-copilot-api
```

使用 Compose：

```bash
docker compose up --build
```

## 测试

核心 RAG：

```bash
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
```

Ticket / AgentOps：

```bash
pytest tests/test_tickets.py
pytest tests/test_agent_ops_service.py
pytest tests/test_agent_ops_api.py
pytest tests/test_ticket_agent_service.py
pytest tests/test_agent_ticket_api.py
```

Document Backend：

```bash
pytest tests/test_document_models.py
pytest tests/test_document_service.py
pytest tests/test_document_api.py
```

Legacy Todo：

```bash
pytest tests/test_todos.py
```

## 验收与 Smoke

```bash
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

## 文档入口

- [docs/project_summary.md](docs/project_summary.md): 当前主基线总结
- [docs/architecture.md](docs/architecture.md): 当前系统结构与边界
- [docs/agent_workflow.md](docs/agent_workflow.md): Ticket Agent 流程
- [docs/security.md](docs/security.md): 当前安全边界
- `docs/*_report.md`: 历史阶段性报告，保留为演进记录

## 当前基线边界

本轮基线冻结明确不做以下改动：

- 不改 HTTP 路径、请求体、响应体、状态码
- 不改业务逻辑
- 保留 `experiments/rag_local` 兼容入口，但正式运行时已切换到 `rag_runtime`
- 不改默认数据库文件名 `todos.db`
- 不删除 Todo / AI Todo 兼容能力

## 推荐项目命名

- 对外展示名：`Enterprise Support AI Copilot`
- 中文名：`企业内部支持 AI Copilot`
- 仓库名：`enterprise-support-ai-copilot-api`
