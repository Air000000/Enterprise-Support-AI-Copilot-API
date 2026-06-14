# Enterprise Support AI Copilot

企业内部知识库与工单 AgentOps 后端系统。

本项目最初由 FastAPI Todo / AI Todo API 演进而来，当前 `learn-rag` 分支已经升级为面向企业内部支持场景的 AI Copilot 后端项目。

当前系统已经完成 Enterprise RAG Core、Ticket CRUD、Ticket Agent preview / confirm，以及 AgentOps 审计记录的 MVP 闭环。项目重点从单纯知识库问答扩展为“知识检索 → 工单预览 → 人工确认 → 工单创建 → AgentOps 审计”的企业支持 Copilot 后端。


当前版本定位：

```text
Enterprise Support AI Copilot
├── Enterprise RAG Core
│   ├── 企业内部支持文档集
│   ├── 文档加载与切块
│   ├── embedding 批处理
│   ├── Chroma 向量库
│   ├── tenant/category metadata filter
│   ├── /rag/search
│   ├── /rag/ask
│   ├── answer + structured sources
│   ├── retrieval eval
│   ├── hit@1 / hit@3 / mrr@3
│   └── category breakdown
├── Ticket CRUD
│   ├── create ticket
│   ├── list ticket
│   ├── get ticket
│   └── update ticket
├── Ticket Agent
│   ├── /agent/ticket/preview
│   ├── /agent/ticket/confirm
│   ├── TicketDraft
│   ├── TicketAgentSource
│   └── approval ownership validation
└── AgentOps
    ├── agent_runs
    ├── approval_requests
    └── tool_calls
```

------

## 1. 项目定位

本项目模拟企业内部支持场景。员工可以围绕 IT、HR、财务、行政、安全等内部制度和流程提出问题，系统基于企业知识库进行检索，并返回带来源依据的回答。

当前项目已经扩展到工单场景：当知识库无法充分解决问题时，系统可以生成工单预览，由用户确认后再创建真实工单，并记录 Agent 执行过程、工具调用参数和审批状态。

目标链路如下：

```text
企业内部文档
↓
文档加载 / 上传
↓
文本切块
↓
embedding
↓
Chroma vector store
↓
tenant/category 过滤检索
↓
RAG answer with sources
↓
retrieval logs / latency / token metrics
↓
Ticket Agent preview
↓
human approval
↓
create ticket
↓
tool_calls / agent_runs audit
```

项目最终希望体现以下工程能力：

| 能力         | 项目体现                                       |
| ------------ | ---------------------------------------------- |
| AI 应用后端  | FastAPI、service 分层、API schema、测试        |
| RAG 工程化   | 文档加载、chunking、embedding、Chroma、sources |
| 企业权限边界 | tenant/category metadata filter                |
| 可评测性     | retrieval eval、hit@1、hit@3、failed cases     |
| Agent 工作流 | Ticket Agent preview / confirm                 |
| AgentOps     | tool_calls、agent_runs、approval_requests      |
| 可维护性     | README、实验记录、测试、分阶段迭代             |

------

## 2. 当前阶段

当前阶段：
Enterprise Support AI Copilot MVP


已完成能力：

| 模块                | 当前状态                                                   |
| ------------------- | --------------------------------------------------------  |
| FastAPI 基础后端    | 已完成                                                     |
| Todo CRUD           | 已保留，作为早期基础后端能力                                |
| AI Todo / LLM 调用  | 已保留，作为早期 LLM API 调用练习                           |
| RAG API             | 已完成 `/rag/search`、`/rag/ask`                           |
| 企业文档集          | 已完成 10 份企业内部支持文档                                |
| 文档分类            | 已支持 `it`、`hr`、`finance`、`admin`、`security`           |
| 文档 metadata       | 已支持 `tenant_id`、`category`                             |
| Chunk metadata      | 已支持 `tenant_id`、`category`                             |
| Chunk 策略          | 已针对企业长文档调优                                        |
| Embedding           | 已支持分批请求                                             |
| Vector store        | 使用 Chroma 持久化向量库                                   |
| Metadata filter     | 已支持 tenant/category 过滤                                |
| API category filter | 已支持 request.category + mock tenant context              |
| Sources             | RAG response 返回结构化来源信息                             |
| 测试                | RAG / Ticket / AgentOps / Todo 测试均已覆盖                 |
| Retrieval eval      | 已支持 hit@1、hit@3、mrr@3、avg latency、category breakdown |
| Ticket CRUD         | 已完成 create / list / get / update                         |
| Ticket Agent        | 已完成 preview / confirm 两阶段流程                         |
| Human approval      | 已通过 approval_requests 表记录                             |
| Tool calls audit    | 已通过 tool_calls 表记录                                    |
| AgentOps records    | 已支持 agent_runs / approval_requests / tool_calls          |

后续开发重点：

| 模块                      | 状态   |
| ------------------------- | ------ |
| Enterprise RAG eval v1    | 已完成 |
| Retrieval metrics         | 已完成 hit@1 / hit@3 / mrr@3 / avg latency |
| Category breakdown        | 已完成 |
| Ticket CRUD               | 已完成 |
| Ticket Agent              | 已完成 MVP |
| Human approval            | 已完成 MVP |
| Tool calls audit          | 已完成 MVP |
| AgentOps records          | 已完成 MVP |
| AgentOps 查询 API         | 下一步 |
| rejected / cancelled flow | 下一步 |
| Document upload API       | 待开发 |
| Docker Compose 最终部署版 | 待整理 |

------

## 3. 技术栈

| 类型                | 技术                                        |
| ------------------- | ------------------------------------------- |
| Web 框架            | FastAPI                                     |
| 数据校验            | Pydantic                                    |
| ORM / 数据库        | SQLModel + SQLite                           |
| LLM / Embedding API | OpenAI-compatible SDK + DashScope / Bailian |
| Vector store        | Chroma                                      |
| 测试                | pytest + FastAPI TestClient                 |
| 部署基础            | Docker                                      |
| 语言                | Python                                      |

------

## 4. 当前项目结构

```text
fastapi-todo-api/
├── main.py
├── routers/
│   └── rag.py
├── schemas/
│   └── rag.py
├── services/
│   └── rag_service.py
├── experiments/
│   ├── docs/
│   │   ├── admin/
│   │   ├── finance/
│   │   ├── hr/
│   │   ├── it/
│   │   └── security/
│   ├── index/
│   ├── rag_local/
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   ├── build_rag_index.py
│   │   ├── query_index.py
│   │   ├── build_chroma_index.py
│   │   ├── query_chroma.py
│   │   └── query_rag_chroma.py
│   ├── evals/
│   └── README.md
├── tests/
│   ├── test_todos.py
│   ├── test_rag_api.py
│   └── test_rag_service.py
├── data/
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitattributes
└── README.md
```

------

## 5. 环境变量

在项目根目录创建 `.env` 文件，并参考 `.env.example` 配置本地运行参数：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
```

------

## 6. 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn main:app --reload
```

打开 Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

------

## 7. 构建 Chroma 向量索引

当前企业文档位于：

```text
experiments/docs/
```

构建 Chroma index：

```bash
python -m experiments.rag_local.build_chroma_index
```

当前企业文档集构建结果：

```text
Loaded documents: 10
Generated chunks: 40
Generated embeddings: 40
Collection count: 40
```

------

## 8. RAG 检索命令示例

### IT 类问题

```bash
python -m experiments.rag_local.query_chroma "VPN 连不上应该先检查什么？" --top-k 3 --tenant-id tenant_demo --category it
```

### HR 类问题

```bash
python -m experiments.rag_local.query_chroma "请假需要在系统里提交吗？" --top-k 3 --tenant-id tenant_demo --category hr
```

### 财务类问题

```bash
python -m experiments.rag_local.query_chroma "差旅报销需要哪些材料？" --top-k 3 --tenant-id tenant_demo --category finance
```

### 行政类问题

```bash
python -m experiments.rag_local.query_chroma "会议室临时不用需要释放吗？" --top-k 3 --tenant-id tenant_demo --category admin
```

### 安全类问题

```bash
python -m experiments.rag_local.query_chroma "外包人员可以共用账号吗？" --top-k 3 --tenant-id tenant_demo --category security
```

------

## 9. RAG API

### POST `/rag/search`

功能：基于 Chroma 向量库执行 top-k 检索，支持 category filter。

请求示例：

```json
{
  "query": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "category": "it"
}
```

字段说明：

| 字段       | 含义                                                         |
| ---------- | ------------------------------------------------------------ |
| `query`    | 用户问题                                                     |
| `top_k`    | 返回的检索结果数量                                           |
| `category` | 可选分类过滤，例如 `it`、`hr`、`finance`、`admin`、`security` |

当前 API 层只暴露 `category` filter。`tenant_id` 暂时由系统内部 mock tenant context 提供：

```text
tenant_demo
```

后续接入登录系统后，`tenant_id` 将从当前用户身份中获取。

响应字段：

| 字段          | 含义                 |
| ------------- | -------------------- |
| `document_id` | 来源文档 ID          |
| `chunk_id`    | 命中的 chunk ID      |
| `title`       | 文档标题             |
| `source_path` | 文档路径             |
| `chunk_index` | chunk 在文档中的顺序 |
| `distance`    | 向量距离             |
| `preview`     | chunk 内容预览       |
| `tenant_id`   | 来源租户             |
| `category`    | 来源分类             |

------

### POST `/rag/ask`

功能：执行完整 RAG 流程，包括 Chroma 检索、上下文构造、LLM 回答生成和 sources 返回。

请求示例：

```json
{
  "question": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "max_distance": 0.9,
  "category": "it"
}
```

响应字段：

| 字段               | 含义                      |
| ------------------ | ------------------------- |
| `answer`           | 基于检索上下文生成的回答  |
| `retrieval_status` | 检索状态                  |
| `top_distance`     | top1 检索结果距离         |
| `sources`          | 回答引用的来源 chunk 列表 |

当检索不到足够相关内容时，系统返回拒答：

```text
我在已提供资料中没有找到足够依据。
```

------

## 10. 当前 API 分层

```text
POST /rag/search
→ routers/rag.py::rag_search()
→ services/rag_service.py::search_documents()
→ experiments/rag_local/query_chroma.py::search_chroma()

POST /rag/ask
→ routers/rag.py::rag_ask()
→ services/rag_service.py::answer_question()
→ experiments/rag_local/query_rag_chroma.py::ask_rag()
```

各层职责：

| 层                        | 职责                                            |
| ------------------------- | ----------------------------------------------- |
| `routers/rag.py`          | 处理 HTTP 请求、调用 service、组装 API response |
| `schemas/rag.py`          | 定义 request / response schema                  |
| `services/rag_service.py` | 提供 RAG 业务入口                               |
| `experiments/rag_local/`  | 执行底层 RAG、Chroma、embedding、LLM 逻辑       |
| `tests/`                  | 验证 API 层和 service 层行为                    |

------

## 11. 测试

运行测试：

```bash
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
pytest tests/test_todos.py
pytest tests/test_tickets.py
pytest tests/test_agent_ops_service.py
pytest tests/test_ticket_agent_service.py
pytest tests/test_agent_ticket_api.py
```

当前结果：

```text
RAG / Ticket / AgentOps / Todo focused test suites are passing.
```

测试覆盖：

| 文件                        | 覆盖内容                                              |
| --------------------------- | ----------------------------------------------------- |
| `tests/test_todos.py`       | Todo 基础 CRUD 和请求校验                             |
| `tests/test_rag_api.py`     | RAG API happy path、422 validation、500 service error |
| `tests/test_rag_service.py` | service 层是否正确调用底层 search / ask 函数          |

RAG 测试使用 `monkeypatch` 隔离真实 Chroma、embedding 和 LLM 调用，因此不会消耗 token。

------

## 12. 当前版本记录

| Version                         | Documents | Categories                           | Chunks | Vector Store  | Eval             |
| ------------------------------- | --------- | ------------------------------------ | ------ | ------------- | ---------------- |
| RAG v0.1 learning-doc baseline  | 5         | general                              | 5      | JSON / Chroma | 15 条旧 eval     |
| RAG Core v0.2 enterprise corpus | 10        | it / hr / finance / admin / security | 40     | Chroma        | 企业 eval 待开发 |

旧 learning-doc baseline 的 eval 结果：

| Retriever         | hit@1 | hit@3 | top1 miss cases | failed cases |
| ----------------- | ----- | ----- | --------------- | ------------ |
| JSON cosine index | 0.93  | 1.00  | 1               | 0            |
| Chroma            | 0.93  | 1.00  | 1               | 0            |

旧 eval 基于 FastAPI、Docker、Embedding、RAG、SQLModel 学习文档，主要用于记录早期 RAG 学习阶段的 baseline。当前企业文档集会使用新的 enterprise RAG eval 进行评测。

------

## 13. 下一步计划

短期下一步：

```text
企业 RAG eval v1
```

计划包括：

```text
1. 新增 enterprise_rag_cases.jsonl
2. 设计 30 条企业支持问题
3. 每条 case 包含 question、expected_document_id、category
4. eval runner 支持 category filter
5. 输出 hit@1、hit@3、top1_miss_cases、failed_cases
6. 分析成功案例和失败案例
```

中期计划：

```text
1. retrieval logs
2. latency / token metrics
3. documents / chunks 后端化
4. Ticket CRUD
5. Ticket Agent preview / confirm
6. tool_calls / agent_runs audit
7. human approval
```

最终目标：

```text
企业内部知识库 RAG
+
受控 Ticket Agent
+
AgentOps 审计与评测
```

------

## 14. 项目说明

仓库名保留为 `fastapi-todo-api`，因为项目最初从 FastAPI Todo API 开始演进。当前 `learn-rag` 分支是 `Enterprise Support AI Copilot` 的开发分支。

后续完成企业 RAG eval、Ticket Agent 和项目文档整理后，再考虑将主项目合并到 `main` 分支或切换默认分支。