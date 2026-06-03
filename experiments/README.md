# RAG Learning Lab

本目录用于学习和验证一个最小可运行的 local-doc RAG pipeline。

当前目标是：基于本地 Markdown 文档，实现文档加载、文本切块、embedding、检索、RAG 问答，以及 retrieval eval。项目同时支持两种检索后端：

1. JSON index：用于理解 embedding 检索的最小实现。
2. Chroma vector store：用于学习持久化向量库和更工程化的检索方式。

当前阶段重点不是追求复杂功能，而是把 RAG 的核心链路跑通、测评清楚、能解释失败案例。

---

## 1. 当前目录结构

```text
experiments/
  docs/
    docker_notes.md
    embedding_notes.md
    fastapi_notes.md
    rag_notes.md
    sqlmodel_notes.md

  index/
    rag_index.json

  rag_local/
    document_loader.py
    text_splitter.py
    build_rag_index.py
    query_index.py
    build_chroma_index.py
    query_chroma.py
    query_rag_chroma.py

  evals/
    eval_core.py
    eval_questions.jsonl
    eval_retrieval.py
    eval_chroma_retrieval.py
    eval_notes.md
```

---

## 2. RAG Pipeline

当前实现的核心流程如下：

```text
本地 Markdown 文档
↓
document_loader
↓
text_splitter
↓
embedding
↓
JSON index / Chroma vector store
↓
top-k retrieval
↓
RAG answer with sources
↓
retrieval eval
```

---

## 3. JSON Index 版本

JSON index 是一个最小向量索引实现，用于理解 RAG 检索的基本原理。

它会把每个 chunk 的文本、metadata 和 embedding 保存到：

```text
experiments/index/rag_index.json
```

### 构建 JSON index

```bash
python -m experiments.rag_local.build_rag_index
```

### 查询 JSON index

```bash
python -m experiments.rag_local.query_index "RAG 为什么需要 chunk？"
```

JSON 检索结果使用 `score` 表示相关性，当前实现中 `score` 越大表示越相关。

---

## 4. Chroma 版本

Chroma 版本用于学习持久化向量库。

相比 JSON index，Chroma 更适合后续扩展到更大的文档集合，也更接近真实 RAG 工程中的 vector store 使用方式。

### 构建 Chroma index

```bash
python -m experiments.rag_local.build_chroma_index
```

### 查询 Chroma index

```bash
python -m experiments.rag_local.query_chroma "RAG 为什么需要 chunk？"
```

Chroma 检索结果使用 `distance` 表示距离，通常 `distance` 越小表示越接近。

---

## 5. Retrieval Eval

当前使用 `eval_questions.jsonl` 作为最小评测集。

每条 eval case 包含：

```json
{"question": "...", "expected_document_id": "..."}
```

评测指标包括：

| 指标              | 含义                   |
| --------------- | -------------------- |
| hit@1           | 正确文档是否排在 top1        |
| hit@3           | 正确文档是否出现在 top3       |
| top1_miss_cases | 正确文档不在 top1，但仍在 top3 |
| failed_cases    | 正确文档没有出现在 top3       |

### 运行 JSON index eval

```bash
python -m experiments.evals.eval_retrieval
```

### 运行 Chroma eval

```bash
python -m experiments.evals.eval_chroma_retrieval
```

---

## 6. 当前 Eval 结果

当前 retrieval eval 数据集包含 15 条 case。

| Retriever         | hit@1 | hit@3 | top1 miss cases | failed cases |
| ----------------- | ----: | ----: | --------------: | -----------: |
| JSON cosine index |  0.93 |  1.00 |               1 |            0 |
| Chroma            |  0.93 |  1.00 |               1 |            0 |

当前 JSON index 和 Chroma 的 retrieval eval 结果一致。

唯一的 top1 miss case 是：

```text
Question: embedding 在 RAG 中有什么作用？
Expected: doc_embedding_notes
Top1: doc_rag_notes
Top2: doc_embedding_notes
```

这个 case 不一定是明确的检索失败。问题同时包含 `embedding` 和 `RAG` 两个主题，而 `doc_rag_notes` 中也解释了 embedding 在 RAG pipeline 中的作用。因此，这更像是一个 query / label 歧义案例，而不是明显的 retrieval failure。

---

## 7. 当前结论

当前 local-doc RAG v0.1 已完成以下能力：

* 读取本地 Markdown 文档
* 切分 chunk
* 生成 embedding
* 构建 JSON index
* 使用 cosine similarity 做 top-k 检索
* 构建 Chroma vector store
* 使用 Chroma 做 top-k 检索
* 基于同一批 eval cases 对比 JSON 和 Chroma
* 输出 hit@1、hit@3、top1 miss cases 和 failed cases
* 对歧义 case 进行分析

当前结果说明：在这个小规模测试集上，JSON index 和 Chroma 的检索效果一致，整体检索表现可接受。

---

## RAG API v0.1

当前 `learn-rag` 分支已经将本地 Chroma RAG pipeline 初步接入 FastAPI，提供了检索接口和问答接口。

### 接口列表

#### POST `/rag/search`

功能：调用 Chroma 向量库进行 top-k 检索，返回相关 chunk 及其来源信息。

请求示例：

```json
{
  "query": "为什么系统能判断两段文字语义相近？",
  "top_k": 3
}
```

响应中包含：

* `document_id`：来源文档 ID
* `chunk_id`：命中的 chunk ID
* `chunk_index`：chunk 在原文档中的顺序
* `distance`：向量检索距离
* `preview`：chunk 内容预览

#### POST `/rag/ask`

功能：执行完整 RAG 流程，包括 Chroma 检索、上下文构造、LLM 回答生成和 sources 返回。

请求示例：

```json
{
  "question": "RAG 为什么需要 chunk？",
  "top_k": 3,
  "max_distance": 0.9
}
```

响应中包含：

* `answer`：基于检索上下文生成的回答
* `retrieval_status`：检索状态
* `top_distance`：top1 检索结果的距离
* `sources`：回答引用的来源 chunk 列表

### 当前验证结果

* `/rag/search` 已在 Swagger 中手动验证通过。
* `/rag/ask` 已在 Swagger 中手动验证通过。
* API 响应已包含来源 metadata 和清理后的 `preview`。
* 当前 retrieval eval 数据集包含 15 条 case。
* JSON index 和 Chroma 当前结果均为 `hit@1 = 0.93`，`hit@3 = 1.00`。

---

## RAG API testing

当前 RAG API 已完成 router/service 分层，并补充了两类自动化测试。

### API layer tests

文件：`tests/test_rag_api.py`

验证内容：

* `/rag/search` 可以接收请求并返回检索结果
* `/rag/ask` 可以接收问题并返回 answer + sources
* 使用 `monkeypatch` mock `routers.rag.search_documents` 和 `routers.rag.answer_question`
* 不调用真实 Chroma、embedding 或 LLM，因此不消耗 token

- 覆盖 200 happy path、422 request validation 和 500 service error handling

### Service layer tests

文件：`tests/test_rag_service.py`

验证内容：

* `search_documents()` 会正确调用 `search_chroma()`
* `answer_question()` 会正确调用 `ask_rag()`
* 使用 `calls` 记录下游函数收到的参数
* 检查返回值和参数传递是否正确

### 当前测试结果

```text
14 passed, 1 warning
```

### 调用链与层次职责

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

分层职责：

* `routers/rag.py`：处理 HTTP 请求、调用 service、组装 API response
* `schemas/rag.py`：定义 RAG API 的 request / response schema
* `services/rag_service.py`：提供 RAG 业务入口
* `experiments/rag_local/`：执行底层 RAG / Chroma / embedding / LLM 逻辑

---

## RAG API structure

当前 `learn-rag` 分支已经将 RAG API 从单文件实现逐步拆分为 router、schema、service 和底层 RAG 实现几层。

当前结构如下：

```text
main.py
→ 创建 FastAPI app，并注册 RAG router

routers/rag.py
→ 定义 `/rag/search` 和 `/rag/ask` endpoint
→ 接收 HTTP 请求
→ 调用 service 层
→ 将 service 返回结果转换为 API response

schemas/rag.py
→ 定义 RAG API 的 request / response schema
→ 包括 RagSearchRequest、RagSearchResponse、RagAskRequest、RagAskResponse 等

services/rag_service.py
→ 提供 RAG 业务入口
→ search_documents() 调用底层 search_chroma()
→ answer_question() 调用底层 ask_rag()

experiments/rag_local/
→ 保留当前本地 RAG 实验实现
→ 包括 Chroma 检索、RAG 问答、embedding 和 LLM 调用逻辑

tests/
→ 包含 API layer tests 和 service layer tests
→ 使用 monkeypatch 隔离真实 Chroma / embedding / LLM 调用
```

这种分层的目标是让各模块职责更清楚：

* `router` 负责 HTTP 请求和响应；
* `schema` 负责 API 数据契约；
* `service` 负责业务入口；
* `experiments/rag_local` 负责底层 RAG 实现；
* `tests` 负责验证 API 层和 service 层行为。

当前重构保持了外部 API 行为不变，`/rag/search` 和 `/rag/ask` 的路径与响应结构保持稳定，并通过全量测试验证。
