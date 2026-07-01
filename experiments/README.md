# 企业支持 RAG 实验说明

本目录用于实现和验证 `Enterprise Support AI Copilot` 的 RAG Core。

当前实验已经从早期 learning-doc RAG 扩展为企业内部支持场景。文档集模拟公司内部 IT、HR、财务、行政、安全等知识库内容，目标是支持基于 category 和 tenant context 的受限检索，并为后续企业 RAG eval、Ticket Agent 和 AgentOps 能力打基础。

当前版本：

```text
RAG Core v1
```

------

## 1. 当前目标

当前阶段重点是将已有 RAG pipeline 工程化为主项目的企业知识库底座。

核心目标：

```text
1. 企业内部支持文档集
2. 稳定文档加载
3. 适合企业长文档的 chunk 策略
4. embedding 批处理
5. Chroma 持久化向量库
6. tenant/category metadata
7. category filter
8. /rag/search
9. /rag/ask
10. structured sources
11. API / service 自动化测试
12. 企业 RAG eval v1 已完成，包括 hit@1、hit@3、mrr@3、avg latency 和 category breakdown
```

------

## 2. 当前目录结构

```text
experiments/
  docs/
    admin/
      device_borrowing.md
      meeting_room.md

    finance/
      invoice_rules.md
      travel_reimbursement.md

    hr/
      leave_policy.md
      onboarding_process.md

    it/
      email_login_faq.md
      vpn_guide.md

    security/
      data_access_policy.md
      outsource_account.md

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

  README.md
```

------

## 3. 当前企业文档集

当前 `experiments/docs/` 下共有 10 份企业内部支持文档。

| Category   | Documents                                       | 说明                           |
| ---------- | ----------------------------------------------- | ------------------------------ |
| `it`       | `vpn_guide.md`、`email_login_faq.md`            | VPN、邮箱、登录、MFA、账号问题 |
| `hr`       | `leave_policy.md`、`onboarding_process.md`      | 请假、休假、入职、前 90 天安排 |
| `finance`  | `travel_reimbursement.md`、`invoice_rules.md`   | 差旅报销、发票、付款规则       |
| `admin`    | `meeting_room.md`、`device_borrowing.md`        | 会议室、设备借用与归还         |
| `security` | `data_access_policy.md`、`outsource_account.md` | 数据访问权限、外包与供应商账号 |

当前文档采用目录推断 category：

```text
experiments/docs/it/vpn_guide.md
→ category = it

experiments/docs/hr/leave_policy.md
→ category = hr
```

当前 tenant 暂时使用 mock 值：

```text
tenant_demo
```

后续接入用户系统后，`tenant_id` 将从当前用户身份中获取。

------

## 4. RAG Pipeline

当前实现的核心流程：

```text
企业内部 Markdown 文档
↓
document_loader
↓
Document(document_id, title, source_path, text, tenant_id, category)
↓
text_splitter
↓
Chunk(chunk_id, document_id, chunk_index, content, tenant_id, category)
↓
embedding
↓
Chroma vector store
↓
top-k retrieval with tenant/category filter
↓
RAG answer with structured sources
↓
API response
```

------

## 5. 文档加载

文件：

```text
rag_runtime/document_loader.py
```

能力：

```text
1. 递归读取 experiments/docs 下的 .md / .txt 文档
2. 生成 Document dataclass
3. 根据文件父目录推断 category
4. 使用 tenant_demo 作为当前 mock tenant
5. 保留 document_id、title、source_path、text 等基础字段
```

当前 `Document` 包含：

```text
document_id
title
source_path
text
tenant_id
category
```

运行：

```bash
python -m rag_runtime.document_loader
```

------

## 6. 文本切块

文件：

```text
rag_runtime/text_splitter.py
```

当前 chunk 参数：

```text
chunk_size = 800
chunk_overlap = 120
min_chunk_size = 150
```

当前切块策略：

```text
1. 优先按空行切自然段
2. 尽量将相关段落合并到同一 chunk
3. 长段落使用字符窗口切分
4. 对过短 chunk 做后处理合并
5. 合并标题-only chunk
6. 合并短残句 chunk
```

当前企业文档集切块结果：

```text
Loaded documents: 10
Generated chunks: 40
```

当前 `Chunk` 包含：

```text
chunk_id
document_id
title
source_path
chunk_index
content
tenant_id
category
```

运行：

```bash
python -m rag_runtime.text_splitter
```

------

## 7. Embedding 批处理

文件：

```text
rag_runtime/build_rag_index.py
```

当前 embedding 接口存在单次 batch size 限制，因此 `embed_texts()` 已支持分批请求。

当前行为：

```text
1. 将所有 chunk content 分成若干 batch
2. 每批调用 embedding API
3. 保持 embedding 顺序与 chunk 顺序一致
4. 合并所有 batch embeddings
```

该能力用于支持更大规模文档集，避免一次性提交过多文本导致接口参数错误。

------

## 8. Chroma Index

文件：

```text
rag_runtime/build_chroma_index.py
```

构建 Chroma index：

```bash
python -m rag_runtime.build_chroma_index
```

当前构建结果：

```text
Loaded documents: 10
Generated chunks: 40
Generated embeddings: 40
Collection count: 40
```

当前 Chroma metadata 包含：

```text
chunk_id
document_id
title
source_path
chunk_index
tenant_id
category
```

这些 metadata 用于检索阶段的 filter：

```text
tenant_id = tenant_demo
category = it / hr / finance / admin / security
```

------

## 9. Chroma 检索

文件：

```text
rag_runtime/query_chroma.py
```

当前 `search_chroma()` 支持：

```text
query
top_k
tenant_id
category
```

字段作用：

| 字段        | 作用                   |
| ----------- | ---------------------- |
| `tenant_id` | 限定当前租户的数据范围 |
| `category`  | 限定文档分类范围       |

示例：

```bash
python -m rag_runtime.query_chroma "VPN 连不上应该先检查什么？" --top-k 3 --tenant-id tenant_demo --category it
```

更多示例：

```bash
python -m rag_runtime.query_chroma "请假需要在系统里提交吗？" --top-k 3 --tenant-id tenant_demo --category hr
python -m rag_runtime.query_chroma "差旅报销需要哪些材料？" --top-k 3 --tenant-id tenant_demo --category finance
python -m rag_runtime.query_chroma "会议室临时不用需要释放吗？" --top-k 3 --tenant-id tenant_demo --category admin
python -m rag_runtime.query_chroma "外包人员可以共用账号吗？" --top-k 3 --tenant-id tenant_demo --category security
```

------

## 10. Chroma RAG Answer

文件：

```text
rag_runtime/query_rag_chroma.py
```

当前能力：

```text
1. 调用 search_chroma() 检索相关 chunks
2. 支持 tenant_id / category filter
3. 根据 max_distance 判断检索结果是否足够相关
4. 构造 RAG context
5. 调用 LLM 生成 answer
6. 返回 answer + retrieval_status + top_distance + sources
```

运行示例：

```bash
python -m rag_runtime.query_rag_chroma "VPN 连不上应该先检查什么？" --top-k 3 --tenant-id tenant_demo --category it
```

当 filter 后没有可用上下文时，系统返回：

```text
我在已提供资料中没有找到足够依据。
```

------

## 11. RAG API

当前 RAG pipeline 已接入 FastAPI。

相关文件：

```text
routers/rag.py
schemas/rag.py
services/rag_service.py
```

调用链：

```text
POST /rag/search
→ routers/rag.py::rag_search()
→ services/rag_service.py::search_documents()
→ rag_runtime/query_chroma.py::search_chroma()

POST /rag/ask
→ routers/rag.py::rag_ask()
→ services/rag_service.py::answer_question()
→ rag_runtime/query_rag_chroma.py::ask_rag()
```

------

### POST `/rag/search`

请求示例：

```json
{
  "query": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "category": "it"
}
```

响应结果包含：

```text
document_id
chunk_id
title
source_path
chunk_index
distance
preview
tenant_id
category
```

------

### POST `/rag/ask`

请求示例：

```json
{
  "question": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "max_distance": 0.9,
  "category": "it"
}
```

响应结果包含：

```text
answer
retrieval_status
top_distance
sources
```

`sources` 中包含：

```text
document_id
chunk_id
title
source_path
chunk_index
distance
preview
tenant_id
category
```

------

## 12. JSON Index 说明

早期 RAG 学习阶段实现过 JSON index：

```text
rag_runtime/build_rag_index.py
rag_runtime/query_index.py
```

JSON index 的作用是帮助理解 embedding 检索的基本原理：

```text
chunk content
↓
embedding
↓
保存到 rag_index.json
↓
query embedding
↓
cosine similarity
↓
top-k retrieval
```

当前主线以后续 Chroma 检索为准，JSON index 作为学习和对照实现保留。

构建 JSON index：

```bash
python -m rag_runtime.build_rag_index
```

查询 JSON index：

```bash
python -m rag_runtime.query_index "VPN 连不上应该先检查什么？"
```

------

## 13. Legacy Retrieval Eval

早期 learning-doc 阶段使用过 15 条 eval cases：

```text
experiments/evals/eval_questions.jsonl
```

旧 eval 文档包括：

```text
docker_notes
embedding_notes
fastapi_notes
rag_notes
sqlmodel_notes
```

旧结果：

| Retriever         | hit@1 | hit@3 | top1 miss cases | failed cases |
| ----------------- | ----- | ----- | --------------- | ------------ |
| JSON cosine index | 0.93  | 1.00  | 1               | 0            |
| Chroma            | 0.93  | 1.00  | 1               | 0            |

唯一 top1 miss：

```text
Question: embedding 在 RAG 中有什么作用？
Expected: doc_embedding_notes
Top1: doc_rag_notes
Top2: doc_embedding_notes
```

该 case 属于 query / label 语义重叠。

旧 learning-doc eval 作为 baseline 记录保留。当前企业文档集需要新的 enterprise RAG eval。

------

## 14. 当前测试

运行 RAG 相关测试：

```bash
pytest tests/test_query_chroma.py
pytest tests/test_rag_api.py
pytest tests/test_rag_service.py
```

当前覆盖：

| 文件                           | 覆盖内容                             |
| ---------------------------- | -------------------------------- |
| `tests/test_query_chroma.py` | Chroma metadata filter           |
| `tests/test_rag_api.py`      | `/rag/search`、`/rag/ask` API 层测试 |
| `tests/test_rag_service.py`  | service 层参数透传和下游调用测试             |

RAG API 测试使用 monkeypatch，不调用真实 embedding、Chroma 或 LLM。


------

## 15. 当前状态表

| Item                    | Current Status                                             |
| ----------------------- | ---------------------------------------------------------- |
| Documents               | 10 enterprise support docs                                 |
| Categories              | `it`、`hr`、`finance`、`admin`、`security`                 |
| Chunk strategy          | `chunk_size=800`、`overlap=120`、`min_chunk_size=150`      |
| Chunks                  | 40                                                         |
| Vector store            | Chroma                                                     |
| Metadata                | `tenant_id`、`category`                                    |
| API filter              | request.category + mock tenant context                     |
| RAG API                 | `/rag/search`、`/rag/ask`                                  |
| Tests                   | RAG API / service / metadata filter tests passing          |
| Legacy eval             | 15 learning-doc eval cases                                 |
| Enterprise eval         | 30 enterprise support eval cases                           |
| Enterprise eval metrics | hit@1=0.97, hit@3=1.00, mrr@3=0.98                         |
| Category breakdown      | admin / finance / hr / it / security                       |
| Main known ambiguity    | finance: `doc_travel_reimbursement` vs `doc_invoice_rules` |
| Current phase           | Enterprise RAG Core v1 completed                           |
| Next step               | Ticket Agent / AgentOps workflow hardening                 |


------

## 16. Enterprise RAG Eval v1

Enterprise RAG eval v1 已完成。

当前企业 eval 覆盖：

```text
1. 30 条企业支持问题
2. 覆盖 it / hr / finance / admin / security 五类
3. 每类 6 条
4. 使用 category filter 检索
5. 输出 hit@1、hit@3、mrr@3
6. 输出 avg_latency_ms
7. 输出 category breakdown
8. 输出 top1_miss_cases
```

当前 Chroma enterprise eval 结果：

| Metric | Result |
| ------ | ------ |
| Total  | 30     |
| hit@1  | 0.97   |
| hit@3  | 1.00   |
| mrr@3  | 0.98   |

当前 category breakdown：

| Category | Total | hit@1 | hit@3 | mrr@3 |
| -------- | ----- | ----- | ----- | ----- |
| admin    | 6     | 1.00  | 1.00  | 1.00  |
| finance  | 6     | 0.83  | 1.00  | 0.92  |
| hr       | 6     | 1.00  | 1.00  | 1.00  |
| it       | 6     | 1.00  | 1.00  | 1.00  |
| security | 6     | 1.00  | 1.00  | 1.00  |

当前唯一主要 top1 miss 位于 finance 场景：

```text
question: 发票报销有哪些要求？
expected_document_id: doc_invoice_rules
top1: doc_travel_reimbursement
top2: doc_invoice_rules
```

该 case 属于差旅报销与发票规则之间的业务语义重叠，不是 Chroma 检索链路故障。


------

## 17. 后续方向

Enterprise RAG eval 已完成，项目已经进入 Ticket Agent / AgentOps 阶段。

后续路线：

```text
RAG Core v1
↓
Ticket CRUD
↓
Ticket Agent preview / confirm
↓
Human approval
↓
Tool calls audit
↓
AgentOps read API
↓
AgentOps metrics
```

最终目标：

```text
企业知识库 RAG
+
受控工单 Agent
+
工具调用审计
+
评测与可观测性
```


## Compatibility

Legacy experiments/rag_local/ modules now wrap rag_runtime/.
Historical python -m experiments.rag_local.* commands should continue to work, but new examples in this document use python -m rag_runtime.*.
