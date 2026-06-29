# Document Backend MVP Report

## 1. 阶段背景

在阶段 7 之前，Enterprise Support AI Copilot 的 RAG 文档主要来自本地目录：

```text
experiments/docs/
```

文档索引主要依赖本地脚本完成：

```text
experiments/rag_local/document_loader.py
experiments/rag_local/text_splitter.py
experiments/rag_local/build_chroma_index.py
experiments/rag_local/query_chroma.py
```

这套方式适合学习、实验和离线 eval，但还不是一个真正的后端知识库模块。用户无法通过 API 上传文档，后端也无法记录文档生命周期，更无法通过 API 删除文档并确保 RAG 不再检索到它。

阶段 7 的目标是把本地脚本式文档处理升级成 Document Backend MVP。

---

## 2. 本阶段目标

本阶段实现了一个最小但完整的文档生命周期闭环：

```text
上传 md/txt 文档
↓
写入 documents 表
↓
触发索引
↓
切分 chunk
↓
写入 document_chunks 表
↓
写入 Chroma
↓
/rag/search 可以检索到上传文档
↓
删除文档
↓
删除 Chroma embeddings
↓
后续 /rag/search 不再返回该文档
```

这个阶段的重点不是做一个完整 CMS，而是让知识库从“静态实验目录”变成“后端 API 可管理的数据源”。

---

## 3. 已实现 API

### 3.1 上传文档

```http
POST /documents/upload
```

作用：

```text
接收 md/txt 文件，校验文件类型和内容，保存到本地 storage，并写入 documents 表。
```

上传成功后，文档状态为：

```text
uploaded
```

这表示文档已经进入系统，但还没有进入向量库。

---

### 3.2 查询文档列表

```http
GET /documents
```

作用：

```text
查询当前 tenant 下的文档列表。
```

支持：

```text
category
status
limit
offset
```

默认不返回 deleted 文档。

---

### 3.3 查询单个文档

```http
GET /documents/{document_id}
```

作用：

```text
查看某个文档的 metadata 和当前状态。
```

这个接口只读，不会触发索引，也不会修改数据库。

---

### 3.4 索引文档

```http
POST /documents/{document_id}/index
```

作用：

```text
读取已上传文档，切分 chunk，生成 embedding，写入 Chroma，并写入 document_chunks 表。
```

索引成功后：

```text
documents.status = indexed
documents.chunk_count > 0
```

这表示文档已经进入 RAG 知识库，可以被 `/rag/search` 检索到。

---

### 3.5 删除文档

```http
DELETE /documents/{document_id}
```

作用：

```text
删除 document_chunks 中记录的 Chroma embeddings，并将 documents.status 标记为 deleted。
```

删除成功后：

```text
documents.status = deleted
后续 GET /documents/{document_id} 返回 404
后续 /rag/search 不再返回该文档
```

---

## 4. 数据表设计

### 4.1 documents

`documents` 表记录文档级生命周期。

核心字段：

| 字段              | 说明                                 |
| --------------- | ---------------------------------- |
| `id`            | 文档 ID                              |
| `tenant_id`     | 租户隔离字段                             |
| `uploaded_by`   | 上传用户                               |
| `filename`      | 原始文件名                              |
| `file_type`     | 文件类型，当前支持 `md` / `txt`             |
| `category`      | 文档分类，如 `it`、`hr`、`finance`、`admin` |
| `source_path`   | 本地保存路径                             |
| `status`        | 文档状态                               |
| `version`       | 文档版本，MVP 默认为 1                     |
| `checksum`      | 文件 sha256                          |
| `chunk_count`   | 当前索引出的 chunk 数量                    |
| `error_message` | 索引失败原因                             |
| `created_at`    | 创建时间                               |
| `updated_at`    | 更新时间                               |

状态流转：

```text
uploaded
↓
indexing
↓
indexed

uploaded / indexing
↓
failed

uploaded / indexed / failed
↓
deleted
```

---

### 4.2 document_chunks

`document_chunks` 表记录文档切分后的 chunk 审计信息。

核心字段：

| 字段              | 说明                       |
| --------------- | ------------------------ |
| `id`            | chunk 数据库 ID             |
| `tenant_id`     | 租户隔离字段                   |
| `document_id`   | 所属 document              |
| `chunk_index`   | 文档内 chunk 序号             |
| `content`       | chunk 文本                 |
| `category`      | 继承 document.category     |
| `metadata_json` | 写入 Chroma 的 metadata 快照  |
| `embedding_id`  | Chroma 中对应的 embedding id |
| `created_at`    | 创建时间                     |

其中 `embedding_id` 是删除闭环的关键。

格式示例：

```text
doc:{document_id}:v1:chunk:{chunk_index}
```

删除文档时，系统会根据 `document_chunks.embedding_id` 删除 Chroma 中的对应向量。

---

## 5. 当前模块结构

本阶段主要涉及以下文件：

```text
models/document.py
schemas/document.py
services/document_service.py
routers/documents.py
tests/test_document_models.py
tests/test_document_service.py
tests/test_document_api.py
```

职责划分：

| 文件                               | 职责                                         |
| -------------------------------- | ------------------------------------------ |
| `models/document.py`             | 定义 `Document` 和 `DocumentChunk` SQLModel 表 |
| `schemas/document.py`            | 定义 API response schema                     |
| `services/document_service.py`   | 处理上传、列表、读取、索引、删除的业务逻辑                      |
| `routers/documents.py`           | 暴露 `/documents` API                        |
| `tests/test_document_models.py`  | 验证 model 和 schema                          |
| `tests/test_document_service.py` | 验证 service 业务规则                            |
| `tests/test_document_api.py`     | 验证 API 路由和 response schema                 |

---

## 6. 核心设计取舍

### 6.1 为什么上传和索引分开

本阶段没有在 `/documents/upload` 中自动索引。

设计为：

```text
POST /documents/upload
只负责上传和登记

POST /documents/{document_id}/index
负责切 chunk、embedding、写 Chroma
```

原因：

```text
1. 上传是 I/O 行为。
2. 索引是计算和向量库写入行为。
3. 两者失败场景不同。
4. 分开后可以清楚记录 document.status。
5. 后续可以自然把索引改成异步任务。
```

这也让系统状态更清楚：

```text
uploaded  表示文件已进入系统
indexed   表示文件已进入 RAG 知识库
failed    表示索引失败，需要排查
deleted   表示文档已从知识库下架
```

---

### 6.2 为什么保留 document_chunks 表

虽然 Chroma 已经存储了 chunk 和 metadata，但仍然保留 `document_chunks` 表。

原因：

```text
1. 后端可以知道一个 document 被切成了多少 chunks。
2. 可以通过 embedding_id 精确删除 Chroma 向量。
3. 可以复盘 chunk 内容和 metadata。
4. 可以支持后续重新索引、chunk-level eval 和审计。
```

Chroma 是向量检索存储，`document_chunks` 是业务数据库中的可审计记录。

---

### 6.3 为什么删除不是只改 status

如果删除时只做：

```text
documents.status = deleted
```

但不删除 Chroma 中的 embedding，那么 `/rag/search` 仍然可能召回旧 chunk。

所以删除必须处理两层：

```text
业务数据库：
documents.status = deleted

向量数据库：
collection.delete(ids=embedding_ids)
```

本阶段删除接口会读取 `document_chunks.embedding_id`，删除 Chroma 中对应向量，再将文档标记为 deleted。

---

### 6.4 为什么测试使用 fake embedding 和 fake Chroma

`tests/test_document_service.py` 中没有真实调用外部 embedding API，也没有依赖真实 Chroma。

测试中使用：

```text
fake_embed_texts
FakeChromaCollection
```

原因：

```text
1. service 测试重点是业务流程。
2. 不应该依赖外部模型 API。
3. 不应该因为 Chroma 本地状态影响单元测试。
4. fake 对象能验证 add/delete 是否被正确调用。
```

真实 Chroma 集成通过手动端到端验证覆盖。

---

## 7. 端到端验证结果

本阶段已完成手动端到端验证：

```text
upload
↓
index
↓
search 能检索到上传文档
↓
delete
↓
search 不再返回该文档
```

验证文档使用唯一关键词：

```text
蓝鲸门禁卡补办流程
```

验证结果：

| 步骤               | 结果                   |
| ---------------- | -------------------- |
| 上传 md 文档         | 成功，状态为 `uploaded`    |
| 查询文档             | 成功，`chunk_count = 0` |
| 触发索引             | 成功，状态变为 `indexed`    |
| RAG search       | 能检索到上传文档             |
| 删除文档             | 成功，状态变为 `deleted`    |
| 删除后查询文档          | 返回 404               |
| 删除后再次 RAG search | 不再返回该上传文档            |

这说明 Document Backend 已经接入现有 RAG 检索链路，而不是孤立的文档管理 API。

---

## 8. 自动化测试覆盖

当前阶段测试覆盖：

```text
1. Document model 默认字段。
2. DocumentChunk 核心字段。
3. DocumentResponse schema 转换。
4. create_document_from_bytes 保存文件并写入 documents 表。
5. 非 md/txt 文件被拒绝。
6. list_documents 按 tenant/category 过滤。
7. get_document 防止跨 tenant 读取。
8. index_document 创建 chunks 并写 Chroma。
9. reindex 时替换旧 chunks。
10. delete_document 删除 Chroma ids 并清空 chunks。
11. delete_document 防止跨 tenant 删除。
12. POST /documents/upload API 成功。
13. GET /documents API 成功。
14. GET /documents/{document_id} API 成功。
15. POST /documents/{document_id}/index API 成功。
16. DELETE /documents/{document_id} API 成功。
```

推荐回归命令：

```powershell
python -m pytest tests/test_document_models.py tests/test_document_service.py tests/test_document_api.py -q
```

完整回归命令：

```powershell
python -m pytest tests/test_todos.py tests/test_rag_api.py tests/test_rag_service.py tests/test_query_chroma.py tests/test_document_models.py tests/test_document_service.py tests/test_document_api.py tests/test_tickets.py tests/test_agent_ops_service.py tests/test_agent_ops_api.py tests/test_ticket_agent_service.py tests/test_agent_ticket_api.py tests/test_agentops_smoke_api.py
```

---

## 9. 当前边界

本阶段仍然有意保留一些边界：

```text
1. 只支持 md/txt，不支持 PDF / Word / Excel。
2. 只使用本地 storage，不接对象存储。
3. 索引是同步执行，不是异步任务。
4. tenant_id 和 user_id 仍然是 mock。
5. 没有真实登录和 RBAC。
6. 没有文档版本回滚。
7. 没有文档批量上传。
8. 没有前端管理页面。
9. 没有物理删除本地文件。
10. 没有对上传内容做敏感信息扫描。
```

这些是 MVP 范围取舍，不是当前阶段缺陷。

---

## 10. 后续扩展方向

后续可以扩展：

```text
1. 增加 PDF 解析。
2. 将索引改成异步任务。
3. 增加文档版本管理。
4. 增加物理删除或归档策略。
5. 增加真实 auth context。
6. 增加用户可见的文档管理 UI。
7. 增加 indexing job logs。
8. 将高频 no-context query 转化为待补充文档任务。
9. 对 document_chunks 增加 chunk-level eval。
10. 增加对象存储支持。
```

---

## 11. 阶段 7 封版结论

阶段 7 Document Backend MVP 已完成：

```text
1. 支持 md/txt 文档上传。
2. 支持 documents 表记录文档元信息。
3. 支持 document_chunks 表记录 chunk 和 embedding_id。
4. 支持手动触发文档索引。
5. 支持写入 Chroma。
6. 支持上传文档被 /rag/search 检索。
7. 支持删除文档。
8. 删除后 Chroma embedding 被清理。
9. 删除后 /rag/search 不再返回该文档。
10. 有 service 和 API 测试覆盖。
11. 有端到端手动验证。
```

下一步建议将 README 和 demo_script 补充 Document Backend 入口，使阶段 7 的 API、验证方式和当前边界可以从项目主页直接访问。
