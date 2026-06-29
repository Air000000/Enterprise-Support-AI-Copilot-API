# Document Backend MVP Design

## 1. 背景

当前项目的 RAG 文档主要来自本地目录：

```text
experiments/docs/
```

索引流程主要依赖脚本：

```text
experiments/rag_local/document_loader.py
experiments/rag_local/text_splitter.py
experiments/rag_local/build_chroma_index.py
experiments/rag_local/query_chroma.py
```

这套方式适合学习和离线实验，但还不是后端知识库模块。对 Enterprise Support AI Copilot 来说，后续需要支持用户通过 API 上传企业内部文档，并将这些文档纳入 RAG 检索。

阶段 7 的目标是把本地脚本式知识库升级为后端化的 Document Backend MVP。

---

## 2. 本阶段目标

阶段 7 MVP 要完成一个小而完整的文档闭环：

```text
上传 md/txt 文档
↓
记录 documents 表
↓
触发 index
↓
切分 chunk
↓
记录 document_chunks 表
↓
写入 Chroma
↓
/rag/search 可以检索到新上传文档
↓
删除文档后不再被检索
```

这一阶段不是做完整 CMS，也不是做复杂权限系统，而是让知识库从“本地脚本数据源”升级为“后端 API 管理的数据源”。

---

## 3. 本轮 MVP 范围

### 3.1 要做

```text
1. 新增 documents 表。
2. 新增 document_chunks 表。
3. 新增 /documents/upload。
4. 新增 /documents。
5. 新增 /documents/{document_id}。
6. 新增 /documents/{document_id}/index。
7. 新增 DELETE /documents/{document_id}。
8. 支持 .md / .txt 文件。
9. 上传文档写入本地 storage 目录。
10. index 时切分 chunk 并写入 Chroma。
11. 删除文档时删除对应 Chroma 向量，保证后续检索不到。
12. 补 document service / API 测试。
```

### 3.2 不做

```text
1. 不做 PDF / OCR。
2. 不做 Word / Excel 解析。
3. 不做真实登录系统。
4. 不做复杂 RBAC。
5. 不做异步任务队列。
6. 不做后台管理 UI。
7. 不做多版本回滚。
8. 不做全文搜索引擎。
9. 不做大文件分片上传。
10. 不做文档在线编辑。
```

---

## 4. 最小数据流

### 4.1 上传流程

```text
用户上传文件
↓
POST /documents/upload
↓
校验文件类型 .md / .txt
↓
读取文件内容
↓
计算 checksum
↓
生成 document_id
↓
保存文件到本地 storage
↓
写入 documents 表，status = "uploaded"
↓
返回 document metadata
```

上传阶段只负责“接收并登记文档”，不自动保证已经进入向量库。

---

### 4.2 索引流程

```text
用户触发索引
↓
POST /documents/{document_id}/index
↓
读取 documents 表
↓
校验 tenant_id
↓
读取 source_path 文件
↓
按现有 text_splitter 逻辑切 chunk
↓
写入 document_chunks 表
↓
写入 Chroma collection
↓
更新 documents.status = "indexed"
↓
返回 chunk_count
```

索引阶段负责把文档变成 RAG 可检索的 chunk。

---

### 4.3 检索流程

```text
用户发起 RAG 检索
↓
POST /rag/search
↓
使用 tenant_id / category filter
↓
Chroma 返回包含 uploaded document metadata 的 chunks
↓
RAG service 组装 sources
↓
写 retrieval log
↓
返回 results
```

为了让上传文档无缝接入现有 RAG，写入 Chroma 时必须保持 metadata 字段兼容当前 `/rag/search` 和 `/rag/ask`。

---

### 4.4 删除流程

```text
用户删除文档
↓
DELETE /documents/{document_id}
↓
校验 tenant_id
↓
查 document_chunks
↓
根据 embedding_id 删除 Chroma 向量
↓
documents.status = "deleted"
↓
document_chunks 标记或保留为历史记录
↓
后续 /rag/search 不再返回该文档
```

MVP 优先保证“删除后不再被检索”。本地源文件可以先保留，用于本地 demo 和审计；后续再增加物理删除或归档策略。

---

## 5. API 设计

### 5.1 上传文档

```http
POST /documents/upload
```

请求类型：

```text
multipart/form-data
```

字段：

```text
file: UploadFile
category: optional string
```

第一版 category 可选。如果用户不传，可以使用：

```text
category = "other"
```

或根据文件路径 / 表单字段显式指定。

成功响应：

```json
{
  "id": "doc_001",
  "tenant_id": "tenant_demo",
  "uploaded_by": "demo_user",
  "filename": "vpn_guide.md",
  "file_type": "md",
  "category": "it",
  "status": "uploaded",
  "version": 1,
  "checksum": "sha256...",
  "created_at": "2026-06-27T10:00:00",
  "updated_at": "2026-06-27T10:00:00"
}
```

失败情况：

```text
1. 文件为空。
2. 文件类型不是 .md / .txt。
3. 文件保存失败。
4. 数据库写入失败。
```

---

### 5.2 查询文档列表

```http
GET /documents
```

可选参数：

```text
category
status
limit
offset
```

成功响应：

```json
{
  "items": [
    {
      "id": "doc_001",
      "filename": "vpn_guide.md",
      "file_type": "md",
      "category": "it",
      "status": "indexed",
      "version": 1,
      "created_at": "2026-06-27T10:00:00",
      "updated_at": "2026-06-27T10:01:00"
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

列表接口必须按 tenant_id 隔离。

---

### 5.3 查询单个文档

```http
GET /documents/{document_id}
```

成功响应：

```json
{
  "id": "doc_001",
  "tenant_id": "tenant_demo",
  "uploaded_by": "demo_user",
  "filename": "vpn_guide.md",
  "file_type": "md",
  "category": "it",
  "source_path": "storage/documents/tenant_demo/doc_001/v1/vpn_guide.md",
  "status": "indexed",
  "version": 1,
  "checksum": "sha256...",
  "chunk_count": 8,
  "created_at": "2026-06-27T10:00:00",
  "updated_at": "2026-06-27T10:01:00"
}
```

如果 document 不属于当前 tenant，返回 404。

---

### 5.4 索引文档

```http
POST /documents/{document_id}/index
```

成功响应：

```json
{
  "document_id": "doc_001",
  "status": "indexed",
  "chunk_count": 8
}
```

索引前状态允许：

```text
uploaded
failed
indexed
```

如果文档已经 indexed，MVP 可以选择：

```text
先删除旧 chunks 和旧 Chroma embeddings，再重新索引。
```

这样行为更确定，方便测试。

---

### 5.5 删除文档

```http
DELETE /documents/{document_id}
```

成功响应：

```json
{
  "document_id": "doc_001",
  "status": "deleted",
  "deleted_embeddings": 8
}
```

删除语义：

```text
1. document.status 改为 deleted。
2. 删除 Chroma 中该 document_id 对应的 embeddings。
3. 后续 /rag/search 不能返回该文档。
4. 文档列表默认不显示 deleted，除非显式 status=deleted。
```

---

## 6. 数据表设计

### 6.1 documents

推荐字段：

```text
id
tenant_id
uploaded_by
filename
file_type
category
source_path
status
version
checksum
error_message
created_at
updated_at
```

字段说明：

| 字段              | 类型          | 说明                                                 |
| --------------- | ----------- | -------------------------------------------------- |
| `id`            | string      | 文档 ID，建议使用 uuid 或 doc_ 前缀 ID                       |
| `tenant_id`     | string      | 租户隔离字段                                             |
| `uploaded_by`   | string      | 上传用户，MVP 可用 demo_user                              |
| `filename`      | string      | 原始文件名                                              |
| `file_type`     | string      | `md` 或 `txt`                                       |
| `category`      | string      | `it`、`hr`、`finance`、`admin`、`security`、`other`     |
| `source_path`   | string      | 本地保存路径                                             |
| `status`        | string      | `uploaded`、`indexing`、`indexed`、`failed`、`deleted` |
| `version`       | int         | 文档版本，MVP 默认 1                                      |
| `checksum`      | string      | 文件 sha256，用于识别重复内容                                 |
| `error_message` | string/null | 索引失败原因                                             |
| `created_at`    | datetime    | 创建时间                                               |
| `updated_at`    | datetime    | 更新时间                                               |

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

### 6.2 document_chunks

推荐字段：

```text
id
tenant_id
document_id
chunk_index
content
category
metadata_json
embedding_id
created_at
```

字段说明：

| 字段              | 类型        | 说明                      |
| --------------- | --------- | ----------------------- |
| `id`            | string    | chunk 数据库 ID            |
| `tenant_id`     | string    | 租户隔离字段                  |
| `document_id`   | string    | 所属 document             |
| `chunk_index`   | int       | 文档内 chunk 序号            |
| `content`       | text      | chunk 文本                |
| `category`      | string    | 继承 document.category    |
| `metadata_json` | text/json | 写入 Chroma 的 metadata 快照 |
| `embedding_id`  | string    | Chroma 中对应的 id          |
| `created_at`    | datetime  | 创建时间                    |

`embedding_id` 建议使用可预测格式：

```text
doc:{document_id}:v{version}:chunk:{chunk_index}
```

这样删除时可以直接根据 document_id 找到所有 Chroma ids。

---

## 7. 文件存储策略

MVP 使用本地目录，不接对象存储。

推荐路径：

```text
storage/documents/{tenant_id}/{document_id}/v{version}/{safe_filename}
```

示例：

```text
storage/documents/tenant_demo/doc_001/v1/vpn_guide.md
```

设计原因：

```text
1. tenant_id 隔离清楚。
2. document_id 避免同名文件冲突。
3. version 为后续版本管理留空间。
4. safe_filename 避免路径注入。
```

安全规则：

```text
1. 不直接相信用户上传的 filename。
2. filename 只用于展示和生成 safe_filename。
3. 实际路径由后端拼接。
4. 不允许使用 ../ 之类的路径片段。
5. 限制文件类型为 .md / .txt。
```

MVP 可以先设置简单大小限制，例如：

```text
max_file_size = 1 MB
```

后续再扩展大文件上传。

---

## 8. Chroma metadata 设计

写入 Chroma 时，每个 chunk 至少包含以下 metadata：

```json
{
  "tenant_id": "tenant_demo",
  "document_id": "doc_001",
  "document_db_id": "doc_001",
  "title": "vpn_guide",
  "filename": "vpn_guide.md",
  "source_path": "storage/documents/tenant_demo/doc_001/v1/vpn_guide.md",
  "category": "it",
  "chunk_index": 0,
  "version": 1,
  "checksum": "sha256...",
  "source_type": "uploaded_document"
}
```

其中：

```text
document_id       用于兼容现有 RAG sources。
document_db_id    明确指向 documents 表。
source_type       区分本地实验文档和上传文档。
tenant_id         用于检索隔离。
category          用于业务分类过滤。
```

---

## 9. 与现有 RAG 的关系

### 9.1 保持 Chroma collection 兼容

Document Backend 不另起一套检索系统，而是写入现有 Chroma collection。

原因：

```text
1. /rag/search 和 /rag/ask 已经基于 Chroma。
2. tenant/category filter 已经存在。
3. retrieval logs / metrics 已经围绕 RAG API 建好。
4. 上传文档只要写入相同 collection，就能进入现有观测体系。
```

---

### 9.2 不在第一轮重构 RAG service

第一轮不重构：

```text
services/rag_service.py
experiments/rag_local/query_chroma.py
```

除非为了让上传文档被正确检索，需要做非常小的兼容调整。

阶段 7 第一轮重点是：

```text
documents registry + upload + list
```

第二轮再做：

```text
index + Chroma write
```

第三轮再做：

```text
delete + retrieval integration test
```

这样可以避免一次改太多。

---

## 10. 模块划分建议

阶段 7 最终可能新增以下文件：

```text
models/document.py
schemas/document.py
services/document_service.py
routers/documents.py
tests/test_documents.py
```

如果需要隔离 Chroma 写入逻辑，可以后续新增：

```text
services/document_index_service.py
```

但 MVP 不建议一开始拆太细。

推荐职责：

| 文件                             | 职责                                  |
| ------------------------------ | ----------------------------------- |
| `models/document.py`           | SQLModel 表结构：Document、DocumentChunk |
| `schemas/document.py`          | 请求和响应 schema                        |
| `services/document_service.py` | 上传、列表、读取、索引、删除的业务逻辑                 |
| `routers/documents.py`         | FastAPI endpoint                    |
| `tests/test_documents.py`      | document API / service 测试           |

---

## 11. 分步实现计划

### Step 1：Document registry MVP

只做：

```text
models/document.py
schemas/document.py
services/document_service.py
routers/documents.py
```

实现：

```text
POST /documents/upload
GET /documents
GET /documents/{document_id}
```

不做：

```text
不写 Chroma
不做 index
不做 delete
```

验收：

```text
1. 可以上传 md/txt。
2. 可以保存文件到 storage。
3. documents 表有记录。
4. 可以按 tenant 查询列表。
5. 其他 tenant 不能读取。
```

推荐 commit：

```text
Add document upload registry
```

---

### Step 2：Document indexing MVP

实现：

```text
POST /documents/{document_id}/index
```

做：

```text
1. 读取上传文件。
2. 切 chunk。
3. 写 document_chunks 表。
4. 写 Chroma。
5. documents.status 更新为 indexed。
```

验收：

```text
1. index 后 chunk_count > 0。
2. document_chunks 有记录。
3. Chroma 中能查到对应 embedding_id。
4. /rag/search 能检索到上传文档。
```

推荐 commit：

```text
Add document indexing into Chroma
```

---

### Step 3：Document deletion MVP

实现：

```text
DELETE /documents/{document_id}
```

做：

```text
1. 校验 tenant。
2. 查 document_chunks。
3. 删除 Chroma embeddings。
4. document.status = deleted。
5. 后续 /rag/search 不返回该文档。
```

验收：

```text
1. 删除前能检索到。
2. 删除后检索不到。
3. 删除其他 tenant 文档返回 404。
```

推荐 commit：

```text
Add document deletion from knowledge base
```

---

## 12. 测试设计

### 12.1 必须覆盖的测试

| 测试                                 | 目的               |
| ---------------------------------- | ---------------- |
| upload md success                  | 验证 Markdown 上传成功 |
| upload txt success                 | 验证 txt 上传成功      |
| reject unsupported file type       | 防止上传 PDF / exe 等 |
| list documents by tenant           | 验证 tenant 隔离     |
| get document by id                 | 验证单文档读取          |
| cannot get other tenant document   | 验证越权访问返回 404     |
| index document success             | 验证索引成功           |
| index missing document returns 404 | 验证不存在文档处理        |
| delete document success            | 验证删除成功           |
| deleted document not retrievable   | 验证删除后不再进入 RAG    |

### 12.2 新增测试文件时的 CI 注意事项

如果新增：

```text
tests/test_documents.py
```

必须同步修改：

```text
.github/workflows/tests.yml
```

原因：

```text
当前 CI 是显式列出测试文件，不是自动运行 tests/ 下全部测试。
```

---

## 13. 手动验证脚本

### 13.1 启动服务

```powershell
uvicorn main:app --reload
```

---

### 13.2 上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -F "file=@experiments/docs/it/vpn_guide.md" `
  -F "category=it"
```

预期：

```text
status = uploaded
file_type = md
category = it
```

---

### 13.3 查看文档列表

```powershell
curl.exe "http://127.0.0.1:8000/documents"
```

预期：

```text
可以看到刚上传的文档。
```

---

### 13.4 触发索引

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/{document_id}/index"
```

预期：

```text
status = indexed
chunk_count > 0
```

---

### 13.5 检索上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"VPN 连接失败怎么办？\",\"top_k\":3,\"category\":\"it\"}"
```

预期：

```text
results 中包含刚上传文档的 source。
```

---

### 13.6 删除文档

```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/documents/{document_id}"
```

预期：

```text
status = deleted
deleted_embeddings > 0
```

---

### 13.7 删除后再次检索

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"VPN 连接失败怎么办？\",\"top_k\":3,\"category\":\"it\"}"
```

预期：

```text
不再返回已删除文档。
```

---

## 14. 关键设计取舍

### 14.1 为什么上传和索引分开

不在 `/documents/upload` 里自动索引，原因是：

```text
1. 上传是文件接收动作。
2. 索引是计算和向量库写入动作。
3. 分开后更容易测试失败场景。
4. 分开后可以清楚表达 status 状态流转。
5. 后续可以把 index 改成异步任务。
```

MVP 中手动触发 index 更直观，也更利于面试解释。

---

### 14.2 为什么先只支持 md/txt

原因：

```text
1. 当前项目已有 md/txt 文档加载经验。
2. md/txt 不需要 OCR 或复杂解析。
3. 可以优先验证文档后端闭环。
4. PDF 解析会引入额外依赖和大量边界问题。
```

PDF 可以作为后续扩展，不应该阻塞 MVP。

---

### 14.3 为什么保留 document_chunks 表

虽然 Chroma 已经存储 chunk 文本和 metadata，但仍然保留 document_chunks 表。

原因：

```text
1. 后端可以查看一个 document 被切成了多少 chunks。
2. 删除文档时可以根据 embedding_id 精确删除 Chroma 向量。
3. 可以复盘 chunk 内容和 metadata。
4. 后续可以做 chunk-level 管理、重建索引和 eval 对齐。
```

Chroma 是向量检索存储，document_chunks 是业务数据库中的可审计记录。

---

### 14.4 为什么删除采用 soft delete + Chroma delete

删除文档时，MVP 采用：

```text
documents.status = deleted
删除 Chroma embeddings
```

原因：

```text
1. Chroma 删除保证后续 RAG 不再检索到。
2. soft delete 保留审计记录。
3. 本地 demo 中更容易排查问题。
4. 后续可以增加物理删除策略。
```

关键验收标准不是文件是否立刻物理删除，而是：

```text
删除后不再被 RAG 返回。
```

---

### 14.5 为什么保持 tenant_id 为核心隔离字段

Document Backend 必须从第一版就带 tenant_id。

原因：

```text
1. RAG 文档属于企业内部数据。
2. 不同 tenant 的文档不能互相检索。
3. 当前 RAG metadata filter 已经支持 tenant/category。
4. 后续接真实 auth 时可以平滑替换 demo tenant。
```

MVP 可以继续使用当前项目已有的 mock tenant 方式。

---

## 15. 失败场景

### 15.1 上传失败

可能原因：

```text
1. 文件为空。
2. 文件类型不支持。
3. 文件名包含非法路径字符。
4. 本地 storage 目录不可写。
5. 数据库写入失败。
```

处理方式：

```text
返回 400 或 500，不写入不完整 document。
```

---

### 15.2 索引失败

可能原因：

```text
1. source_path 文件不存在。
2. 文件编码无法解析。
3. chunk 结果为空。
4. Chroma 写入失败。
5. embedding 服务异常。
```

处理方式：

```text
documents.status = failed
documents.error_message = 失败原因
不留下半完成状态
```

如果 Chroma 已写入部分 chunks，应该尽量回滚删除已经写入的 embedding_id。

---

### 15.3 删除失败

可能原因：

```text
1. document 不存在。
2. tenant_id 不匹配。
3. Chroma 删除失败。
4. document_chunks 缺少 embedding_id。
```

处理方式：

```text
1. 不存在或越权返回 404。
2. Chroma 删除失败时返回 failed，不直接标记 deleted。
3. 保留 error_message 方便排查。
```

---

## 16. 面试表达

可以这样解释阶段 7：

```text
我在 RAG 和 AgentOps 之后补了 Document Backend，把原来 experiments/docs 下的本地脚本文档处理升级成后端知识库模块。这个模块支持 md/txt 上传、documents 表登记、document_chunks 表记录、手动触发索引、写入 Chroma，并保持 tenant_id 和 category metadata 与现有 /rag/search、/rag/ask 兼容。

我特意把 upload 和 index 拆开，因为上传是 I/O 行为，index 是向量化和检索入库行为，失败场景不同。这样可以清楚记录 document.status，例如 uploaded、indexing、indexed、failed、deleted，也方便后续把 index 改成异步任务。

删除时我不是只删数据库记录，而是根据 document_chunks 中的 embedding_id 删除 Chroma 向量，保证后续 RAG 不再检索到这份文档。这体现的是知识库生命周期管理，而不是简单文件上传 demo。
```

---

## 17. 阶段 7 MVP 完成标准

阶段 7 完成时，应该满足：

```text
1. 可以通过 API 上传 md/txt 文档。
2. documents 表能记录文档元信息。
3. document_chunks 表能记录切分结果。
4. 可以手动触发 index。
5. index 后 /rag/search 能检索上传文档。
6. 支持 tenant/category metadata。
7. 删除文档后 /rag/search 不再返回该文档。
8. 至少 5 个 document 相关测试通过。
9. GitHub Actions 包含新增 document 测试文件。
10. README 或 docs 有阶段说明。
```

---

## 18. 第一轮代码落点

第一轮代码只做 Document registry，不碰 Chroma。

预计新增或修改：

```text
models/document.py
schemas/document.py
services/document_service.py
routers/documents.py
main.py
tests/test_documents.py
.github/workflows/tests.yml
```

第一轮只实现：

```text
POST /documents/upload
GET /documents
GET /documents/{document_id}
```

第一轮不实现：

```text
POST /documents/{document_id}/index
DELETE /documents/{document_id}
Chroma 写入
/rag/search 联动
```

第一轮验收：

```text
1. 上传 md 成功。
2. 上传 txt 成功。
3. 非 md/txt 被拒绝。
4. 列表按 tenant 隔离。
5. 查询其他 tenant 文档返回 404。
6. 新增测试进入 GitHub Actions。
```

推荐第一轮 commit：

```text
Add document upload registry
```
