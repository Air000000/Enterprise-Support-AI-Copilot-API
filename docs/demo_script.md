# Demo Script

Enterprise Support AI Copilot 演示脚本。

本脚本用于展示当前 `main` 分支的核心能力：

```text
Enterprise RAG Core
Document Backend
Ticket CRUD
Ticket Agent preview / confirm
Human approval
AgentOps tool_calls audit
AgentOps metrics summary
```

---

## 1. Demo Goal

本次 demo 展示一条完整企业支持流程：

```text
员工提出内部支持问题
↓
系统基于企业知识库检索相关资料
↓
Document Backend 支持上传、索引、删除并影响 RAG 检索结果
↓
Ticket Agent 判断是否需要创建工单
↓
生成 ticket preview
↓
用户人工确认
↓
系统创建真实 ticket
↓
AgentOps 记录 agent_run / tool_calls / approval_request
↓
通过 metrics summary 查看整体运行状态
```

重点展示：

```text
1. RAG 返回 answer + sources
2. Document Backend 支持 upload / index / delete，并能影响 RAG 检索结果
3. Ticket Agent 不直接创建工单，而是先生成 preview
4. create_ticket 必须经过 approval_request 确认
5. search_kb / classify_ticket / create_ticket 均有 tool_call 审计记录
6. /agent-ops/metrics/summary 能汇总 AgentOps 状态
```

---

## 2. Prerequisites

本 demo 支持两种运行方式：

```text
方式 A：本地开发运行
uvicorn main:app --reload

方式 B：Docker Compose 本地运行
docker compose up --build
```

### 2.1 通用环境变量

项目根目录需要存在 `.env` 文件。可以从 `.env.example` 复制：

```powershell
Copy-Item .env.example .env
```

然后配置模型参数：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
EMBEDDING_MODEL=text-embedding-v4
```

注意：

```text
.env 只用于本地运行，不应提交到 GitHub。
不要把包含真实 API key 的 docker compose config 完整输出粘贴到公开位置。
```

### 2.2 本地开发运行前置条件

如果使用本地开发方式，需要先安装依赖：

```bash
pip install -r requirements.txt
```

如果本地 SQLite 表结构较旧，可以在开发环境中删除本地数据库后重新启动服务。

PowerShell 示例：

```powershell
dir *.db -Recurse
dir *.sqlite -Recurse
```

找到本地开发数据库后删除，例如：

```powershell
Remove-Item .pp.db
```

实际文件名以本地项目为准。

### 2.3 Docker Compose 运行前置条件

如果使用 Docker Compose 方式，需要先启动 Docker Desktop，并确认 Docker Engine 正常运行：

```powershell
docker info
```

当前 Compose 配置只绑定本机地址：

```text
127.0.0.1:8000
```

因此服务只在本机可访问，不会直接暴露给局域网或公网。

Docker Compose 使用独立运行目录：

```text
docker_data/
docker_storage/
docker_chroma_db/
```

这些目录用于保存容器运行产生的 SQLite 数据库、上传文档和 Chroma 向量库，不会污染本地开发数据目录。

---

## 3. Start API Server

### 3.1 方式 A：本地开发启动

启动 FastAPI 服务：

```bash
uvicorn main:app --reload
```

打开 Swagger UI：

```text
http://127.0.0.1:8000/docs
```

先检查健康接口：

```powershell
curl.exe http://127.0.0.1:8000/health
```

预期结果：

```json
{
  "status": "ok"
}
```

### 3.2 方式 B：Docker Compose 启动

先检查 Compose 配置：

```powershell
docker compose config
```

如果配置正常，构建并启动服务：

```powershell
docker compose up --build
```

另开一个 PowerShell 窗口检查健康接口：

```powershell
curl.exe http://127.0.0.1:8000/health
```

预期结果：

```json
{
  "status": "ok"
}
```

Swagger UI：

```text
http://127.0.0.1:8000/docs
```

停止服务：

```powershell
docker compose down
```

如果 `docker compose up --build` 运行在前台，也可以在对应窗口按：

```text
Ctrl + C
```

### 3.3 Docker 常见问题

#### 端口 8000 被占用

如果启动时报错：

```text
ports are not available
```

说明本机已有进程占用 `127.0.0.1:8000`，常见原因是本地 `uvicorn main:app --reload` 还在运行。

查看占用进程：

```powershell
netstat -ano | findstr :8000
```

根据 PID 停止进程：

```powershell
Stop-Process -Id <PID> -Force
```

然后重新启动：

```powershell
docker compose up --build
```

#### Docker Chroma index 为空

Docker 使用独立目录 `docker_chroma_db/`，第一次运行时可能还没有 Chroma index。

可以在 Docker 环境中构建索引：

```powershell
docker compose run --rm api python -m experiments.rag_local.build_chroma_index
```

然后重新启动服务：

```powershell
docker compose up --build
```

---

## 4. Run RAG Eval

运行 Enterprise RAG eval：

```bash
python -m experiments.evals.eval_enterprise_chroma_retrieval
```

预期关注指标：

```text
hit@1
hit@3
mrr@3
avg latency
category breakdown
failed cases
```

当前企业文档集预期结果接近：

```text
Total: 30
hit@1: 0.97
hit@3: 1.00
mrr@3: 0.98
```

讲解点：

```text
RAG 不只提供接口，还配套了 retrieval eval，用于验证检索效果。
当前 eval 覆盖 IT、HR、finance、admin、security 等企业内部支持分类。
```

---

## 5. RAG Search Demo

Swagger 中打开：

```http
POST /rag/search
```

请求体：

```json
{
  "query": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "category": "it"
}
```

预期返回字段：

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

预期讲解：

```text
/rag/search 展示底层检索结果。
它支持 category filter，同时 tenant_id 由 mock tenant context 提供。
返回结果中包含 source_path、document_id、chunk_id 和 distance。
```

截图建议：

```text
截取 /rag/search response 中的 sources / document_id / distance。
```

---

## 6. RAG Ask Demo

Swagger 中打开：

```http
POST /rag/ask
```

请求体：

```json
{
  "question": "VPN 连不上应该先检查什么？",
  "top_k": 3,
  "max_distance": 0.9,
  "category": "it"
}
```

预期返回字段：

```text
answer
retrieval_status
top_distance
sources
```

预期讲解：

```text
/rag/ask 执行完整 RAG 流程：
先检索企业知识库，再基于检索上下文生成回答，并返回 sources。
如果检索不到足够依据，系统会返回拒答。
```

截图建议：

```text
截取 answer + sources。
```

---

## 7. Document Backend Demo

本节用于验证 Document Backend MVP 的完整文档生命周期：

```text
upload
↓
index
↓
rag search can retrieve uploaded document
↓
delete
↓
rag search no longer returns deleted document
```

当前 Document Backend 支持：

```http
POST   /documents/upload
GET    /documents
GET    /documents/{document_id}
POST   /documents/{document_id}/index
DELETE /documents/{document_id}
```

### 7.1 准备唯一测试文档

在项目根目录创建一份临时 Markdown 文档：

```powershell
New-Item -ItemType Directory -Force tmp_manual_docs

@"
# 蓝鲸门禁卡补办流程

如果员工的蓝鲸门禁卡丢失，需要先在行政系统登记遗失信息，然后联系行政支持团队冻结旧卡，并提交补办申请。

补办申请必须包含员工姓名、部门、遗失时间、遗失地点和直属主管确认。
"@ | Set-Content -Encoding UTF8 tmp_manual_docs\blue_whale_access_card.md
```

这里使用“蓝鲸门禁卡”作为唯一关键词，方便确认 RAG 检索结果来自刚上传的文档。

---

### 7.2 上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -F "file=@tmp_manual_docs/blue_whale_access_card.md" `
  -F "category=admin"
```

预期返回重点：

```json
{
  "id": "doc_xxx",
  "filename": "blue_whale_access_card.md",
  "file_type": "md",
  "category": "admin",
  "status": "uploaded",
  "chunk_count": 0
}
```

记录返回中的文档 ID：

```text
DOCUMENT_ID = <document_id>
```

上传成功只代表文档已经进入系统，还没有进入向量库。

---

### 7.3 查看 uploaded 状态

```powershell
curl.exe "http://127.0.0.1:8000/documents/<document_id>"
```

预期重点：

```json
{
  "id": "<document_id>",
  "status": "uploaded",
  "category": "admin",
  "chunk_count": 0
}
```

---

### 7.4 触发索引

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/<document_id>/index"
```

预期重点：

```json
{
  "document_id": "<document_id>",
  "status": "indexed",
  "chunk_count": 1
}
```

索引成功表示系统已经完成：

```text
读取 source_path 文件
↓
切分 chunk
↓
生成 embedding
↓
写入 Chroma
↓
写入 document_chunks
↓
更新 documents.status = indexed
```

---

### 7.5 查看 indexed 状态

```powershell
curl.exe "http://127.0.0.1:8000/documents/<document_id>"
```

预期重点：

```json
{
  "id": "<document_id>",
  "status": "indexed",
  "category": "admin",
  "chunk_count": 1
}
```

---

### 7.6 使用 RAG search 检索上传文档

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"蓝鲸门禁卡丢失后如何补办？\",\"top_k\":5,\"category\":\"admin\"}"
```

预期结果中应能看到刚上传的文档信息，例如：

```json
{
  "document_id": "<document_id>",
  "filename": "blue_whale_access_card.md",
  "category": "admin",
  "source_type": "uploaded_document"
}
```

字段名可能根据 response schema 略有差异，核心检查点是：

```text
1. 返回结果包含 <document_id>
2. 返回结果包含 blue_whale_access_card.md
3. 返回结果 category = admin
4. 返回结果能对应刚上传的文档内容
```

---

### 7.7 删除文档

```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/documents/<document_id>"
```

预期重点：

```json
{
  "document_id": "<document_id>",
  "status": "deleted",
  "deleted_embeddings": 1
}
```

`deleted_embeddings` 大于 0 表示 Chroma 中对应的向量已经被删除。

---

### 7.8 删除后查询文档

```powershell
curl.exe "http://127.0.0.1:8000/documents/<document_id>"
```

预期返回：

```json
{
  "detail": "Document not found"
}
```

这是正常结果。当前 `GET /documents/{document_id}` 默认不返回 `deleted` 文档。

---

### 7.9 删除后再次检索

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"蓝鲸门禁卡丢失后如何补办？\",\"top_k\":5,\"category\":\"admin\"}"
```

预期结果中不应再出现：

```text
<document_id>
blue_whale_access_card.md
source_type = uploaded_document
```

这说明删除不仅修改了业务数据库状态，也清理了 Chroma embeddings，RAG 不会继续召回已删除文档。

---

### 7.10 验收标准

本 demo 通过的标准：

```text
1. POST /documents/upload 返回 status = uploaded。
2. GET /documents/{document_id} 能看到 uploaded 状态。
3. POST /documents/{document_id}/index 返回 status = indexed。
4. GET /documents/{document_id} 能看到 chunk_count > 0。
5. /rag/search 能检索到刚上传的文档。
6. DELETE /documents/{document_id} 返回 status = deleted。
7. deleted_embeddings > 0。
8. 删除后 GET /documents/{document_id} 返回 404。
9. 删除后 /rag/search 不再返回该文档。
```

这个闭环证明 Document Backend 已经接入 RAG 知识库生命周期，而不是孤立的文件上传 API。

---

## 8. Ticket CRUD Demo

### 8.1 Create Ticket

Swagger 中打开：

```http
POST /tickets
```

请求体：

```json
{
  "title": "VPN 无法连接",
  "description": "用户反馈 VPN 客户端无法连接，重启后仍失败。",
  "category": "it",
  "priority": "high"
}
```

预期结果：

```text
返回新建 ticket
status 默认为 open
包含 ticket_id / created_at / updated_at
```

---

### 8.2 List Tickets

Swagger 中打开：

```http
GET /tickets
```

预期结果：

```text
可以看到刚才创建的 ticket
```

---

### 8.3 Get Ticket

Swagger 中打开：

```http
GET /tickets/{ticket_id}
```

填入刚才创建的 ticket id。

预期结果：

```text
返回指定 ticket 详情
```

---

### 8.4 Update Ticket

Swagger 中打开：

```http
PATCH /tickets/{ticket_id}
```

请求体示例：

```json
{
  "status": "in_progress"
}
```

预期结果：

```text
ticket status 更新为 in_progress
```

讲解点：

```text
Ticket CRUD 是后续 Agent 工具 create_ticket 的业务落地点。
Agent 最终不是只返回文字，而是可以接入后端业务对象。
```

---

## 9. Ticket Agent Preview Demo

Swagger 中打开：

```http
POST /agent/ticket/preview
```

请求体：

```json
{
  "message": "VPN 连不上，重启客户端也没用",
  "category": "it"
}
```

预期返回字段：

```text
agent_run_id
approval_request_id
should_create_ticket
reason
draft
sources
```

预期结果：

```text
should_create_ticket = true
approval_request_id 不为空
draft 包含 title / description / category / priority
sources 包含 RAG 检索来源
```

记录以下值：

```text
agent_run_id = <PREVIEW_AGENT_RUN_ID>
approval_request_id = <PREVIEW_APPROVAL_REQUEST_ID>
draft = <PREVIEW_DRAFT>
```

讲解点：

```text
preview 阶段不会创建真实 ticket。
它只创建 agent_run、tool_calls、ticket draft 和 approval_request。
真实 ticket 必须等 confirm 阶段才会创建。
```

截图建议：

```text
截取 preview response 中的 agent_run_id、approval_request_id、draft、sources。
```

---

## 10. Inspect Preview Tool Calls

Swagger 中打开：

```http
GET /agent-ops/runs/{agent_run_id}/tool-calls
```

填入 preview 返回的：

```text
agent_run_id
```

预期看到两条 tool_call：

```text
search_kb
classify_ticket
```

预期状态：

```text
search_kb.status = success
classify_ticket.status = success
```

检查字段：

```text
tool_name
tool_input_json
tool_output_json
status
error_message
latency_ms
```

讲解点：

```text
search_kb 记录知识库检索动作。
classify_ticket 记录工单判断和分类动作。
preview 阶段的中间决策不是只存在内存里，而是进入 tool_calls 审计记录。
```

截图建议：

```text
截取 search_kb 和 classify_ticket 两条 tool_call。
```

---

## 11. Approval Reject Demo

本节用于展示 rejected flow。

注意：不要使用后续要 confirm 的 approval_request 做 reject。为了保留 confirm 演示，建议重新发起一次 preview。

### 11.1 Create another preview

Swagger 中再次打开：

```http
POST /agent/ticket/preview
```

请求体：

```json
{
  "message": "我需要申请外包账号访问测试环境",
  "category": "security"
}
```

记录返回：

```text
reject_demo_agent_run_id
reject_demo_approval_request_id
```

---

### 11.2 Reject approval request

Swagger 中打开：

```http
POST /agent-ops/approval-requests/{approval_request_id}/reject
```

填入：

```text
reject_demo_approval_request_id
```

预期结果：

```text
approval_request.status = rejected
```

---

### 11.3 Inspect rejected approval

Swagger 中打开：

```http
GET /agent-ops/runs/{agent_run_id}/approval-requests
```

填入：

```text
reject_demo_agent_run_id
```

预期结果：

```text
status = rejected
```

讲解点：

```text
Agent 建议执行动作后，用户可以拒绝审批。
rejected approval_request 不应再被 confirm 执行。
```

截图建议：

```text
截取 rejected approval_request。
```

---

## 12. Ticket Agent Confirm Demo

回到第 9 节的 preview 结果，使用未被 reject 的：

```text
PREVIEW_AGENT_RUN_ID
PREVIEW_APPROVAL_REQUEST_ID
PREVIEW_DRAFT
```

Swagger 中打开：

```http
POST /agent/ticket/confirm
```

请求体必须使用 preview 返回的 draft，不要修改 draft 内容。

示例：

```json
{
  "agent_run_id": 1,
  "approval_request_id": 10,
  "draft": {
    "title": "VPN 连不上，重启客户端也没用",
    "description": "这里必须复制 preview 返回的 description",
    "category": "it",
    "priority": "high"
  }
}
```

预期结果：

```text
返回创建后的 ticket
approval_request 更新为 approved
新增 create_ticket tool_call
agent_run 状态为 completed
```

讲解点：

```text
confirm 阶段会做三类校验：
1. approval_request 必须属于当前 agent_run
2. approval_request.status 必须是 pending
3. confirm draft 必须与服务端保存的 approval_request.draft_json 一致

校验通过后，系统使用服务端保存的 draft 创建真实 ticket。
```

截图建议：

```text
截取 confirm response 中的 ticket_id / status / category / priority。
```

---

## 13. Inspect Full Tool Calls

Swagger 中打开：

```http
GET /agent-ops/runs/{agent_run_id}/tool-calls
```

填入第 9 节 preview 返回的：

```text
PREVIEW_AGENT_RUN_ID
```

confirm 成功后，预期看到三条 tool_call：

```text
search_kb
classify_ticket
create_ticket
```

预期状态：

```text
search_kb.status = success
classify_ticket.status = success
create_ticket.status = success
```

讲解点：

```text
一次完整 Ticket Agent 流程包含三个可审计工具调用：
search_kb 解释查了什么知识库；
classify_ticket 解释为什么建议创建工单；
create_ticket 记录用户确认后真正执行的业务动作。
```

截图建议：

```text
截取完整三条 tool_call。
```

---

## 14. Inspect Approval Requests

Swagger 中打开：

```http
GET /agent-ops/runs/{agent_run_id}/approval-requests
```

填入第 9 节 preview 返回的：

```text
PREVIEW_AGENT_RUN_ID
```

confirm 成功后，预期结果：

```text
approval_request.status = approved
```

讲解点：

```text
approval_requests 记录高风险动作的人工确认状态。
create_ticket 不会在 preview 阶段直接执行，而是在 approval_request approved 后执行。
```

截图建议：

```text
截取 approved approval_request。
```

---

## 15. Inspect AgentOps Metrics

Swagger 中打开：

```http
GET /agent-ops/metrics/summary
```

如果只跑了一条完整 confirm flow，预期接近：

```json
{
  "total_agent_runs": 1,
  "running_agent_runs": 0,
  "completed_agent_runs": 1,
  "failed_agent_runs": 0,
  "cancelled_agent_runs": 0,
  "total_tool_calls": 3,
  "pending_tool_calls": 0,
  "successful_tool_calls": 3,
  "failed_tool_calls": 0,
  "total_approval_requests": 1,
  "pending_approval_requests": 0,
  "approved_approval_requests": 1,
  "rejected_approval_requests": 0,
  "cancelled_approval_requests": 0
}
```

如果同时跑了 rejected demo，预期数字会更多，例如：

```text
total_agent_runs >= 2
total_tool_calls >= 5
total_approval_requests >= 2
approved_approval_requests >= 1
rejected_approval_requests >= 1
```

讲解点：

```text
metrics summary 用于展示 AgentOps 的整体运行状态。
它可以快速回答：
当前有多少 agent_runs？
有多少 tool_calls？
多少成功、失败、pending？
有多少 approval_requests 被 approved / rejected / cancelled？
```

截图建议：

```text
截取 /agent-ops/metrics/summary response。
```

---

## 16. Optional: Confirm Draft Tampering Demo

本节用于展示 confirm 阶段的 payload consistency check。

先创建一个新的 preview：

```http
POST /agent/ticket/preview
```

请求体：

```json
{
  "message": "生产系统无法访问，客户现场已经阻塞",
  "category": "it"
}
```

记录返回：

```text
agent_run_id
approval_request_id
draft
```

然后调用 confirm，但故意修改 draft，例如把 priority 改成：

```json
"priority": "low"
```

预期结果：

```text
confirm 被拒绝
不会创建真实 ticket
不会执行 create_ticket
```

预期错误信息：

```text
Confirm draft does not match approval draft
```

讲解点：

```text
confirm 阶段不会信任客户端临时提交的 draft。
系统会和服务端保存的 approval_request.draft_json 做一致性校验。
```

---

## 17. Optional: Repeated Confirm Demo

本节用于展示 pending approval validation。

对同一个已经 confirm 成功的 approval_request，再次调用：

```http
POST /agent/ticket/confirm
```

预期结果：

```text
confirm 被拒绝
不会重复创建 ticket
```

预期错误信息：

```text
Approval request is not pending
```

讲解点：

```text
approval_request 只能从 pending 被确认一次。
已经 approved / rejected / cancelled 的请求不能重复执行。
```

---

## 18. Expected Demo Summary

完成 demo 后，系统应展示以下能力：

```text
1. RAG 可以基于企业内部文档进行检索和问答
2. RAG response 返回结构化 sources
3. Document Backend 可以上传、索引、删除文档，并影响 RAG 检索结果
4. Ticket CRUD 可以创建、查询、更新工单
5. Ticket Agent preview 生成 draft 和 approval_request
6. preview 阶段记录 search_kb 和 classify_ticket tool_call
7. reject / cancel API 可以改变 approval_request 状态
8. confirm 阶段经过审批校验后创建真实 ticket
9. confirm 阶段记录 create_ticket tool_call
10. AgentOps API 可以查看 agent_runs、tool_calls、approval_requests
11. metrics summary 可以汇总 AgentOps 状态
```

建议最终展示顺序：

```text
RAG ask
↓
Document Backend upload / index / delete
↓
Ticket Agent preview
↓
Tool calls: search_kb + classify_ticket
↓
Approval request pending
↓
Confirm
↓
Tool calls: search_kb + classify_ticket + create_ticket
↓
Metrics summary
```

---

## 19. Demo Talking Points

简短讲解版本：

```text
这个项目是一个企业内部支持 AI Copilot 后端系统。系统首先通过 RAG 检索企业内部知识库，并返回带 sources 的回答。当用户问题更适合进入支持流程时，Ticket Agent 会生成工单 preview，而不是直接创建工单。只有用户确认后，后端才会创建真实 ticket。整个过程中，search_kb、classify_ticket、create_ticket 都会记录到 tool_calls，审批状态记录到 approval_requests，整体运行状态可以通过 metrics summary 查看。
```

技术讲解版本：

```text
后端采用 FastAPI + SQLModel + Chroma。RAG 层负责知识库检索和 sources 返回；Ticket Agent 层负责 preview / confirm 两阶段工单流程；AgentOps 层负责 agent_runs、tool_calls、approval_requests 和 metrics summary。create_ticket 属于状态变更动作，因此必须经过 approval_request，并在 confirm 阶段校验 ownership、pending 状态和 draft payload 一致性。
```

---

## 20. Optional: Run Smoke Script

如果不想手动通过 Swagger 点击完整流程，可以运行 smoke script。

### 20.1 本地开发方式运行

先启动服务：

```bash
uvicorn main:app --reload
```

然后运行：

```bash
python scripts/smoke_agentops_flow.py
```

如需指定 API 地址：

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/smoke_agentops_flow.py
```

### 20.2 Docker Compose 方式运行

先启动容器：

```powershell
docker compose up --build
```

另开一个 PowerShell 窗口，确认服务可访问：

```powershell
curl.exe http://127.0.0.1:8000/health
```

然后在宿主机运行 smoke script：

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/smoke_agentops_flow.py
```

### 20.3 Smoke Script 验证内容

该脚本会自动验证：

```text
preview
search_kb tool_call
classify_ticket tool_call
pending approval_request
confirm
create_ticket tool_call
approved approval_request
metrics summary
```

预期输出：

```text
Smoke test passed.
Validated: preview -> search_kb/classify_ticket -> approval -> confirm -> create_ticket -> metrics
```