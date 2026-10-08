# Retrieval Logs / Metrics Report

## 1. 背景

本项目的 RAG API 已经支持 `/rag/search` 和 `/rag/ask`。在早期版本中，接口可以返回检索结果和回答，但一次请求内部到底检索到了什么、耗时多久、是否因为上下文不足而拒答、哪些文档经常被召回，并没有形成统一的审计记录。

阶段 6 的目标是补齐 Retrieval Logs / Metrics，让 RAG 不只是“能回答”，还可以被追踪、被排查和被评估。

当前阶段没有引入复杂 tracing 系统，也没有接入外部监控平台，而是优先用数据库表和 AgentOps API 实现最小可用的 RAGOps 证据层。

---

## 2. 本阶段目标

阶段 6 主要解决以下问题：

1. 一次 RAG 请求检索了哪些 sources。
2. 本次请求 top distance 是多少。
3. 本次请求耗时多少。
4. 本次请求是正常命中、无上下文，还是失败。
5. 哪些问题经常没有上下文。
6. 哪些错误经常导致检索失败。
7. 哪些文档最常被检索。
8. 后续如何根据日志和指标优化 RAG。

---

## 3. 当前数据流

### 3.1 `/rag/search`

```text
用户请求
↓
POST /rag/search
↓
routers/rag.py
↓
services/rag_service.search_documents()
↓
Chroma 检索
↓
构造搜索结果
↓
写入 retrieval_logs
↓
返回 query、top_k、total_hits、results
```

`/rag/search` 会在检索成功后记录：

```text
endpoint = "search"
query_text = request.query
top_k = request.top_k
category = request.category
retrieval_status = "ok" 或 "no_context"
total_hits = len(results)
top_distance = 第一条结果的 distance
source_documents_json = 本次返回的 sources
scores_json = 本次返回的 distance 列表
latency_ms = 请求耗时
```

如果检索过程抛出异常，会记录：

```text
retrieval_status = "failed"
total_hits = 0
error_message = 异常信息
latency_ms = 失败前耗时
```

---

### 3.2 `/rag/ask`

```text
用户请求
↓
POST /rag/ask
↓
routers/rag.py
↓
services/rag_service.answer_question()
↓
Chroma 检索
↓
根据 max_distance 判断是否上下文足够
↓
有上下文：生成回答和 sources
↓
无上下文：返回拒答状态
↓
写入 retrieval_logs
↓
返回 answer、retrieval_status、top_distance、sources
```

`/rag/ask` 会记录：

```text
endpoint = "ask"
query_text = request.question
top_k = request.top_k
category = request.category
retrieval_status = rag_result.retrieval_status
total_hits = len(sources)
top_distance = rag_result.top_distance
source_documents_json = sources
scores_json = sources 对应 distance 列表
latency_ms = 请求耗时
```

其中 `retrieval_status` 用来区分：

```text
ok          有可用上下文
no_context  检索不到足够可靠的上下文
failed      检索或问答链路执行失败
```

---

## 4. Retrieval Log 字段说明

当前 retrieval log 记录的核心字段如下：

| 字段                      | 含义               | 用途                   |
| ----------------------- | ---------------- | -------------------- |
| `id`                    | 日志 ID            | 排序、定位单次请求            |
| `tenant_id`             | 租户 ID            | 保证不同 tenant 的日志隔离    |
| `user_id`               | 用户 ID            | 当前可为空，后续接真实用户体系      |
| `endpoint`              | 请求入口             | 区分 `search` 和 `ask`  |
| `query_text`            | 用户查询文本           | 分析用户真实问题             |
| `top_k`                 | 检索返回数量           | 分析参数对召回效果的影响         |
| `category`              | 文档分类过滤           | 分析 IT、HR、Finance 等场景 |
| `retrieval_status`      | 检索状态             | 区分成功、无上下文和失败         |
| `total_hits`            | 返回结果数量           | 判断是否召回到内容            |
| `top_distance`          | 第一条结果距离          | 判断 top1 相关性          |
| `source_documents_json` | 本次返回 sources     | 复盘检索结果               |
| `scores_json`           | 本次结果 distance 列表 | 分析分数分布               |
| `latency_ms`            | 请求耗时             | 排查慢请求                |
| `error_message`         | 错误信息             | 分析失败原因               |
| `created_at`            | 创建时间             | 按时间排查请求              |

---

## 5. 当前 AgentOps Retrieval API

当前 retrieval logs / metrics 统一放在 AgentOps 路由下。

### 5.1 查询明细日志

```http
GET /agent-ops/retrieval-logs
```

支持过滤：

```text
endpoint
retrieval_status
category
limit
offset
```

示例：

```http
GET /agent-ops/retrieval-logs?endpoint=ask&retrieval_status=no_context&limit=10
```

用途：

```text
查看最近哪些 RAG 请求发生了 no_context 或 failed。
```

---

### 5.2 查询检索汇总指标

```http
GET /agent-ops/metrics/retrieval
```

支持过滤：

```text
endpoint
category
```

返回核心指标：

```text
total_retrieval_logs
ok_retrieval_logs
no_context_retrieval_logs
failed_retrieval_logs
average_latency_ms
average_top_distance
endpoint_counts
category_counts
```

用途：

```text
快速判断当前 RAG 的整体检索健康状态。
```

---

### 5.3 查询高频来源文档

```http
GET /agent-ops/metrics/retrieval/sources
```

支持过滤：

```text
endpoint
category
limit
```

返回核心字段：

```text
document_id
title
source_path
retrieval_count
average_distance
```

用途：

```text
分析哪些文档最常被检索，以及这些文档平均距离是否稳定。
```

这可以帮助判断：

```text
1. 是否有文档被过度召回。
2. 是否有高频文档质量较差。
3. 是否需要拆分、重写或补充某类知识文档。
```

---

### 5.4 查询高频无上下文问题

```http
GET /agent-ops/metrics/retrieval/no-context-queries
```

支持过滤：

```text
endpoint
category
limit
```

返回核心字段：

```text
query_text
endpoint
category
no_context_count
latest_latency_ms
```

用途：

```text
发现哪些用户问题经常检索不到可靠上下文。
```

这类问题通常说明：

```text
1. 知识库缺文档。
2. 文档有内容但 chunk 不好。
3. query 表达和文档表达差异过大。
4. max_distance 阈值过严。
5. category filter 过窄。
```

---

### 5.5 查询失败类型

```http
GET /agent-ops/metrics/retrieval/failures
```

支持过滤：

```text
endpoint
category
limit
```

返回核心字段：

```text
error_message
endpoint
category
failed_count
latest_latency_ms
```

用途：

```text
聚合检索失败原因，帮助定位系统级问题。
```

常见失败类型可能包括：

```text
1. Chroma collection 不存在。
2. embedding 服务异常。
3. index 文件缺失。
4. metadata 格式异常。
5. service 层参数传递错误。
```

---

## 6. 手动验证方法

### 6.1 启动服务

```powershell
uvicorn main:app --reload
```

---

### 6.2 触发一次 `/rag/search`

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"VPN 连接失败怎么办？\",\"top_k\":3,\"category\":\"it\"}"
```

预期：

```text
返回 total_hits 和 results。
同时写入一条 endpoint=search 的 retrieval log。
```

---

### 6.3 触发一次 `/rag/ask`

```powershell
curl.exe -X POST "http://127.0.0.1:8000/rag/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"VPN 连接失败怎么办？\",\"top_k\":3,\"category\":\"it\"}"
```

预期：

```text
返回 answer、retrieval_status、top_distance、sources。
同时写入一条 endpoint=ask 的 retrieval log。
```

---

### 6.4 查看 retrieval logs

```powershell
curl.exe "http://127.0.0.1:8000/agent-ops/retrieval-logs?limit=10"
```

重点检查：

```text
endpoint
query_text
retrieval_status
total_hits
top_distance
source_documents_json
scores_json
latency_ms
```

---

### 6.5 查看 retrieval summary

```powershell
curl.exe "http://127.0.0.1:8000/agent-ops/metrics/retrieval"
```

重点检查：

```text
total_retrieval_logs
ok_retrieval_logs
no_context_retrieval_logs
failed_retrieval_logs
average_latency_ms
average_top_distance
endpoint_counts
category_counts
```

---

### 6.6 查看高频来源文档

```powershell
curl.exe "http://127.0.0.1:8000/agent-ops/metrics/retrieval/sources?limit=10"
```

重点检查：

```text
document_id
title
retrieval_count
average_distance
```

---

### 6.7 查看无上下文问题

```powershell
curl.exe "http://127.0.0.1:8000/agent-ops/metrics/retrieval/no-context-queries?limit=10"
```

重点检查：

```text
query_text
no_context_count
latest_latency_ms
```

---

### 6.8 查看失败原因

```powershell
curl.exe "http://127.0.0.1:8000/agent-ops/metrics/retrieval/failures?limit=10"
```

重点检查：

```text
error_message
failed_count
latest_latency_ms
```

---

## 7. 自动化测试

本阶段已经覆盖以下测试方向：

```text
1. /rag/search 写 retrieval log。
2. /rag/ask 写 retrieval log。
3. RAG 请求失败时写 failed log。
4. /agent-ops/retrieval-logs 可以查询日志。
5. /agent-ops/metrics/retrieval 可以统计 summary。
6. /agent-ops/metrics/retrieval/sources 可以聚合来源文档。
7. /agent-ops/metrics/retrieval/no-context-queries 可以聚合无上下文问题。
8. /agent-ops/metrics/retrieval/failures 可以聚合失败原因。
9. tenant 隔离。
10. endpoint / category / status 过滤。
```

推荐本地验证命令：

```powershell
python -m pytest tests/test_rag_api.py tests/test_rag_service.py tests/test_agent_ops_service.py tests/test_agent_ops_api.py tests/test_query_chroma.py
```

完整回归命令：

```powershell
python -m pytest tests/test_todos.py tests/test_rag_api.py tests/test_rag_service.py tests/test_query_chroma.py tests/test_tickets.py tests/test_agent_ops_service.py tests/test_agent_ops_api.py tests/test_ticket_agent_service.py tests/test_agent_ticket_api.py tests/test_agentops_smoke_api.py
```

---

## 8. 设计取舍

### 8.1 为什么 retrieval logging 是 best-effort

当前 `/rag/search` 和 `/rag/ask` 中写 retrieval log 使用 best-effort 策略。

也就是说：

```text
日志写入失败，不应该影响用户正常使用 RAG API。
```

原因：

```text
1. RAG 回答是用户主链路。
2. retrieval log 是可观测性辅助链路。
3. 辅助链路失败不能让主链路不可用。
```

后续如果进入生产级系统，可以把日志写入失败记录到应用日志或监控系统中。

---

### 8.2 为什么先用数据库表而不是外部 tracing 系统

当前实现用于本地 MVP 验证，不是生产级监控平台。

第一版优先选择数据库表：

```text
retrieval_logs
```

原因：

```text
1. 容易测试。
2. 容易通过 API 查询。
3. 支持直接复核检索轨迹。
4. 不引入额外部署复杂度。
5. 能和 AgentOps 的 agent_runs、tool_calls、approval_requests 形成统一审计视角。
```

---

### 8.3 为什么记录 source_documents_json 和 scores_json

RAG 失败排查时，只看最终 answer 不够。

需要同时看到：

```text
1. 召回了哪些文档。
2. 每个文档的 distance 是多少。
3. sources 是否属于正确 category。
4. top1 是否明显优于后续结果。
5. 是否存在“检索到了但回答仍拒答”的情况。
```

所以当前记录：

```text
source_documents_json
scores_json
top_distance
```

这让一次 RAG 请求可以被复盘。

---

### 8.4 为什么区分 ok / no_context / failed

三种状态代表三类不同问题：

| 状态           | 含义        | 排查方向                    |
| ------------ | --------- | ----------------------- |
| `ok`         | 检索到可用上下文  | 关注回答质量、引用质量、latency     |
| `no_context` | 没有足够可靠上下文 | 关注知识库覆盖、chunk、阈值、filter |
| `failed`     | 系统执行失败    | 关注异常、依赖服务、代码路径          |

这样设计可以避免把所有异常都混成“RAG 不好用”。

---

## 9. 当前边界

当前阶段仍有一些有意保留的边界：

```text
1. 没有真实登录系统，tenant_id 当前使用 mock tenant。
2. 没有记录真实 token usage。
3. 没有接 Prometheus / Grafana / OpenTelemetry。
4. 没有异步日志队列。
5. 没有模型调用日志表。
6. 没有复杂 dashboard。
7. 没有对 source_documents_json 建结构化子表。
```

这些不是当前阶段缺陷，而是 MVP 取舍。

当前目标是：

```text
先让 RAG 请求可追踪、可解释、可测试。
```

---

## 10. 后续扩展方向

后续可以扩展：

```text
1. 增加 model_call_logs，记录 provider、model、prompt_tokens、completion_tokens、total_tokens、latency_ms。
2. 增加 request_id，把一次 RAG 请求、LLM 调用、Agent run 串起来。
3. 增加用户维度统计，分析不同用户的问题类型。
4. 增加 p95 / p99 latency。
5. 增加按天聚合的 metrics。
6. 增加 dashboard。
7. 将 retrieval_logs 和 eval cases 打通，用线上 no_context query 反哺 eval 数据集。
8. 将高频 no_context query 转化为待补充知识文档清单。
```

---

## 12. 阶段 6 封版结论

阶段 6 已经完成以下能力：

```text
1. /rag/search 写 retrieval log。
2. /rag/ask 写 retrieval log。
3. RAG 失败写 failed log。
4. 支持查询 retrieval logs 明细。
5. 支持 retrieval summary。
6. 支持 source metrics。
7. 支持 no-context query metrics。
8. 支持 failure metrics。
9. 支持 endpoint / category / status 过滤。
10. 支持 tenant 级数据隔离。
11. 有自动化测试覆盖。
12. 有文档说明阶段价值、验证方式和设计取舍。
```

下一阶段不要直接默认进入阶段 7。进入阶段 7 前，应先确认：

```text
1. 当前 main 干净。
2. GitHub Actions 通过。
3. README 或 demo_script 是否需要补入口链接。
4. 是否要先做阶段 6 的 commit/tag。
```
