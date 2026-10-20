# Enterprise Support AI Copilot（企业技术支持 AI Copilot）

[![Tests](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml)

面向**企业技术支持**场景，构建知识检索、回答 / 拒答与工单升级一体化的 **RAG + 受控工单 Agent 后端**。

系统首先从技术支持知识库检索相关证据，返回带来源的回答或在上下文不足时拒答；对于需要进一步处理的问题，通过工单分类、预览和人工确认后再执行真实建单，并以 AgentOps 记录关键运行与审批链路。这里的目标不是让模型直接接管业务状态，而是把知识问答与后续问题升级连接成一条可控、可追踪的技术支持流程。

在这条应用链路之上，项目长期以 **TechQA** 作为核心技术支持语料与统一评测基准，持续建立检索、生成、拒答与失败归因闭环，用受控评测回答“系统是否真的变好、失败在哪里、某次工程改动是否值得保留”。

> **当前定位：** TechQA 是主技术支持语料与长期主评测基准，不再把它视为迁移到另一套主数据集之前的临时阶段。未来若增加多来源、冲突处理或 Agent 压力测试，只作为补充评测，不替换现有 TechQA 主线。

---

## 30 秒看项目

```text
技术支持请求
          │
          ▼
   当前在线运行时：
 Dense Chroma 检索
          │
          ├──────────────► 回答 + 来源
          │
          └──────────────► 上下文不足时拒答
          │
          ▼
   工单 Agent 预览
     ├─ search_kb
     ├─ classify_ticket
     └─ approval_request.pending
          │
          ▼
      人工确认
     ├─ 运行归属校验
     ├─ 待审批状态校验
     └─ 服务端草稿一致性校验
          │
          ▼
      create_ticket
          │
          └──────────────► AgentOps 运行轨迹
                           ├─ Agent 运行记录
                           ├─ 工具调用
                           ├─ 审批
                           └─ 检索日志 / 指标

评测与迭代
────────────────────────────────────────────────────────────
                         TechQA
        28,481 篇 Technote / 610 条检索问题
         910 条生成 / 拒答问答记录
                           │
                           ▼
                   离线评测
                  Dense / 重排 / Hybrid
                  证据级审计
                  生成 / 拒答
                           │
                           ▼
                    失败归因
                           │
                           ▼
                     系统迭代
                           │
                           └────► RAG / 上下文策略 / 评测闭环
```

### 在线运行时与冻结候选边界

- **当前在线运行时：** Dense Chroma 检索.
- **冻结工程候选：** Dense100 + BM25100 -> RRF60 -> 融合 Top100 -> `qwen3-rerank` -> 直接 Top14.
- **集成状态：** 尚未切换到在线运行时。
- **拒答 / 生成：** 尚未冻结。

### 当前核心能力

| 能力 | 当前实现 |
| --- | --- |
| RAG 在线运行时 | Chroma Dense 检索、租户 / 类别过滤、来源返回、低相关拒答 |
| 受控工单 Agent | `search_kb` / `classify_ticket` / `create_ticket`，预览-确认 + 人工审批 |
| 核心技术支持数据 | TechQA 28,481 Technotes、610 条可回答检索问题、910 条生成 / 拒答问答记录 |
| RAG 评测 | 冻结 TRAIN / DEV、文档级 Recall@K / MRR、生成 / 拒答评测框架 |
| 重排 / Hybrid | 保留冻结离线实验；可选 Hybrid 搜索、实验问答默认关闭；默认检索/问答及工单仍为 Dense |
| 失败归因 | 候选覆盖、chunk 拥挤、证据级审计、路线准入门槛 |
| AgentOps | Agent 运行记录 / 工具调用 / 审批 / 检索轨迹与聚合指标 |
| 工程化 | Alembic、Pytest、Ruff、GitHub Actions、Docker Compose、冒烟测试 |

---

# 1. TechQA：核心技术支持语料与统一评测基准

TechQA 不是单独外挂的评测数据集，而是当前项目的数据与评测主线。

## 检索语料

- **28,481** 篇 Technote 技术支持文档；
- **610** 条可回答检索问题；
- 每条问题恰好 1 个相关文档；
- qrels 为文档级标注；
- 实际检索器返回 chunk，因此正式检索评测会先保留原始 chunk 排序，再按 `document_id` 首次出现位置去重成文档排序。

## 生成与拒答数据集

- **610** 条可回答；
- **300** 条不可回答；
- 共 **910** 条 问答记录。

这使同一 技术支持领域 可以连续支撑：

```text
检索
  ↓
重排 / Hybrid 对比
  ↓
证据质量诊断
  ↓
生成正确性 / 忠实性
  ↓
拒答 / 幻觉评测
```

完整数据版本、SHA256、split 和评测契约见：

- [experiments/evals/README.md](experiments/evals/README.md)
- [experiments/evals/datasets/techqa/manifest.json](experiments/evals/datasets/techqa/manifest.json)

---

# 2. 受控工单 Agent

Ticket Agent 的核心目标不是让模型直接修改业务状态，而是将预览 / 审批阶段与真实写操作分离。

```text
用户请求
   │
   ▼
search_kb
   │
   ▼
classify_ticket
   │
   ├─ 无需建单 ──► 返回判定
   │
   └─ 需要建单
          │
          ▼
      工单草稿
          │
          ▼
approval_request.pending
          │
          ▼
   人工确认
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

其中 `classify_ticket` 当前是可解释的规则化决策步骤，不把它包装成自主 LLM 规划。

## 预览 / 确认

### Preview

`POST /agent/ticket/preview`

预览阶段会：

1. 创建 `agent_run`；
2. 执行并记录 `search_kb`；
3. 根据用户请求与 RAG 来源 执行并记录 `classify_ticket`；
4. 若需要建单，生成工单草稿；
5. 将草稿持久化到 `approval_request.draft_json`，状态保持 `pending`；
6. 返回预览结果，不产生真实工单写操作。

预览默认仍为 Dense Top3；可显式复用冻结 Hybrid Top14 检索：

```json
{"message": "VPN 连不上", "category": "it", "retrieval_mode": "hybrid"}
```

需先批准查询及当前租户候选正文的外发与费用，并开启现有 `RAG_HYBRID_SEARCH_ENABLED`。不需要 ask 开关或 Phase B 分类配置，因为这里只检索并执行原有工单规则，不调用 v5.x 证据 gate 或答案生成模型。`other` 分类继续表示不限制检索分类，租户始终来自 JWT。响应返回 `retrieval_mode`；来源保留可空 `distance` 和独立重排/融合分数及排名，`search_kb` 与运行摘要记录模式和分数类型，Hybrid 汇总 `top_distance=null`。开关关闭或提供方失败返回脱敏 503，记录失败且不产生后续分类/审批，不自动回退 Dense。

`confirm` 契约不变：只消费当前租户的 pending 审批及服务端冻结草稿，不重跑检索、不调用模型。本文新增接线仅有离线模型替身回归，不是工单真实提供方或质量验收；默认开关不变。

工单描述仍限制为 2000 字符。用户原文加来源附录若超限，描述只保留完整用户原文，不截断问题；完整来源列表仍在 preview 的 `sources` 中，AgentOps 运行摘要与检索工具记录保留来源 ID。该规则同时适用于 Dense 和 Hybrid，不改变用户输入上限、审批草稿冻结或确认建单规则。

离线集成检查：`pytest tests/test_hybrid_search.py -k ticket_hybrid_document_to_decision`。通过真实 Uvicorn 回环 HTTP、JWT、上传/索引、临时 SQLite 与 Chroma，分别验证 Hybrid 预览后的确认、拒绝和取消，以及 trace/metrics 查询、租户与角色隔离；只有显式确认会在临时库产生测试工单。外部 embedding/rerank 使用替身，不是检索质量或并发审批验收，不外发资料、不修改日常数据。

### Confirm

`POST /agent/ticket/confirm`

只有以下条件全部成立时才创建真实工单：

```text
approval_request.agent_run_id == request.agent_run_id
approval_request.status == "pending"
request.draft == server-side approval_request.draft_json
```

真正用于创建工单的是服务端持久化的审批草稿，而不是客户端临时传入的数据。

这组校验用于拒绝：

- 跨 Agent 运行记录 使用其他审批请求；
- rejected / cancelled / already-approved 等非 `pending` 审批再次确认；
- 预览后由客户端篡改草稿内容。

审批更新通过数据库条件更新，限定审批 ID、租户和旧状态 `pending`。
并发 confirm / reject / cancel 只有一个决策成功，其他决策返回 HTTP 400；终态不可覆盖或重新打开。

> 这是审批决策至多一次消费，不是工单与审计写入的跨事务恰好一次保证，也不包含崩溃恢复。

更多实现细节见 [docs/agent_workflow.md](docs/agent_workflow.md)。

---

# 3. RAG 在线运行时

API 默认保持 **Dense Chroma 检索**；搜索和实验问答另提供默认关闭的 Hybrid 可选路径。

| 方案 | API 接入状态 | 仓库默认行为 |
| --- | --- | --- |
| 冻结 Hybrid：Dense100 + BM25100 → RRF(k=60) Top100 → qwen3-rerank → 直接 Top14 | 已接入可选搜索和实验问答 | 两个开关均关闭，请求默认 Dense |
| v5.3 结构化证据拒答 | 已接入 Hybrid 实验问答，策略名 `v5_3_experimental` | 不替换默认 Dense 拒答 |
| v5.3.1 版本审计候选 | 已接入显式实验选项 `v5_3_1_experimental` | 不自动选择，不替换 v5.3 或 Dense |
| Ticket Agent | preview 可显式选择 Hybrid 检索；confirm 不变 | 默认 Dense，不调用 v5.x 拒答 gate |

以上是代码及仓库默认配置状态，不代表某台机器的实际 `.env`、运行进程或公网部署状态。启用实验问答还需两个开关、专用分类凭据及显式请求参数，见 §3.2；接入不等于质量晋级。

主要能力：

- `/rag/search`
- `/rag/ask`
- 文档 chunk 检索
- `tenant_id` / `category` 元数据过滤
- 结构化 sources 返回
- 无上下文与低相关拒答
- 检索日志

问答路径仅根据检索上下文生成答案，并返回对应 sources。默认 Dense 路径的低相关拒答使用 Top-1 距离作为工程信号；实验 Hybrid 路径使用 v5.3 证据 gate，不使用该距离阈值。

> **运行时边界：** Hybrid 搜索和实验问答需要显式选择并开启服务端开关，不接管默认问答或 Ticket Agent。实验问答可显式选择尚未独立验收的 v5.3 / v5.3.1 gate，不代表稳定拒答策略已升级。代码接入不等于已部署、已开启或已证明整链质量。

## 3.1 可选 Hybrid 搜索

先通过 Document Backend 上传并显式索引文档，再在服务端 `.env` 配置已有 embedding 凭据、专用 `DASHSCOPE_RERANK_API_KEY` / `DASHSCOPE_RERANK_BASE_URL`，并设置 `RAG_HYBRID_SEARCH_ENABLED=true`。地址格式参见 [.env.example](.env.example)；重启服务或重建 Compose API 容器使配置生效。

**开启前须批准数据外发与费用：** 查询会发往 embedding 服务；查询及最多 100 条当前租户候选正文会发往配置的 rerank 服务。每次搜索都会调用模型，未配置全局费用额度；不要把密钥交给浏览器或提交到 Git。

使用现有 JWT 调用 `POST /rag/search`，请求体：

```json
{"query": "VPN error 809", "category": "it", "retrieval_mode": "hybrid", "top_k": 14}
```

- 路线固定为 `Dense100 + BM25100 → RRF(k=60) Top100 → qwen3-rerank → Top14`，返回其前 `top_k` 条；Hybrid 省略 `top_k` 时为 14，允许 1–14。Dense 仍默认 3、允许 1–10。
- 只检索 SQL 文档后端标记为 `indexed`、当前版本、当前租户且匹配 category 的 chunk；两个通道共享此范围。仅存在于旧静态 Chroma 索引、未通过文档后端登记的内容不进入 Hybrid；TechQA 离线语料也不会自动导入。
- `rerank_score` 越大越相关，但不是概率。`fusion_score`、`dense_rank`、`bm25_rank` 用于解释来源；BM25 独有命中的 `distance` / `dense_rank` 为 `null`。**不得把 Dense 的 0.9 距离阈值用于这些分数。** 审计 JSON 明确记录 Hybrid 模式和分数类型，Hybrid 的 `top_distance` 留空。
- 失败返回脱敏 HTTP 503，不暗中切换检索方案。显式改为 `retrieval_mode: "dense"` 可回退；关闭开关并重启可禁用 Hybrid。重排客户端不自动重试，超时 15 秒；embedding 沿用原客户端策略。
- Demo 上限为每租户/所选分类 5000 条可见 chunk，BM25 每请求重建，无持久化第二索引。英文技术词分词不覆盖中文；纯中文查询跳过 BM25，保留 Dense + 重排。规模或中文检索需要扩大时再单独测量和改进。
- 发送候选正文前及返回结果前复查 SQL 内容/可见性，发现删除或变更则终止请求；这不是 SQL 与 Chroma 的跨存储原子快照。上传、重建中、失败和删除状态不会被当作可见文档。

搜索接口不调用答案生成、不改变拒答逻辑、不创建工单；实验问答见下一节，完整问答与审批演示仍需独立验收。

小范围接线验收脚本（默认只展示计划，零模型调用且不读取 `.env`）：

```bash
python -m scripts.smoke_hybrid_runtime
# 仅在明确批准本次数据外发和费用后执行；此参数本身不构成授权：
python -m scripts.smoke_hybrid_runtime --execute --env-file .env
```

真实模式只发送脚本内固定的四篇虚构说明及两个查询，使用独立临时 SQLite、Chroma 与上传目录，经过 ASGI 路由及 Demo JWT 认证，不连接已运行的服务、不改现有数据库和开关。检查默认 Dense、Hybrid 简单排序、租户/分类隔离、删除生效、空语料零调用及检索审计；不是 HTTP 网络部署验收或准确率评测。
固定使用北京 `text-embedding-v4`（最多 8 次）及新加坡 `qwen3-rerank`（最多 3 次），禁用 SDK/传输层重试与重定向。发送前校验端点、模型及文本白名单；单次请求体最多 4 KiB、累计 16 KiB。控制台只记录尝试次数、字节、状态、耗时和服务端 usage，不打印密钥或提供方错误正文。任何失败即停止，不自动重新运行。
预算按 [官方价格](https://www.alibabacloud.com/help/en/model-studio/model-pricing) 保守预留 ¥1，短文本预计不足 ¥0.1；次数/字节限制不是提供方账单硬限额。中断后的请求可能已计费，重新执行须重新确认，不能把脚本重跑当成免费续跑。

2026-10-16 已获授权完成一次真实提供方冒烟：8 次 embedding + 3 次重排全部 HTTP 200，服务端 usage 分别为 126 / 162 tokens，9 项接线检查通过；[逐次结果与边界](docs/reports/hybrid_runtime_smoke_2026-10-16.json)。这是临时 ASGI 环境的小范围验证，不证明 Hybrid 优于 Dense，也不代表已开启现有服务或验收问答、拒答、工单整链。

## 3.2 实验 Hybrid 问答（默认关闭）

该路径验证集成，不是 v5.3 的正式晋级：其独立语义质量尚未确认，历史误放行/误拒仍是已知风险。默认 `/rag/ask` 继续使用 Dense + Top1 距离阈值，Ticket Agent 不变。

只在批准数据外发及费用后，在服务端同时开启 `RAG_HYBRID_SEARCH_ENABLED=true`、`RAG_HYBRID_ASK_ENABLED=true` 并重启。须另配置专用 `DASHSCOPE_PHASE_B_API_KEY` / `DASHSCOPE_PHASE_B_BASE_URL`（运行时不回退到重排或通用密钥）；生成沿用 `DASHSCOPE_API_KEY` / `DASHSCOPE_BASE_URL`，模型按 `CHAT_MODEL` 或 `DASHSCOPE_MODEL` 选择。默认示例分类端点为新加坡，通用生成端点为北京；批准外发时应分别核对。

使用现有 JWT 请求 `POST /rag/ask`：

```json
{"question": "How do I resolve VPN error 809?", "category": "it", "retrieval_mode": "hybrid", "refusal_policy": "v5_3_experimental"}
```

也可显式指定 `refusal_policy="v5_3_1_experimental"`，仅将分类提示词换为已有的版本审计澄清候选；检索、严格解析器、生成及安全检查复用原链路。响应和成功/拒答/失败审计保留所选策略名。两个策略都要求原有两个开关及专用分类凭据；没有新增默认策略或自动回退。回到 v5.3 只需显式选择原策略。新提示词更长，分类输入 tokens 会增加，未新增全局费用硬限额。v5.3.1 的已知分项错误与未独立验收状态不因接入而消失；本次仅完成离线接线回归，没有新真实模型结果。原有界冒烟脚本仍只允许原 v5.3 提示词，不能据其历史报告声称已验收 v5.3.1。

Hybrid 默认 `top_k=14`，允许 1–14，显式较小值只使用该 Top14 前缀；Dense 仍默认 3、最多 10。Hybrid 必须显式声明实验策略，且**不能传 `max_distance`**（包括原默认值 0.9），否则 HTTP 422；不把任何距离或重排分数作为充分性阈值。

```text
Hybrid 检索（同一租户/分类、已索引当前版本，最多 Top14）
→ 空语料：no_context，零模型请求
→ 所选 v5.3 / v5.3.1 需求/适用性/版本覆盖检查 + 原文行引用校验
→ 不足/冲突/条件缺失/歧义：refused_insufficient_evidence，不生成
→ 充分：基于完整有序来源生成，校验 [Source N] 属于 gate 支持的来源
→ 生成器自行拒答：refused_by_generator；合法答案：ok
```

- 固定分类模型 `qwen3.5-plus-2026-04-20`，按显式策略复用原 v5.3 或 v5.3.1 提示词和同一严格解析器，不读取评测样本、标签或 DEV；分类最多 2048 输出 tokens，生成最多 1024，两者超时各 30 秒、无 SDK 自动重试。分类及 Qwen3.5 Plus 生成（含日期版本）显式设置 `enable_thinking=false`，其他生成模型不附加该提供方参数。检索侧 embedding 仍沿用原客户端策略。
- 问题和最多 14 条全文会额外发往分类服务；放行后再发往配置的生成服务。非空请求比搜索最多增加两次 chat 调用；没有新增全局费用硬限额。首次真实问答验收未通过，见下一节。
- 分类格式错误、非法行引用、截断、超时、空生成或答案缺少/含未批准的 `[Source N]` 均为脱敏 HTTP 503 + `failed` 审计，不算正确拒答、不暗中放行或回退 Dense。显式选择 Dense 并移除实验策略可回退；关闭 ask 开关只禁用实验问答。
- 发送分类/生成上下文前及返回前复查 SQL 可见性和内容，变化则终止；仍不是跨 SQL/Chroma 原子快照。返回 `top_distance=null`，sources 保留距离与重排分数的不同语义。
- 响应与成功/正常拒答审计包含 `refusal_policy`、`evidence_gate`（需求、覆盖检查、原文证据、缺口与歧义）；失败审计记录模式、策略、`error_stage` 和最多 6 层 `error_types`，不记录异常正文或任意异常备注。阶段为分类、生成或未细分的 `service`；错误仍返回通用 503。AgentOps 可按两个新增拒答状态过滤，汇总 `retrieval_status_counts` 显示全部状态。

**校验只证明格式、引用出处和状态接线，不证明语义蕴含、逐句忠实或抗提示注入。** 不把有效引用或实验 gate 放行写成答案正确率，也不改写历史 NO_PROMOTION / FAIL。

## 3.3 集成验收状态（更新至 2026-10-18）

**当前不接受为已验收发布版。** 已补齐仓库声明的评测测试依赖，在隔离 Windows / Python 3.12.4 环境运行完整 `tests/`：最新 659 项通过、零失败/跳过；不只原重点回归集。主 pytest 进程禁止外部 socket 与读取项目私有 `data/`，测试使用临时数据；CI 包含相同完整离线检查。离线成功不是模型质量验收。

真实问答计划复用同一脚本，默认仍然零调用：

```bash
python -m scripts.smoke_hybrid_runtime --ask
# 仅在单独授权后执行；异常即停，不自动续跑或重新执行：
python -m scripts.smoke_hybrid_runtime --ask --execute --env-file .env
```

固定 7 篇虚构文档、8 类案例，预期标签只在本地检查。独立上限为 14 次北京 embedding、6 次新加坡重排、6 次新加坡分类、2 次北京生成，禁用重试和重定向；请求体最多 24 KiB、累计 96 KiB，保守预算 ¥2 而非账单硬限额。生成只获准用于两个预设可回答案例；若负例错误放行，发送守卫会阻断生成并判验收失败，不改写为正常拒答。旧搜索模式仍保留原 8/3 上限。

首次获授权执行一次，在首个可回答案例停止：9 次 embedding、1 次重排、1 次分类均 HTTP 200；生成尝试约 30 秒未获得可用响应，接口 HTTP 503。已报告 usage 合计 2639 tokens，**该次生成用量未知，可能计费，不能当作零费用**。当时剩余 7 类真实案例未执行，没有重试；[完整失败记录](docs/reports/hybrid_acceptance_2026-10-17.json) 保留输出、调用计数及执行时脚本哈希，不被后续复测覆盖。

另有语义警示：问题明确指定版本 1.0，分类器却把 `version_and_time` 标为 `NOT_REQUIRED`。来源确实覆盖该版本，但这个审计判断不符合提示词约定，格式和引用通过不能证明语义判断可靠。
该次生成请求没有显式关闭思考模式，而 [官方文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) 说明 Qwen3.5 Plus 默认开启思考。这是配置风险，**不是本次超时唯一原因的证据**。

随后修复显式关闭 Qwen3.5 Plus 生成思考模式，并补齐脱敏诊断。验收脚本区分等待响应头、读取响应体、读取完成，保留已收到的 HTTP 状态、异常类型链、usage 分项及 `finish_reason`；不记录隐藏思考或提供方错误正文。等待响应头不代表服务器已接受请求，超时缺少 usage 时仍不能推断零计费。30 秒超时、输出上限、重试策略、提示词与拒答规则均未变。

修复后另获授权执行一次[真实提供方复测](docs/reports/hybrid_acceptance_retest_2026-10-17.json)：8 类流程检查均通过（英文/中文可回答各 1、缺资料/版本不符/冲突/无关拒答各 1、空范围/删除后无上下文各 1）。实际 14 次 embedding、6 次重排、6 次分类、2 次生成，共 28 次，全部 HTTP 200，无重试；usage 合计 14525 tokens，请求体合计 63541 bytes。两次生成分别 3480 / 1889 ms，正常结束；本次未复现超时，但单次成功不证明历史超时的唯一原因或稳定延迟，也不等于总体答案准确率。

复测仍有版本审计语义警示：NovaVPN 1.0 的显式版本约束仍被标成 `NOT_REQUIRED`；AcornExpense 2.0 的版本覆盖被标成 `SUPPORTED`，理由却承认来源只覆盖 1.0。后者整体因其他检查为 `MISSING` 而正确拒答，但不能据此认定每项审计可靠。保留这两项警示；该次复测未针对案例改提示词或规则，历史 NO_PROMOTION / FAIL 不变。

开发阶段新增 [v5.3.1 版本审计澄清候选](experiments/evals/refusal_version_audit_clarification.py)：只追加通用说明，区分“来源声明自己的版本范围”和“证据满足问题所需版本”，保留可见兼容性依据，不退回版本关键词/字符串匹配。没有加入上述验收案例答案，沿用原输入、状态与严格引用解析器，不修补或重标模型输出。手写边界测试只证明接线与限制；当时仅为离线候选，后续显式 API 接入见 §3.2，原 `v5_3_experimental` 提示词、运行行为与历史报告均不变。

随后单独授权完成一次[6 题分类配对开发检查](docs/reports/version_audit_paired_2026-10-17.json)：输入、预期与请求守卫在调用前提交冻结，新旧提示词使用相同输入/参数、交替首跑顺序。版本审计符合预期为 v5.3 **3/6**、v5.3.1 **5/6**；整体放行/拒答均 **6/6**，没有决策差异。候选在版本不符、兼容范围两题改善分项审计，但仍把“版本已匹配、功能信息缺失”的版本检查标成 `NOT_REQUIRED`。这只是少量已知虚构开发场景，不是独立验收，也不证明回答准确率提升；不改预期、不重标结果、不因局部改善切换服务策略。

本次仅 12 次新加坡分类，均 HTTP 200，无重试；usage 合计 27481 tokens，请求体 119304 bytes，没有重跑检索或生成。`python -m scripts.probe_version_audit` 默认只输出零调用计划；真实执行还要求事先授权、配置专用新加坡凭据并绑定计划哈希。单次/累计请求体上限为 24/192 KiB，最多 12 次调用；发送守卫逐次比对完整请求体，预期标签不进入模型请求。任何网络、提供方或解析异常即停，不自动续跑。候选仅部分有效，下一步应先判断审计改进的收益，不自动进入新一轮付费调参。

2026-10-18 启动 Docker 后完成[隔离本地容器验收](docs/reports/local_container_acceptance_2026-10-18.json)：使用原 Dockerfile/Alembic/Uvicorn、假凭据、3 个独立卷和 `--network none`，通过容器内真实回环 TCP HTTP 检查迁移 head、JWT/角色/租户隔离、两篇虚构文档的元数据/原文校验和、独立 Chroma 预制向量及审计记录。在重启、删除重建保留卷、临时开启空范围 Hybrid、恢复关闭配置后均通过，最终审计 10 条，工单/Agent 执行表仍为零；没有付费模型调用。

该次验收不发布宿主机端口、不加载真实 `.env`；Chroma 预制向量不是文档索引/检索质量验证，空范围 `no_context` 不是非空问答验证。其范围不含完整 Compose、宿主机访问或非空问答。固定检查入口为 `python -m scripts.smoke_local_persistence`，首次全新验收卷加 `--seed`；**只允许规定假凭据/数据库路径、Docker 无网络环境，拒绝 `-O`，不能直接针对日常服务运行**。临时容器和仅含虚构数据的卷已清理，专用镜像保留供复现。Ticket Agent 仍使用 Dense，默认 Hybrid 开关和 v5.3.1 候选接线均未改变。

后续已分别补充 [Compose / 宿主机回环、重启与容器重建验收](docs/reports/compose_loopback_acceptance_2026-10-20.json)，以及 `tests/test_hybrid_search.py::test_nonempty_hybrid_over_real_uvicorn_tcp` 的非空真实 HTTP 检查（模型使用测试替身）。这两项证据本身不能拼成同一次 Docker + 真实提供方端到端通过。

2026-10-20 经单独授权完成一次[容器内真实 HTTP + 真实提供方联测](docs/reports/container_real_http_acceptance_2026-10-20.json)：调用前提交固定协议，使用同一镜像的 Alembic、Uvicorn 回环 TCP、临时 SQL/上传目录与真实临时 Chroma，运行原 7 篇虚构文档、8 类用例。28 次调用全部 HTTP 200（14 embedding / 6 rerank / 6 分类 / 2 生成），8 类流程检查通过；模型 usage 合计 14507 tokens，请求体 63541 bytes，不代表账单金额。回答 2 类、证据不足拒答 4 类、空/删除范围 2 类，工单与 Agent 动作表均为零。未发布宿主机端口，临时容器和合成运行数据已清理，未重跑或改持久开关。

该次使用有界验收入口，不是原 Compose 启动命令；没有把宿主机发布、持久化重启、Dense 答案或工单整链加入本次付费运行。版本 2.0 问题仍被错误标成版本覆盖 `SUPPORTED`（来源仅有 1.0），最终由其他缺证检查拒答。因此结论仍是**限定范围的集成通过，不是独立拒答质量晋级或生产发布**；v5.3.1 未采用。

---

# 4. RAG 评测与迭代：从 Dense 基线到当前状态

`experiments/evals/` 是正式离线评测入口。默认 API 与默认问答/工单仍使用 **Dense Chroma 检索**；可选 Hybrid 搜索及实验问答见上一节。下面记录的是离线实验与冻结工程选择，不是新运行时语料的质量验收或公网部署证明。

## 4.1 评测契约

| 数据划分 | 可回答 | 不可回答 | 用途 |
| --- | ---: | ---: | --- |
| TRAIN | 450 | 150 | 开发、失败归因、方案选择 |
| DEV | 160 | 150 | 冻结验证，只用于正式独立对比 |

TechQA 的 qrel 是文档级，而检索器返回 chunk。正式 IR 评测会保留原始 chunk 排序，再按 `document_id` 首次出现位置去重成文档排序，计算 Document Recall@5、Recall@20 和 MRR@10。每条 可回答问题 只有 1 个 相关文档，因此这里 Recall@K 与 Hit@K 数值相同。

---

## 4.2 核心检索迭代

| 阶段 | 为什么做 / 本次改动 | 评测口径 | 核心结果 | 失败归因与决策 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **E0：Dense 基线** | `问题 → Dense Top100 chunks → 文档去重排序`，先建立统一基线 | TRAIN + 冻结 DEV | **TRAIN**：R@5 61.3%，R@20 74.0%，MRR 0.510；**DEV**：R@5 64.4%，R@20 81.9%，MRR 0.519 | 总指标只能说明效果不足，不能区分“候选没召回”和“候选已召回但排序靠后”，因此先做失败归因 | [E0 失败分析](experiments/evals/reports/e0_dense/failure_analysis.md) |
| **E0：失败归因** | 对失败按 标准相关文档排名分桶，再抽固定样本人工审计 | TRAIN 450 条 answerable | 排除 5 条仅由查询文本尾部空白差异造成的跨运行漂移后：排名 4–5 有 **20** 条，排名 6–20 有 **57** 条，共 **77/450** 个明确排序问题；另有 117 个 Top20 未命中 | 30/117 个 miss 中：17 个 qrel / 问题歧义、7 个明显词法未命中、6 个语义未命中。另抽 30 个低正确率 样本：14 个评测 / 参考答案问题、12 个证据覆盖问题，仅 4 个明显生成问题。**排序问题是当时最强、最确定的可操作失败类，因此 E1 先做 重排** | [完整归因与样本审计](experiments/evals/reports/e0_dense/failure_analysis.md) |
| **E1：Dense + 重排** | `Dense Top100 → qwen3-重排 → 文档去重排序`；**只加重排器** | 冻结 DEV 正式对比 | R@5 **64.4% → 72.5%**；R@20 **81.9% → 84.4%**；MRR **0.519 → 0.561** | DEV Top5：21 个改善、8 个退化；Top20：5 个改善、1 个退化。重排有明显净收益，但不是单调改善。冻结 DEV 不用于逐样本反向调参，因此没有针对这 8 个 DEV 退化继续做正式根因拟合 | [E0 / E1 对比](experiments/evals/reports/e1_rerank/comparison.md) |
| **R3：BM25 互补性验证** | 剩余失败里出现 错误码、版本号、CVE、固定技术词等精确词法查询，因此加入 BM25，用 RRF 验证候选互补性 | TRAIN，文档级互补性验证 | Dense 命中@100 **387**，BM25 **375**，融合 **402**；救回 Dense 未命中 **19** 个，净增 **15** 个 | BM25 单独不优于 Dense，但确实补回一批 Dense 漏掉的精确词法候选，因此允许进入正式 Hybrid + 重排 实验 | [R3 准入结果](experiments/evals/reports/r3_hybrid/admission_decision.md) |
| **R4 C1：Hybrid + 重排** | `Dense100 + BM25100 → RRF60 → 融合 Top100 → qwen3-rerank`；与 **E1 TRAIN** 对比 | TRAIN 450 条 | E1：R@5 / R@20 / MRR = **.691 / .816 / .567**；C1：**.702 / .831 / .571** | 三项都上涨，但预注册 MRR 门槛为 **.577**，实际只有 .571。最终 Top20 未命中中 **56 个候选缺失、20 个排序不足**；固定 Top100 融合预算既会救回候选，也会压掉部分单源尾部候选。**正式结论 FAIL，不继续调 RRF k、权重、深度等参数** | [正式对比](experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md) · [失败案例与归因](experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md) |

> **查询文本尾部空白漂移：** 历史 retrieval 与 generation 数据中，有些问题语义完全相同，只在末尾多了空格或换行。250/450 条 可回答问题 存在这种差异，其中 74 条 Top3 排序发生变化、5 条 标准相关文档准入 发生变化。由于这不是语义变化，这 5 条不用于因果失败计数；后续 模型服务调用前统一对 query 做 `rstrip()`。

R4 C1 的正式历史状态仍是 **FAIL**。后续冻结工程候选保留固定 Hybrid 路线，是基于整体证据做出的工程选择；可选搜索接入不改写历史结论，也不代表默认 API 或问答已切换到 Hybrid。

---

## 4.3 从“命中文档”到“命中答案证据”

Document Recall 只能回答“相关文档是否出现”，但 RAG 真正交给生成模型的是 chunk，因此还需要回答：

> **命中了 标准相关文档，真正能回答问题的 chunk 是否进入了高位 上下文？**

### 人工证据标注

从 TRAIN 可回答样本 中确定性抽取 60 条，在 标准相关文档 内定位候选 chunk，共人工标注 187 个 chunk：

| 标签 | 含义 | 评测中的作用 |
| --- | --- | --- |
| `0 = 弱证据` | 相关性弱，不能作为有效回答证据 | 不计入 有用证据 / 答案证据命中 |
| `1 = 有用证据` | 对解决问题有帮助，但本身不直接承载完整答案 | 计入 有用证据命中 |
| `2 = 答案证据` | 直接承载回答问题所需的关键证据 | 同时计入 有用证据命中 与 答案证据命中 |

60 条中有 6 条被标记为 `questionable_gold=true`：标准相关文档与问题主题不匹配、条件冲突，或不足以支撑 标准答案，因此正式证据指标只评 **54 条**。

[人工标签](experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl) · [证据指标](experiments/evals/reports/r1_evidence_audit/evidence_metrics.json)

### 同一批人工标签上的三条排序链

| 排序链 | 答案证据命中@5 | 答案证据命中@20 | 有用证据命中@5 | 有用证据命中@20 |
| --- | ---: | ---: | ---: | ---: |
| **E0 Dense** | 44.4% | 61.1% | 63.0% | 77.8% |
| **C1 融合后、重排 前** | 46.3% | 53.7% | 68.5% | 77.8% |
| **C1 重排 后** | **53.7%** | **64.8%** | **72.2%** | **79.6%** |

这里比较的不是三套不同人工标签，而是**用同一批人工 chunk 标签去评三条不同的 chunk 排序**。这也解释了为什么上面不能只写 “Dense 44.4% → C1 重排 53.7%”：中间的“融合后、重排 前”本身也是一条被测链路，而且它在 答案证据命中@20 上反而从 Dense 的 61.1% 降到 53.7%，随后 重排器 又把它提升到 64.8%。

因此后续实验不再只看文档 Recall/MRR，还单独追踪“真正能回答问题的证据是否进入高位 上下文”。

---

## 4.4 上下文策略：G1 / G2

### 历史 E1 上下文 对照

G1 并不是拿“Top5 文档”去对比 E1。历史 E1 生成评测链路 的 上下文 策略是：

```text
Dense Top100
→ 全局 qwen3-rerank
→ 重排后 Top3 chunk 作为锚点
+ Dense 排名第1的 chunk 兜底
→ 每个锚点最多带 3 个向后相邻 chunk
→ 去重
→ 最多 16 chunks
```

它的特点是：**先让全局 重排器 看完整个 Dense Top100，再围绕高位 anchor 做局部扩展。**

### G1 / G2 对比

| 阶段 | 为什么做 / 本次改动 | 核心结果 | 失败案例与归因 | 决策 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **G1：文档内证据扩展** | `Dense 排名 → 前5篇去重文档 → 展开文档内 chunks → 合并后重排 → Top16`。目标是解决“关键证据藏在同一文档更远位置”的问题 | 30 条：充分 **24→28**；宏平均关键事实覆盖率 **0.825→0.956**；6 个改善 / 22 个持平 / 2 个退化 | **Q346**：关键 chunk 原 Dense 排名 72，历史 E1 的全局 重排器 本来能在裁剪前把它救上来；G1 却先按 Dense 只保留5篇文档，关键文档在 重排 前已被永久裁掉。Q492 另有连续证据丢失 | **NO_GO**：平均提升明显，但出现 充分→不充分 的灾难性退化 | [G1 结果与 Q346/Q492 分析](experiments/evals/reports/g1_document_local/final_decision.md) |
| **G2-A：重排 后再做文档准入** | `Dense Top100 → 全局 重排 → 前5篇去重文档 → 文档内展开 → 合并后重排 → Top16`；**只改文档准入顺序** | 新的 TRAIN 30 条：**2 个改善 / 25 个持平 / 3 个退化**；充分 19→19；宏平均关键事实覆盖率 **.667→.650** | Q287/Q578 被全局 重排 救回，但 Q090/Q500 又因新的前5篇预算把原本有效文档挤出去。主因是**准入排序不稳定 + 固定5篇预算放大这种不稳定** | **NO_GO**，不根据已经看过的失败 样本 继续补规则 | [G2 五个变化 样本 逐条归因](experiments/evals/reports/g2_rerank_informed_admission/post_hoc_forensic_analysis.md) |

**宏平均关键事实覆盖率**：先把标准答案拆成若干关键事实点，计算每条问题的事实覆盖率，再对问题等权平均。例如两个问题分别覆盖 100% 和 25%，宏平均关键事实覆盖率 为 62.5%。因此 G1 的 0.956 是**平均关键事实覆盖率**，不是“95.6% 准确率”。

---

## 4.5 上下文收口：为什么冻结为 直接 Top14

### 直接 TopK 是什么

**直接 TopK** 指：重排器 输出一条全局 chunk 排名后，**直接取前 K 个 chunk 作为最终 上下文**。这里的 “直接”表示不再做层级式或局部式扩展：不追加 相邻 chunk、不展开整篇文档、不做硬性文档准入，也不再进行第二次 重排。

当前冻结候选使用：

```text
重排后的 chunks
→ 直接取前 14 个
→ final 上下文
```

历史 E1 / G1 实验曾使用最多 16 个 chunk；当前冻结工程候选是 **直接 Top14**，两者不是同一个上下文策略。

| 阶段 | 为什么做 / 本次改动 | 核心结果 | 失败归因与决策 | 详情 |
| --- | --- | --- | --- | --- |
| **检索前沿审计** | G1/G2 都说明不同排序存在互补，但继续在已看过的 TRAIN 样本 上调 K、RRF、权重容易过拟合，因此只对冻结结果做零模型服务调用 的反事实分析 | 观察到 Dense 与全局 重排 确实存在互补，文档级 RRF 是未来可能方向 | 这些 30 条已经是设计/诊断数据，不能继续拿来证明新方案有效。**关闭当前检索参数搜索** | [前沿审计](experiments/evals/reports/retrieval_frontier_freeze/final_offline_frontier_audit.md) |
| **最终上下文预算** | 比较不同 直接 TopK，找达到当前证据命中上限的最小 K | 54 条证据样本：**Top14 = 答案证据 35/54、有用证据 43/54；Top20 完全相同** | 从14增加到20没有新增 证据命中，只增加上下文。**选择 直接 Top14**；这是当前 TRAIN 开发集选择，不是通用最佳 K | [架构冻结](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |
| **局部二次 重排** | 尝试对 Top14 未覆盖、且具备局部恢复条件的残留 样本 再做局部候选恢复与第二次 重排 | 答案证据 **35→35**；有用证据 **43→43**；**7 个具备局部恢复条件的 残留样本，0/7 被救回** | 没有新增 证据命中，还增加一次 重排；p50 约 661 ms。**拒绝** | [架构冻结 §4.1](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |
| **结构保持扩展** | 尝试通过同文档 / 结构扩展恢复被固定 chunk 切分打散的信息 | 答案证据 **35→32**；有用证据 **43→38**；2 个未命中→命中，但 5 个命中→未命中 | 5/5 退化都来自**上下文预算挤占**：同一文档放入更多内容后，跨文档证据 被挤掉；上下文 中文档数中位数 **12→4.5**。**拒绝，保留 直接 Top14** | [架构冻结 §4.2–5](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |

最终冻结的检索 / 上下文 工程候选：

```text
Dense Top100
+
BM25 Top100
    ↓
等权 RRF (k=60)
    ↓
fused Top100
    ↓
qwen3-rerank
    ↓
直接 Top14
```

- 检索参数研究：**CLOSED**
- 上下文 组装研究：**CLOSED**
- 当前默认运行时、问答和工单：仍是 **Dense Chroma 检索**
- Hybrid + 重排 + 直接 Top14：**已接入默认关闭的可选搜索和实验问答，工单不变；整链质量尚未验收**

---

## 4.6 当前拒答线：证据充分性

冻结检索 / 上下文 后，离线研究转向判断 **直接 Top14 是否包含足够证据**；默认低相关拒答仍保留 Dense Top1 距离阈值（默认 0.9），没有被开发候选替换。另有显式开启的 v5.3 实验问答路径，不代表其通过独立质量验收。

| 阶段 | 本次方案 | 结果 | 失败归因 | 结论 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **拒答 v1.1** | `直接 Top14 → qwen3.5-plus 证据充分性分类` | 平衡准确率 **0.736**；充分证据召回率 97.1%；不足证据召回率仅 **50.0%** | 16 个不足证据样本 中有 8 个被错误放行为“充分” | **FAIL**，不进入生成与 运行时集成 | [v1.1 正式结果](experiments/evals/reports/refusal_evidence_sufficiency/phase_b_v1_1_result.md) |
| **拒答 v2.1** | 新分类契约 + 新的 AI 草稿代理标签 | 平衡准确率 **0.775**，门槛 0.80；充分召回 .85，不足召回 .70 | 只差 1 个 样本 即可过门槛，但后验检查发现部分 分歧 来自**标注契约与问题实际要求不一致**，不能简单继续调 提示词 | **FAIL**，不修改门槛后重跑 | [v2.1 结果](experiments/evals/reports/refusal_evidence_sufficiency/v2_ai_proxy_v2_1_result.md) |
| **拒答 v3** | 从盲标 AI 草稿中冻结 50 条 TRAIN，25 充分 / 25 不充分 | 与 AI proxy 一致 **38/50**；平衡准确率 **0.76**；两类召回均 0.76 | 充分题误拒 6 条、不足题误放行 6 条；标签不是独立人工金标 | **FAIL**，不进入生成验收或 runtime | [v3 结果](experiments/evals/reports/refusal_evidence_sufficiency/v3_ai_proxy_result.md) |
| **v4 / v5.1 配对开发诊断** | 20 条 TRAIN 同题对照；v5.1 选择来源行，由代码提取原文并推导决策 | 40 次调用全部有效；AI proxy 一致 **16/20 → 17/20**；估算总费 ¥0.741975 | 不足题误放行 **4/6 → 2/6**，但新增 1 个充分题误拒；费用约 2.06 倍、p50 延迟约 1.82 倍 | **NO_PROMOTION**；非独立 AI 标签、小型已见 TRAIN，不能宣称全面改善 | [v5.1 冻结结果](experiments/evals/reports/refusal_evidence_sufficiency/v5_1_paired_result.md) |
| **v5.2 必填覆盖审计** | 强制检查请求结果、适用性、版本与时效；复用行引用校验 | 14 条虚构控制题真实模型对照：v5.1 有效且符合构造标签 **14/14**，v5.2 **12/14**，另有 2 次格式失败 | 模型将已有证据的审计栏位输出为空引用；格式失败不是正确拒答 | **NO_PROMOTION**；控制题不是自然问题准确率 | [控制题结果](experiments/evals/reports/refusal_evidence_sufficiency/v5_2_control_result.md)、[零调用机制演示](docs/demo_script.md#21-离线拒答-gate-演示) |
| **v5.2.1 引用示例调整** | 仅修改提示词中的引用示例，不放宽 schema / 引用校验 | 同一批已见虚构控制题，新跑 v5.2 / v5.2.1 均 **14/14** 有效且符合构造标签 | 旧版本的 2 次格式失败在同期基线中也未重现，不能归功于新示例 | **未证明格式修复收益 / NO_PROMOTION** | [同期回归结果](experiments/evals/reports/refusal_evidence_sufficiency/v5_2_1_regression_result.md) |
| **v5.1 / v5.2.1 自然 TRAIN 诊断** | 30 条冻结自然问题、完整 Top14、两组新跑 | 25 条二元 AI proxy 标签一致 **19/25 → 21/25**；充分题误拒 **5/21 → 3/21** | 修正 4 次误拒、引入 2 次；不足题误放行仍 **1/4**；费用约 1.36 倍 | **仅开发诊断 / NO_PROMOTION** | [自然问题结果](experiments/evals/reports/refusal_evidence_sufficiency/natural_v5_2_1_result.md) |
| **v5.2.1 / v5.3 问题范围回归** | 明确只检查必要请求，并要求具体事故的证据适用性；相同 30 题、两组重新运行 | 二元 AI proxy 标签一致 **22/25 → 23/25**；充分题误拒 **2/21 → 1/21**；60 次调用估算 ¥1.917294 | 修正 2 次误拒、新增 1 次；**同一个不足案例仍误放行（1/4）**，部分标签一致的推理仍有无依据外推 | **仅已见题开发回归 / NO_PROMOTION**，未解决语义可靠性 | [v5.3 完整结果](experiments/evals/reports/refusal_evidence_sufficiency/v5_3_regression_result.md) |

后两次自然问题实验使用同一套冻结 AI proxy 标签：21 条充分、4 条不足、5 条有歧义；
5 条有歧义只作描述，不计入二元一致率。这不是独立专家金标，也没有生成答案或验证线上整链。
v5.3 的设计已经参考这些题；其同期 v5.2.1 基线为 22/25，不能拿历史 21/25 来夸大增益。
目前不继续在这 30 题上追加提示词或重复付费调用；后续语义验证需先明确新样本与冻结口径，
另行授权数据发送和费用。v5.2～v5.3 均未替换线上 Dense 距离拒答。

完整实验契约、索引与冻结产物见 [experiments/evals/README.md](experiments/evals/README.md)。

---

# 5. 文档生命周期

文档后端提供从知识入库到下架的显式生命周期：

```text
Upload
  ↓
文档记录
  ↓
显式索引
  ↓
切片 + 向量化
  ↓
RAG 检索
  ↓
Delete
  ├─ 关系库 chunk 删除
  └─ Chroma 向量删除
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

# 6. AgentOps / 可观测性

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

- Agent 运行记录 状态；
- 工具调用 成功 / 失败与 error type；
- 审批 状态；
- 检索无上下文 / 拒答 / 失败；
- 检索来源分布；
- 单次运行轨迹。

---

# 7. 认证与租户范围

项目包含用于工程验证的 Demo JWT Auth：

- Bearer token；
- `user_id`；
- `tenant_id`；
- `role`；
- `support` / `admin` 角色检查；
- 按租户隔离的 Document / AgentOps / RAG 访问。

> 这里验证的是认证上下文和 租户范围 在应用链路中的传递，不把它描述为完整生产级 IAM / RBAC 或数据库级多租户隔离方案。

安全边界见 [docs/security.md](docs/security.md)。

---

# 8. 项目结构

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

- `rag_runtime/`：正式在线 RAG 运行时；
- `experiments/evals/`：TechQA 主评测、受控实验与产物；
- `experiments/rag_local/`：早期兼容入口；
- Todo / AI Todo 路径保留为历史兼容，不作为当前项目定位。

---

# 9. Quick Start

## Environment

在仓库根目录创建或编辑未提交的 `.env`，参考 `.env.example`（不要覆盖已有配置）：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
EMBEDDING_MODEL=text-embedding-v4
DATABASE_URL=sqlite:///data/todos.db
SQL_ECHO=true
DOCUMENT_STORAGE_ROOT=storage/documents
```

## 安装与数据库迁移

以下命令在仓库根目录、同一已激活的 Python 环境中运行，确保 `python` 与 `alembic` 属于该环境。
先创建默认 SQLite 父目录，再将 `.env` 载入迁移进程；
若修改 `DATABASE_URL`，需创建对应父目录。迁移与启动都优先使用已有 shell 环境变量。

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
python -m dotenv -f .env run --no-override -- alembic upgrade head
```

## 启动 API

```bash
python -m uvicorn main:app --reload --env-file .env
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

镜像构建排除 `.env`/其变体、开发及 Docker 的数据库/上传文件/Chroma 目录，保留公开 `.env.example`。Git 忽略不等于 Docker 构建忽略；不要把日常运行数据复制进镜像。

零模型调用的隔离 Compose 启动验收（Compose ≥ 2.24.4）：

```bash
docker compose --env-file .env.example -p copilot-acceptance -f docker-compose.yml -f docker-compose.acceptance.yml up --build --pull never -d
curl http://127.0.0.1:18080/health
```

该覆盖配置移除真实 `.env`、代理和业务目录挂载，使用假凭据、容器回环不可用模型地址与项目专属命名卷，Hybrid 两个开关保持关闭。普通桥接网络用于宿主机端口访问，不提供网络层外连阻断。它验证运行接线，不验证模型质量。不要用于正常问答或公网部署。完成后仅清理该独立项目：

```bash
docker compose --env-file .env.example -p copilot-acceptance -f docker-compose.yml -f docker-compose.acceptance.yml down --volumes
```

实际宿主机回环、重启及容器重建结果与未覆盖范围见 [Compose 验收记录](docs/reports/compose_loopback_acceptance_2026-10-20.json)。

非空 Hybrid 的真实 HTTP 回归见 `tests/test_hybrid_search.py::test_nonempty_hybrid_over_real_uvicorn_tcp`：启动仅回环的 Uvicorn，使用真实 JWT、临时 SQL 和 Chroma，验证回答、证据不足、提供方错误、空范围及开关关闭的状态与审计。外部模型响应使用测试替身，不调用付费模型；这不是同次 Docker + 真实提供方端到端验证，也不证明模型语义质量或 Dense 答案质量。

同一有界冒烟脚本可用 `--live-http` 将 TestClient 换成真实回环 Uvicorn/TCP；以下仍只输出零调用计划：

```bash
python -m scripts.smoke_hybrid_runtime --ask --live-http
```

真实执行另需单独授权及 `--execute`，原有固定 7 文档/8 用例、28 次调用及请求体守卫不变。该模式使用临时 Alembic 数据库和真实临时 Chroma；容器执行时可不发布任何宿主机端口。`--env-file -` 只接受标准输入中三个提供方配置的六项 JSON 字段（`DASHSCOPE` / `DASHSCOPE_RERANK` / `DASHSCOPE_PHASE_B` 的 `_API_KEY`、`_BASE_URL`），不把凭据放进 Docker 环境或命令行。它运行验收入口，不是原 Compose 启动命令或持久化/重启测试的替代。

---

# 10. 测试 / CI

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

下面的脚本需要 API 已启动，会调用已配置的向量模型并写入测试数据，请使用隔离的运行目录。
工单脚本会调用 confirm，保留测试工单与审计记录；文档脚本会移除测试向量，但保留软删除记录和原文件。
首次运行可先执行文档脚本建立 Chroma 集合。若要验证工单 preview 命中来源，应先上传并索引相关 IT 文档；
工单脚本本身不要求来源命中，不能据其成功宣称检索质量已通过验证。

```bash
python scripts/smoke_document_backend_flow.py
python scripts/smoke_agentops_flow.py
```

零调用拒答 gate 演示（合成资料、手写响应；不是模型效果或线上端到端验证）：

```bash
python -m scripts.demo_refusal_coverage
```

运行步骤及已知错误放行示例见 [离线 gate 演示](docs/demo_script.md#21-离线拒答-gate-演示)。

GitHub Actions 的 `test` 作业执行 Python 3.11 环境配置、依赖安装、`compileall`、Ruff 和核心定向测试。

Workflow：

- [.github/workflows/tests.yml](.github/workflows/tests.yml)

---

# 11. 文档索引

推荐阅读顺序：

1. [README.md](README.md) — 项目定位与能力总览；
2. [experiments/evals/README.md](experiments/evals/README.md) — TechQA 长期主评测与实验契约；
3. [RAG 冻结架构](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) — 当前冻结的检索 / 上下文工程决策；
4. [docs/architecture.md](docs/architecture.md) — 系统结构与边界；
5. [docs/agent_workflow.md](docs/agent_workflow.md) — 工单 Agent 预览 / 确认；
6. [docs/security.md](docs/security.md) — 当前认证边界提示与早期 MVP 安全记录；
7. `experiments/evals/reports/` — 检索 / Hybrid / 证据 / 生成实验产物。

`docs/*_report.md` 与 `docs/superpowers/` 中保留历史阶段报告、设计与实验计划，用于追溯项目演进；历史路线图 不自动代表当前产品方向。

---

# 12. 当前边界与非声明

当前项目对外表述保持以下边界：

- TechQA 是长期主技术支持语料与主评测基准，不再计划迁移到另一套 primary corpus；
- 不把离线 BM25 / RRF / Hybrid 实验或默认关闭的可选搜索写成默认线上已切换或公网已部署；
- portfolio-v1 的 Hybrid + 重排已冻结并接入可选搜索、默认关闭的实验问答及显式可选的工单预览检索；默认仍为 Dense，工单不使用 v5.x gate；R4 C1 的正式结论仍为 FAIL；
- 直接 Top14 是 TRAIN 开发阶段的选择，不是通用最优值；
- 局部扩展 / 结构扩展的负结果不能证明 Parent-Child 或文档结构方法普遍无效；
- 拒答策略和最终生成验证尚未冻结；
- 不把规则化 Ticket 分类写成自主 ReAct / 自主规划；
- 不把 Demo JWT + 租户范围 写成完整生产级 IAM / 多租户隔离；
- 不把 Docker Compose 写成生产部署；
- 不把 approval `pending` 校验写成并发恰好一次保证；
- 不把 G1 的条件性证据充分性改善写成生产生成效果提升，也不声称 G1 已替代 E1；
- 不针对 Q346 / Q492 已知 TRAIN 退化样本 做 样本-specific patch 后再把原 30-样本 样本当作 新的独立验证；
- 不声称当前系统已具备完整 多来源 / 冲突处理 / 自主 Agentic RAG 能力。

项目当前关注的是：

> **围绕企业技术支持中的“知识检索 → 回答 / 拒答 → 工单升级”建立可控业务闭环，并用同一 TechQA 主线持续回答“系统是否真的变好、失败在哪里、某次工程改动是否值得保留”。**

---

## Project Name

- 对外展示名：**Enterprise Support AI Copilot**
- 中文定位：**企业技术支持 RAG + 受控工单 Agent**
- 仓库：`Enterprise-Support-AI-Copilot-API`
