# Enterprise Support AI Copilot

[![Tests](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Air000000/enterprise-support-ai-copilot-api/actions/workflows/tests.yml)

面向**企业技术支持（Technical Support）**场景，构建知识检索、回答 / 拒答与工单升级一体化的 **RAG + 受控工单 Agent 后端**。

系统首先从技术支持知识库检索相关证据，返回带来源的回答或在上下文不足时拒答；对于需要进一步处理的问题，通过工单分类、预览和人工确认后再执行真实建单，并以 AgentOps 记录关键运行与审批链路。这里的目标不是让模型直接接管业务状态，而是把知识问答与后续问题升级连接成一条可控、可追踪的技术支持流程。

在这条应用链路之上，项目长期以 **TechQA** 作为核心技术支持语料与统一评测基准，持续建立检索、生成、拒答与失败归因闭环，用受控评测回答“系统是否真的变好、失败在哪里、某次工程改动是否值得保留”。

> **当前定位：** TechQA 是主技术支持语料与长期主评测基准，不再把它视为迁移到另一套主数据集之前的临时 Phase。未来若增加 multi-source / conflict / agentic stress 测试，只作为补充评测，不替换现有 TechQA 主线。

---

## 30 秒看项目

```text
Technical Support Request
          │
          ▼
   Current Runtime:
 Dense Chroma Retrieval
          │
          ├──────────────► Answer + Sources
          │
          └──────────────► Refusal when context is insufficient
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
      create_ticket
          │
          └──────────────► AgentOps Trace
                           ├─ Agent Run
                           ├─ Tool Call
                           ├─ Approval
                           └─ Retrieval Log / Metrics

Evaluation & Iteration
────────────────────────────────────────────────────────────
                         TechQA
        28,481 Technotes / 610 retrieval queries
         910 generation & abstention QA records
                           │
                           ▼
                   Offline Evaluation
                  Dense / Rerank / Hybrid
                  Evidence-level Audit
                  Generation / Abstention
                           │
                           ▼
                    Failure Diagnosis
                           │
                           ▼
                     System Iteration
                           │
                           └────► RAG / context policy / evaluation loop
```

### Runtime / Portfolio-v1 boundary

- **Current runtime:** Dense Chroma Retrieval.
- **Frozen portfolio-v1 integration target:** Dense100 + BM25100 -> RRF60 -> fused100 -> `qwen3-rerank` -> Flat Top14.
- **Integration status:** not yet promoted into online runtime.
- **Refusal / generation:** not yet frozen.

### 当前核心能力

| 能力 | 当前实现 |
| --- | --- |
| RAG Runtime | Chroma Dense Retrieval、tenant/category filter、sources、低相关拒答 |
| Controlled Ticket Agent | `search_kb` / `classify_ticket` / `create_ticket`，preview-confirm + Human-in-the-loop |
| Primary Technical Support Data | TechQA 28,481 Technotes、610 条 answerable retrieval queries、910 条 generation/abstention QA |
| RAG Evaluation | Frozen TRAIN / DEV、Document Recall@K / MRR、generation / abstention harness |
| Rerank / Hybrid Research | Dense Top-100 + `qwen3-rerank` 正式 held-out 对照；BM25 / RRF / Hybrid 为离线受控实验 |
| Failure Diagnosis | candidate coverage、chunk crowding、evidence-level audit、route-selection gate |
| AgentOps | Agent Run / Tool Call / Approval / Retrieval Trace 与聚合指标 |
| Engineering | Alembic、Pytest、Ruff、GitHub Actions、Docker Compose、Smoke |

---

# 1. TechQA：核心技术支持语料与统一评测基准

TechQA 不是单独外挂的评测数据集，而是当前项目的数据与评测主线。

## Retrieval corpus

- **28,481** 篇 Technote 技术支持文档；
- **610** 条 answerable retrieval queries；
- 每条 query 恰好 1 个 relevant document；
- qrels 为 document-level；
- 实际 retriever 返回 chunk，因此正式 IR 评测会先保留原始 chunk ranking，再按 `document_id` 首次出现位置 collapse 成 document ranking。

## Generation / abstention set

- **610** 条 answerable；
- **300** 条 impossible；
- 共 **910** 条 QA records。

这使同一 technical-support domain 可以连续支撑：

```text
Retrieval
  ↓
Rerank / Hybrid comparison
  ↓
Evidence quality diagnosis
  ↓
Generation correctness / faithfulness
  ↓
Abstention / hallucination evaluation
```

完整数据版本、SHA256、split 和评测契约见：

- [experiments/evals/README.md](experiments/evals/README.md)
- [experiments/evals/datasets/techqa/manifest.json](experiments/evals/datasets/techqa/manifest.json)

---

# 2. Controlled Ticket Agent

Ticket Agent 的核心目标不是让模型直接修改业务状态，而是将预览 / 审批阶段与真实写操作分离。

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

当前使用三个业务工具语义：

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

只有以下条件全部成立时才创建真实工单：

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

> 当前流程不被描述为并发场景下的 exactly-once side-effect guarantee。

更多实现细节见 [docs/agent_workflow.md](docs/agent_workflow.md)。

---

# 3. Enterprise RAG Runtime

当前在线 / API serving 路径保持为 **Dense Chroma Retrieval**。

主要能力：

- `/rag/search`
- `/rag/ask`
- 文档 chunk 检索
- `tenant_id` / `category` metadata filter
- 结构化 sources 返回
- 无上下文与低相关拒答
- retrieval logging

问答路径仅根据检索 Context 生成答案，并返回对应 sources。当前低相关拒答使用 Dense Top-1 distance 作为工程信号。

> **在线 / 离线边界：** BM25 / RRF / Hybrid / `qwen3-rerank` 当前用于 `experiments/evals/` 的离线评测与受控对照，不把它们描述成线上 serving 已切换到 Hybrid Retrieval。

---

# 4. RAG 评测与迭代：从 Dense 基线到当前状态

`experiments/evals/` 是正式离线评测入口。当前在线 API 仍使用 **Dense Chroma Retrieval**；下面的 Hybrid、rerank、Flat Top14 属于离线评测与冻结工程候选，不等同于已上线 serving。

## 4.1 评测契约

| Split | Answerable | Impossible | 用途 |
| --- | ---: | ---: | --- |
| TRAIN | 450 | 150 | 开发、失败归因、方案选择 |
| DEV | 160 | 150 | 冻结验证，只用于正式 held-out 对比 |

TechQA 的 qrel 是文档级，而检索器返回 chunk。正式 IR 评测会保留原始 chunk 排序，再按 `document_id` 首次出现位置去重成文档排序，计算 Document Recall@5、Recall@20 和 MRR@10。每条 answerable query 只有 1 个 relevant document，因此这里 Recall@K 与 Hit@K 数值相同。

---

## 4.2 核心检索迭代

| 阶段 | 为什么做 / 本次改动 | 评测口径 | 核心结果 | 失败归因与决策 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **E0：Dense 基线** | `Query → Dense Top100 chunks → 文档去重排序`，先建立统一基线 | TRAIN + frozen DEV | **TRAIN**：R@5 61.3%，R@20 74.0%，MRR 0.510；**DEV**：R@5 64.4%，R@20 81.9%，MRR 0.519 | 总指标只能说明效果不足，不能区分“候选没召回”和“候选已召回但排序靠后”，因此先做失败归因 | [E0 失败分析](experiments/evals/reports/e0_dense/failure_analysis.md) |
| **E0：失败归因** | 对失败按 gold 文档排名分桶，再抽固定样本人工审计 | TRAIN 450 条 answerable | 排除 5 条仅由查询文本尾部空白差异造成的跨运行漂移后：rank 4–5 有 **20** 条，rank 6–20 有 **57** 条，共 **77/450** 个明确排序问题；另有 117 个 Top20 miss | 30/117 个 miss 中：17 个 qrel / 问题歧义、7 个明显词法 miss、6 个语义 miss。另抽 30 个低正确率 case：14 个评测/reference 问题、12 个证据覆盖问题，仅 4 个明显生成问题。**排序问题是当时最强、最确定的可操作失败类，因此 E1 先做 rerank** | [完整归因与样本审计](experiments/evals/reports/e0_dense/failure_analysis.md) |
| **E1：Dense + rerank** | `Dense Top100 → qwen3-rerank → 文档去重排序`；**只加 reranker** | frozen DEV 正式对比 | R@5 **64.4% → 72.5%**；R@20 **81.9% → 84.4%**；MRR **0.519 → 0.561** | DEV Top5：21 个改善、8 个退化；Top20：5 个改善、1 个退化。rerank 有明显净收益，但不是单调改善。冻结 DEV 不用于逐 case 反向调参，因此没有针对这 8 个 DEV 退化继续做正式根因拟合 | [E0 / E1 对比](experiments/evals/reports/e1_rerank/comparison.md) |
| **R3：BM25 互补性验证** | 剩余失败里出现 error code、版本号、CVE、固定技术词等精确词法查询，因此加入 BM25，用 RRF 验证候选互补性 | TRAIN，文档级互补性验证 | Dense hit@100 **387**，BM25 **375**，融合 **402**；救回 Dense miss **19** 个，净增 **15** 个 | BM25 单独不优于 Dense，但确实补回一批 Dense 漏掉的精确词法候选，因此允许进入正式 Hybrid + rerank 实验 | [R3 准入结果](experiments/evals/reports/r3_hybrid/admission_decision.md) |
| **R4 C1：Hybrid + rerank** | `Dense100 + BM25100 → RRF60 → fused100 → qwen3-rerank`；与 **E1 TRAIN** 对比 | TRAIN 450 条 | E1：R@5 / R@20 / MRR = **.691 / .816 / .567**；C1：**.702 / .831 / .571** | 三项都上涨，但预注册 MRR 门槛为 **.577**，实际只有 .571。最终 Top20 miss 中 **56 个候选缺失、20 个排序不足**；固定 Top100 融合预算既会救回候选，也会压掉部分单源尾部候选。**正式结论 FAIL，不继续调 RRF k、权重、深度等参数** | [正式对比](experiments/evals/reports/r4_c1_hybrid_rerank/comparison.md) · [失败案例与归因](experiments/evals/reports/r4_c1_hybrid_rerank/postmortem_decision.md) |

> **查询文本尾部空白漂移：** 历史 retrieval 与 generation 数据中，有些问题语义完全相同，只在末尾多了空格或换行。250/450 条 answerable query 存在这种差异，其中 74 条 Top3 排序发生变化、5 条 gold admission 发生变化。由于这不是语义变化，这 5 条不用于因果失败计数；后续 provider 调用前统一对 query 做 `rstrip()`。

R4 C1 的正式历史状态仍是 **FAIL**。后续冻结工程候选保留固定 Hybrid 路线，是基于整体证据做出的工程选择，不改写这次正式实验结论，也不代表当前在线 API 已切换到 Hybrid。

---

## 4.3 从“命中文档”到“命中答案证据”

Document Recall 只能回答“相关文档是否出现”，但 RAG 真正交给生成模型的是 chunk，因此还需要回答：

> **命中了 gold document，真正能回答问题的 chunk 是否进入了高位 Context？**

### 人工证据标注

从 TRAIN answerable 中确定性抽取 60 条，在 gold document 内定位候选 chunk，共人工标注 187 个 chunk：

| 标签 | 含义 | 评测中的作用 |
| --- | --- | --- |
| `0 = weak` | 相关性弱，不能作为有效回答证据 | 不计入 Useful / Answer hit |
| `1 = useful` | 对解决问题有帮助，但本身不直接承载完整答案 | 计入 UsefulEvidenceHit |
| `2 = answer-bearing` | 直接承载回答问题所需的关键证据 | 同时计入 UsefulEvidenceHit 与 AnswerEvidenceHit |

60 条中有 6 条被标记为 `questionable_gold=true`：gold 文档与问题主题不匹配、条件冲突，或不足以支撑 gold answer，因此正式证据指标只评 **54 条**。

[人工标签](experiments/evals/reports/r1_evidence_audit/evidence_labels.jsonl) · [证据指标](experiments/evals/reports/r1_evidence_audit/evidence_metrics.json)

### 同一批人工标签上的三条排序链

| 排序链 | AnswerEvidenceHit@5 | AnswerEvidenceHit@20 | UsefulEvidenceHit@5 | UsefulEvidenceHit@20 |
| --- | ---: | ---: | ---: | ---: |
| **E0 Dense** | 44.4% | 61.1% | 63.0% | 77.8% |
| **C1 融合后、rerank 前** | 46.3% | 53.7% | 68.5% | 77.8% |
| **C1 rerank 后** | **53.7%** | **64.8%** | **72.2%** | **79.6%** |

这里比较的不是三套不同人工标签，而是**用同一批人工 chunk 标签去评三条不同的 chunk ranking**。这也解释了为什么上面不能只写 “Dense 44.4% → C1 rerank 53.7%”：中间的“融合后、rerank 前”本身也是一条被测链路，而且它在 AnswerEvidenceHit@20 上反而从 Dense 的 61.1% 降到 53.7%，随后 reranker 又把它提升到 64.8%。

因此后续实验不再只看文档 Recall/MRR，还单独追踪“真正能回答问题的证据是否进入高位 Context”。

---

## 4.4 上下文策略：G1 / G2

### 历史 E1 Context 对照

G1 并不是拿“Top5 文档”去对比 E1。历史 E1 generation harness 的 Context 策略是：

```text
Dense Top100
→ 全局 qwen3-rerank
→ rerank Top3 chunk 作为 anchor
+ Dense rank1 rescue
→ 每个 anchor 最多带 3 个向后相邻 chunk
→ 去重
→ 最多 16 chunks
```

它的特点是：**先让全局 reranker 看完整个 Dense Top100，再围绕高位 anchor 做局部扩展。**

### G1 / G2 对比

| 阶段 | 为什么做 / 本次改动 | 核心结果 | 失败案例与归因 | 决策 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **G1：文档内证据扩展** | `Dense 排名 → 前5篇 unique docs → 展开文档内 chunks → 合并 rerank → Top16`。目标是解决“关键证据藏在同一文档更远位置”的问题 | 30 条：COMPLETE **24→28**；Macro Claim Coverage **0.825→0.956**；6 wins / 22 ties / 2 losses | **Q346**：关键 chunk 原 Dense rank 72，历史 E1 的全局 reranker 本来能在裁剪前把它救上来；G1 却先按 Dense 只保留5篇文档，关键文档在 rerank 前已被永久裁掉。Q492 另有连续证据丢失 | **NO_GO**：平均提升明显，但出现 COMPLETE→INSUFFICIENT 的灾难性退化 | [G1 结果与 Q346/Q492 分析](experiments/evals/reports/g1_document_local/final_decision.md) |
| **G2-A：rerank 后再做文档准入** | `Dense Top100 → 全局 rerank → 前5篇 unique docs → 文档内展开 → 合并 rerank → Top16`；**只改文档准入顺序** | fresh TRAIN 30 条：**2 wins / 25 ties / 3 losses**；COMPLETE 19→19；Macro Claim Coverage **.667→.650** | Q287/Q578 被全局 rerank 救回，但 Q090/Q500 又因新的前5篇预算把原本有效文档挤出去。主因是**准入排序不稳定 + 固定5篇预算放大这种不稳定** | **NO_GO**，不根据已经看过的失败 case 继续补规则 | [G2 五个变化 case 逐条归因](experiments/evals/reports/g2_rerank_informed_admission/post_hoc_forensic_analysis.md) |

**Macro Claim Coverage**：先把标准答案拆成若干关键事实点，计算每条问题的事实覆盖率，再对问题等权平均。例如两个问题分别覆盖 100% 和 25%，Macro Claim Coverage 为 62.5%。因此 G1 的 0.956 是**平均关键事实覆盖率**，不是“95.6% 准确率”。

---

## 4.5 上下文收口：为什么冻结为 Flat Top14

### Flat TopK 是什么

**Flat TopK** 指：reranker 输出一条全局 chunk 排名后，**直接取前 K 个 chunk 作为最终 Context**。这里的 “Flat” 表示不再做层级式或局部式扩展：不追加 sibling、不展开整篇文档、不做硬性文档准入，也不再进行第二次 rerank。

当前冻结候选使用：

```text
reranked chunks
→ 直接取前 14 个
→ final Context
```

历史 E1 / G1 实验曾使用最多 16 个 chunk；当前冻结工程候选是 **Flat Top14**，两者不是同一个上下文策略。

| 阶段 | 为什么做 / 本次改动 | 核心结果 | 失败归因与决策 | 详情 |
| --- | --- | --- | --- | --- |
| **检索前沿审计** | G1/G2 都说明不同排序存在互补，但继续在已看过的 TRAIN case 上调 K、RRF、权重容易过拟合，因此只对冻结结果做零 provider 的反事实分析 | 观察到 Dense 与全局 rerank 确实存在互补，document-level RRF 是未来可能方向 | 这些 30 条已经是设计/诊断数据，不能继续拿来证明新方案有效。**关闭当前检索参数搜索** | [前沿审计](experiments/evals/reports/retrieval_frontier_freeze/final_offline_frontier_audit.md) |
| **最终上下文预算** | 比较不同 Flat TopK，找达到当前证据命中上限的最小 K | 54 条证据样本：**Top14 = answer 35/54、useful 43/54；Top20 完全相同** | 从14增加到20没有新增 evidence hit，只增加上下文。**选择 Flat Top14**；这是当前 TRAIN 开发集选择，不是通用最佳 K | [架构冻结](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |
| **局部二次 rerank** | 尝试对 Top14 未覆盖、且具备局部恢复条件的残留 case 再做局部候选恢复与第二次 rerank | answer **35→35**；useful **43→43**；**7 个具备局部恢复条件的 residual，0/7 被救回** | 没有新增 evidence hit，还增加一次 rerank；p50 约 661 ms。**拒绝** | [架构冻结 §4.1](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |
| **结构保持扩展** | 尝试通过同文档 / 结构扩展恢复被固定 chunk 切分打散的信息 | answer **35→32**；useful **43→38**；2 个 miss→hit，但 5 个 hit→miss | 5/5 退化都来自**上下文预算挤占**：同一文档放入更多内容后，跨文档 evidence 被挤掉；Context 中文档数中位数 **12→4.5**。**拒绝，保留 Flat Top14** | [架构冻结 §4.2–5](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) |

最终冻结的检索 / Context 工程候选：

```text
Dense Top100
+
BM25 Top100
    ↓
equal-weight RRF (k=60)
    ↓
fused Top100
    ↓
qwen3-rerank
    ↓
Flat Top14
```

- 检索参数研究：**CLOSED**
- Context 组装研究：**CLOSED**
- 当前在线 runtime：仍是 **Dense Chroma Retrieval**
- Hybrid + rerank + Flat Top14：**冻结工程候选，尚未推广到在线 runtime**

---

## 4.6 当前拒答线：证据充分性

冻结检索 / Context 后，下一步不再沿用历史 `Dense Top1 distance > 0.9` 作为最终拒答契约，而是直接判断 **Flat Top14 是否包含足够证据**。

| 阶段 | 本次方案 | 结果 | 失败归因 | 结论 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **拒答 v1.1** | `Flat Top14 → qwen3.5-plus 证据充分性分类` | Balanced Accuracy **0.736**；充分证据召回 97.1%；不足证据召回仅 **50.0%** | 16 个不足证据 case 中有 8 个被错误放行为“充分” | **FAIL**，不进入生成与 runtime 集成 | [v1.1 正式结果](experiments/evals/reports/refusal_evidence_sufficiency/phase_b_v1_1_result.md) |
| **拒答 v2.1** | 新分类契约 + fresh AI 草稿代理标签 | Balanced Accuracy **0.775**，门槛 0.80；充分召回 .85，不足召回 .70 | 只差 1 个 case 即可过门槛，但后验检查发现部分 disagreement 来自**标注契约与问题实际要求不一致**，不能简单继续调 prompt | **FAIL**，不修改门槛后重跑 | [v2.1 结果](experiments/evals/reports/refusal_evidence_sufficiency/v2_ai_proxy_v2_1_result.md) |
| **拒答 v3：当前阶段** | 新选 80 条盲样本，只给 Question + Top14，按 sufficient / insufficient / questionable 重新标注 | 已冻结 AI 草稿：**44 sufficient / 27 insufficient / 9 questionable** | 当前仍是开发用 AI 草稿标签，不是独立人工金标；尚无新的 v3 classifier 正式结果 | **拒答策略尚未冻结，也未接入 runtime** | [v3 当前冻结状态](experiments/evals/reports/refusal_evidence_sufficiency/v3_ai_draft_freeze.md) |

完整实验契约、索引与冻结 artifact 见 [experiments/evals/README.md](experiments/evals/README.md)。

---

# 5. Document Lifecycle

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

# 6. AgentOps / Observability

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

- Agent Run 状态；
- Tool Call 成功 / 失败与 error type；
- Approval 状态；
- Retrieval no-context / refused / failed；
- Retrieval source distribution；
- 单次 Run Trace。

---

# 7. Authentication / Tenant Scope

项目包含用于工程验证的 Demo JWT Auth：

- Bearer token；
- `user_id`；
- `tenant_id`；
- `role`；
- `support` / `admin` 角色检查；
- tenant-scoped Document / AgentOps / RAG 访问。

> 这里验证的是认证上下文和 tenant scope 在应用链路中的传递，不把它描述为完整生产级 IAM / RBAC 或数据库级多租户隔离方案。

安全边界见 [docs/security.md](docs/security.md)。

---

# 8. Project Structure

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

- `rag_runtime/`：正式在线 RAG runtime；
- `experiments/evals/`：TechQA 主评测、受控实验与 artifacts；
- `experiments/rag_local/`：早期兼容入口；
- Todo / AI Todo 路径保留为历史兼容，不作为当前项目定位。

---

# 9. Quick Start

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

## Install & migrate

```bash
pip install -r requirements.txt
alembic upgrade head
```

## Run API

```bash
uvicorn main:app --reload
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

---

# 10. Tests / CI

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

```bash
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

GitHub Actions `test` job 执行 Python 3.11 setup、dependency install、`compileall`、Ruff 和核心 focused tests。

Workflow：

- [.github/workflows/tests.yml](.github/workflows/tests.yml)

---

# 11. Documentation

推荐阅读顺序：

1. [README.md](README.md) — 项目定位与能力总览；
2. [experiments/evals/README.md](experiments/evals/README.md) — TechQA 长期主评测与实验契约；
3. [Portfolio-v1 RAG Architecture Freeze](experiments/evals/reports/portfolio_v1_rag_freeze/architecture_freeze.md) — 当前冻结的 retrieval / context 工程决策；
4. [docs/architecture.md](docs/architecture.md) — 系统结构与边界；
5. [docs/agent_workflow.md](docs/agent_workflow.md) — Ticket Agent preview / confirm；
6. [docs/security.md](docs/security.md) — 当前认证与权限边界；
7. `experiments/evals/reports/` — retrieval / hybrid / evidence / generation artifacts。

`docs/*_report.md` 与 `docs/superpowers/` 中保留历史阶段报告、设计与实验计划，用于追溯项目演进；历史 roadmap 不自动代表当前产品方向。

---

# 12. Current Scope / Non-Claims

当前项目对外表述保持以下边界：

- TechQA 是长期主技术支持语料与主评测基准，不再计划迁移到另一套 primary corpus；
- 不把离线 BM25 / RRF / Hybrid 实验写成线上 Hybrid Serving；
- portfolio-v1 Hybrid+rereank is evaluated/frozen, not serving; R4 C1 remains formal FAIL；
- Flat Top14 is a TRAIN-development selection, not a universal optimum；
- locality/structure-expansion negatives do not prove Parent-Child or document structure generally ineffective；
- refusal policy and final generation validation are not yet frozen；
- 不把规则化 Ticket 分类写成自主 ReAct / autonomous planning；
- 不把 Demo JWT + tenant scope 写成完整生产级 IAM / multi-tenant isolation；
- 不把 Docker Compose 写成生产部署；
- 不把 approval `pending` 校验写成并发 exactly-once guarantee；
- 不把 G1 conditional evidence-sufficiency 改善写成 production generation uplift，也不声称 G1 已替代 E1；
- 不针对 Q346 / Q492 已知 TRAIN regression 做 case-specific patch 后再把原 30-case 样本当作 fresh validation；
- 不声称当前系统已具备完整 multi-source / conflict-resolution / autonomous Agentic RAG 能力。

项目当前关注的是：

> **围绕企业技术支持中的“知识检索 → 回答 / 拒答 → 工单升级”建立可控业务闭环，并用同一 TechQA 主线持续回答“系统是否真的变好、失败在哪里、某次工程改动是否值得保留”。**

---

## Project Name

- 对外展示名：**Enterprise Support AI Copilot**
- 中文定位：**企业技术支持 RAG + 受控工单 Agent**
- Repository：`Enterprise-Support-AI-Copilot-API`
