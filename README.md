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
| 重排 / Hybrid 实验 | Dense Top-100 + `qwen3-rerank` 正式冻结验证集对比；BM25 / RRF / Hybrid 为离线受控实验 |
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

> 当前流程不被描述为并发场景下的 严格的并发恰好一次副作用保证。

更多实现细节见 [docs/agent_workflow.md](docs/agent_workflow.md)。

---

# 3. RAG 在线运行时

当前在线 API 路径保持为 **Dense Chroma 检索**。

主要能力：

- `/rag/search`
- `/rag/ask`
- 文档 chunk 检索
- `tenant_id` / `category` 元数据过滤
- 结构化 sources 返回
- 无上下文与低相关拒答
- 检索日志

问答路径仅根据检索 上下文 生成答案，并返回对应 sources。当前低相关拒答使用 Dense Top-1 距离作为工程信号。

> **在线 / 离线边界：** BM25 / RRF / Hybrid / `qwen3-rerank` 当前用于 `experiments/evals/` 的离线评测与受控对照，不把它们描述成线上检索已切换到 Hybrid。

---

# 4. RAG 评测与迭代：从 Dense 基线到当前状态

`experiments/evals/` 是正式离线评测入口。当前在线 API 仍使用 **Dense Chroma 检索**；下面的 Hybrid、rerank、直接 Top14 属于离线评测与冻结工程候选，不等同于已上线能力。

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

R4 C1 的正式历史状态仍是 **FAIL**。后续冻结工程候选保留固定 Hybrid 路线，是基于整体证据做出的工程选择，不改写这次正式实验结论，也不代表当前在线 API 已切换到 Hybrid。

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
- 当前在线运行时：仍是 **Dense Chroma 检索**
- Hybrid + 重排 + 直接 Top14：**冻结工程候选，尚未推广到在线运行时**

---

## 4.6 当前拒答线：证据充分性

冻结检索 / 上下文 后，下一步不再沿用历史 `Dense Top1 distance > 0.9` 作为最终拒答契约，而是直接判断 **直接 Top14 是否包含足够证据**。

| 阶段 | 本次方案 | 结果 | 失败归因 | 结论 | 详情 |
| --- | --- | --- | --- | --- | --- |
| **拒答 v1.1** | `直接 Top14 → qwen3.5-plus 证据充分性分类` | 平衡准确率 **0.736**；充分证据召回率 97.1%；不足证据召回率仅 **50.0%** | 16 个不足证据样本 中有 8 个被错误放行为“充分” | **FAIL**，不进入生成与 运行时集成 | [v1.1 正式结果](experiments/evals/reports/refusal_evidence_sufficiency/phase_b_v1_1_result.md) |
| **拒答 v2.1** | 新分类契约 + 新的 AI 草稿代理标签 | 平衡准确率 **0.775**，门槛 0.80；充分召回 .85，不足召回 .70 | 只差 1 个 样本 即可过门槛，但后验检查发现部分 分歧 来自**标注契约与问题实际要求不一致**，不能简单继续调 提示词 | **FAIL**，不修改门槛后重跑 | [v2.1 结果](experiments/evals/reports/refusal_evidence_sufficiency/v2_ai_proxy_v2_1_result.md) |
| **拒答 v3：当前阶段** | 新选 80 条盲样本，只给 问题 + Top14，按 充分 / 不充分 / 存疑 重新标注 | 已冻结 AI 草稿：**44 充分 / 27 不充分 / 9 存疑** | 当前仍是开发用 AI 草稿标签，不是独立人工金标；尚无新的 v3 分类器 正式结果 | **拒答策略尚未冻结，也未接入 runtime** | [v3 当前冻结状态](experiments/evals/reports/refusal_evidence_sufficiency/v3_ai_draft_freeze.md) |

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

参考 `.env.example`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.5-plus
DATABASE_URL=sqlite:///data/todos.db
SQL_ECHO=true
DOCUMENT_STORAGE_ROOT=storage/documents
```

## 安装与数据库迁移

```bash
pip install -r requirements.txt
alembic upgrade head
```

## 启动 API

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

```bash
python scripts/smoke_agentops_flow.py
python scripts/smoke_document_backend_flow.py
```

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
6. [docs/security.md](docs/security.md) — 当前认证与权限边界；
7. `experiments/evals/reports/` — 检索 / Hybrid / 证据 / 生成实验产物。

`docs/*_report.md` 与 `docs/superpowers/` 中保留历史阶段报告、设计与实验计划，用于追溯项目演进；历史路线图 不自动代表当前产品方向。

---

# 12. 当前边界与非声明

当前项目对外表述保持以下边界：

- TechQA 是长期主技术支持语料与主评测基准，不再计划迁移到另一套 primary corpus；
- 不把离线 BM25 / RRF / Hybrid 实验写成已经上线的 Hybrid 检索；
- portfolio-v1 的 Hybrid + 重排已评测并冻结为工程候选，但尚未上线；R4 C1 的正式结论仍为 FAIL；
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
