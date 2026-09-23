# TechQA 评测总览

本目录是 Enterprise Support AI Copilot 的长期主评测入口，用于保存冻结实验、评测契约、失败归因与工程决策。这里记录的是离线证据，不是在线运行时配置。

## 当前冻结结论

### 当前在线运行时

当前在线 API 仍使用 **Dense Chroma 检索**。BM25、RRF、Hybrid、rerank 和直接 Top14 目前都属于离线评测与冻结工程候选，不描述为已上线能力。

### 冻结 DEV 对比

Dense Top100 + `qwen3-rerank` 的 冻结 DEV 结果：

| 指标 | Dense | Dense + rerank |
| --- | ---: | ---: |
| Recall@5 | 0.643750 | 0.725000 |
| Recall@20 | 0.818750 | 0.843750 |
| MRR@10 | 0.518931 | 0.560841 |

这是冻结验证集上的检索提升，不是线上效果声明。

### 冻结工程候选

```text
Dense chunk Top100 + BM25 chunk Top100
    → 等权 chunk RRF (k=60)
    → 融合 Top100
    → qwen3-rerank
    → 直接 Top14
```

R4 C1 Hybrid + rerank 的正式历史结论仍然是 **FAIL**：三项 TRAIN 聚合指标都上涨，但预注册的 MRR 门槛没有通过。后来保留这条固定 Hybrid 路线，是基于整体证据做出的工程冻结，不改写原实验结论，也不意味着已经上线。

### 尚未冻结的部分

- 证据充分性拒答策略；
- 最终生成验收协议；
- 最终 DEV 生成验证；
- 在线运行时集成；
- Ticket Agent 复用同一检索链。

历史 `dense_top1_distance > 0.9` 不自动成为最终拒答契约，因为冻结工程候选已经不再只有 Dense Top1 这一种检索信号。

---

## 数据与划分契约

TechQA 提供：

- 28,481 篇 Technote；
- 610 条可回答检索问题；
- 910 条生成 / 拒答问答记录，其中 610 条可回答、300 条不可回答；
- 每条可回答检索问题只有 1 个 相关文档。

| 数据划分 | 可回答 | 不可回答 | 用途 |
| --- | ---: | ---: | --- |
| `TRAIN_*` | 450 | 150 | 开发、失败归因、方案选择 |
| `DEV_*` | 160 | 150 | 冻结验证 |

TRAIN 是开发面，DEV 是冻结验证面。正式对比冻结后，不通过查看 individual DEV failure 来继续调参，也不把 TRAIN 调优结果包装成 DEV 提升。

数据身份和 SHA256 见：

- `datasets/techqa/manifest.json`
- `datasets/techqa/corpus_manifest.json`

## 检索指标契约

运行时检索 chunk，但 TechQA qrel 是文档级。因此正式 IR 评测会：

```text
保留原始 chunk 排序
→ 按 document_id 首次出现位置去重
→ 得到文档排序
→ 计算 Recall@5 / Recall@20 / MRR@10
```

TechQA 每条问题只有 1 个 相关文档，因此这里 Recall@K 与 Hit@K 数值相同。

---

## 实验决策表

| 阶段 | 核心问题 | 关键证据 | 正式结论 | 当前工程决策 |
| --- | --- | --- | --- | --- |
| E0 Dense | 基线有多强？ | 建立冻结 Dense 基线 | 基线 | 作为历史对照 |
| E1 Dense + rerank | 重排能否带来 冻结验证集提升？ | DEV R@5 .643750→.725000；R@20 .818750→.843750；MRR .518931→.560841 | **PASS** | 保留为最重要的冻结验证证据 |
| R4 C1 Hybrid + rerank | Hybrid + rerank 能否通过预注册门槛？ | TRAIN E1 .691111/.815556/.567206；Hybrid .702222/.831111/.570929 | **FAIL** | 固定 Hybrid 路线后来作为工程候选保留，但不继续调融合参数 |
| R1 证据审计 | 命中文档是否等于命中答案证据？ | 60 条人工标注，54 条有效 | Diagnostic | 增加证据级指标 |
| G1 文档内扩展 | 文档内展开能否改善最终上下文？ | 聚合指标上涨，但出现灾难性退化 | **NO_GO** | 不采用 |
| G2-A 重排后准入 | 把文档准入放到全局 rerank 之后是否更稳？ | 2 改善 / 25 持平 / 3 退化，其中 2 个灾难性退化 | **NO_GO** | 不采用 |
| 检索前沿冻结 | 是否还值得继续调检索参数？ | 只做 事后诊断 反事实分析，不再开启新的参数搜索 | **CLOSED** | 关闭 RRF-k、候选深度、权重、配额、每文档上限等调参 |
| 最终上下文预算 | 多大的直接 TopK 能达到当前证据命中上限？ | Top14 与 Top20 都是 answer 35/54、useful 43/54 | Selected | 选择直接 Top14 |
| 局部二次 rerank | 局部恢复能否救回残留证据？ | 答案证据 35→35；有用证据 43→43；0/7 可恢复样本 被救回 | 拒绝 | 不加第二次 rerank |
| 结构保持扩展 | 静态 parent / 文档结构扩展是否有净收益？ | 答案证据 35→32；有用证据 43→38；2 未命中→命中，5 命中→未命中 | 拒绝 | 不采用 parent expansion |
| 失败归因 | 结构扩展为什么退化？ | 5/5 答案证据退化 = 上下文预算挤占；2/2 收益 = 整文档恢复；文档数中位数 12→4.5 | 已确认 | 保留直接 Top14 |
| 检索 / Context 冻结 | 当前工程候选是什么？ | 固定检索路线 + `flat_rerank_top14_v1` | **CLOSED** | 停止当前检索与上下文研究线 |

结构扩展的负结果只说明：在当前固定上下文预算和这套静态 parent expansion 下，增加同文档深度会挤占跨文档覆盖。它不证明 Parent-Child、small-to-big 或文档结构方法普遍无效。

---

## 历史生成 Context

`document_aware_forward_expansion_v1` 是真实存在过的历史 生成评测 Context 策略，相关产物仍保留；它不是当前冻结的最终 Context。

当前冻结 Context 是 `flat_rerank_top14_v1`。G1 / G2-A 都是历史 TRAIN 实验，正式结论为 NO_GO。

---

## 评测演进链

```text
冻结 TechQA 契约
  → E0 Dense
  → E1 Dense + rerank：DEV 明显提升
  → BM25 / RRF 互补性验证
  → R4 C1：正式 FAIL
  → 证据级人工审计
  → G1 / G2：NO_GO
  → 检索前沿关闭
  → 直接 TopK 上下文预算分析
  → 选择 Top14
  → 局部二次 rerank：拒绝
  → 结构保持扩展：拒绝
  → 归因为上下文预算挤占
  → 检索 / Context 冻结
```

`retrieval_parameter_research = CLOSED`，`context_assembly_research = CLOSED`。只有出现当前冻结范围之外的新失败模式，才重新开启这一研究线。

---

## 关键产物索引

- 当前冻结架构：[`reports/portfolio_v1_rag_freeze/architecture_freeze.md`](reports/portfolio_v1_rag_freeze/architecture_freeze.md)
- 机器可读冻结契约：`reports/portfolio_v1_rag_freeze/freeze.json`
- 证据演进时间线：[`reports/portfolio_v1_rag_freeze/evidence_timeline.md`](reports/portfolio_v1_rag_freeze/evidence_timeline.md)
- E1 冻结 DEV：[`reports/e1_rerank/comparison.md`](reports/e1_rerank/comparison.md)
- R4 C1：[`reports/r4_c1_hybrid_rerank/`](reports/r4_c1_hybrid_rerank/)
- 证据审计：[`reports/r1_evidence_audit/`](reports/r1_evidence_audit/)
- G1 / G2：[`reports/g1_document_local/`](reports/g1_document_local/) / [`reports/g2_rerank_informed_admission/`](reports/g2_rerank_informed_admission/)
- 检索前沿审计：[`reports/retrieval_frontier_freeze/`](reports/retrieval_frontier_freeze/)

---

## 评测边界

当前明确不做以下声明：

- 不把离线 Hybrid + rerank 描述为已经上线；
- 不改写 R4 C1 的正式 FAIL；
- 直接 Top14 是 TRAIN 开发阶段选择，不是通用最优值；
- 局部 / 结构扩展的负结果不等于这些方法普遍无效；
- 拒答策略和最终生成验证尚未冻结。

应用架构与 Agent 工作流见仓库根目录 [README](../../README.md)。
