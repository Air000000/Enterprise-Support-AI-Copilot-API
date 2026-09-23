# RAG 冻结证据时间线

本文件按“问题 → 证据 → 决策”记录从 Dense 基线到当前冻结工程候选的关键节点。

## E0：Dense 基线

**问题：** TechQA 冻结契约上的起点是什么？

**证据：** 建立 Dense 基线。

**决策：** 保留为历史对照。

---

## E1：Dense + rerank

**问题：** rerank 能否带来 frozen DEV 提升？

**证据：**

- Recall@5：0.643750 → 0.725000
- Recall@20：0.818750 → 0.843750
- MRR@10：0.518931 → 0.560841

**决策：** 保留为当前最强的 held-out 检索提升证据。

---

## R4 C1：Hybrid + rerank

**问题：** Hybrid + rerank 能否通过预注册晋级门槛？

**TRAIN 证据：**

- E1：R@5 0.691111，R@20 0.815556，MRR 0.567206
- C1：R@5 0.702222，R@20 0.831111，MRR 0.570929
- Recall@20 门槛：PASS
- MRR@10 门槛：FAIL

**决策：** 正式实验仍为 **FAIL**。固定 Hybrid 路线后来被保留为工程候选，但不继续调融合参数。

---

## R1：证据级人工审计

**问题：** 命中正确文档是否等于答案证据真的进入高位 Context？

**证据：** 对 TRAIN 子集人工标注 weak / useful / answer-bearing chunk。

**决策：** 文档 Recall/MRR 继续保留用于 benchmark 对比，同时增加证据级命中指标。

---

## G1 / G2：文档内扩展与文档准入

**问题：** 更大范围的文档内重建或 rerank 后准入能否改善最终证据？

**证据：** 两条路线都出现聚合指标变化，但也都出现预注册不可接受退化。

**决策：** G1、G2-A 均为 **NO_GO**。

---

## 检索前沿关闭

**问题：** 是否值得继续搜索 RRF k、候选深度、来源权重等参数？

**证据：** post-hoc 审计显示 Dense / 其他排序存在互补，但当前设计样本已被用于诊断。

**决策：** 当前检索参数研究关闭，不继续在已见 TRAIN case 上调参。

---

## 最终上下文预算

**问题：** 多大的直接 TopK 能达到当前人工证据审计中的 Top20 上限？

**证据：**

- Top14：answer 35/54，useful 43/54
- Top20：answer 35/54，useful 43/54

**决策：** 选择 `flat_rerank_top14_v1`。

---

## 局部二次 rerank

**问题：** rerank 后的局部恢复能否救回残留 answer-bearing evidence？

**证据：**

- answer：35 → 35
- useful：43 → 43
- 可操作 residual：0/7 被救回
- 增加额外 rerank 延迟和模型调用成本

**决策：** 拒绝，保留直接 Top14。

---

## 结构保持扩展

**问题：** 在同等上下文字符预算下，静态整文档 / 结构恢复是否有净收益？

**证据：**

- answer：35 → 32
- useful：43 → 38
- miss→hit：2
- hit→miss：5

**决策：** 净退化，拒绝。

---

## 失败归因

**问题：** 结构保持方案为什么退化？

**证据：**

- 5/5 answer regression = `BUDGET_CROWD_OUT`
- 2/2 gain = `WHOLE_DOCUMENT_RECOVERY`
- 基线 Context 中 unique documents 中位数：12
- challenger：4.5

**结论：** 在当前固定预算下，增加同文档深度会减少跨文档广度；本次审计中损失大于收益。

---

## 当前冻结工程候选

```text
Dense Top100 + BM25 Top100
    → 等权 chunk RRF (k=60)
    → fused Top100
    → qwen3-rerank
    → 直接 Top14
```

当前状态：

- 检索参数研究：**CLOSED**
- Context 组装研究：**CLOSED**
- 拒答 / 证据充分性：尚未冻结
- 最终生成验收：尚未完成
- 在线运行时集成：尚未完成
- Ticket Agent 共享检索：尚未完成
