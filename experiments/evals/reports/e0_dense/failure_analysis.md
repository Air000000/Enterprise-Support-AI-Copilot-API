# E0 Dense：TRAIN 失败归因

范围：仅 TRAIN  
基线：E0 Dense Retrieval + E0 Generation  
日期：2026-08-24  
目的：用可测量的失败证据选择一个 E1 优化方向；DEV 保持冻结。

## 1. 基线结果

450 条可回答 TRAIN 检索问题：

| 指标 | 数值 |
| --- | ---: |
| Document Recall@5 | 0.613333 |
| Document Recall@20 | 0.740000 |
| MRR@10 | 0.510477 |
| p50 检索延迟 | 991.995 ms |
| p95 检索延迟 | 1819.083 ms |

E0 generation 在 600 条 TRAIN 上的自动 correctness 均值为 0.312，但这个分数只作为筛选信号使用，因为人工校准显示自动 judge 与人工判断并非完全一致。Faithfulness 也只作为辅助指标。

## 2. 查询尾部空白漂移

历史 retrieval 和 generation 数据中，同一个问题有时只是在末尾多了空格或换行。

- 250 / 450 条可回答问题存在这种尾部差异；
- 74 条 Top3 排序因此发生变化；
- 5 条 gold admission 因此发生变化。

这不是语义变化，所以这 5 条不用于后续因果失败计数。之后 provider 调用前统一对 query 做 `rstrip()`。

## 3. 检索失败分桶

在排除上述 5 条漂移 case 后，确定性排名证据为：

- relevant document rank 4–5：20 条
- relevant document rank 6–20：57 条
- 因此 candidate 已在较大池中、但排序不足的明确 case：**77 / 450**
- Top20 miss：117 条

这 77 条直接满足“候选已经召回，只是排得不够前”的条件，因此是非常干净的 rerank 假设来源。

## 4. Top20 miss 人工审计

对 117 个 Top20 miss 固定抽样 30 条人工检查：

- qrel / 问题本身歧义：17
- 明显词法型 miss：7
- 语义间接匹配型 miss：6

结论：

- error code、版本号、CVE、产品标识等精确 token 的确构成一类真实失败；
- 但在这批诊断样本里，词法失败不是主导问题；
- 很多表面上的 Dense miss 受到 qrel 不完整或 gold 本身可疑的影响；
- 因此不能仅凭 117 个 Top20 miss 就直接把 BM25 / Hybrid 作为 E1。

## 5. gold 已进入 Context 但 correctness 低

另抽取 30 条“gold 已在 Context，但自动 correctness 较低”的 case：

| 主要归因 | 数量 |
| --- | ---: |
| 评测 / 参考答案不匹配 | 14 |
| Context 证据覆盖不足 | 12 |
| 生成模型明显误用证据 | 2 |
| 生成不完整 | 2 |

这说明：

- 单纯 prompt / model 优化不是当时最强问题；
- 很多“生成错”其实来自证据没覆盖完整；
- 评测 reference 本身也需要审计。

## 6. Impossible case 审计

150 条 impossible case 的人工复核发现：

- 70 条语义上确实拒答；
- 58 条其实语料中存在可支持答案，虽然 benchmark 标成 impossible；
- 21 条完全匹配预期拒答；
- 1 条确认是不安全回答。

因此 benchmark 的 impossible 标签不能直接等同于“回答即幻觉”。

## 7. E1 选择

候选优化方向包括：

- rerank
- Hybrid / BM25
- prompt / generation
- chunk / Context 策略
- refusal / grounding

最终选择 **rerank** 作为 E1，原因不是它“看起来先进”，而是：

1. 有 **77/450** 个明确 candidate-in-pool ranking case；
2. 这是可确定、可复现、最容易做单变量验证的失败类；
3. Top20 miss 人工审计并不支持立即把 BM25 作为首个正式优化；
4. generation 低分里大量问题实际来自评测 reference 或 Context coverage，而不是单纯生成模型。

因此 E1 假设为：

> 在固定 Dense Top100 候选池上增加 reranker，应提升早期文档排序，尤其是 Recall@5 与 MRR@10，同时保留 Recall@20 作为安全边界。

最终是否成立，由 frozen DEV 对比决定。
