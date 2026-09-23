# RAG 冻结架构

状态：**portfolio-v1 已冻结**
范围：检索候选 + 上下文组装
在线集成：**尚未完成**

本文件记录 TechQA 检索 / Context 研究线最终形成的工程决策。它不改写历史实验结论，也不表示冻结候选已经上线。

## 1. 当前在线运行时与冻结目标

当前在线 API：

```text
Dense Chroma 检索
```

冻结工程候选：

```text
Dense chunk Top100
+
BM25 chunk Top100
        ↓
等权 chunk RRF (k=60)
        ↓
融合 Top100
        ↓
qwen3-rerank
        ↓
直接 Top14
```

这是一条用于后续集成的冻结工程候选，不代表完整 Hybrid + rerank + Top14 已经获得新的 frozen DEV 整链确认。

## 2. 检索冻结

冻结参数：

- Dense chunk 候选：Top100
- BM25 chunk 候选：Top100
- BM25：使用现有 TechQA BM25 路线
- 融合：等权 reciprocal-rank fusion
- RRF 常数：60
- 融合候选预算：Top100
- reranker：`qwen3-rerank`
- 最终 Context 不使用 Dense Top1 rescue
- 不再搜索 RRF-k、来源权重、候选深度、来源配额、每文档上限

### R4 C1 历史状态

R4 C1 Hybrid + rerank 在预注册门槛下仍然是 **FAIL**。

TRAIN：

| 方法 | Recall@5 | Recall@20 | MRR@10 |
| --- | ---: | ---: | ---: |
| E1 Dense + rerank | 0.691111 | 0.815556 | 0.567206 |
| C1 Hybrid + rerank | 0.702222 | 0.831111 | 0.570929 |

C1 三项指标都上涨，但没有达到预注册 MRR 门槛。后来保留固定 Hybrid 路线，是单独的工程冻结决策，不等于把 C1 改写成 PASS。

## 3. Context 冻结

冻结策略：

```text
policy = flat_rerank_top14_v1
final_context = reranked_chunks[:14]
```

以下策略不属于冻结方案：

- Dense Top1 rescue
- forward-only sibling expansion
- symmetric locality expansion
- second locality rerank
- whole-document expansion
- section expansion
- Parent-Child retrieval
- hard document admission

### 为什么是 Top14

在 54 条冻结 TRAIN 证据审计样本上：

| 上下文预算 | 答案证据命中 | 有用证据命中 |
| --- | ---: | ---: |
| Top14 | 35 / 54 | 43 / 54 |
| Top20 | 35 / 54 | 43 / 54 |

Top14 是达到当前 Top20 证据命中上限的最小 K。它不是通用或生产最优值。

## 4. 被拒绝的 Context 方案

### 4.1 局部二次 rerank

直接 Top14 vs 局部二次 rerank：

- 答案证据：35 → 35
- 有用证据：43 → 43
- answer miss→hit：0
- answer hit→miss：0
- 可操作 residual：0 / 7 被救回
- 二次 rerank p50：661.314 ms
- p95：1163.955 ms
- provider tokens：659,386

结论：**REJECTED_NO_GAIN**

### 4.2 结构保持扩展

在相同每问题字符预算下：

- 答案证据：35 → 32
- 有用证据：43 → 38
- miss→hit：2
- hit→miss：5
- targeted structural residual recovery：2 / 6
- 所有问题均满足原始字符预算

结论：**REJECTED_NET_REGRESSION**

## 5. 失败归因

结构保持方案逐 case 审计：

退化归因：

- 5 / 5 answer regression：`BUDGET_CROWD_OUT`
- assembly mapping miss：0
- span / 实现错误：0
- 其他：0

收益归因：

- 2 / 2 gain：`WHOLE_DOCUMENT_RECOVERY`
- section recovery：0
- atomic recovery：0

广度 / 深度变化：

- 基线 Context 中 unique documents 中位数：12
- challenger：4.5
- 基线 atomic chunks 中位数：14
- challenger processed anchors 中位数：5
- challenger whole-document parents 中位数：2.5

支持的解释是：

> 在固定 Context 预算下，静态 parent expansion 用跨文档广度换取同文档深度。它确实能在部分 case 中恢复整文档证据，但同时挤掉了更多后排跨文档证据，因此在当前审计 TRAIN 集上净退化。

这不等于 Parent-Child、small-to-big 或文档结构方法普遍无效。

## 6. 研究关闭

portfolio-v1 当前状态：

- 检索参数研究：**CLOSED**
- Context 组装研究：**CLOSED**

除非出现当前冻结范围之外的新失败模式，否则不重新开启这两条研究线。

## 7. 最终集成前仍未解决

- 拒答 / 证据充分性策略
- 最终生成验收协议
- 最终 DEV 生成验证
- 在线运行时集成
- Ticket Agent 共享检索集成

历史 `dense_top1_distance > 0.9` 不自动继承为最终拒答策略，因为冻结候选已经是多阶段 Hybrid + rerank，而 Dense Top1 distance 只是第一阶段信号之一。
