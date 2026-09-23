# G1：文档内证据扩展最终决策

## 1. 实验问题

历史 E1 Context：

```text
Dense Top100
→ 全局 qwen3-rerank
→ rerank Top3 anchor
+ Dense rank1 rescue
→ 每个 anchor 最多带 3 个向后相邻 chunk
→ 去重
→ 最多16个 chunk
```

G1：

```text
历史 E0 Dense 排序
→ 前5篇去重文档
→ 展开这5篇文档内的全部候选 chunk
→ 合并 qwen3-rerank
→ Top16
```

目标是验证：

> 如果答案证据不在 anchor 附近，而在同一文档更远位置，整篇文档范围的候选重建是否比历史 E1 的局部 sibling expansion 更好？

## 2. 正式结果

30 条 TRAIN：

| 指标 | E1 | G1 | 变化 |
| --- | ---: | ---: | ---: |
| COMPLETE | 24 / 30 | **28 / 30** | +4 |
| PARTIAL | 1 / 30 | 1 / 30 | 0 |
| INSUFFICIENT | 5 / 30 | **1 / 30** | -4 |
| 宏平均关键事实覆盖率 | 0.825000 | **0.955556** | +0.130556 |

逐 case：

- G1 改善：6
- 持平：22
- 退化：2

退化 case：

- `TRAIN_Q346`
- `TRAIN_Q492`

预注册灾难性退化定义：

```text
E1 = COMPLETE
且
G1 = INSUFFICIENT
```

实际发生 1 次：`TRAIN_Q346`

因此虽然大部分 aggregate gate 通过，但 catastrophic regression gate 失败，正式结论：

**NO_GO**

## 3. claim 级变化

65 个冻结 claim：

| 变化 | 数量 |
| --- | ---: |
| E1 covered → G1 covered | 57 |
| E1 covered → G1 not covered | 2 |
| E1 not covered → G1 covered | 6 |
| E1 not covered → G1 not covered | 0 |

净变化：

- 丢失 claim：2
- 新增 claim：6
- 净增：+4

收益并非来自单一机制：

- merged-rerank gain：3
- document-local expansion gain：2
- multi-chunk evidence gain：1

## 4. Q346：文档准入过早

Q346 的关键 claim：

> 缺陷在 IBM Rational DOORS Version 9.4.0.1 中得到修复。

决定性证据：

- document：`swg1PM50525.txt`
- chunk：`swg1PM50525.txt_chunk_0`

这个 chunk 在历史 Dense 排名里只有 **72**，但 E1 的全局 reranker 能把它提升到 Top3 anchor。

G1 却在 reranker 之前先按 Dense 只保留前5篇文档。`swg1PM50525.txt` 不在这5篇里，于是关键 chunk 根本没有进入 G1 的后续候选池。

根因：

```text
CANDIDATE_DOCUMENT_MISS
```

更准确的诊断不是“Top5 太小”，而是：

> 一个较弱的一阶段文档排序，不应该在更强的全局 reranker 之前拥有不可逆的文档否决权。

这也再次说明：

```text
document-level qrel hit != answer-bearing evidence hit
```

## 5. Q492：连续证据丢失

E1：COMPLETE（3/3 claim）  
G1：PARTIAL（2/3 claim）

E1 保留了：

- `swg21972012.txt_chunk_1`
- forward sibling `swg21972012.txt_chunk_2`

G1 虽然准入了该文档，候选池里也同时有 chunk1 / chunk2，但最终 Top16 只保留了 chunk1，没有保留决定性的 chunk2。

这说明 G1 的问题不只有 document admission，也存在 final evidence continuity 风险。

## 6. 最终结论

G1 的 aggregate 结果看起来非常好，但由于出现 COMPLETE→INSUFFICIENT 的灾难性退化，因此正式判定 **NO_GO**。

下一步不能针对 Q346 / Q492 写特判后继续拿同一批30条作为“新验证”。后续若继续研究，必须更换 fresh TRAIN 样本并重新预注册。
