# G2-A：重排后文档准入复盘

## 1. 冻结正式结果

| 指标 | G1 | G2-A |
| --- | ---: | ---: |
| COMPLETE | 19 | 19 |
| PARTIAL | 2 | 1 |
| INSUFFICIENT | 9 | 10 |
| UNRESOLVED | 0 | 0 |
| 宏平均关键事实覆盖率 | 0.666667 | 0.650000 |

逐 case：

- 改善：2
- 持平：25
- 退化：3
- 其中灾难性退化：2

正式结论：**NO_GO**

这份文档是解盲后的 post-hoc 机制分析，不改写正式结果，也不作为新的 confirmatory evidence。

## 2. G1 与 G2 唯一差异

G1：

```text
Dense Top100
→ 按 Dense 顺序取前5篇去重文档
→ 文档内展开
→ merged rerank
→ Top16
```

G2：

```text
Dense Top100
→ shared global rerank
→ 按 rerank 顺序取前5篇去重文档
→ 文档内展开
→ merged rerank
→ Top16
```

也就是说，G2 **只改文档准入顺序**。其余文档展开、500 chunk pool cap、merged rerank 和 Top16 选择保持一致。

## 3. 五个发生变化的 case

| Case | 结果变化 | 主要机制 |
| --- | --- | --- |
| `TRAIN_Q287` | INSUFFICIENT → COMPLETE | global rerank 把 gold 文档从 Dense rank 20/58 提升到 global rank 1/8，成功准入 |
| `TRAIN_Q578` | INSUFFICIENT → COMPLETE | gold 文档从 Dense 14/67 提升到 global 1/11，成功准入 |
| `TRAIN_Q090` | COMPLETE → INSUFFICIENT | gold 文档 Dense rank 2/11，但 global Top5 文档预算将其挤出 |
| `TRAIN_Q500` | COMPLETE → INSUFFICIENT | gold 文档 Dense rank 3、global rank 8，G1 能准入，G2 被挤出 |
| `TRAIN_Q367` | PARTIAL → INSUFFICIENT | G1/G2 都没准入 gold 文档，但准入集合不同，导致后续候选池和 Top16 不同 |

## 4. 机制归因

两次改善都来自：

> global rerank 把 Dense 顺序中较靠后的相关文档救进了5篇预算。

两次灾难性退化都来自：

> rerank 后的前5篇预算把原本 Dense 顺序里非常靠前、且实际有用的文档挤掉了。

Q367 则说明，即使两边都没 gold，文档准入集合变化也会改变 downstream candidate pool。

因此主机制是：

**文档准入排序不稳定，固定5篇预算放大了这种不稳定。**

当前证据不足以把责任归结为 merged-rerank 本身不稳定。

## 5. 能支持和不能支持的结论

可以支持：

- global rerank admission 确实能 rescue 一些 Dense admission miss；
- 同时也会 displacement 一些 Dense 高位有用文档；
- 固定5篇准入预算会把这种排序差异变成不可逆后果。

不能支持：

- “global rerank 一般是有害的”；
- “merged rerank 是主要问题”；
- “只要继续调一个更好的 K 就能解决”；
- 用这5个 post-hoc case 直接设计并验证 G3。

## 6. 决策

保持 E1 reference，不推广 G1 / G2-A。

如未来继续研究，需要：

- 更广的冻结 trace 诊断；
- fresh TRAIN confirmation cases；
- 新的预注册；
- 不复用当前已用于机制分析的 case 作为 confirmatory evidence。
