# R4 C1 Hybrid + rerank：失败复盘与最终决策

## 1. 正式结果

范围：TechQA TRAIN，450 条问题。  
DEV 未在复盘中打开。  
复盘阶段 provider 调用：0。  
E1 没有重跑。

### E1 TRAIN

| 指标 | E1 |
| --- | ---: |
| Recall@5 | 0.691111 |
| Recall@20 | 0.815556 |
| MRR@10 | 0.567206 |

### R4 C1 TRAIN

| 指标 | C1 | 相对 E1 |
| --- | ---: | ---: |
| Recall@5 | 0.702222 | +0.011111 |
| Recall@20 | 0.831111 | +0.015556 |
| MRR@10 | 0.570929 | +0.003723 |

预注册门槛：

- Recall@20 >= 0.811111：PASS
- MRR@10 >= 0.577206：FAIL

因此正式结论：**FAIL**

正确理解不是“Hybrid 没用”，而是：

> Hybrid + rerank 三项聚合指标都上涨，但早期排序收益不足以通过预注册 MRR 门槛。

## 2. Dense / BM25 互补性

冻结复盘：

- Dense-only gold hit：52
- BM25-only gold hit：26
- 两者都命中：335
- 两者都没命中：37
- Hybrid 救回 Dense miss：19
- Hybrid 丢掉 Dense hit：12
- 净候选收益：+7

BM25 主要补充：

- error code / SQL state
- 产品与版本标识
- 配置术语
- 命令路径
- API / class 名
- 固定技术短语

Dense 则更擅长弱词面重合下的症状、根因、修复语义。

## 3. 最终 Top20 残留失败

450 条中：

- 最终 Top20 命中：374
- gold 已在 fused candidate 中，但最终 rank > 20：20
- gold 不在 fused Top100：56

因此 76 个最终 Top20 miss 分成：

- candidate miss：56 / 76
- ranking miss：20 / 76

当前 C1 的主要绝对瓶颈已经从 E0 的明显排序问题，转向了**候选覆盖不足**。

## 4. chunk 拥挤并不是主因

融合 Top100 中：

- unique documents p50：75.0
- p95：90.55
- duplicate ratio p50：0.25
- p95：0.44
- max：0.69
- 每文档 chunk 数 p95：8

虽然 chunk crowding 确实存在，但分桶分析不支持它是 C1 的主要失败根因，因此不据此开启 per-document cap 调参。

## 5. RRF Top100 压缩损失

Dense Top100 与 BM25 Top100 的完整 union 被压缩成一个 fused Top100。

复盘发现 19 个 case：

> 至少一个来源检索到了 gold document，但固定 Top100 RRF 压缩后把它挤出了候选池。

来源：

- Dense-only：11
- BM25-only：7
- 两者都有：1

大多数损失来自较弱的单源尾部候选：

- Dense gold rank 中位数：67.5
- BM25 gold rank 中位数：72.5
- full-RRF gold rank 中位数：123

因此这更像**固定候选预算下的保留权衡**，不是 RRF 实现 bug。

## 6. 38 条人工诊断

人工对比：

- 19 个 Dense miss → Hybrid rescue
- 19 个 source hit → fusion loss

### Hybrid rescue

主标签：

- 精确词法：12 / 19
- chunk boundary：1 / 19
- weak document hit：3 / 19
- ambiguous / questionable gold：3 / 19

其中：

- 明确值得保留：13 / 19
- 条件性有价值：3 / 19
- 不适合驱动架构：3 / 19

这进一步确认 BM25 对精确技术字符串具有真实互补价值。

### Fusion loss

- 明确值得保留：9 / 19
- 条件性有价值：3 / 19
- 较弱 / 可疑：7 / 19

11 个 Dense-only loss 中有 7 个被判断为明确有价值的语义证据。

因此未来更准确的假设不是“RRF 会丢好候选”，而是：

> 固定预算的等权 RRF 可能压掉少量有价值的单源语义尾部候选。

但当前 TRAIN 证据不足以授权继续搜索 RRF 参数。

## 7. 文档命中 != 答案证据命中

TechQA qrel 是文档级，但运行时检索 chunk，因此：

```text
gold document hit != answer-bearing evidence hit
```

一个 chunk 可以属于官方 gold document，却完全不包含用户问题需要的答案。

这也是为什么后续保留官方 Recall / MRR，同时新增人工 evidence-level audit。

## 8. 最终决策

R4 C1 在预注册门槛下正式关闭为 **FAIL**。

不继续搜索：

- RRF k
- Dense / BM25 权重
- candidate depth
- per-document cap
- source quota
- gate threshold

下一阶段不再继续调融合参数，而是转向“检索到的 chunk 是否真的能回答问题”的证据级评测。
