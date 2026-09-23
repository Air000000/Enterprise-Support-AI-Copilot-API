# 检索前沿冻结：最终离线审计

日期：2026-09-10

这是对冻结 G2 30 条 TRAIN 设计 / 诊断样本的**零 provider、post-hoc 反事实分析**。它只用于理解检索前沿，不是新的 confirmatory benchmark。

## 1. 正式结果保持不变

- E1 仍是参考策略；
- G1 仍为 NO_GO；
- G2-A 仍为 NO_GO；
- 本审计不产生新的正式 GO / NO_GO。

provider 调用：

- reranker：0
- embedding：0
- generation：0
- judge：0

## 2. 已知诊断

冻结30条中：

- BOTH：20
- G1_ONLY_DISPLACED：2
- G2_ONLY_RESCUED：2
- NEITHER：6

G1 / G2 的5个已知 movement 保持不变：

- 2 个 rescue
- 2 个 displacement
- 1 个 NEITHER loss

## 3. 反事实策略

使用同一份冻结 Dense Top100、shared global rerank、corpus 和 splitter，对以下策略做离线重放：

- Dense K=5..10
- Global K=5..10
- document-level RRF K=5..10，RRF 常数固定60
- Dense5 UNION Global5

这里的 raw chunks 只是 merged-rerank 工作量代理，不等于真实延迟、token 成本或生产成本。

关键结果：

| 策略 | Gold / 30 | Rescue | Displacement | Mean chunks |
| --- | ---: | ---: | ---: | ---: |
| Dense5 | 22 | 0 | 0 | 30.5 |
| Global5 | 22 | 2 | 2 | 29.6 |
| Global8 | 27 | 5 | 0 | 51.6 |
| RRF5 | 24 | 2 | 0 | 31.9 |
| RRF6 | 25 | 3 | 0 | 37.6 |
| RRF7 | 25 | 3 | 0 | 44.3 |
| Dense5 UNION Global5 | 24 | 2 | 0 | 50.2 |

观察到：

- Dense5 与 Global5 总 gold 数相同，但 Global5 有 rescue 也有 displacement；
- RRF5 / RRF6 在接近 Dense 的 workload 下增加了 rescue，同时这30条中没有观察到 displacement；
- Global8 达到 27/30，但依赖单一 global ranking，并没有显式保留 Dense / Global 两种方向；
- Union 能保留互补性，但 workload 比 RRF5..8 更大。

这些都只是**设计集上的描述性结果**，不称为最优、已验证或生产可用。

## 4. 未来假设

当前数据支持：

> Dense 与 global rerank 存在真实互补，而固定 admission budget 会把排序差异放大。

在允许的策略族中，document-level RRF 是一个值得未来重新验证的方向，因为它在这批设计 case 上表现出“保留 Dense 覆盖 + 增加 rescue”的特征。

但这只是**未来假设**：

> document-level RRF admission 可能比单纯增大某一个 ranking 的 cutoff，更稳定地保留 Dense / Global 的互补覆盖。

本文件不：

- 冻结新的 K；
- 设计 G3；
- 授权付费 rerank；
- 打开 DEV；
- 把这30条再次当作 confirmation sample。

## 5. 研究关闭

当前检索研究线关闭。

不再对当前 artifacts 继续搜索：

- RRF k
- candidate depth
- source weight
- quota
- per-document cap

未来如果重新开启，必须使用新问题、新预注册和 fresh TRAIN confirmation cases。
