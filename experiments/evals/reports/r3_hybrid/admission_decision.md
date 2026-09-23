# R3：Hybrid 候选互补性准入

## 预注册门槛

- 至少救回 Dense miss：10 条
- 净增命中：至少 7 条

## 实际结果

- Dense hit@100：387
- BM25 hit@100：375
- Hybrid hit@100：402
- 救回 Dense miss：19
- 净增命中：15

### Dense

- Recall@20：0.740000
- Recall@100：0.860000
- MRR@10：0.510477

### BM25

- Recall@20：0.695556
- Recall@100：0.833333
- MRR@10：0.513429

### Hybrid

- Recall@20：0.777778
- Recall@100：0.893333
- MRR@10：0.547090

## 结论

BM25 单独并不优于 Dense，但对错误码、版本号、CVE、命令路径和固定技术短语等词法型问题存在真实互补。

**状态：ADMIT_PAID_R4**

因此允许进入下一阶段 Hybrid + rerank 的正式受控实验。
