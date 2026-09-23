# R4 C1：Hybrid + rerank 正式对比

## 对比口径

本实验在 **TRAIN 450 条可回答问题**上比较：

```text
E1：
Dense Top100
→ qwen3-rerank

R4 C1：
Dense100 + BM25100
→ 等权 RRF(k=60)
→ fused Top100
→ 同一个 qwen3-rerank
```

## E1 TRAIN

- Recall@5：0.691111
- Recall@20：0.815556
- MRR@10：0.567206

## R4 C1 TRAIN

- Recall@5：0.702222
- Recall@20：0.831111
- MRR@10：0.570929

## 预注册门槛

- Recall@20 >= 0.811111：**PASS**
- MRR@10 >= 0.577206：**FAIL**

## 正式结论

**R4 C1 = FAIL**

C1 的三项聚合指标都比 E1 TRAIN 高，但早期排序提升不足以通过预注册 MRR 门槛。因此不进入原计划中的进一步融合参数搜索。
