# 拒答 v1.1 正式结果

状态：`COMPLETE_FORMAL_FAIL`

日期：2026-09-22

正式结论：`REJECT_EVIDENCE_SUFFICIENCY_V1`

## 执行条件

- Run ID：`refusal_evidence_sufficiency_phase_b_v1_1`
- TRAIN 分类输入：54
- 模型：`qwen3.5-plus-2026-04-20`
- Region：Singapore / International
- Temperature：0.0
- Thinking：关闭
- Context：`flat_rerank_top14_v1`
- provider 调用：54
- 失败调用：0
- DEV：未打开
- 本阶段 retrieval / rerank / generation / judge 调用：0

## 正式门槛

| 指标 | 实际 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| 平衡准确率 | 0.735714 | >= 0.80 | FAIL |
| 充分证据召回率 | 0.971429 | >= 0.85 | PASS |
| 不足证据召回率 | 0.500000 | >= 0.70 | FAIL |
| 充分→不充分误拒 | 1 | <= 5 | PASS |

51 条正式计分样本上的 Accuracy 为 0.823529。

混淆矩阵：

| 真实标签 | 预测充分 | 预测不充分 |
| --- | ---: | ---: |
| 充分代理 | 34 | 1 |
| 不充分代理 | 8 | 8 |

另有 3 条描述性 ambiguous multi-chunk case，全部被预测为 `SUFFICIENT`，不进入正式门槛。

## 成本与延迟

- Prompt tokens：147,090
- Completion tokens：8,923
- 总 tokens：156,013
- 估算成本：CNY 0.589026
- p50：3668.418 ms
- p95：4885.734 ms

## 结论

v1.1 的核心问题是**过度放行不足证据**：16 个 insufficient proxy 中有 8 个被判为 `SUFFICIENT`。

因此不足证据召回率和平衡准确率都没有达到预注册门槛，不能进入 Phase C，也不能接入 runtime。

失败后不得直接调 prompt 或改门槛后把新结果继续包装成同一个正式实验。
