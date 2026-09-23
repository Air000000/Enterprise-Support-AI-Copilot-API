# 拒答 v2.1：AI 代理标签结果

状态：`COMPLETE_PROXY_FAIL`

日期：2026-09-23

结论：`REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY`

## 执行条件

- Run：`refusal_v2_ai_proxy_holdout_v2_1`
- 模型：`qwen3.5-plus-2026-04-20`
- Region：Singapore / International
- 冻结输入：40 条 AI 草稿代理样本，其中 20 充分、20 不充分
- 沿用原预测：38 条
- 新增 provider 调用：2
- 总 provider 尝试：41（40 个成功预测，保留 1 个 schema failure）
- v2.1 最大输出 tokens：512
- prompt、模型、输入、targets、Context 策略和质量门槛：均未改
- DEV：未打开
- retrieval / rerank / generation / judge 调用：0

## 代理标签门槛

| 指标 | 实际 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| 平衡准确率 | 0.775000 | >= 0.80 | FAIL |
| 充分证据召回率 | 0.850000 | >= 0.85 | PASS |
| 不足证据召回率 | 0.700000 | >= 0.70 | PASS |
| 充分→不充分误拒 | 3 | <= 3 | PASS |

总 Accuracy：0.775000（31 / 40）。

混淆矩阵：

| 真实标签 | 预测充分 | 预测不充分 |
| --- | ---: | ---: |
| 充分代理 | 17 | 3 |
| 不充分代理 | 6 | 14 |

类别是平衡的，因此只要再多 1 条分类正确，平衡准确率就会达到 0.80。但实验解盲后没有修改门槛。

## 成本与延迟

- Prompt tokens：110,403
- Completion tokens：6,799
- 总 tokens：117,202
- 估算成本：CNY 0.443901
- p50：3584.833 ms
- p95：6223.223 ms

## 结论

v2.1 没有达到预注册平衡准确率门槛，因此不能进入 Phase C 或 runtime 集成。

这批标签是 **AI 草稿代理标签，不是独立人工金标**。后验错误审计可以用于分析 9 个 disagreement，但新的分类器必须使用新的独立样本和新的预注册；不能继续在这 40 条上调到通过。
