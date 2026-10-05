# v4 单次开发探针结果

2026-10-05，用户确认 #51 已合并并授权后执行。状态：
`COMPLETE_POST_HOC_DEVELOPMENT_PROBE_NO_PROMOTION`；本轮未定义正式 PASS gate。
机器记录见 [结果冻结](v4_development_result_freeze.json)，设置见
[运行前契约](v4_development_run_contract.json)。

## 真实运行与费用

新加坡 `qwen3.5-plus-2026-04-20`，temperature 0、thinking disabled、JSON object、max output 512。
显式使用主仓库已配置的 .env，预检验证输入及契约 SHA、新 checkpoint 为零。
不复制或打印密钥。此前无调用的环境来源问题没有被计作本次 provider 失败。

50 条全部完成，50 次真实调用、0 次失败；50 个唯一题目 ID、50 个唯一 provider request ID。
paid loop 未加载 AI 标签，退出后才单独 evaluate；没有重新生成或重试成功结果。

- 输入 154,706 tokens；输出 9,822；合计 164,528。
- 固定会计折算费用 **¥0.640914**，低于 ¥1.5 上限；不是实际账单声明。
- 延迟 p50 4,016.54 ms、p95 6,088.97 ms。
- DEV 未打开；retrieval/rerank/generation/judge 新调用均为 0。

## 与后验 AI 复核的一致性

| AI 复核主标签 | 模型判充分 | 模型判不足 | 合计 |
| --- | ---: | ---: | ---: |
| 充分 | 17 | 0 | 17 |
| 不足 | 13 | 18 | 31 |
| 二分类合计 | 30 | 18 | 48 |

一致 35/48 = 72.92%，平衡一致性 79.03%；相对 AI 复核的充分召回为 100%，
不足召回为 58.06%。另两条歧义仅描述：Q049 判不足、Q549 判充分，不进入上述分母。
有限排查维度没有送给分类器，也没有被偷换成完整回答许可。

这些标签由看过旧结论的当前助手复核，题目和来源参与了候选开发；不是人工金标或独立样本。
不能将 13 条分歧全部断言为真实业务中的错误，更不能直接称为 13 次幻觉——本轮没生成回答。
同样不能只因 17/17 充分都获准就称拒答机制已修复。与旧 v3 的 0.76 不作改进幅度比较，
因为标签与评分口径不同；不把旧预测换新标签重算成绩。

## 一个可见的逻辑问题

Q112 明确要求减少/避免远程 JMS 写入的三秒等待。Source 1/2/4 的建议是增加等待，
以便 WLM/HA manager 有更多时间定位 messaging engine、减少报错。
本轮模型理由承认来源建议增加等待，却仍把“识别可调属性”当作完整解决加速要求。
这至少表明所需效果与建议方向仍可能混淆，不能靠提示词边界文字自动保证准入正确。

其他分歧保存在本地 summary 和预测中；其中一些还涉及此前披露的回答范围/版本解释敏感性。
本轮不再重新标注、改 prompt 或重复付费探针来追求 PASS。

## 保存与工程决定

完整本地产物：`data/refusal_v4_development/run_v1/{run_identity.json,predictions.jsonl,summary.json}`。
无失败，故不存在 failed_attempts.jsonl；各 SHA 已冻结。Git 只保存紧凑记录和报告。
确认旧 input/target/prediction/summary 与 AI 复核记录 SHA 均未变；本地离线回归 50 passed，
编译和 diff check 通过。线上代码与 portfolio refusal freeze 不变，旧 v3 FAIL 保留。

决定：记录开发分歧，**不晋升**，不进入 Phase C/generation/runtime 集成。
针对求职目标，建议停止在这 50 条上反复研究，下一阶段优先完成现有线上 Dense 路径的
端到端演示与边界说明；Hybrid/Top14 和新拒答候选仍如实列为离线工作。
这不要求用户追加人工金标，也不意味着新候选已经生产可用。
