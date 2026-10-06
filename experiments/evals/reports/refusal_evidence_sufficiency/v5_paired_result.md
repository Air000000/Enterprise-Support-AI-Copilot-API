# v4/v5 新 TRAIN 成对诊断：格式失败停机结果

2026-10-06。用户确认 PR 55 已合并并授权执行后，按
[v1.1 冻结契约](v5_train_run_contract_v1_1.json)实际运行。
状态 **STOPPED_V5_FORMAT_FAILURE_BUDGET_EXHAUSTED_NO_PROMOTION**，不是完整评测、
不是质量 PASS/FAIL，也不是拒答策略已修复。
[授权记录](v5_paired_execution_authorization.json)、
[逐次异常审计](v5_paired_execution_audit.md)、
[机器结果冻结](v5_paired_result_freeze.json)可追溯。

## 真实运行与费用

执行版本 `7c7de82a0c40e0ff183c4233587baecd1eb6283d`（PR 55 合并版本），
契约 canonical SHA256
`ed66bfbbb841a199670e6922f29cdd30ed1e17ab5d9f3cfec6f633f504064032`。
Singapore `qwen3.5-plus-2026-04-20`，temperature=0、thinking=false、JSON object。
v4/v5 输出上限分别为 512/2048，仍按题号交替臂顺序。
没有更改 prompt、schema、校验器、输入、标签、候选或历史结果。

| 项目 | v4 | v5 | 合计 |
| --- | ---: | ---: | ---: |
| 预注册病例分母 | 20 | 20 | 同一批 20 |
| 请求次数 | 5 | 7 | 12 |
| 已请求不同病例 | 5 | 6 | 6 |
| 有效预测 | 5 | 5 | 10 / 40 |
| 格式失败尝试 | 0 | 2 | 2 |
| 已观测格式失败病例 | 0 | 1 | 同一 Q193 |
| 尚未请求病例 | 15 | 14 | 不计正确或拒答 |
| input tokens | 14,027 | 23,687 | 37,714 |
| output tokens | 758 | 3,274 | 4,032 |
| total tokens | 14,785 | 26,961 | 41,746 |
| 固定记账折算 CNY | 0.055725 | 0.129993 | **0.185718** |

12 个唯一 request ID，usage 全部已知，provider 失败 0。费用包括失败和重试，
低于 ¥3 hard cap；按冻结 ¥3/百万 input、¥18/百万 output 与固定 7.5 换算记账，
不是实际账单保证。先停止、核实已知账务和原始响应、记录审计后才显式重试；
自动重试 0，成功预测重试 0。

## 为什么停止

前 5 对有效病例为 Q029/Q059/Q168/Q172/Q188；两臂均报告 SUFFICIENT。
第 6 题 Q193 按交替顺序先执行 v5，两次尝试均在原文引文校验处失败。
第一次在 Source 2 与 Source 1 引文中加省略号；原样重试换成了 Source 12 的
标题/后续句拼接。引用内容相关，并不代表它是指定来源的连续原文。
两次均正常 finish、无 token 截断，模型都将全部要求标 SUPPORTED、无歧义，
但形式校验后的 `result=null`。这是格式/原文定位失败，不是一次有效 INSUFFICIENT。

本题至少暴露出 **让模型直接逐字抄引文的执行可靠性问题**。停止的不是 API 或
账务错误，证据也没有指向 AI 标注：执行及结果分析未打开 targets 或标注冻结。
原样重试没消除该错误；不能把一次剩余费用充足解释为应持续重复请求。
v5 已达冻结的每臂 2 次失败上限，因此结束本轮，而不是原地放宽校验或加预算。

## 不报告哪些分数

仅 5 个共同有效对、总计 10/40 有效预测，不满足完整 evaluator 的 40 条条件。
不打开标签，不给出混淆矩阵、充分/不足召回、AI 一致率、成对改对/改错或质量提升。
20 条主分母仍保留；“观测 1 个失败病例”不能当成完整样本的 5% 失败率，
14 条 v5 未请求病例不是零错误。两次无效输出也不能算两次正确拒答。
相关问题家族和范围敏感病例的预注册不变，但本次运行未完成对应整体诊断。

## 产物与决定

完整原始检查点已按原字节保存在工作树与主项目：
`D:\文档\GitHub\fastapi-todo-api\data\refusal_v5_train\paired_run_v1`，
包含 run_identity.json 和两臂 attempts.jsonl；SHA 已核对一致并记在机器冻结中。
不在 Git 提交完整本地账本、数据包或 env；Git 只保存紧凑冻结、审计和结果说明。
两个账本保留原始 JSON、逐项检查、request ID、usage、延迟和成本；不存在完整
summary.json，不能伪装成完成评测。原运行不可用 resume 标志绕过失败预算。

按 Ponytail 复用已合并运行器，本轮没有写新 runner、修改运行时或添加依赖。
本地 refusal 离线回归 **95 passed**，编译及 diff check 通过；新增结果回归
校验未完成分母、两次失败、完整计费与不晋升声明。核心 API/Ruff 看结果 PR CI，
这些工程检查不能替代未完成的模型质量评测。
DEV 未打开，retrieval/rerank/generation/judge 新调用均为 0；不更改历史 FAIL，
不晋升、不进入 Phase C，不宣称 production-safe。

下一步建议先离线改变引文定位机制，而不是继续重试本题：让模型选择可定位的
来源句段，由代码取回原文，减少模型抄写/拼接导致的形式失败；必须另行验证
定位错误与语义误支持，不把“引用真实”等同“语义正确”。该变化会改变 prompt/
payload/schema，需新的候选、冻结契约与付费授权。本轮不会自行执行这一扩展。
已见预测和失败题仍是开发材料，不能包装成新的独立验收。
