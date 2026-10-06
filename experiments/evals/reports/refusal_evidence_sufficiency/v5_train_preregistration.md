# v5 新 TRAIN 小样本与成对诊断预注册

2026-10-06。`PREREGISTERED_NOT_RUN`，准备阶段调用 0、费用 ¥0，未打开 DEV。
输入/标签已冻结；新的付费 runner 尚未实现，不能立即执行此契约。

## 为什么换数据、为什么同时跑 v4

旧 50 条已用于标注复核、开发和 v4 评测，不再充当独立验收。本轮从已有的
450 条 frozen TRAIN 检索结果中，排除旧证据审计 60、v2 全部标注包 80、
v3 全部标注包 80（不只排除各自付费子集），剩余 230 条。
以 `refusal-v5-new-train-diagnostic-v1` 为种子，选 SHA256(seed:question_id)
最低的 20 条，按题目 ID 排序；不按标签或模型结果补齐类别、换题或加量。
已读题后不改抽样规则，保留实际产生的相似题。

只从 snapshot 取 question 和 frozen reranked Top14 正文，校验 chunk/doc 身份与
results 一致。未读取 generation metadata、gold answer 或 DEV 文件；文档 ID
仅用于数据身份校验，不进入标注包或模型消息。来源文本保持原顺序，不重跑检索、
rerank、splitter 或 context assembly。

v4 和 v5 必须对同一批输入运行，才能描述成对变化；不能将 v5 新集成绩直接与
v4 旧 50 条分数比较并宣称改善。只变分类候选的 prompt/payload/schema/checks，
不把额外输出预算造成的成本差异藏起来：v4 512、v5 2048 output tokens。

## AI 草稿标注与局限

当前助手已逐题阅读 20 条问题与全部 280 段可见来源，冻结 14 SUFFICIENT / 6
INSUFFICIENT / 0 QUESTIONABLE。标注合同沿用完整回答充分性：文档位置请求可由
对应链接支持；一般解释/方法可保留文档条件；实例确诊/修复不能只靠相同错误码或
相关主题；不能额外发明隐藏参考答案要求。未浏览资料中的链接、图片或外部正文。

评审者参与了候选开发且看过 v4/v5 规则及历史分析，不是独立盲评或人工专家金标。
标签冻结发生在本轮任何新模型预测之前，但这并不消除评审者偏差。
[逐题理由](v5_train_ai_annotations.jsonl)、[目标](v5_train_targets.jsonl)、
[标注冻结](v5_train_annotation_freeze.json)与[抽样记录](v5_train_selection.json)可核对。

Q029/Q490 是近重复的 TWS/DWC 升级连接问题；Q348/Q394 是同一 DSM SQLCODE=-206
问题家族，前者另问实例修复，后者按一般原因说明解释。两组共用大量来源，不能把
20 条视为 20 个独立领域确认。主结果保留全部 20 条；另分别列出这两组的结果，不
根据结果删除重复题。Q029/Q339/Q348/Q394/Q490/Q593 等范围判断存在解释敏感性，
分歧应列出原证据及限制，而不是自动宣称模型错或事后更改本轮标签。

标注验证器的 minimum_per_class=5 只是小样本支持度提示，不是新正式 PASS gate。
6 条不足样本的一个变化即影响该类比例 16.7 个百分点，不能推广为总体准确率。

## 冻结运行方案

以 [机器契约](v5_train_run_contract.json) 为 authority，canonical SHA256：
`bd0edf1e130bb625f93f078098382c5c896349e66dd7aca744ea19214a2bb76d`。

- Singapore / Alibaba Cloud Model Studio / `qwen3.5-plus-2026-04-20`。
- temperature=0、thinking=false、JSON object；v4/v5 prompt SHA 与 v5 校验源码 SHA 固定。
- 每臂 20 个成功预测，最多 22 次尝试，总计最多 44 次；每臂错误预算 2。
- 按题号顺序，偶数索引先 v4 后 v5，奇数索引先 v5 后 v4，减少固定运行顺序偏差。
- 每臂独立新目录 `data/refusal_v5_train/paired_run_v1/v4|v5`，绑定同一契约身份。
- 复用现有认证、计费、请求错误记录/checkpoint 机制，不复制新模型客户端。
- 付费循环不读取标签，不发送题号、标注或私密元数据；预测完成后单独评价。
- 保存每次原始响应、request ID、usage、latency、成本；v5 还保存逐项检查和派生结果。
- 不重试成功题；provider/schema/auth/SHA/checkpoint 异常立即停止并审计。
  若 usage 未知，先核实账务，不能默认为零费用后续跑。预算允许重试不意味着自动重试。
- 格式失败、截断或引文错误不能伪装成正确拒答；保留失败记录与全 20 条分母。

官方 International 价格在 2026-10-06 核对为 input $0.4 / output $2.4 每百万 tokens，
0<Token≤256K：[官方价格页](https://www.alibabacloud.com/help/en/model-studio/model-pricing)。
按固定记账换算 7.5，记 input ¥3 / output ¥18 每百万 tokens；不使用折扣/免费额度。
模型文本 UTF-8 字节数 + 512 framing 预留作为输入 token 的保守代理上界，不是实测
token 数。全部 40 次输出用满预算且每臂额外 2 次按最大片段计，折算界为 ¥2.836473，
总 hard cap ¥3。该换算不是实时汇率或账单保证，未知 usage 必须停机审计。

## 结果口径与出口

报告两臂全 20 条格式/provider 失败数、共同有效对上的混淆矩阵、充分召回、
不足召回、AI 标签一致性，以及逐题成对“改对/改错/仍错”的变化。
另外报告 v5 模型状态判断的初步准入和原文/格式校验后的最终准入，明确失败是否
来自格式校验，而非声称语义判定改善。报告完整病例、分母、tokens/latency/cost 差异。
不足被放行与充分被拒答同时观察，不靠全拒答获得好看数字。

本阶段没有正式 PASS/promotion gate：小样本 AI 代理成对诊断不足以清除 refusal
freeze、覆盖历史 FAIL、进入 Phase C 或宣称 production-safe/DEV validated。
无新增 provider 请求授权。下一步先实现并离线测试新 schema 的 raw/checkpoint
保存与续跑完整性；就绪后再请求该明确数据/模型/预算的付费授权。
