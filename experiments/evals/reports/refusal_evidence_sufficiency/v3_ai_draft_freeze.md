# 拒答 v3：盲标 AI 草稿冻结

状态：`AI_DRAFT_TARGETS_FROZEN_FOR_DEVELOPMENT`

日期：2026-09-23

## 冻结结果

- 标注总数：80
- 充分：44
- 不充分：27
- 存疑：9
- 两个可评分类别都达到至少 25 条的最低样本要求
- 标注包 SHA-256：`a3a1226343f5facd51b1fe714974aa66ade947a52710198effcaba867c142218`
- 紧凑 targets SHA-256：`5b0d8b5b167e8ae742210c2deb8ff91c711143bf8da10451bc15f835d9ca8d68`

校验器确认：

- 80 条问题与对应 Top14 Context 没有变化；
- 顺序保持不变；
- 每条都有非空 rationale；
- 每条“充分”标注至少引用一个有效 source；
- rationale 不引用 gold / reference / ground-truth answer。

## 来源与边界

这些标注由外部 GPT 工作流生成，但该外部工作流的模型、prompt、解码参数、调用与成本没有记录在本仓库中。

因此它们目前只作为**开发阶段 AI 草稿标签**，不是独立人工金标。

当前类别数量足以支持后续 v3 AI-proxy classifier 的预注册，但**不能据此进入 Phase C、生成验证、拒答策略冻结或在线集成**。
