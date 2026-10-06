# v5 结构化证据检查：离线开发候选

2026-10-06。状态：`OFFLINE_CANDIDATE_NO_MODEL_EVALUATION`。
调用 0，新增费用 ¥0，DEV 未打开，不改变 v3/v4 分数、标签或历史结论。

## 最小改动

`experiments/evals/refusal_structured_evidence.py` 复用原 ClassifierInput 与 v4 范围规则。
模型只收到 question 和有序 source_id/content；不发送题目 ID、chunk/doc ID、
gold、标签或标注理由。输入改为 JSON，v4 的原消息格式、prompt 和付费契约不修改。

模型返回 requirements（每项含 requirement、status、evidence、reason）与
unresolved_ambiguities。状态只有 SUPPORTED / MISSING / CONFLICTING / CONDITIONAL。
每个 evidence 含 source_id 和连续原文 quote，不允许翻译、改写或省略号替代原文。
列出完整必要要求，但不额外发明要求；核对目标效果、适用范围、必要步骤和条件。

校验函数验证对象字段、状态、非空要求/理由、来源身份和引文确实出现在对应正文。
SUPPORTED/CONFLICTING 必须引证；允许缺失项的 evidence 为空。
任一非 SUPPORTED 要求或未解决歧义 => INSUFFICIENT；否则 => SUFFICIENT。
不接受模型另报的总体 decision。重复 JSON 字段、空要求、虚构引用等抛出 ValueError；
调用方应停止并审计，不得在异常时回退到生成回答。

调用方式（无模型客户端）：

```python
messages = build_classifier_messages_v5_dev(classifier_input)
# raw_response must come from a separately preregistered, authorized future run.
checked = validate_evidence_response(raw_response, classifier_input)
```

## 验证边界

测试使用手写的理想响应，不是假装模型已经生成的预测。四对合成边界案例覆盖：
等待增减方向、版本/平台范围、单步激活与完整迁移、条件已确认与未确认。
另检验跨来源完整支持、部分支持不能掩盖缺失项、冲突、歧义、异常响应和元数据隔离。
CI 显式执行该测试文件。

本轮本地检查：全部 refusal 离线回归 76 passed；新文件编译及 git diff --check 通过。
本机 Python 环境没有 Ruff，不能记本地 Ruff PASS；Ruff 与核心接口回归交由 PR CI 验证。

**这不是语义真值校验器，也没有证明拒答质量改善。** 若模型遗漏必要要求，或把
“增加等待”错误标为支持“减少等待”，真实引文仍会通过形式校验。
测试特意保留此反例，记录实现能力上限；需要真实模型评测检验该错误是否减少。
合成案例、旧 50 条及已见标签都是开发材料，不是独立验收集或人工专家金标。

## 下一步

先完成独立的新 TRAIN 输入/AI 代理标签准备与冻结（不称人工真值）；再单独预注册
v5 prompt/schema/input SHA、模型/地区/解码、输出预算、调用/成本上限及 checkpoint。
结构化输出通常更长，不能直接沿用 v4 的 512-token 预算或旧 run_v1 输出目录。
未来复用 checkpoint/accounting 逻辑，但保留新的运行身份、原始检查及派生二分类结果。
同时报告不足被放行、充分被拒答和格式失败；格式失败不是正确拒答，不删除失败行。
旧 50 条若再次运行只记开发诊断，不反复调整直到 PASS。

本次不新增付费 runner、不执行模型、不打开 DEV、不改检索/context、生成或 runtime。
不进入 Phase C，不更改 refusal freeze 的 unresolved 状态。新实验仍需授权。
