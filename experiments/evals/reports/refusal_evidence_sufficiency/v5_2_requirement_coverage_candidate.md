# v5.2 必填覆盖审计候选（离线）

2026-10-08。`OFFLINE_CANDIDATE_NOT_MODEL_EVALUATED_NO_PROMOTION`。
新增 provider calls 0 / 费用 ¥0；没有读取 targets、标注理由或 DEV。
本阶段只实现可检测的覆盖栏位缺失，不声称三个真实分歧已经修好。

## 最小改动

新模块 `experiments/evals/refusal_requirement_coverage.py` 复用 v4 的完整回答范围规则、
v5.1 的完整有序上下文/无损行号、原文提取和 v5 状态 gate；无新依赖、检索、
第二次模型请求或另一个客户端。v5/v5.1 源码、提示词、契约、标签、结果和 journals
均不修改；新候选不接入 serving 或现有 paid runner。

原来的任意 requirements 列表可能遗漏适用性、时效或版本要求，且遗漏本身不触发
机械校验。v5.2 仍保留该列表，并要求输出额外的固定 `coverage_checks`：

- `requested_outcomes`：是否覆盖所有请求结果和必要步骤，不把一般机制、许可或
  获取方法代替具体可获得性/交付结果。
- `applicability`：检查相关资料到目标产品/环境/事件的适用桥梁。区分一般可能
  原因与具体事件的已成立原因/修复；一般解释不因未逐字重复目标名就自动失败。
- `version_and_time`：保留明确版本、latest/current 和时间要求；历史/家族级机制
  不能替代时效证明。只有该项可用 `NOT_REQUIRED`，需空 evidence 和非空理由。

三项均是固定字典键，不是模型可随意漏写的 requirements。每项使用
`status/evidence/reason`，引用仍为 source-local `source_id/segment_ids`。
除上述 NOT_REQUIRED 外，只接受原 SUPPORTED/MISSING/CONFLICTING/CONDITIONAL。
把适用审计转换成内部 `Coverage audit: ...` 检查，与原材料要求一起交给既有 gate；
返回的 requirements/citation_selections 包含这些审计项，coverage_checks 另保留
模型原始定位与状态以供审计。NOT_REQUIRED 不伪造引用、不作为一个有证据的要求。

任一要求或审计声明 MISSING/CONDITIONAL/CONFLICTING、或有未解决歧义，最终为
INSUFFICIENT。缺失审计键、未知/额外字段、无效状态/定位、SUPPORTED 没有证据等
是格式错误，必须停止审计，不能兜底成正确拒答。仍不做产品名黑名单、句子情感判断、
日期 regex 或“latest 一出现就拒答”的捷径；这些不能证明实际适用性/时效。

## 已验证与未验证

新增 8 项纯合成测试、手写响应：完整 payload 不变且无元数据泄漏；一般原因解释
可通过；三种必填项分别缺失均报错；声明版本/时效或事件适用缺口会挡住放行；
显式受支持版本可通过；状态/歧义、无效定位/schema 仍遵循旧 gate。
全部 refusal 离线回归 **123 passed**，编译/diff check PASS。
本地无 Ruff，由 PR CI 验证 lint 与核心 API 回归。
旧 v5.1 检查点零调用回放 PASS：40/40、原估算 ¥0.741975，契约 SHA 不变。

测试特意保留两个负面对照：模型错误地把明确时效要求标为 NOT_REQUIRED，或者
把真实历史引文错误地标为最新状态的 SUPPORTED，仍可能机械通过。
所以本次能验证**格式上不能省略覆盖检查、已声明缺口会进入 gate**，不能证明模型
能正确识别每个语义要求、NOT_REQUIRED 是否真实、完整 requirements 是否遗漏或
引用是否蕴含结论。没有把手写测试当新的 provider 预测、真实题质量分数或独立 gold。

## 决定与求职用途

这是 post-hoc 开发候选，不改 PR59 的 80%/85% AI-proxy 一致率，不重标或重算
Q348/Q394/Q396，不声称 v5.2 在这三题成功；Q394 的适用性阈值争议仍未被解决。
旧停止实验、历史 FAIL、Dense 在线拒答和业务 API 不变，不晋升。
公开测试只含合成问题与来源，未提交真实题目/来源正文、标签理由、原始响应或密钥。

下一步优先把这些合成对照用于可复现的离线演示：来源真实不等于支持真实；一般
解释与具体诊断的范围不同；缺字段与证据不足不是同一种失败。对求职项目无需马上
增加付费轮次或更大的检索链。若要衡量真实改善，须先重新冻结新 prompt/schema/
校验源码、预算和独立验收口径并单独授权，不能继承 PR58 的已用完授权。
