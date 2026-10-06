# v4 / v5.1 新配对实验准备

2026-10-06。`PREREGISTERED_NOT_RUN`。本阶段模型调用 0、费用 ¥0；
不读 targets / annotation freeze，不打开 DEV，不接入 serving。

## 冻结范围与兼容

契约：`v5_1_train_run_contract.json`，canonical SHA-256：
`76210c6cf96c9bd57986dccc6847170ec178be25bc4512fece2e2bda1ad6d8fb`。
命令入口：`python -m experiments.evals.refusal_v5_1_paired_runner`，默认仅离线预检。

复用原执行器的 Singapore 客户端（SDK 重试关闭）、请求参数、逐次 durable journal、
pending marker、独占锁、预算预约与原始响应回放。只绑定新的两臂 builder/parser/契约；
旧 v4/v5 默认行为和原 checkpoint 身份不变，没有新依赖或第二套记账逻辑。
新目录 `data/refusal_v5_train/paired_run_v5_1`，不能拿旧 v5 journal 续跑。
schema、模型、prompt、定位/原文校验源码、输入 SHA、顺序和消息上界都在预检验证。
定位、实际原文引用、原始响应、usage/cost/request_id 都随本次结果保留并重新派生比对。

两臂仍用同一 20 条既有 TRAIN 开发输入 / 每题完整有序 Top14，模型
`qwen3.5-plus-2026-04-20`，temperature 0、thinking false、json_object。
v4 输出上限 512，v5.1 2048；偶数题先 v4，奇数题先 v5.1。
v4 不复用旧预测，而是在同一新协议下重新调用。付费消息只含 question/来源正文与
定位编号，不含题号、标签、标注理由或密钥；模型仍会收到完整正文，不能称脱敏数据。
公开仓库只增加合成测试与元数据，不提交真实题目/来源正文或本地原始响应。

## 新失败出口（不追溯修改旧 v5）

每个 question/arm **只调用一次**，20 × 2，最多 40 次；无重试额度。
已知 usage 的 schema / 截断 / 无效编号等失败记为 `result=null`，保留原始响应和费用，
**立即停止本次进程并审计**。获准以 `--resume-after-audit` 继续时，走到下一个 pair，
不再请求失败 pair，不自动把失败转成 INSUFFICIENT。失败上限每臂 20 是完整诊断的
计数边界，不是允许无人值守越过错误或质量通过线。

provider 错误、未知 usage、未落盘 pending、身份/输入/回放/成本异常继续 fail closed，
不能靠该 flag 跳过未知费用。成功项和失败项均不会再次调用；第 41 次不允许。
若无法对账或预算不足，保留不完整结果并停止，不自动修复或修改已冻结契约。
本阶段没有授权开始新调用，也没有授权跳过未来一次实际错误的审计。

新协议完成条件是 40 个尝试结果全部落盘，**不是 40 个有效预测**。
只在全部完成后另行执行 `--evaluate` 才读取冻结 AI targets / annotation freeze。
质量混淆矩阵、两类 recall 与 agreement 仅在两臂共同有效题上计算，并明确真实题数和
类分母；无共同有效题/某类时为 null。每臂另报全部 20 题有效数、失败数和
valid-and-agree 数/20（失败无有效 agreement，但不被当成一个拒答类别）。
转移表单列 missing_valid_prediction；raw status-only admission 不用作有效预测。
这样不会通过删除失败题把条件分数包装为全样本准确率。

## 预算与验证

按 [官方 International 价格](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
于 2026-10-06 核对：≤256K 档输入 $0.4 / 输出 $2.4 每百万 tokens；
固定会计换算 7.5（非实时汇率、非账单保证），对应 ¥3 / ¥18。
每次输入代理上界为 UTF-8 消息字节 + 512 framing，输出取 max_tokens，不假设折扣。
v4 20 次保守上界 ¥0.938556，v5.1 20 次 ¥2.043603；合计 **¥2.982159**，
冻结估算硬上限 **¥3**。每次调用前预约、之后用返回 usage 对账；不够则停止。
编号增大输入，不能沿用旧 v5 消息预算，也不能宣称更便宜或语义更准。

真实冻结输入的离线预检 PASS，prior calls 0 / checkpoint 0/40；不创建输出目录。
合成测试覆盖新响应提取与回放、失败后跳到下一 pair、不重试成功/失败项、
标签盲区、旧身份/未知 usage/pending 拒绝恢复、混合成功失败的全样本分母、
全部失败时空质量指标、预算/源码/prompt 冻结及原 v5 默认兼容。
本地全部 refusal 回归 **114 passed**；本地未安装 Ruff，由 PR CI 验证。

## 决定

`READY_FOR_SEPARATE_AUTHORIZATION_NO_PROMOTION`。下一步合并本准备 PR 后，
对这份新冻结契约另行请求发送 20 条 TRAIN question/Top14 正文及执行最多 40 次调用、
估算上限 ¥3 的授权。旧已见材料仍是开发诊断，AI 标注非独立、非人工 gold；
即使运行完整也不能称独立验收或自动晋升。v5 STOPPED/12 calls/¥0.185718、
历史 FAIL、拒答 freeze、Dense 在线路径、检索/rerank/generation 均不变。
