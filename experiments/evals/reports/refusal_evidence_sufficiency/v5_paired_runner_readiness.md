# v4/v5 成对运行器就绪记录

2026-10-06。`OFFLINE_RUNNER_READY_AWAITING_AUTHORIZATION`。真实 provider 调用 0，
费用 ¥0，DEV 未打开。仅实现冻结的 20 条 TRAIN 成对诊断，不接入在线服务。

[预注册](v5_train_preregistration.md)及 v1.1 机器契约保持原样；其中 runner_ready=false
描述准备时状态，不为实现进度重新签订或修改冻结契约。本记录描述后续实现状态。
authority canonical SHA256 仍为
`ed66bfbbb841a199670e6922f29cdd30ed1e17ab5d9f3cfec6f633f504064032`。

## 最小实现

`experiments/evals/refusal_v5_paired_runner.py` 复用现有 Singapore 鉴权/客户端（SDK
max_retries=0）、输入加载、v4 解析与 v5 校验器；没有新依赖、第二个模型客户端、
检索/生成请求或新 judge。按预注册题号与交替臂顺序请求，成功预测不重试。
预检核对契约、输入、抽样、两套 prompt、v5 校验源码及各题 UTF-8 字节上界。
预检和付费循环均不读取标签或标注冻结，发送内容只含 question 与 Top14 来源。

每臂 `attempts.jsonl` 是唯一逐次账本，保存 raw_response、request ID、usage、latency、
成本、finish_reason、错误阶段及类型；v5 另存逐项检查和派生结果。
解析失败不记为 INSUFFICIENT，截断即使留下可解析 JSON 也拒绝作为有效预测。
`raw_model_admission` 只按模型状态做初步描述，不是有效预测、语义真值或上线准入。

根目录身份绑定冻结契约；读取时按全局 call_index 重放交替顺序、原始解析、引文与
派生结果，并复核 usage、计费及预算。未知文件、重复/越序/改写的成功预测均停机。
请求前独占本地 runner.lock 并写 pending_request.json；响应账本 flush/fsync 后才清除
pending。若进程在请求与记录之间中断，未对应账本的 pending 阻止再发请求，避免
把潜在已计费调用当作零调用。残留锁也需人工/助手审计，不自动抢锁或恢复请求。
若 pending 恰对应完整有效账本，重放核对后可清理该冗余标记；不重复已记录调用。
这是单机实验文件锁，不声称分布式恰好一次或断电后绝对文件系统持久性。

每次 provider/schema 错误立即停止，无自动重试。已知 usage 的 schema 失败经审计后
可显式续跑；每臂达到 2 次失败即停止，不能赌下一次成功而发出潜在第 3 次失败。
未知/异常 usage、未记录请求或 provider 异常不能靠 resume 标志绕过：先核对账务和
调用状态，必要时另行记录契约修订；不要删除记录、填零或修改已保存 usage。
SDK 异常文字可能带鉴权信息，账本只存错误类型，不存异常文字。

## 命令与授权

从仓库根目录执行，仅预检（不需要密钥，不创建运行输出）：

```bash
python -m experiments.evals.refusal_v5_paired_runner
```

以下仅在用户另行明确授权后执行；本次没有运行：

```bash
python -m experiments.evals.refusal_v5_paired_runner --run-paid --env-file /absolute/path/to/.env
```

原项目本地 env 路径为 `D:\文档\GitHub\fastapi-todo-api\.env`，只本地加载，不复制或
提交密钥。初次输出固定为 `data/refusal_v5_train/paired_run_v1/v4|v5`；不要借用旧目录。
普通中断只有已知完整检查点时才直接续跑；已知计费的格式失败，审计后在付费命令
上加 `--resume-after-audit`。该标志不等于授权，也不会绕过未知用量、锁或预算检查。

两臂全部 40 个有效预测完成后，才单独打开冻结 AI targets：

```bash
python -m experiments.evals.refusal_v5_paired_runner --evaluate
```

summary.json 独占创建，不覆盖既有结果。报告全 20 条分母、各臂失败病例/尝试次数、
共同有效对混淆矩阵、两类召回、逐题 AI 一致性得失、初步/最终准入、tokens/latency/
成本以及两组相关问题家族。成功续跑后的分类与此前格式失败分开，不能隐藏失败。
发生未知计费/provider 错误时没有最终完整分类报告，原始账本用于停机审计，不给
失败病例填假拒答或正确率。

## 离线验证与结论

模拟客户端测试覆盖交替顺序、原始检查保存、引文失败、已知计费的审计续跑、
成功不重试、标签隔离、provider 异常、未知 usage、截断、非法 JSON、失败预算、
请求前成本预留、身份/输入契约及账本篡改、残留锁和未记录 pending。
全部 refusal 离线回归 **94 passed**；预检 **PASS**，调用 0。
本地 Ruff 不可用；实现提交 `854587d` 的
[PR CI](https://github.com/Air000000/Enterprise-Support-AI-Copilot-API/actions/runs/37419819121)
已确认编译、Ruff **PASS**，核心 API 与拒答回归 **197 passed / 1 warning**。
warning 为既有 Starlette TestClient/httpx 弃用提示，不借本轮引入依赖迁移。

请求未来授权范围保持预注册：20 条冻结 TRAIN 问题和每条 Top14 来源正文，同题
v4/v5；Singapore `qwen3.5-plus-2026-04-20`，最多 44 次总调用，固定记账折算 hard
cap ¥3，不发送标签、理由或密钥。AI 草稿不是独立人工金标，20 条存在相关题。
本轮没有模型效果证据，不改变历史 FAIL、不晋升或打开 Phase C/DEV。
