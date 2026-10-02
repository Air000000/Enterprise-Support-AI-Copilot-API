# 拒答 v3：运行器与复现方式

运行合同沿用已合并的 `v3_ai_proxy_run_contract.json`，不改样本、标签、
提示词或质量门槛。模型只接收问题与有序 Top14 来源正文，不接收记录 ID、
标签、标注理由、gold 字段或来源元数据。

## 命令

先准备 SHA 匹配的 `data/refusal_v3_ai_proxy/holdout_inputs.jsonl`、
`holdout_targets.jsonl`，并按已有规则配置新加坡凭证配对。
若这两个文件尚不存在，可用原冻结盲包重建；另指定本地 manifest，
避免覆盖已提交的预注册 manifest：

```bash
python -m experiments.evals.refusal_v3_ai_proxy --manifest-path data/refusal_v3_ai_proxy/rebuilt_manifest.json
```

```bash
python -m experiments.evals.refusal_v3_ai_proxy_runner
python -m experiments.evals.refusal_v3_ai_proxy_runner --run-paid
python -m experiments.evals.refusal_v3_ai_proxy_runner --evaluate
```

默认结果目录为 `data/refusal_v3_ai_proxy/run_v1`。输入和结果目录可通过
`--inputs-path`、`--targets-path`、`--output-dir` 指定；SHA 校验不因此放宽。
付费命令不读取标签，只有完整 50 个有效预测落盘后，独立评测命令才读标签。

## 执行保护

- 每个成功预测及可捕获的失败响应都即时落盘并 fsync；成功预测不重复。
- 禁用 SDK 自动重试；provider/schema 异常立即停止，保留日志。
- 续跑检查记录 ID、模型、结构、token、费用及尝试次数。
- 用量不明的 provider 失败必须先审计，不能按零成本继续。
- 请求前按 UTF-8 字节数、消息封装余量及 512 输出 token 保守预留预算。
- 这仍是单进程实验脚本；不要对同一结果目录并发执行，也不声称崩溃窗口内
  外部请求 exactly-once。异常或进程被强制中断后先核对 provider 账目。

## 价格口径

2026-10-02 核对[阿里云官方价格表](https://www.alibabacloud.com/help/en/model-studio/model-pricing)：
`qwen3.5-plus-2026-04-20` 国际区、输入不超过 256K 的公开价格仍为
输入 USD 0.4、输出 USD 2.4／百万 token，与历史美元价格一致。
本实验仍以合同冻结的 CNY 2.936 / 17.614 估算并限制在 CNY 1.5，
不声称该换算等于当前汇率或用户实际账单。

通过仅允许进入独立人工确认，不允许进入 Phase C、generation 或 runtime 集成。
