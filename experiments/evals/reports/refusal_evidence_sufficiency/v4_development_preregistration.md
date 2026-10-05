# v4 单次开发探针

2026-10-05：运行前固定 [契约](v4_development_run_contract.json)。没有 PASS/promotion gate。
候选 prompt SHA 为 `7b8ff671d54b929f0877ddb82a41f1adc33558569aa1ce70af25ac2f1c459cd3`，
canonical JSON 契约 SHA 为 `d7bd916c48fdafb05f48d4684666b574e43183cdc2a5c68df9f201fbfecc8e38`。

复用全部 50 条历史 TRAIN 问题及冻结 Top14，不重跑 retrieval/rerank。
模型只接收 question 和 Source N 正文，不接收题目 ID、标签、标注理由、gold 或密钥。
付费循环不加载 AI 复核文件；50 条成功预测完成后，单独 evaluate。
17 充分 + 31 不足仅用于报告与后验 AI 复核的一致性；2 条歧义单独描述。
不是独立准确率、人工金标或新样本确认，不改 v3 FAIL，也不进入 Phase C/generation/runtime。

## 固定运行参数

- Alibaba Cloud Model Studio，新加坡；`qwen3.5-plus-2026-04-20`。
- temperature 0、thinking disabled、JSON object、max output 512 tokens。
- 50 次成功预测；所有失败也计入最多 55 次尝试，不重试已成功题目。
- 折算预算上限 ¥1.5；官方 International ≤256K 定价为每百万输入 $0.4、输出 $2.4，
  查验于 2026-10-05：[官方价格表](https://www.alibabacloud.com/help/en/model-studio/model-pricing)。
- 固定会计折算 7.5 CNY/USD，对应输入 3、输出 18 CNY/百万 token，等效 USD 预算 $0.20；
  这不是即期汇率或最终账单保证，不假设缓存/免费额度优惠。

复用 v3 的预算预留、token/cost 校验、响应 schema、请求 ID 与成功/失败 checkpoint。
v3 默认入口仍使用旧冻结 prompt 和契约；这里只抽出原调用循环，原回归继续通过。
新目录 `data/refusal_v4_development/run_v1` 用不可覆盖的 run_identity 绑定契约 SHA，
发现旧 checkpoint 无身份、身份/SHA不一致、未知失败 usage 或预算超限即停止。
provider/schema 错误记账后停止，不在同一进程自动重试；须审计后才能在原冻结配置下续跑。
evaluate 的 summary 也拒绝覆盖。使用单进程操作，不声明并发 exactly-once。

## 零调用认证预检记录

初次默认查找未找到工作树 .env，继承进程只有旧 rerank 配置，地址不是 chat endpoint，
在 provider 创建之前停止：真实调用 0、没有预测或失败 checkpoint；没有更换密钥或端点重试。
只读检查确认主仓库 `.env` 已有完整 Phase B key/base 配对和有效新加坡 chat 地址格式。
新入口允许显式 `--env-file`，仅当前进程读取，不复制或写入 artifacts；地址格式不证明真实鉴权成功。

在工作树根目录，先预检：

```powershell
python -m experiments.evals.refusal_development_probe --env-file "D:\文档\GitHub\fastapi-todo-api\.env"
```

只有用户另行授权本次数据发送/运行后才加 `--run-paid`；完成后用 `--evaluate` 单独读标签。
不打印密钥、不打开 DEV，不按本轮结果调 prompt 或改标签。
