# v5.1 引用定位候选：模型选编号，代码取原文

2026-10-06。`OFFLINE_CANDIDATE_NO_MODEL_EVALUATION`。新增调用 0、费用 ¥0，
DEV 未打开。本阶段针对 v5/Q193 的抄写/拼接引文失败，不重新运行已停止的契约。
v5 原结果仍是 12 次调用 / ¥0.185718 / 10 个有效预测，不原地补成完整评测。

## 最小改动与协议

新模块 `experiments/evals/refusal_citation_selection.py` 复用 ClassifierInput、v4
完整回答范围规则及原 v5 的结构/状态/原文校验与二分类派生。原模块、prompt、
运行器、输入、标签及历史契约均不修改；没有新依赖、NLP splitter 或模型客户端。

每个 source 的正文用 `splitlines(keepends=True)` 按原有行界编号，source 内
1-based segment_id。这里 segment 是显示行，不是新的检索 chunk、句法句子或
新的上下文筛选：source 顺序不变，全部字符、空白、换行和原本的断句保留。
将这些 line content 原样拼回必须与原 source content 完全相等，不重跑检索、
rerank、splitter/context selection，不缩短或重新拼接上下文。

模型消息仅含 question 和有序 source_id/segments；不带题号、chunk/doc ID、
标签、标注理由或金标。新输出仍是逐项 requirement/status/reason 和 ambiguities，
只把证据从模型抄写 quote 改为定位引用：

```json
{"source_id": "Source 12", "segment_ids": [3, 4]}
```

代码按本次完整输入的 source-local 编号取回每一行，分别形成原文 quote；
非相邻行也保持分别引证，不补省略号、不合成模型改写后的“原文”。
重复选择去重；空白行保留在输入但不充当证据。SUPPORTED/CONFLICTING 如果只选
空白行，仍因没有有效证据而报错。保留 citation_selections 以便与原始模型响应
对照定位，并复用 v5 gate 得出最终二分类。

只接受已显示的整数编号；bool、数字字符串、小数、越界编号、错 source ID、
跨 source 冒用编号、重复 JSON 键、额外 quote/offset/decision 字段等报错。
没有“引用失败就改成 INSUFFICIENT”的兜底；失败不是正确拒答。
引用正确的 SUPPORTED 也不等于人工真值。

## 离线验证与边界

一个聚焦测试文件（12 项）覆盖无损编号、不同换行/Unicode/空白、source-local
身份、非相邻证据不拼接、重复/空白选择、缺失/条件/冲突/歧义、非法选择、
元数据隔离和继承的语义能力上限。全部 refusal 离线回归 **107 passed**，
新文件编译及 diff check 通过；Ruff/核心 API 回归交由 PR CI。

公开测试只使用合成文本复现 Q193 类型的标题/后续句拼接：伪连续引文仍在 v5 gate
失败，手工构造的有效编号可分别提取给定原文；不提交真实题目或来源正文。这不是新的 provider 预测，
不将旧响应自动转换成“修复后的成功”，不修改原 journal 或重算质量指标。
测试明确保留“增加等待”被模型错标支持“减少等待”的反例：原文真实时仍可
通过机械校验；必要要求遗漏、条件/范围误判或错误选句仍依赖模型识别。

从已有 20 条 TRAIN 开发输入读取 question/Top14，不读 targets 或 DEV；共
280 个 source / 3937 个显示行，全部无损 round-trip **PASS**。
这些来源已经参与准备、候选开发或部分实际预测，不称新的独立验收。

编号并非零成本。在同一 20 条消息上按 UTF-8 字节 + 512 framing 计算：
v5.1 单请求代理上界 12,556–29,458，合计 435,441；原 v5 合计 288,019。
这是格式机械检查，不是实测 tokens，也不是新付费契约/价格报价。
原 v5 prompt/source SHA、请求上界和旧预算不能直接用于 v5.1；下一阶段必须
重新冻结全部提示词、定位/校验源码、消息大小、解码设置、输出及调用/成本预算。

## 决定与下一步

先保留为离线候选，不接入 serving，不启动付费 runner、不打开标签/DEV、不晋升。
此阶段只证明代码能从有效定位提取原文；**尚未证明模型能稳定选对编号，
也未证明不足误放行或充分误拒减少**。仅继续调整提示词或多次重试不能代替检验。

下一步可准备新的 v4/v5.1 对照契约与兼容该响应的 checkpoint（不覆盖 v5）：
继续把旧已见材料标为开发诊断，报告格式失败、语义分歧和成本差异；定义不同
失败/重试出口，不让抄写错误与语义错误混成好看的拒答分数。付费运行需另行授权。
原始 v5 停机结果、历史 FAIL、在线 Dense 路径和 refusal freeze 均保持不变。
