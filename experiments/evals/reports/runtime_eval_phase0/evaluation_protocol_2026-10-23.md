# 归档索引集成评测：检索预注册与端到端边界

日期：2026-10-23。状态：**检索方案已登记，执行适配/离线验证及新授权待完成；未运行 DEV，未授予付费权限**。
这是实验协议，不是上线、质量 PASS 或预算授权。协议修改须保留版本与理由，不能观察本轮 DEV 结果后调参重跑。

## 1. 要回答的问题与证据边界

先回答“同一归档快照上，固定 Hybrid 相比 Dense / Dense+rerank 的文档检索有无改善”，
再单独回答“受限 Dense 默认问答配置与固定 Hybrid+v5.3.1 的整系统差异”。两阶段分别授权、记账；
阶段1不调用分类/生成/标注，不能据此写拒答或答案质量提升。

历史 [E1 DEV](../e1_rerank/comparison.md) 的 R@5 .643750→.725000、R@20 .818750→.843750、
MRR@10 .518931→.560841 仍只是旧索引上的 Dense→Dense+rerank 证据。
R4 C1 TRAIN 的预注册 FAIL、拒答 TRAIN50 两版均43/50且各2次格式失败的 NO_PROMOTION、
v5.4 合成探针停止结论不变，见[评测总览](../../README.md)。

DEV160 已用于 E1；历史 impossible 审计也已暴露标签噪声。本轮是**复用既有 DEV 的固定方案配对评测**，
不是从未使用的独立盲测。新快照重建了 HNSW，不能将新旧数字相减当成线上提升；不自动晋级默认策略。

## 2. 固定资产与运行配置

- 数据文件身份、revision、SHA256、划分沿用 [TechQA manifest](../../datasets/techqa/manifest.json)。
  检索只用按 question_id 排序的全部160条 DEV answerable 与其文档 qrel；不采样、不挑题。
  问题来自 retrieval queries 的原文，仅统一 `rstrip()`；保留内部空白，不借助 gold 改写查询。
- 两阶段共用[归档受管快照](archival_import_2026-10-22.json)：28,481文档、172,614 chunk、1024维、L2，
  collection=`local_doc_rag`，tenant=`tenant_eval_techqa_archival`，category=`other`，
  managed-records SHA256=`7ea91b0d516761bdf80e65397379c2e3a934e105a30ca956846e4a025e8e8fc5`。
  源索引6个文件指纹须符合[资产审计](asset_audit_2026-10-22.json)。只从已验证快照制作隔离工作副本。
- 归档向量输入是纯正文，模型/端点身份**未知**；它不是当前上传的“文件名+正文”索引重建。
  新查询明确使用北京 `text-embedding-v4`、1024维、float，但这不补证历史向量的模型身份或语义兼容性。
  结果仅适用于这个归档向量+新查询模型组合；若要证明正常上传链路，须另立索引重建实验。
- 检索/提示词/校验器以 main `02f51fb59374442c71ae2d984461d5300fbd14a0` 为基线。
  只在独立评测进程替换客户端和增加观测，不改算法/提示词；适配器提交、所有依赖源码/提示词哈希、
  Python/Chroma/SDK/HTTPX/ranx版本与快照文件指纹必须在首次发送前写入运行 manifest。
  不能直接拿旧 e76d159 日常镜像执行。适配后如发现语义变化，须先修订协议，不能静默继承冻结资格。
- 隔离 Docker：单 worker、串行、4 CPU / 5 GiB、`RAG_HYBRID_MAX_VISIBLE_CHUNKS=200000`；
  显式指向副本 SQL/Chroma，不读取工作目录 `.env`，不挂载日常数据库、上传目录或生产凭据。
  原始快照只读；归档 SQL 仅含文档/分块表，阶段2需在新库完成迁移后复制数据并重新核验。
  日常服务、默认开关、原始索引、Ticket/Agent 流程不在本次修改/执行范围。

## 3. 阶段1：DEV160 文档检索（下一次拟授权范围）

| arm | 固定流程 | 每题新增付费请求上限 |
| --- | --- | ---: |
| D | 相同查询向量 → Dense chunk Top100 | 1次共享 query embedding |
| DR | D的同一完整100候选 → qwen3-rerank | 1次 rerank |
| H | 同一 Dense100 + BM25100 → 等权 RRF(k=60) Top100 → qwen3-rerank | 1次 rerank |

最多 **160 embedding + 320 rerank = 480次**。不做全库 embedding，不用旧候选/预测替代新对照。
每题 embedding 只发送一次并供三臂复用；H 应执行现有 `search_hybrid`，通过评测进程的观测包装保存
其实际 Dense、BM25、融合及完整重排输出。D复用实际捕获的 Dense100，DR重排这个池；不另写一套 Hybrid。
在 H 最终截断前保存完整100排名，核对其前14与服务返回完全一致。完整观测缺失则不能计算正式IR。

RRF同分按现有 `fuse_rrf` 的 best_rank、chunk_id 排序；Dense保持Chroma返回顺序，
rerank保持现有客户端的提供方排列，不事后按score另排。同题按question_id排序；偶数序号先DR后H、奇数先H后DR
（序号从0开始），共享向量/Dense结果须完全一致，否则停。缓存复用属于离线检索对照，不冒充真实HTTP延迟。

### 指标与预注册结论

- 从各臂**完整 chunk Top100**按 document_id 首次出现去重，再算 Document R@5、R@20、MRR@10；
  每题一个 qrel，R@K数值等于Hit@K。少于K个不同文档不补检索、不改变分母。
  复用 `ir/ranx_adapter.py`；报告每臂160题聚合、绝对差与百分点，不只公布最好的一臂。
- 主要对照 **H−DR**，主要指标 **MRR@10**；DR−D为次要重排对照，H−D为整条检索链对照。
  R@5/R@20为次要指标，不在看结果后切换主要指标。记录qrel文档是否进入实际Top14上下文，
  作为单独的 context-hit@14，不称作 Top14 文档 R@20。
- 固定配对bootstrap：按题有放回重采样160题，10,000次、stdlib `random.Random(20261023)`，
  同一索引序列应用于所有臂；差值样本排序后取下标249/9749作95%区间。
  只在 H−DR 的MRR差值区间下界>0时写“本基准配对改善”；否则写“未证实改善”，负结果同样发布。
  此门槛不是多指标整体PASS、统计独立性证明或上线门槛，不覆盖/更改历史R4门槛。
- 记录全部实际调用usage、估算费用、提供方延迟；冷/暖态单列，P50/P95标明样本数及统计口径。
  不用共享embedding/候选的耗时宣称服务端端到端延迟或吞吐。

## 4. 当前价格与保守预留（不是预计实际账单）

2026-10-23 核对[阿里云中国站按量价格](https://help.aliyun.com/zh/model-studio/model-pricing)，
单位均为**人民币/百万tokens**，不计免费额度、缓存折扣、Batch或活动优惠。
此口径适用于人民币按量账户，不把国际站美元价格直接当人民币；账户币种/费率不同须先重算并重新确认。

| 用途/地域 | 模型 | 输入/输出价 | 每次规划 token 预留 |
| --- | --- | ---: | ---: |
| 查询/北京 | text-embedding-v4 | 0.5 / 无 | 8192 / 0 |
| 重排/新加坡 | qwen3-rerank | 0.74942 / 无 | 120000 / 0 |
| 分类/新加坡（阶段2） | qwen3.5-plus-2026-04-20 | 2.936 / 17.614 | 65536 / 2048 |
| 生成/北京（阶段2） | qwen3.5-plus | 0.8 / 4.8 | 65536 / 1024 |

分类预留低于256K档、生成低于128K档；超出预留须停止，而非自动按更高档继续。
[embedding规格](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)支持显式1024维，单行上限8192。
[rerank API规格](https://help.aliyun.com/zh/model-studio/rerank)列单条4000、请求120000，
总量按query tokens×候选数+文档tokens计算，不能只按wire query出现一次估价。
[模型概览页](https://help.aliyun.com/zh/model-studio/qwen3-rerank)另列30000输入限制，与API页不一致；
预算取较大的120000保守预留，不据此保证请求可被接受。执行适配必须先固定限长校验口径；
若无法在发送前验证实际输入符合端点限制，停在预检并报告，不截断/丢题/付费试探端点来解决文档冲突。

阶段1满额token预留 **¥29.433088**，拟记账限额 **¥35**。
这不是根据DEV长度得到的预测（本轮未读DEV），实际usage通常不等于满额预留；也不是账单硬上限。
发送前须确认现行价格未变，并实现“已记账+本次满额预留≤阶段限额”的检查；完成后按usage释放未使用预留。
若提供方usage超预留、计费口径变化或不可判断，保留pending并停，不能继续花剩余额度。

可复算检查（只用stdlib，无网络/数据/凭据读取）：

```python
from decimal import Decimal as D

def cost(n, prompt, output, pi, po="0"):
    return D(n) * (D(prompt) * D(pi) + D(output) * D(po)) / D(1_000_000)

p1 = cost(160, 8192, 0, ".5") + cost(320, 120000, 0, ".74942")
p2 = (cost(620, 8192, 0, ".5") + cost(310, 120000, 0, ".74942")
      + cost(310, 65536, 2048, "2.936", "17.614")
      + cost(620, 65536, 1024, ".8", "4.8"))
assert 160 + 320 == 480
assert 620 + 310 + 310 + 620 == 1860
assert p1 == D("29.433088") < D("35")
assert p2 == D("136.80224608") < D("150")
print("P1 reservation:", p1, "P2 service-only reservation:", p2)
```

## 5. 输入外发与断点边界

- 拟固定北京 `https://dashscope.aliyuncs.com/compatible-mode/v1`（embedding/生成），
  新加坡 `https://dashscope-intl.aliyuncs.com/compatible-api/v1`（rerank）及
  `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`（分类）；地域Key不混用。
  [端点说明](https://help.aliyun.com/zh/model-studio/base-url)及上述rerank API为依据。
  不因环境变量不同自动换主机、地域、模型或兼容路径；不调用模型列表/探针做付费连通检查。
- 阶段1仅外发该160条查询及每题两套最多100条完整候选正文、固定rerank指令/协议字段；
  不发qrel、gold answer、answerable/impossible标签、理由、业务资料、其他租户内容或密钥正文。
  qrel只用于发送后本地评分，与provider消息构造分离。
- 新授权后、首次发送前：先无网络核验数据SHA/160唯一ID/每题一个有效gold doc、非空查询、全库SQL/向量映射、
  模型输入限制与完整候选可见性；固定全部question ID顺序、问题投影哈希、源码/配置/预算哈希。
  此后候选按固定算法产生、核验授权租户/当前版本与原文；每次发送前保存payload哈希。
  某题超限/资产不一致则全局停止，先报告预检失败，不能悄悄截断问题/来源、跳题或缩减候选数。
- 评测wire body上限：embedding 32 KiB，rerank 512 KiB；阶段1累计上限
  `160*32768 + 320*524288 = 173015040` bytes。字节限制不是token上限或价格保证；两种检查都要做。
  每个question/arm/stage预分配唯一call ID，串行使用 [runtime_transport](../../runtime_transport.py)，
  SDK/HTTP重试0、无redirect/代理环境继承；embedding各HTTP阶段30秒，rerank15秒，不是总deadline。
- 完整响应和usage已持久化：无额外请求恢复原始结果，不重复花钱。
  网络/HTTP/响应结构/重排排列异常、缺usage、truncation、超预留、写盘/哈希/资产变化一律停整轮；
  不自动重跑，不修补提供方输出。正常完整排列的语义好坏如实记分，不因为指标差而停止/追加请求。
  有pending/旧锁或可能未知计费时先审计提供方记录和本地持久化文件；不自动删锁/清pending，也不把超时视作免费。
  清晰无未知计费且身份验证通过的未发送ID，才可在仍有效授权范围内继续；无法确认须另请用户决定。
- 未完成整160题不得声称正式全量验收完成；可公布已完成数量/错误/费用，指标只作明确标注的部分结果。
  原始请求/响应/逐题来源放在受控Git忽略目录，公共仓库只提交协议、指纹及聚合报告；不提交凭据/原始语料。

## 6. 阶段2：端到端服务与独立评审，尚不申请执行

候选固定为 `v5_3_1_experimental`，不是质量晋级；不追加v5.4或看DEV后择优换策略。
预定对310题（160 answerable+150 impossible）走隔离loopback真实HTTP `/rag/ask`，服务端JWT解析评测租户：
D使用Dense Top3、距离0.9；H使用Hybrid Top14、显式v5.3.1、不传距离阈值。
两臂同用上述受限生成（关闭思考、max_tokens=1024），Dense仍沿用现有提示词，H沿用现有完整Top14提示词/校验器。
这比较的是**上下文数量、检索、gate、提示词共同不同的整系统**，不是单独拒答消融或原封不动的普通Dense生成。
阶段2从generation元数据question取原文后 `rstrip()`；不拿阶段1缓存替代真实HTTP内embedding/rerank调用。

服务侧最多620 embedding、310 rerank、310分类、620生成，合计1860次；满额预留 **¥136.80224608**，
服务侧拟记账限额¥150。**不含证据标注/答案评审，不能把¥150说成全套验收总价**；不申请这笔授权。
65536输入预留、HTTP body上限、输入身份/运行库迁移、call-ID恢复适配需在阶段2协议中做可运行验证。
`qwen3.5-plus`是非日期模型别名，须记录响应model/执行时间，跨时段不能保证后端版本不变。

评审另冻结模型、完整提示词、输出schema、来源、次数与价格后单独授权；最多需620份按实际上下文的盲证据标注，
以及最多620份答案评审，实际可去重/拒答免答案评审但不能暗中追加调用。盲标不看arm/gate/答案/旧标签，
助手在不知道系统输出的条件下复核全部证据标签并冻结；之后再做答案评审，判定冲突保留歧义而非强行金标。
这是AI proxy+助手复核，不是独立领域专家gold；同族judge偏差必须披露，不让用户承担大量专业复核。

下一份阶段2协议必须在付费前固定：答案正确/依据支持/关键条件覆盖的逐题评分规则、
实际上下文充分/不足/歧义的分母、gate误放/误拒和最终回答行为的区别、格式/API失败按失败计入全部310题、
answered-only指标及回答覆盖率同时公布、逐臂错误数、分层160/150结果、配对区间、HTTP P50/P95及全部服务/评审费用。
原始impossible标签只作分层，不能把这些题一切回答都判幻觉；少答不是自动更安全或更正确。
在上述规则和评审费用未冻结前，不具备拒答/答案质量正式验收资格。

## 7. 执行前关口

本提交只有协议和可复算预算，没有DEV读取器/执行器，没有模型调用。
下一步先完成阶段1薄适配器：复用当前检索与wire守卫，仅加观测/预留记账/原始响应恢复，
用合成数据与MockTransport验证480调用/字节/费用上限、全排名与Top14一致、qrel不外发、超限停止和恢复不重发。
通过离线检查后，附具体提交与剩余预检风险请求**新DEV读取/外发/阶段1预算授权**；
未经明确授权不读取真实DEV、不发一个请求、不改日常服务。阶段1完成后如实报告，再决定是否投入阶段2，不能据此自动开放或部署。
