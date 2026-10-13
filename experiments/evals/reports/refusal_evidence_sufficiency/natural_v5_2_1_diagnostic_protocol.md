# Natural v5.1 / v5.2.1 exploratory development diagnosis

Status: **prepared, not run or authorized; no paid entrypoint**.

Reuse all 30 frozen natural TRAIN questions and their unchanged full ordered Top14
contexts. Compare current experimental v5.1 with prompt-only v5.2.1 under identical
decoding; the original v5.2 is not another arm. No new retrieval, model annotation,
relabeling, class balancing, replacement, context truncation or DEV access.

## Why this is diagnosis, not representative acceptance

The frozen AI-proxy labels are 21 SUFFICIENT, 4 INSUFFICIENT and 5 AMBIGUOUS.
They were annotated by Qwen and reviewed by the current assistant before any
classifier predictions, not independently adjudicated by domain experts. Every
TRAIN question already participated in project retrieval/automated analysis.
The known near-duplicate pair and absent conflict-category examples limit coverage.
Neither enterprise traffic representativeness nor independence is established.

The [original v5.2 protocol](v5_2_train_protocol.md) requires five cases in each
binary class. That rule and the old labels remain unchanged and unmet. This new
v5.1/v5.2.1 diagnostic has **no formal pass gate**, regardless of the scores.
It cannot retrospectively pass the original experiment or authorize deployment.
Its purpose is to identify where coverage auditing changes admissions in natural
contexts, and whether those changes agree with the frozen proxy, at what cost.

| Reporting population | Denominator / treatment |
| --- | --- |
| All requests and failures per arm | 30; null failures never count as refusals |
| Binary proxy target agreement | 25; retain binary failures in the denominator |
| Class-level proxy agreement | 21 sufficient / 4 insufficient; show counts first |
| Ambiguous targets | 5; decisions/failures descriptive, never correct/incorrect gold |
| Common-valid binary pairs | Actual subset size and class counts explicitly reported |

Report paired decisions, false admission/refusal counts relative to the proxy,
admission coverage on all30, proxy agreement among admitted binary cases separately,
and ambiguous admission counts. One insufficient-case change is 1/4, not evidence
of a robust gain. Null binary outcomes also produce lower/upper possible agreement
bounds, not fabricated decisions. Empty metric populations return undefined, not
zero accuracy. Do not claim IID significance/confidence intervals from these data.

Retain all30 as primary. Separately recompute metrics dropping each of the known
correlated members TRAIN_Q383 and TRAIN_Q417 in turn, without paid repeats or label
changes. This limited sensitivity is not a complete family partition or independent
validation. Do not pool fictional control scores into these real-context denominators.

## Proposed execution and privacy

Singapore `qwen3.5-plus-2026-04-20`, temperature 0, thinking off, JSON output,
2048-token output cap per arm. Frozen question-ID order; alternate arm order by
case index. At most 60 calls, no retries; proposed fixed-FX accounting cap CNY8.
Exact byte/framing planning estimate is frozen in the preparation JSON. Official
[International pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
checked 2026-10-13: $0.4/$2.4 per million input/output tokens up to 256K input,
7.5 CNY/USD accounting, no discounts. The planning proxy is not certified tokens,
spot FX or an invoice guarantee; record actual usage and charges separately.

Only question text and numbered Source 1..14 lines enter classifier requests, not
question IDs, labels, annotation reasons, retrieval metadata or credentials. Source
text is untrusted data. Public artifacts contain IDs/hashes/counts/budgets only;
question/source bodies, per-case target rows and raw model responses remain ignored.

Future paid execution requires separate scoped authority and a runner-bound contract.
Use a new checkpoint, exclusive local lock, pending-before-request and durable raw
response/usage before parsing. Schema failures are terminal; successes never repeat;
unknown response/charge blocks resume. Labels stay outside the paid loop until all
60 outcomes are complete. Prior experiments and checkpoints remain immutable.
No post-unblinding tuning/relabeling, automatic recharge or serving integration.
