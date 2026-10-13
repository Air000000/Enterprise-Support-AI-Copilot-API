# Frozen natural v5.1 / v5.2.1 exploratory diagnosis

Decision: **development diagnosis complete, no acceptance or production promotion**.
The candidate has two more binary agreements with the frozen AI proxy, primarily
through fewer sufficient-case refusals. It does not reduce insufficient admissions.
Known correlated cases, new disagreements and higher cost/latency limit this gain.

## What ran

All 30 frozen natural TRAIN questions with their original ordered Top14, no new
retrieval/truncation/annotation, relabeling, balancing or replacement. Labels remain
21 SUFFICIENT, 4 INSUFFICIENT and 5 AMBIGUOUS: Qwen draft plus current-assistant
evidence review before predictions, not independent expert gold or representative
enterprise traffic. All TRAIN previously participated in project retrieval/analysis.
The [original five-per-class rule](v5_2_train_protocol.md) stays unmet and unchanged.
The [separate exploratory protocol](natural_v5_2_1_diagnostic_protocol.md) has no pass gate.

Singapore `qwen3.5-plus-2026-04-20`, temperature 0, thinking off, JSON response,
2048-token cap per arm, alternating arm order. Exactly 60 calls, 30 per arm,
no retries/recharge, provider/usage/schema failures or DEV access. Requests contain
question text and numbered source lines, not IDs, labels, reasons or retrieval
metadata. Evaluation opened target rows only after all 60 terminal outcomes.

## Results relative to the frozen AI proxy

| Population / outcome | v5.1 | v5.2.1 |
| --- | ---: | ---: |
| All-case failures | 0/30 | 0/30 |
| Binary target agreement | 19/25 (76%) | 21/25 (84%) |
| Sufficient targets admitted | 16/21 | 18/21 |
| Sufficient targets refused | 5/21 | 3/21 |
| Insufficient targets admitted | 1/4 | 1/4 |
| Insufficient targets refused | 3/4 | 3/4 |
| All-case admission coverage | 19/30 | 22/30 |
| Admitted binary proxy agreement | 16/17 | 18/19 |
| Ambiguous admissions / refusals | 2/3 | 3/2 |

Five ambiguous targets are descriptive only, not correct/incorrect binary labels.
Common-valid binary pairs include all25. Paired agreement counts: both agree17,
candidate only4, baseline only2, neither2. There are no null outcomes in this run,
so binary failure-sensitivity lower/upper bounds coincide at 76% and 84%.
These percentages are **proxy agreement**, not measured real-world answer accuracy.

Private paired attribution shows four sufficient-case refusals corrected and two
new sufficient-case refusals introduced; every insufficient-case decision is
unchanged, including the same remaining false admission relative to the proxy.
One ambiguous case changes from refusal to admission; it is not another success.

Two of the four corrected refusals are the already-known near-duplicate pair.
Preregistered drop-one sensitivity: removing either member gives baseline19/24
versus candidate20/24, with unchanged 1/4 insufficient admissions. The apparent
net gain falls from two cases to one. This is limited sensitivity, not a complete
family partition, IID significance, unseen validation or proof of generalization.
One insufficient-case outcome would change that class's rate by 1/4.

## Targeted local diagnosis, not label repair

For the two new proxy disagreements, all mandatory coverage audits say SUPPORTED;
additional material requirements are CONDITIONAL and correctly trigger the gate.
Targeted inspection suggests requirement inflation from alternative/off-scope
search instructions and fix-version comparisons for different symptoms. This is
an interpretation of selected local evidence, not independent expert adjudication.
It does not establish that every proxy label is correct. Frozen labels and outputs
are neither changed nor rescored.

The shared insufficient admission still treats related product/security-update
and classloader guidance as enough to resolve a specific incident despite absent
patch/configuration/classpath evidence. Literal product/update matching and valid
citations do not establish the incident's cause or remedy. Mandatory audit fields
have not eliminated falsely SUPPORTED judgments. Do not weaken citation checks or
silently add requirements/repair evidence after seeing these outcomes.

## Cost, latency and evidence boundary

Total estimated fixed-FX accounting cost **CNY1.663062**, below CNY8; not an invoice.
Frozen [International pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
checked 2026-10-13: $0.4/$2.4 per million input/output tokens, 7.5 CNY/USD, no discounts.

| Usage | v5.1 | v5.2.1 |
| --- | ---: | ---: |
| Input/output tokens | 162,730 / 11,960 | 178,030 / 23,639 |
| Estimated cost | CNY0.703470 | CNY0.959592 |
| Median provider latency | 6.90 s | 11.31 s |

Candidate cost is about1.36x and median latency1.64x in this run. Net two proxy
agreements and unchanged insufficient admissions do not establish that this tradeoff
is worthwhile or justify replacing the experimental baseline/online policy.
No fictional scores are pooled into these denominators. No post-result paid tuning
or new labels are included; future work requires a separate scope and authority.

The contract pins preparation, runner/dependency sources, inputs, prompts, requests,
decoding and budgets. Existing durable checkpoint machinery is reused in an isolated
namespace. Raw response/usage precedes parsing; schema outcomes are terminal,
successes never repeat and unknown charge blocks resume. Completed offline replay
adds zero calls, with no pending/lock/raw-stage leftovers. Old experiment and all
natural annotation/input/recovery hashes remain unchanged; original rules remain.

Public JSON contains aggregates/hashes only, never real questions/source bodies,
target rows or raw responses. Per-case transitions and evidence remain ignored
under `data/refusal_natural_v5_2_1/run_v1`. Private summary byte SHA256:
`3855e2de65fe3d5c191e68233e1a0081717b78eaaaa086ecbb32d23866217e36`.
A fresh checkout lacks the private corpus/journals; metadata alone cannot reproduce
or audit the original execution. The 60-call authorization is exhausted; do not
start an empty new checkpoint or overwrite the frozen result to rerun it.
