# Evaluation Cost Ledger

This ledger tracks model/API spend at experiment-stage granularity. Historical spend is approximate unless a provider invoice or exact stage-level token record is available.

| Stage | Question answered | API | Estimated cap | Actual spend | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Completed E0/E1 | establish the baseline and test the ranking-error rerank hypothesis | mixed historical | historical | ≈ CNY 41 cumulative | frozen |
| R2 | characterize E1 TRAIN residuals | local | CNY 0 | CNY 0 | pending |
| R3 | test lexical complementarity with BM25/RRF | local | CNY 0 API | CNY 0 API | ADMIT_PAID_R4 |
| R4 C1 | test Hybrid candidate pool + rerank | qwen3-rerank | USD 1.50 safety envelope | 12,080,467 formal provider tokens; monetary billing not yet reconciled | STOP ? C1 failed pre-registered MRR gate |
| G1 | test generator bottleneck on frozen context | candidate LLM | freeze before run | — | gated |
| D1 | validate the final frozen generation configuration | final LLM/eval | final approval | — | gated |

## Governance rule

Before any new paid stage begins, freeze its hypothesis, reusable artifacts, minimum model-call count, hard cost cap, stop condition, expected quantitative output, and resume/interview evidence. Record actual spend after the stage finishes. Bulk LLM-as-Judge evaluation is not a default action.

## R4 C1 accounting note

The formal 450-query C1 TRAIN run recorded 12,080,467 provider tokens.
A separate successful three-document connectivity diagnostic recorded
88 tokens and is excluded from the formal C1 metric/token total.
Earlier rejected HTTP 400 and client-side URL/TLS failures do not have
reliable provider token or billing evidence, so no monetary amount is
invented for them.
