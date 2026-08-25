# Evaluation Cost Ledger

This ledger tracks model/API spend at experiment-stage granularity. Historical spend is approximate unless a provider invoice or exact stage-level token record is available.

| Stage | Question answered | API | Estimated cap | Actual spend | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Completed E0/E1 | establish the baseline and test the ranking-error rerank hypothesis | mixed historical | historical | ≈ CNY 41 cumulative | frozen |
| R2 | characterize E1 TRAIN residuals | local | CNY 0 | CNY 0 | pending |
| R3 | test lexical complementarity with BM25/RRF | local | CNY 0 API | CNY 0 API | pending |
| R4 | test Hybrid candidate pool + rerank | qwen3-rerank | freeze before run | — | gated |
| G1 | test generator bottleneck on frozen context | candidate LLM | freeze before run | — | gated |
| D1 | validate the final frozen generation configuration | final LLM/eval | final approval | — | gated |

## Governance rule

Before any new paid stage begins, freeze its hypothesis, reusable artifacts, minimum model-call count, hard cost cap, stop condition, expected quantitative output, and resume/interview evidence. Record actual spend after the stage finishes. Bulk LLM-as-Judge evaluation is not a default action.
