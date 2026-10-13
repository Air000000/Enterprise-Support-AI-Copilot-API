# Frozen v5.2 / v5.2.1 development regression

Decision: **no demonstrated format-fix gain** under the preregistered comparison.
Keep v5.1 as the experimental baseline; no production promotion. The v5.2.1
example correction remains an experimental prompt candidate, not a proven
reliability or semantic-coverage improvement.

## Execution and result

14 already-observed fictional controls, seven correlated pairs, two sources per
case, construction targets rather than independent expert gold. Fresh concurrent
v5.2 versus v5.2.1, same numbered user payload and unchanged validator. Singapore
`qwen3.5-plus-2026-04-20`, temperature 0, thinking off, JSON output, 2048-token cap;
alternate arm order. 28 calls, no retries, recharge or extra annotation. Targets
were excluded from requests and the paid loop, then opened for evaluation after
all terminal outcomes. No natural TRAIN pooling, label changes, DEV or serving.

| All-case outcome | v5.2 | v5.2.1 |
| --- | ---: | ---: |
| Valid and target-aligned | 14/14 | 14/14 |
| SUFFICIENT targets admitted | 7/7 | 7/7 |
| INSUFFICIENT targets refused | 7/7 | 7/7 |
| Format/provider failures | 0 | 0 |
| Valid false admission / false refusal | 0 / 0 | 0 / 0 |
| Input/output tokens | 16,643 / 6,398 | 21,599 / 6,246 |
| Estimated cost | CNY0.165093 | CNY0.177225 |
| Median provider latency | 7.70 s | 7.60 s |

Total estimated accounting cost: **CNY0.342318**, below the proposed CNY3 cap;
not an invoice. Frozen [official International pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
snapshot checked 2026-10-13: $0.4/$2.4 per million input/output tokens, fixed
7.5 CNY/USD, no discounts. v5.2.1 cost about 1.07x v5.2 in this run; the small
latency difference is not evidence of a general speed improvement.

The common-valid subset is all 14 cases, both arms agree with all construction
targets, and there are no new valid semantic errors relative to those targets.
However, both arms have zero failures: the required strict reduction in failures
was not observed, so the result is inconclusive for the citation-example hypothesis.
These fixture outcomes are not real-data accuracy or a formal acceptance score.
The validator still cannot detect false SUPPORTED/NOT_REQUIRED semantic judgments.

## Relation to PR70

Original v5.2 had two audit-citation format failures in the [earlier run](v5_2_control_result.md).
The same frozen v5.2 requests now pass both `CONTROL_STEPS_SUPPORTED` and
`CONTROL_CONFLICT_UNRESOLVED`, citing evidence for their SUPPORTED/CONFLICTING
audits. Both v5.2.1 outputs also provide the required citations. Original input
tokens, source fingerprints and request hashes match; original failures and raw
journals remain unchanged. The failures were not repaired or reclassified.

The old failures were not reproduced here. Cross-run outcome variation is observed;
its cause is not established. Do not attribute the historical 12/14 to current
14/14 change to the new prompt, since the concurrent unchanged arm also got 14/14.
Do not spend more calls on these same fixtures until a desired result appears.
Natural insufficient-class count remains four; this result does not resolve that
shortfall. Broader claims require a separately scoped representative evaluation.

## Persistence boundary

The new contract binds PR72 preparation, source dependencies, runner source,
requests, inputs, decoding and budgets. A stdlib-isolated namespace reuses the
frozen checkpoint engine without modifying its original module bindings. Raw
response/usage is durably saved before parsing; final journals, pending markers,
exclusive lock and unknown-charge blocking retain the existing audited behavior.
Each successful or schema-failed outcome is terminal and is never repeated.

The completed checkpoint replays with zero extra provider calls. Original journals
and all natural annotation/input/recovery hashes were verified unchanged. New raw
journals remain ignored under `data/refusal_v5_2_1_controls/run_v1`; compact public
JSON includes outcomes, metrics and hashes only. A fresh checkout lacks the ignored
inputs/journals, so it cannot reproduce the original execution from scores alone.
Do not overwrite the run contract/result or use an empty checkpoint to spend a new
28 calls under the exhausted authorization. Future execution needs new authority.
