# Frozen controlled v5.1/v5.2 model diagnostic

Decision: **no evidence to adopt v5.2 from this diagnostic**. Keep v5.1 as the
experimental baseline, not a production-approved refusal policy. No runtime
integration, prompt change, natural TRAIN rescoring or DEV access occurred.

## What ran

14 fictional FixtureDesk cases, seven correlated positive/negative pairs; two
sources per case, not retrieval Top14. Targets authored by construction, not
independent human gold. Same Singapore `qwen3.5-plus-2026-04-20` snapshot, temperature
0, thinking disabled, JSON response and 2048 output cap. Alternate arm order per
case as preregistered. 28 calls; no retries, additional annotation or recharge.
Expected targets/rationales/IDs were excluded from requests. Per-case targets were
not opened by the paid loop; evaluation occurred only after all 28 terminal outcomes.

| Per-arm outcome | v5.1 | v5.2 |
| --- | ---: | ---: |
| All-case denominator | 14 | 14 |
| Valid and target-aligned | 14 | 12 |
| Valid SUFFICIENT targets admitted | 7/7 | 6/7 |
| Valid INSUFFICIENT targets refused | 7/7 | 6/7 |
| Format failures | 0 | 2 |
| False admission among valid outputs | 0 | 0 |
| False refusal among valid outputs | 0 | 0 |
| Median provider latency | 3.96 s | 7.40 s |
| Input/output tokens | 14,459 / 2,715 | 16,643 / 6,081 |
| Estimated accounting cost | CNY0.092247 | CNY0.159387 |

Total: 28 calls, estimated CNY0.251634. Rates: $0.4/$2.4 per million input/output
tokens, fixed 7.5 CNY/USD accounting, no discounts assumed
([official pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)).
This is not an invoice or general latency estimate. v5.2 cost is about 1.73x
v5.1, with about 1.87x median latency in this small run.

These are **controlled-fixture outcomes**, not real-data accuracy. The v5.2
valid subset has complete target agreement but excludes two failures: do not
report that as 14/14, conceal the denominator, or count errors as correct refusals.
The natural 30-case dataset remains 21 sufficient / 4 insufficient / 5 ambiguous;
its insufficient-class shortfall is unchanged. Neither result set is pooled.

## Failure evidence

Both failures are schema/provenance-reference failures, not provider failures,
unknown charges, truncated output or an observed false semantic status:

- `CONTROL_STEPS_SUPPORTED`: material requirements cite the creation and activation
  lines correctly, but coverage `applicability=SUPPORTED` has empty evidence.
- `CONTROL_CONFLICT_UNRESOLVED`: the material requirement correctly identifies
  the contradictory current policies and cites both sources; mandatory coverage
  checks declare CONFLICTING/SUPPORTED but leave their evidence arrays empty.

The frozen validator requires references for every SUPPORTED/CONFLICTING audit;
both fail with `supported/conflicting requirement needs evidence`. One failed
positive case and one failed negative case remain explicit null outcomes. No
post-hoc reference repair, prompt tuning, retry or reclassification was applied.
Raw responses, usage and failures remain under ignored local run data; public
JSON records per-case outcomes, confusion counts, hashes and cost only.

The coverage JSON example in the v5.2 prompt uses empty evidence arrays while
the surrounding instructions require nonempty evidence for supported/conflicting
audits. Example anchoring is a plausible contributor, not a proven causal finding.
If continuing v5.2 development, clarify this serialization example first; do not
weaken reference checks, add more audits, modify the frozen result, or treat a
rerun on these now-observed fixtures as an unseen validation set.

## Persistence and reproduction boundary

Execution contract pins inputs, source files, prompts, request hashes, decoding,
28-call/CNY3 accounting limits and separately frozen preparation. An exclusive
local lock prevents concurrent dispatch. Pending markers precede each request;
raw response/usage is fsynced before parsing, then final attempts are fsynced.
A saved raw response replays locally without a new request; unknown outcomes or
usage block resume until accounting audit. Schema failures are terminal outcomes
and never retried. Successful outcomes are never repeated. Stale locks require
confirming the original process has stopped before removing the exact lock file.

```powershell
python -m experiments.evals.refusal_v5_2_control_runner --run-paid --env-file 'path\to\.env'
```

The completed local checkpoint returns without extra provider calls. A fresh
checkout does not contain ignored inputs/raw journals: reconstruct the exact
question/source-only projection of public control fixtures and verify its SHA
against the contract before using it; do not overwrite the public contract or
present a new execution as the original run. Local raw journals are required to
audit/replay original responses. Future changed prompts or new calls require a
distinct registered development run and separately scoped user authority.
