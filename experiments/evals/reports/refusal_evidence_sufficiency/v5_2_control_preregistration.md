# Controlled v5.1/v5.2 stress-diagnostic preparation

Status: fixtures and preflight only, zero provider calls. No paid runner or
execution authorization, real-data augmentation, DEV access or serving change.

The natural 30-case TRAIN diagnostic remains frozen at 21 sufficient, 4
insufficient and 5 ambiguous. Adding one convenient insufficient example would
not make that sample independent or statistically reliable. Do not resample it,
relabel it to meet the five-case rule, or add controlled cases to its denominators.

## Fixed construction

Fourteen explicitly fictional FixtureDesk cases form seven positive/negative
pairs. Each pair holds the question and one source fixed, changes one source,
and flips whether the requested supported answer can be established:

- Steps: token creation versus creation plus existing-worker activation.
- Applicability: confirmed target incident versus an unrelated incident.
- Version: requested-release source bundle versus old-release source/binary-only support.
- Time: explicit requested-date latest-release evidence versus an older snapshot.
- Condition: confirmed mandatory memory condition versus an unmeasured node.
- Conflict: distinct archived/current scopes versus unresolved equally current contradictions.
- Negative answer: an explicit supported No versus a default that proves neither Yes nor No.

Expected classes/rationales are authored by construction, not independently
annotated human gold. Fixtures are intentionally simple and correlated within
pairs; not 14 independent trials. Two supplied sources per case, not frozen
retrieval Top14. Fictional source bodies are safe to publish; real questions,
source bodies and annotations from the natural diagnostic remain local.

This differs from the existing gate demo: that demo feeds hand-written model
responses to the parser. These fixtures are potential **model inputs**; only a
future actual model response can test whether it recognizes a gap/conflict.
Structural tests and preflight are not model accuracy scores.

## Future execution boundary

Reuse the frozen v5.1 and v5.2 prompts/validators with identical visible input,
model snapshot, decoding and 2048 output cap. Plan 28 calls, alternate arm order
by case index. Project only question and source text; expected targets, rationale,
family and case IDs never enter model requests. Targets must remain outside
the paid loop. Report every attempted outcome and each family/arm separately:
false admission, false refusal, format/provider failure, paired discrimination,
latency, tokens and estimated cost. A schema failure is not a correct refusal.

Passing controls demonstrates only basic fixture handling, not real-data
accuracy, superiority or deployment readiness. Simple-control failures warrant
investigation before spending on a larger model comparison. Never tune prompts
or rewrite fixtures after seeing predictions and then reuse them as an unseen
benchmark; retain results and register a distinct development revision instead.
Do not pool controlled and natural results to satisfy the natural decision rule.

```powershell
python -m experiments.evals.refusal_v5_2_controls
```

The offline preparer refuses existing output. The manifest pins canonical fixture
and request hashes, prompt/validator fingerprints and budget projections. Prices
are the existing Singapore international qwen3.5-plus-2026-04-20 rates, checked
2026-10-13 ([official pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)).
UTF-8 byte+framing cost is a planning proxy, not a tokenizer or invoice cap. A
future executable contract must set spend/call limits, fingerprint the runner,
and provide durable raw-response/usage checkpoints plus unknown-charge audit.
No annotation calls, automatic recharge, successful-outcome retries or production
promotion are authorized by this preparation.
