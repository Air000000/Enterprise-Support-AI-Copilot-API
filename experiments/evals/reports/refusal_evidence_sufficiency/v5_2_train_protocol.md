# v5.2 additional TRAIN diagnostic preparation

Status: packets selected, annotations pending, no provider execution authorized.
This is step 1 only; neither a model result nor production acceptance.

## Population and privacy

Reuse the frozen 450-case TRAIN snapshot and flat Top14, without new retrieval,
truncation, gold-based strata, DEV, or changing historical experiments.
Exclude all earlier refusal packets (not just paid cases), recorded targeted
research samples, normalized duplicate questions and shared leading retrieved
documents. The latter is a conservative **family proxy**, not proof of semantic
independence. All TRAIN already participated in retrieval and automated failure
analysis. This sample is additional diagnosis, not an unseen project holdout.

Select the lowest SHA256(seed:question_id), then question-ID order. Do not balance,
replace or resample after labels/predictions. Category coverage is unknown until
annotation. Public metadata contains IDs, hashes and budgets only; questions,
Top14 text, labels and rationales stay under ignored `data/refusal_v5_2_train/`.

Regenerate once in a clean checkout with the frozen research artifacts available:

```powershell
python -m experiments.evals.refusal_v5_2_preparation --research-root 'D:\文档\GitHub\fastapi-todo-api'
```

The tool refuses existing outputs. Its public manifest is already frozen; do not
overwrite it to repair an experiment. Explicit amendments must preserve originals.

## Blind annotation contract

The annotator sees only the question and ordered Source 1..14 text, not model
predictions, previous labels, gold answers, retrieval scores or document IDs.
Return one compact row per packet in the same order:

```json
{"question_id":"TRAIN_Qxxx","annotation":{"target_class":"SUFFICIENT|INSUFFICIENT|AMBIGUOUS","supporting_source_ids":["Source 1"],"notes":"Requirements, direct evidence and material gaps"}}
```

SUFFICIENT: all material requested outcomes, applicability and explicit version/
time requirements supported by visible evidence. General explanations need not
diagnose a particular incident. Missing literal names alone is not a gap.
INSUFFICIENT: at least one necessary step, applicability bridge, version/time
constraint, condition or conflict remains unsupported/unresolved. AMBIGUOUS:
the scope or evidence admits materially different defensible judgments.
Notes must identify source-local original lines or a concrete missing requirement;
never fill a gap using outside knowledge. Empty source IDs are allowed for absence
of support; references must exist. No fabricated quote or forced binary label.

Use a separately recorded annotator identity/model and an evidence review before
any classifier predictions; record disagreements rather than silently overriding.
AI labels plus assistant review remain an AI proxy, not independent human gold.
Freeze local label hashes, category counts and provenance first. Category tags
(steps, applicability, version/time, condition, conflict) may overlap; they do not
alter the sample. Insufficient category coverage limits conclusions, not selection.

## Prospective paired execution and decision rule

Not implemented or authorized by this preparation. A later execution contract
must bind frozen labels, inputs, prompts, validators and shared dependencies.
Same model, ordered full context, temperature 0, thinking off, JSON response and
2048 output-token cap for both v5.1 and v5.2; alternate arm order by case index.
Plan 60 calls. Additional attempts require an explicit budget/authorization amendment;
do not retry successful predictions or treat transport/schema failures as refusals.
Checkpoint raw responses/usage before parsing; keep labels outside the paid loop.

Report all 30 outcomes per arm, valid-pair confusion, ambiguous and failed counts,
false admissions/refusals, paired transitions, tokens, cost and latency. Report
admission coverage together with accuracy among admitted cases. Missing/ambiguous/
failed cases never disappear from denominators. Keep controlled synthetic tests
separate from real-data results.

Only consider further validation if v5.2 reduces false admissions on common valid
non-ambiguous pairs, increases false refusals by at most one case, and introduces
no extra failures; require at least five cases in each binary class. Otherwise
report no demonstrated gain or inconclusive evidence. Also report worst-case
failure sensitivity; passing these small-sample rules is **not** deployment approval.
No post-result prompt tuning, label repair or promotion on this diagnostic sample.

## Budget preflight

The manifest records request fingerprints and UTF-8 byte counts plus 512 framing
as a conservative planning proxy, not a certified tokenizer or hard invoice bound.
Singapore international pricing checked 2026-10-13: $0.4 input / $2.4 output per
million tokens in the <=256K input tier ([official pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)).
No discounts assumed. Annotation, retries, FX and provider changes are excluded.
Refresh prices and establish an enforceable spend cap before requesting execution.
Provider calls so far: zero. Runtime refusal and ticket preview/confirm are unchanged.
