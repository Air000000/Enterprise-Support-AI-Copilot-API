# Refusal v3 AI-proxy development preregistration

Status: `PREREGISTERED_NOT_RUN`

## Scope

This is a TRAIN development probe against blind GPT-produced proxy labels.
It is not independent human confirmation and cannot admit Phase C, generation,
refusal-policy freeze, or runtime integration.

The only positive outcome is
`ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION`.

## Why the classifier prompt is unchanged

The v2.1 disagreement audit found a mismatch between the old gold-equivalent
annotation contract and the classifier's visible-question task. v3 corrected
the labels and physically hid gold fields from the annotator. The v2 classifier
prompt already judges completeness against the exact visible question, so
changing it here would confound the label repair with prompt tuning.

The prompt remains frozen at SHA-256
`eff5768921067cc04a0686fff0ff50599f4bf295f3ad263e63a143f110fe3c20`.

## Frozen proxy holdout

The 71 usable blind AI-draft labels contain 44 sufficient and 27 insufficient
cases. Nine questionable cases are excluded. Within each binary class, cases
are ordered by
`SHA256("refusal-v3-ai-proxy-holdout-v1:" + question_id)`, and the lowest 25
are selected.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| Classifier inputs | 50 | `87f91ba55b78bfc16db2c018b092212e39cf9d171b52e62853a5e536b6fd344d` |
| Hidden proxy targets | 50 | `a1965b36786b96d956a63b2af6533d14c1182fb3b42155ab8cd15879b7823c83` |
| Source compact targets | 80 | `5b0d8b5b167e8ae742210c2deb8ff91c711143bf8da10451bc15f835d9ca8d68` |

The classifier payload contains only the question and ordered `Source 1`
through `Source 14` contents. The record ID is retained for local checkpoint
joins but must not be sent to the provider. Targets, annotation rationales,
source metadata, selection strata, and gold fields are excluded.

## Frozen classifier

- Provider: Alibaba Cloud Model Studio, Singapore
- Model: `qwen3.5-plus-2026-04-20`
- Temperature: `0.0`
- Thinking: disabled
- Response format: JSON object
- Maximum output: 512 tokens
- Required valid predictions: 50
- Maximum provider attempts: 55
- Hard accounting cap: CNY 1.5
- Context: frozen `flat_rerank_top14_v1`

Every provider attempt counts toward the call and accounting limits. Valid
predictions are checkpointed and never retried. The five-attempt margin is
only for provider or schema failures; it does not permit resampling a valid
prediction.

## Development gate

The 50-case holdout is class-balanced by construction:

- balanced accuracy >= 0.80
- sufficient recall >= 0.85
- insufficient recall >= 0.70
- sufficient to insufficient <= 3

The existing three-case false-refusal ceiling is retained, not scaled up.
With 25 sufficient cases, recall changes in 0.04 increments. Both the 0.85
recall threshold and the three-case ceiling therefore require at least
22/25 (0.88); these constraints are equivalent at this sample size.

Targets remain unopened by the paid loop and are loaded only by a separate
evaluation command after 50 valid predictions exist. Failed attempts and
their costs remain part of the final usage report.

Any failed metric yields `REJECT_EVIDENCE_SUFFICIENCY_V3_AI_PROXY`. A pass
only admits independent human confirmation. It does not admit Phase C.

## Controls

- DEV remains unopened.
- Retrieval, rerank, generation, and judge calls are forbidden.
- The labels remain `AI_DRAFT_DEVELOPMENT_ONLY`.
- The external GPT model, prompt, decoding settings, calls, and cost are
  unknown and must not be reconstructed or claimed.
- This preregistration and input construction make zero provider calls.
- The CNY 1.5 cap uses the frozen v2 accounting estimate, not verified current
  provider billing. Current pricing must be checked before the paid run.
