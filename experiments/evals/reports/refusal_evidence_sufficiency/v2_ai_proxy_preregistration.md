# Refusal v2 AI-proxy development preregistration

Status: `PREREGISTERED_NOT_RUN`

## Scope

This is a TRAIN development probe against GPT-produced proxy labels. It is
not an independent confirmation set and cannot promote the refusal policy,
admit Phase C, open generation work, or change runtime behavior.

The only allowed positive outcome is
`ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION`.

## Why v2 differs from v1

The v1.1 aggregate result showed over-admission: 8 of 16 insufficient
proxies were classified `SUFFICIENT`. The v2 system prompt therefore makes
one change: it requires direct support for every material requirement in the
exact question and explicitly rejects neighboring-topic evidence, omitted
claims, and plausible unsupported inference.

No AI-draft case-specific rationale was used to write the prompt. During the
integrity audit, details for `TRAIN_Q000` and `TRAIN_Q005` were displayed;
both are deterministically excluded from this holdout.

## Frozen proxy holdout

The 74 usable AI-draft labels contain 38 sufficient and 36 insufficient
cases. Six questionable cases are excluded. Within each binary class, cases
are ordered by
`SHA256("refusal-v2-ai-proxy-holdout-v1:" + question_id)`, and the lowest 20
are selected.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| Classifier inputs | 40 | `0b5f9cd0c797930c7d7efefe5f9721435cb94de3bb98721d2a703d50b4cb269d` |
| Hidden proxy targets | 40 | `b4828f0b9870c53774c5b78722ddf2f3dc0d23a85aeea71281a4cda6f1603fd7` |
| v2 system prompt | — | `eff5768921067cc04a0686fff0ff50599f4bf295f3ad263e63a143f110fe3c20` |

The classifier payload contains only the question and ordered `Source 1`
through `Source 14` contents. The record ID is retained for checkpoint joins
but must not be sent to the provider. Gold answers, qrels, document/chunk
IDs, sampling strata, AI targets, source rationales, and notes are excluded.

## Frozen classifier

- Provider: Alibaba Cloud Model Studio, Singapore
- Model: `qwen3.5-plus-2026-04-20`
- Temperature: `0.0`
- Thinking: disabled
- Response format: JSON object
- Maximum output: 256 tokens
- Maximum calls/attempts: 40
- Hard cost cap: CNY 1.5
- Context: frozen `flat_rerank_top14_v1`

The Singapore-scoped key/base-URL rules from v1.1 remain mandatory. Every
provider attempt counts toward the call and cost limits and is checkpointed.
Successful cases are never retried.

## Development gate

The 40-case proxy holdout is class-balanced by construction:

- balanced accuracy >= 0.80
- sufficient recall >= 0.85
- insufficient recall >= 0.70
- sufficient to insufficient <= 3

Accuracy is reported but cannot be interpreted as production-distribution
accuracy. Targets remain unopened by the paid loop and are loaded only by a
separate evaluation command after all predictions are checkpointed.

If any gate fails, the decision is
`REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY`. The same holdout may not be reused
for prompt tuning and another claimed v2 run.

If every gate passes, the decision is
`ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION`. Phase C, generation, refusal
freeze, and runtime integration remain blocked.

## Controls

- DEV artifacts remain unopened.
- Retrieval, rerank, generation, and judge calls are forbidden.
- The external GPT labels remain `AI_DRAFT_DEVELOPMENT_ONLY`.
- The external GPT model, prompt, decoding settings, calls, and cost are
  unknown and must not be reconstructed or claimed.
- Preflight and input construction make zero provider calls.
