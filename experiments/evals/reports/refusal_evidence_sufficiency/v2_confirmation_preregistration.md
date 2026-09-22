# Refusal v2 confirmation-set preregistration

Status: `PREREGISTERED_NOT_ANNOTATED`

## Why this phase exists

Refusal Evidence Sufficiency v1.1 is rejected: balanced accuracy was
`0.735714` and insufficient-context recall was `0.50`. The failed 54-case
evaluation is not reusable as a v2 confirmation set. Phase C, generation
experiments, and runtime integration remain blocked.

This phase constructs a fresh, human-labelled confirmation set. It does not
define, run, tune, or evaluate a v2 classifier.

## Frozen inputs

Only these frozen TRAIN artifacts may be read:

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| TechQA generation metadata | 910 | `69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7` |
| R4 C1 fused snapshot | 450 | `12db56e50efaf11dab4a28ff3c1b4df223e2ad985a8b48021f4c7dd9fdd889d2` |
| R4 C1 rerank results | 450 | `12b312a8403cda2c5fe8afe53aa18853891a7e87feaf863e2d438ca15367ee5b` |
| Prior evidence labels | 60 | `d522eff8daba8435d34ee0e51ad8c56fbbbb759b3f71dbf3a253cd9a1b506013` |

DEV artifacts must not be opened. Selection and annotation make no provider
calls.

## Deterministic selection

1. Start from answerable TechQA TRAIN cases present in both frozen retrieval
   artifacts.
2. Exclude all 60 cases from the prior evidence audit, including cases not
   used in the v1.1 score.
3. Partition the remaining cases into two sampling strata according to
   whether the single gold document occurs in the frozen reranked Top14.
   These strata are sampling aids, not target labels.
4. Within each stratum, sort by
   `SHA256("refusal-v2-confirmation-set-v1:" + question_id)` and take the
   lowest 40 hashes.
5. Freeze the 80 IDs and the annotation-packet SHA before any v2 classifier
   prediction exists.

The selection report may contain the sampling stratum for auditability. It
must never be supplied to a classifier.

Because this is a deliberately stratified confirmation set, it cannot
estimate the natural prevalence of sufficient or insufficient production
contexts. Any later promotion gate must use per-class metrics; aggregate
accuracy must not be presented as a production-distribution estimate.

## Human annotation contract

The annotator receives the question, gold answer, gold document ID, and the
complete frozen Top14 context in reranked order. Each usable case must have
one target and a short rationale:

- `SUFFICIENT`: the Top14 context directly supports every material claim
  needed for a correct gold-equivalent answer.
- `INSUFFICIENT`: at least one material claim is absent, contradictory, or
  requires unsupported inference.
- `QUESTIONABLE`: the question/gold pair is materially ambiguous, broken, or
  cannot be judged reliably. These cases are excluded from scoring.

For `SUFFICIENT`, at least one valid `Source N` identifier is required. For
`INSUFFICIENT`, source IDs may be empty; the notes must state what material
support is missing. The annotator must not run or inspect v2 predictions.

## Leakage barrier and freeze rule

Before classifier design or prediction, annotations are validated and frozen
with a SHA-256 digest. Classifier payloads may contain only `question` and the
ordered `Source 1` through `Source 14` contents. They must exclude gold
answers, gold/qrel document IDs, sampling strata, selection hashes, human
targets, notes, and prior predictions.

If either usable target class has fewer than 25 cases, this confirmation run
is cancelled. Any expansion requires a new deterministic preregistration
before additional cases are opened; observed labels may not be used to
cherry-pick cases or tune a classifier.

## Exit condition

This phase ends after the 80 packets are selected. Its decision is
`PROCEED_TO_BLIND_HUMAN_ANNOTATION`, not approval of a v2 classifier. The v2
prompt, model, decoding settings, cost cap, and promotion gate require a
separate preregistration after targets are frozen and before predictions.
