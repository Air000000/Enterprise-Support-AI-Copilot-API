# Refusal v3 blind annotation preregistration

Status: `SELECTED_NOT_ANNOTATED`

## Purpose

v3 repairs the construct mismatch found after the v2.1 AI-proxy run. The
target is runtime answerability from the exact visible question and frozen
Top14 context. It is not equivalence to a hidden gold answer.

This stage selects and packages fresh TRAIN cases. It makes no classifier,
generation, retrieval, rerank, or judge calls and does not open DEV.

## Fresh deterministic selection

- Exclude all 60 cases from the earlier evidence audit.
- Exclude all 80 cases from the v2 confirmation set.
- Use seed `refusal-v3-confirmation-set-v1`.
- Select the lowest 40 hashes from each retrieval stratum: gold document in
  Top14 and gold document outside Top14.
- Keep the stratum only in the tracked selection manifest; do not expose it
  to the annotator.

The gold-document strata are used only to ensure retrieval-condition
coverage. They do not define the target label.

Selection completed with 80 unique cases and zero overlap with either
excluded population. The local annotation packet SHA-256 is
`5afa8c766bccfb2f8a1e2703d6d7777ec37548accdb40c5d279541cb72db1ac1`.

## Blind annotation packet

The annotator receives exactly:

- `question_id`
- the exact user `question`
- ordered `Source 1` through `Source 14`, each containing only `source_id`
  and `content`
- an empty `annotation` object

The packet physically excludes the gold answer, gold/qrel document IDs,
selection stratum, selection hash, chunk IDs, document IDs, prior labels,
and classifier predictions.

## Corrected label contract

- `SUFFICIENT`: the visible Top14 directly supports every material
  requirement explicitly present in the visible question.
- `INSUFFICIENT`: at least one explicitly requested cause, condition,
  version, value, or resolution step is absent, conflicting, or requires an
  unsupported inference.
- `QUESTIONABLE`: the visible question is materially ambiguous, broken, or
  does not identify which of several plausible answers is intended.

Operational clarifications:

- If the question asks where to find information, the correct directly
  supported document or URL can be sufficient; the linked document does not
  need to be reproduced in full.
- If the question asks for actual details or a recommended fix, a pointer to
  another document is not sufficient unless the requested details are also
  present in Top14.
- Do not invent requirements from an unseen reference answer.
- A `SUFFICIENT` label requires at least one supporting `Source N` ID.
- `INSUFFICIENT` and `QUESTIONABLE` may use an empty source-ID list.
- Every label requires a short rationale based only on visible content.

## Annotation handoff

Edit only the `annotation` object in each row. Preserve every other field,
the row order, and the 80 question IDs exactly. Save an AI-produced draft as
`data/refusal_v3_confirmation_set/annotation_packets_annotated_ai_draft.jsonl`.

After annotation, validate and freeze it with:

```bash
python -m experiments.evals.refusal_v3_confirmation_set \
  --metadata-path <TRAIN_METADATA_JSON> \
  --freeze-annotated-packet \
  data/refusal_v3_confirmation_set/annotation_packets_annotated_ai_draft.jsonl \
  --annotation-source ai-draft
```

## Provenance and exit

AI-produced annotations remain development-only even when they follow the
corrected contract. Only independently human-produced annotations may enter
a human confirmation result.

After annotation, the validator freezes compact targets and provenance. A
classifier prompt, model, thresholds, and paid run require a later, separate
preregistration after targets are frozen.
