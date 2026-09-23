# Refusal v3 blind AI-draft freeze

Status: `AI_DRAFT_TARGETS_FROZEN_FOR_DEVELOPMENT`

Date: `2026-09-23`

## Validation

- Annotated cases: 80
- Sufficient: 44
- Insufficient: 27
- Questionable: 9
- Minimum usable cases per scored class: 25, met
- Annotated packet SHA-256: `a3a1226343f5facd51b1fe714974aa66ade947a52710198effcaba867c142218`
- Compact targets SHA-256: `5b0d8b5b167e8ae742210c2deb8ff91c711143bf8da10451bc15f835d9ca8d68`

The validator confirmed that all 80 frozen questions and Top14 contexts were
unchanged and in the original order. Every row has a non-empty rationale,
every sufficient row cites at least one valid source, and no rationale refers
to a gold, reference, or ground-truth answer.

## Provenance and consequence

The annotations were produced by an external GPT workflow whose model,
prompt, decoding settings, calls, and cost are not recorded here. They remain
development-only and are not independent human confirmation.

The class-support condition permits a later v3 AI-proxy classifier
preregistration. It does not admit Phase C, generation, refusal-policy freeze,
or runtime integration.
