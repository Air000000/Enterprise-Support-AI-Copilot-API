# Refusal v2 AI-draft annotation amendment

Status: `AI_DRAFT_ACCEPTED_FOR_DEVELOPMENT_ONLY`

## Reason for the amendment

The frozen v2 confirmation protocol requires blind human annotations. After
the 80-case packet was selected, an external GPT system produced a completed
annotation draft. The user explicitly requested that this draft be used.

This is a post-selection provenance amendment, not a preregistered human
confirmation result. No v2 classifier prediction existed when the draft was
accepted into the repository workflow.

## Frozen draft

- Local file: `data/refusal_v2_confirmation_set/annotation_packets_annotated_ai_draft.jsonl`
- Rows: 80
- SHA-256: `2d5ee8f2a63db722a2ce664bbff026149adcac8db9dcae9b4a693937cd1974b0`
- Frozen packet content changes: 0
- `SUFFICIENT`: 38
- `INSUFFICIENT`: 36
- `QUESTIONABLE`: 6
- Invalid targets, source IDs, or missing rationales: 0

The external GPT model version, prompt, decoding settings, calls, and cost
were not captured in the repository artifact and therefore remain unknown.
The repository process made no provider calls and opened no DEV artifact.

## Allowed use

The compact labels may be used to inspect errors and develop or preregister a
v2 evidence-sufficiency classifier. They may not be described as human gold,
independent confirmation, or evidence that the refusal policy is ready for
runtime promotion.

The original human-confirmation requirement remains unresolved. Refusal
freeze, Phase C admission, generation, and runtime integration remain
blocked until an independent confirmation protocol is completed and passes.
