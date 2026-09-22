# Phase B v1.1 Schema and Attempt-Audit Amendment

Status: `PREREGISTERED_NOT_RUN`

Date: `2026-09-22`

## Why v1 ended

Phase B v1 produced no checkpointed prediction. Its three provider attempts were:

1. `401 invalid_api_key` from a cross-region key/endpoint mismatch.
2. `404` from using a rerank `compatible-api/v1` endpoint for chat completions.
3. A valid JSON response whose `supporting_source_ids` contained the bare ID `"1"` instead of the required `"Source 1"`.

The third response was rejected before checkpointing. Its request ID, usage, latency, raw response, and exact cost were not retained. This is an audit gap. Phase B v1 is terminated and must not be reported as a completed formal run.

## v1.1 amendment

Phase B v1.1 changes only response handling and failed-attempt accounting:

- JSON numbers `1` through `14` and decimal strings `"1"` through `"14"` are losslessly canonicalized to `"Source 1"` through `"Source 14"`.
- All other source IDs remain invalid.
- The Singapore chat base URL must end in `compatible-mode/v1`; the rerank-only `compatible-api/v1` path is rejected during preflight.
- Provider, schema, and post-response cost failures are appended to `failed_attempts.jsonl` before the runner exits.
- Failed attempts retain the available request ID, token usage, latency, estimated cost, and error; schema failures also retain the raw model response.
- Checkpointed predictions plus failed attempts count toward the 54-call maximum and CNY 3 hard estimated cost cap.
- Evaluation is forbidden when the v1.1 failed-attempt log is non-empty.

## Unchanged frozen contract

- TRAIN population: 54 total, 51 gated, 3 descriptive ambiguous cases.
- Classifier input SHA256: `bf0164d096758bf0c4cdda3ea0e106a3914e6e43aa1b627f13e8d6cd7efb9fa9`.
- Target SHA256: `7b1535d033da3b8bf6b6d94c9a6f93105c76e593828dd53e1b0873d35b89e655`.
- Context: `flat_rerank_top14_v1`.
- Model: `qwen3.5-plus-2026-04-20` in Singapore.
- Temperature: `0.0`; thinking disabled; JSON object; maximum output 256 tokens.
- Prompt and decision semantics are unchanged.
- Gate thresholds are unchanged.
- DEV remains closed; retrieval, reranking, generation, and judging remain disabled.

Formal v1.1 outputs use `data/refusal_evidence_sufficiency_phase_b_v1_1/`; v1 artifacts are not reused or overwritten.
