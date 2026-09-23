# Refusal v2 AI-proxy run incident

Status: `INCOMPLETE_SCHEMA_FAILURE_NO_EVALUATION`

Date: `2026-09-23`

Decision: `STOP_DO_NOT_EVALUATE_OR_RESUME`

## Frozen run state

- Run: `refusal_v2_ai_proxy_holdout_v1`
- Model: `qwen3.5-plus-2026-04-20`
- Region: Singapore / International
- Frozen inputs: 40
- Successful checkpointed predictions: 38
- Failed attempts: 1
- Total provider attempts: 39 of 40
- Remaining attempt budget: 1, intentionally unused
- Targets opened: no
- Evaluation executed: no
- Summary artifact created: no
- DEV opened: no

## Failure

The provider returned a response that could not be parsed as JSON for
`TRAIN_Q584`.

- Stage: `schema`
- Error: `RuntimeError: classifier returned invalid JSON`
- Request ID: `chatcmpl-3c4ead67-438a-9ee6-83af-ababc01f539e`
- Usage available: yes
- Failed-attempt tokens: 2,838
- Failed-attempt estimated cost: CNY 0.012090

The raw provider response remains only in the ignored local failed-attempt
artifact. It was not printed, committed, repaired, or converted into a
prediction.

## Usage before stop

- Prompt tokens: 105,009
- Completion tokens: 6,525
- Total tokens: 111,534
- Estimated cost: CNY 0.423238

## Local artifact manifest

| Artifact | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `data/refusal_v2_ai_proxy/run_v1/predictions.jsonl` | 38 | 35,133 | `bbfd3ff87e332b516fae0e78afba588453cde113d9caefbd7bc6c17e26f1c6ca` |
| `data/refusal_v2_ai_proxy/run_v1/failed_attempts.jsonl` | 1 | 1,381 | `d5ece92619caab15a1129b5c37144021d5166bc94f923aedf84cbd20f7c5dc1a` |

## Consequence

The preregistered run is incomplete and has no gate result. The hidden proxy
targets must remain unopened. The remaining provider-attempt allowance must
not be used to retry or continue this run.

Any schema-handling amendment requires a separately reviewed v2.1 contract
that supersedes this run and preserves all prior attempts and costs. It must
not be presented as the original v2 holdout run. Phase C, generation,
refusal freeze, and runtime integration remain blocked.
