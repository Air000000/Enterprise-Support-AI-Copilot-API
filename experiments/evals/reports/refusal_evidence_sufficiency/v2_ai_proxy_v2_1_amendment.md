# Refusal v2.1 AI-proxy execution amendment

Status: `PREREGISTERED_RESUME_NOT_RUN`

## Cause

The original v2 run stopped after 38 successful predictions and one schema
failure. The failed response used exactly 256 completion tokens and ended
inside `supporting_source_ids`, proving that the JSON was truncated by the
frozen output limit. It was not a model-authentication, input, retrieval, or
target mismatch.

The hidden targets remain unopened. The raw failed response was inspected
only to diagnose its serialization boundary; it was not repaired or treated
as a prediction.

## Frozen v2.1 changes

- Carry forward the 38 successful v2 predictions unchanged.
- Preserve the original failed-attempt record and its cost.
- Increase maximum output tokens from 256 to 512.
- Increase the total provider-attempt ceiling from 40 to 41.
- Permit exactly two additional calls: retry `TRAIN_Q584` once and run the
  one remaining never-called case once.
- Keep the CNY 1.5 total hard cost cap.

The model, provider region, prompt and prompt SHA, temperature, thinking
mode, response schema, 40 inputs, hidden targets, context policy, and all
quality gates remain unchanged.

## Resume controls

The original v2 artifacts are copied to a new local `run_v2_1` directory
before execution. Preflight must match both prior SHA-256 values exactly:

- 38-row checkpoint: `bbfd3ff87e332b516fae0e78afba588453cde113d9caefbd7bc6c17e26f1c6ca`
- 1-row failure log: `d5ece92619caab15a1129b5c37144021d5166bc94f923aedf84cbd20f7c5dc1a`

All 41 attempts and their costs are reported. Evaluation is permitted only
after exactly 40 valid predictions exist. A proxy PASS still admits only an
independent human confirmation; it does not admit Phase C.
