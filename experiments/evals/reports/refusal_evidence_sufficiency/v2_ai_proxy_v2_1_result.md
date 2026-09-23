# Refusal v2.1 AI-proxy result

Status: `COMPLETE_PROXY_FAIL`

Date: `2026-09-23`

Decision: `REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY`

## Execution controls

- Run: `refusal_v2_ai_proxy_holdout_v2_1`
- Model: `qwen3.5-plus-2026-04-20`
- Region: Singapore / International
- Frozen inputs: 40 AI-draft proxy cases (20 sufficient, 20 insufficient)
- Carried-forward predictions: 38, byte-for-byte unchanged
- Additional provider calls: 2
- Total provider attempts: 41 (40 successful predictions, 1 preserved schema failure)
- Maximum output tokens for v2.1 calls: 512
- Prompt, model, inputs, targets, context policy, and quality gates changed: no
- Targets opened only by the separate evaluation command: yes
- DEV opened: no
- Retrieval, rerank, generation, and judge calls: 0

## Proxy gate

| Metric | Actual | Gate | Result |
| --- | ---: | ---: | --- |
| Balanced accuracy | 0.775000 | >= 0.80 | FAIL |
| Sufficient recall | 0.850000 | >= 0.85 | PASS |
| Insufficient recall | 0.700000 | >= 0.70 | PASS |
| Sufficient to insufficient | 3 | <= 3 | PASS |

Accuracy was `0.775000` (31 of 40 cases).

Confusion matrix:

| Target | Predicted sufficient | Predicted insufficient |
| --- | ---: | ---: |
| Sufficient proxy | 17 | 3 |
| Insufficient proxy | 6 | 14 |

Because the classes are balanced, one additional correct classification
would have raised balanced accuracy to the 0.80 threshold. The threshold was
not changed after unblinding.

## Usage and latency

- Prompt tokens: 110,403
- Completion tokens: 6,799
- Total tokens: 117,202
- Estimated cost: CNY 0.443901
- Latency p50: 3,584.833 ms
- Latency p95: 6,223.223 ms

The two v2.1 calls added 5,668 tokens and an estimated CNY 0.020663.

## Local artifact manifest

| Artifact | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `data/refusal_v2_ai_proxy/run_v2_1/predictions.jsonl` | 40 | 36,655 | `ce041d2c0bce2e3b7870e3da54dfb7f9146febc370ac4d01839a57a42abef754` |
| `data/refusal_v2_ai_proxy/run_v2_1/failed_attempts.jsonl` | 1 | 1,381 | `d5ece92619caab15a1129b5c37144021d5166bc94f923aedf84cbd20f7c5dc1a` |
| `data/refusal_v2_ai_proxy/run_v2_1/summary.json` | 1 | 948 | `b0c3e50e7fbb07a32bc0b0407eab3857ebb593506e51156adad84c484ab59b2c` |
| `v2_ai_proxy_v2_1_run_contract.json` | 1 | 3,308 | `10be4d51bdbb7432ab8c49747d294ef861ef97be19a3bbc1b4c8753f18b43fb3` |

## Consequence

This development proxy failed its preregistered balanced-accuracy gate, so it
does not admit Phase C or runtime integration. It must not be rerun or have
its threshold changed and then be presented as the same confirmatory run.

The labels are AI-draft proxy labels rather than independent human ground
truth. A post-hoc error review may diagnose the nine disagreements, but any
revised classifier requires fresh held-out cases and a new preregistration.
