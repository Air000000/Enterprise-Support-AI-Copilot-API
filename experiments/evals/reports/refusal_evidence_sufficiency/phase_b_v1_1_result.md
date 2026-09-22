# Phase B v1.1 Formal Result

Status: `COMPLETE_FORMAL_FAIL`

Date: `2026-09-22`

Formal decision: `REJECT_EVIDENCE_SUFFICIENCY_V1`

## Execution controls

- Run ID: `refusal_evidence_sufficiency_phase_b_v1_1`
- TRAIN classifier inputs: 54
- Input SHA256: `bf0164d096758bf0c4cdda3ea0e106a3914e6e43aa1b627f13e8d6cd7efb9fa9`
- Model: `qwen3.5-plus-2026-04-20`
- Region: Singapore / International
- Temperature: `0.0`
- Thinking: disabled
- Context: `flat_rerank_top14_v1`
- Provider calls: 54
- Failed attempts: 0
- Checkpointed predictions: 54
- Targets opened only by the separate evaluation command: yes
- DEV opened: no
- Retrieval, rerank, generation, and judge calls: 0

## Formal gate

| Metric | Actual | Gate | Result |
|---|---:|---:|---|
| Balanced accuracy | 0.735714 | >= 0.80 | FAIL |
| Sufficient recall | 0.971429 | >= 0.85 | PASS |
| Insufficient recall | 0.500000 | >= 0.70 | FAIL |
| Sufficient to insufficient | 1 | <= 5 | PASS |

Accuracy was `0.823529` on the 51 gated cases.

Confusion matrix:

| Target | Predicted sufficient | Predicted insufficient |
|---|---:|---:|
| Sufficient proxy | 34 | 1 |
| Insufficient proxy | 8 | 8 |

All 3 descriptive ambiguous multi-chunk cases were classified `SUFFICIENT`; they were excluded from the gate.

## Usage and latency

- Prompt tokens: 147,090
- Completion tokens: 8,923
- Total tokens: 156,013
- Estimated cost: CNY 0.589026
- Latency p50: 3,668.418 ms
- Latency p95: 4,885.734 ms

## Artifact manifest

Large run artifacts remain local under `data/refusal_evidence_sufficiency_phase_b_v1_1/`.

| Artifact | SHA256 | Size |
|---|---|---:|
| `predictions.jsonl` | `d7ad10221b563d250de1927cc2406dd73cafa4c9cbd87429280567491dc962d6` | 49,286 bytes |
| `summary.json` | `bf5c37d6e8327073e70fe28692f7bd927185c2f5e992253612f99b9dd37d1978` | compact local result |
| `phase_b_v1_1_run_contract.json` | `af258ee3a81ddaf0753f1a45e1fa6fb649befd04fb4fd9895c29128a1d6fe8d0` | tracked contract |

## Consequence

The v1 classifier over-admits insufficient evidence: 8 of 16 insufficient proxies were classified `SUFFICIENT`. The preregistered insufficient-recall and balanced-accuracy gates failed.

Phase C is not admitted. The refusal policy remains unresolved, and this result must not be followed by prompt tuning or a rerun presented as the same formal experiment.
