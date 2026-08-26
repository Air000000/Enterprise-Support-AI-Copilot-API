# E0 Dense vs E1 Rerank Evidence

- Reranker: `qwen3-rerank`
- Provider region: `ap-southeast-1`
- DEV evidence is aggregate-only; individual DEV failure IDs are intentionally omitted.

## TRAIN

| Metric | E0 Dense | E1 Rerank |
| --- | ---: | ---: |
| Recall@5 | 0.613333 | 0.691111 |
| Recall@20 | 0.740000 | 0.815556 |
| MRR@10 | 0.510477 | 0.567206 |

- Top-5 fixed/regressed: 51/16
- Top-20 fixed/regressed: 37/3
- E1 full retrieval p50: 4114.488 ms
- E1 full retrieval p95: 4959.818 ms
- Provider total tokens: 11972838

## DEV

| Metric | E0 Dense | E1 Rerank |
| --- | ---: | ---: |
| Recall@5 | 0.643750 | 0.725000 |
| Recall@20 | 0.818750 | 0.843750 |
| MRR@10 | 0.518931 | 0.560841 |

- Top-5 fixed/regressed: 21/8
- Top-20 fixed/regressed: 5/1
- E1 full retrieval p50: 4111.483 ms
- E1 full retrieval p95: 4657.189 ms
- Provider total tokens: 4163611
