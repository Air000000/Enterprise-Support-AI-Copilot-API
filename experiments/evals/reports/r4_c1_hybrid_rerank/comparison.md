# R4 C1 Hybrid + Rerank vs E1

## Frozen E1 TRAIN baseline

- Recall@5: 0.691111111111
- Recall@20: 0.815555555556
- MRR@10: 0.567206349206

## R4 C1 TRAIN

- completed queries: 450
- Recall@5: 0.702222222222
- Recall@20: 0.831111111111
- MRR@10: 0.570929453263
- provider tokens: 12080467
- stopped reason: None

## Pre-registered executable gate

- MRR@10 >= 0.5772063492063492
- Recall@20 >= 0.8111111111111111
- C1 PASS: False

Complete historical E1 per-query evidence is not preserved,
so C1 does not rerun E1 solely to recreate paired diagnostics.

## Decision

C1 improved aggregate retrieval effectiveness over E1 on TRAIN:

- Recall@5: 0.691111111111 -> 0.702222222222
- Recall@20: 0.815555555556 -> 0.831111111111
- MRR@10: 0.567206349206 -> 0.570929453263

However, C1 failed the pre-registered MRR@10 threshold
of 0.5772063492063492.

Therefore the pre-registered C1 gate is FAIL.
Paid R4 stops here and no C2 paid optimization is admitted.

