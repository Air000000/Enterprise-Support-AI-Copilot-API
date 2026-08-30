# R4 C1 Zero-Cost Postmortem

## Scope

- TRAIN queries: 450
- Provider calls: 0
- DEV artifact opened: False

## Candidate complementarity

- Dense-only gold hits: 52
- BM25-only gold hits: 26
- Both-source gold hits: 335
- Neither-source gold hits: 37
- Hybrid rescued Dense misses: 19
- Hybrid lost Dense hits: 12
- Net candidate gain: 7

## Candidate miss vs ranking miss

- Resolved in final Top-20: 374
- Gold in fused candidates but final rank > 20: 20
- Gold absent from fused Top-100: 56

## Chunk crowding

- Fused unique documents p50: 75.000
- Fused unique documents p95: 90.550
- Duplicate ratio p50: 0.250000
- Duplicate ratio p95: 0.440000
- Duplicate ratio max: 0.690000
- Max chunks per document p95: 8.000

## Source composition

- Dense-only fused chunks: 16570
- BM25-only fused chunks: 16345
- Shared fused chunks: 12085

## Interpretation boundary

This report is a zero-provider diagnostic over the already frozen
R4 C1 TRAIN candidate snapshot and completed C1 rerank results.

It can distinguish candidate coverage failures from post-candidate
ranking failures and quantify chunk/document crowding.

It does not by itself prove that BM25, RRF, crowding, or the reranker
is the causal root cause of any observed failure. No E1 rerun,
parameter search, DEV inspection, or provider call is performed.
