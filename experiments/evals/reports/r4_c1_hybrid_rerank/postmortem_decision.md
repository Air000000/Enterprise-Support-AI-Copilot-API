# R4 C1 Hybrid Rerank Postmortem Decision

## Scope

- Benchmark split: TechQA TRAIN
- Query count: 450
- DEV artifact opened during postmortem: false
- Provider calls during zero-cost postmortem: 0
- E1 was not rerun.
- No RRF parameter search, source weighting search, depth search, or post-hoc gate adjustment was performed.

This document closes the R4 C1 retrieval experiment and records the interpretation supported by the frozen C1 artifacts and the bounded manual case study.

## Formal C1 result

Frozen E1 reference:

| Metric | E1 |
| --- | ---: |
| Recall@5 | 0.691111 |
| Recall@20 | 0.815556 |
| MRR@10 | 0.567206 |

R4 C1:

| Metric | C1 | Delta vs E1 |
| --- | ---: | ---: |
| Recall@5 | 0.702222 | +0.011111 |
| Recall@20 | 0.831111 | +0.015556 |
| MRR@10 | 0.570929 | +0.003723 |

Preregistered C1 gate:

- Recall@20 >= 0.8111111111111111
- MRR@10 >= 0.5772063492063492

Outcome:

- Recall@20 gate: PASS
- MRR@10 gate: FAIL
- Overall C1 decision: **FAIL**

The correct interpretation is not that Hybrid retrieval failed. C1 improved all three aggregate metrics, but the gain was insufficient to clear the preregistered early-rank MRR gate.

## Candidate complementarity

Frozen zero-cost postmortem:

- Dense-only gold hits: 52
- BM25-only gold hits: 26
- Both-source gold hits: 335
- Neither-source gold hits: 37
- Hybrid rescued Dense misses: 19
- Hybrid lost Dense hits: 12
- Net candidate gain: +7

This confirms genuine lexical complementarity between BM25 and Dense retrieval.

BM25 contributes evidence that Dense can miss, especially for technical-support queries containing exact identifiers such as:

- error codes and SQL states;
- product and version identifiers;
- configuration terms;
- command paths;
- API/class names;
- fixed technical phrases.

Dense remains complementary because it can retrieve evidence connected through symptom-to-mechanism, root-cause, or remediation semantics even when lexical overlap is weak.

## Final Top-20 residual attribution

Of 450 TRAIN queries:

- resolved in final Top-20: 374
- gold present in fused candidates but final rank > 20: 20
- gold absent from fused Top-100: 56

Therefore the 76 final Top-20 misses decompose into:

- candidate miss: 56 / 76
- ranking miss: 20 / 76

For the absolute C1 residual, candidate coverage is the larger bottleneck.

This does not establish a paired causal explanation for the C1-vs-E1 MRR difference because full historical E1 per-query rerank results were not preserved.

## Rank-distribution validation

The reconstructed postmortem cases exactly reproduce the formal C1 metrics:

- rank 1: 212
- rank 2-3: 79
- rank 4-5: 25
- rank 6-10: 32
- rank 11-20: 26
- rank 21+: 20
- missing: 56

Derived totals:

- Top-3: 291
- Top-5: 316
- Top-10: 348
- Top-20: 374
- reconstructed MRR@10: 0.5709294532627865

No metric-reconstruction mismatch was found.

## Chunk crowding

Across fused Top-100 candidates:

- unique documents p50: 75.0
- unique documents p95: 90.55
- duplicate ratio p50: 0.25
- duplicate ratio p95: 0.44
- duplicate ratio max: 0.69
- max chunks per document p95: 8

Chunk crowding is measurable, but bucketed analysis does not support it as the primary C1 failure cause.

Candidate-miss queries did not show systematically higher crowding than resolved queries, and lost-Dense-hit cases did not show a higher median duplicate ratio than rescued-Dense-miss cases.

Therefore no per-document cap tuning is admitted from this evidence.

## Fusion-loss audit

A zero-provider reconstruction of the full Dense/BM25 RRF union identified 19 cases where at least one source retrieved the gold document but fixed Top-100 RRF compression removed it.

Source pattern:

- Dense-only: 11
- BM25-only: 7
- both sources: 1

Evidence overlap pattern:

- single-source-only: 18
- same gold document via different source chunks: 1

Source gold ranks were generally weak/tail:

- Dense best-gold-rank median: 67.5
- BM25 best-gold-rank median: 72.5

Full-RRF gold ranks:

- minimum: 102
- median: 123
- maximum: 171

Only four loss cases were within ten positions of the Top-100 cutoff.

The dominant mechanism is therefore a fixed-budget candidate-retention tradeoff: two Top-100 source rankings are compressed into one Top-100 pool, and weak single-source tail candidates can lose to candidates supported more strongly or by both sources.

This is not evidence of an RRF implementation bug.

## 38-case manual diagnostic study

A bounded manual study compared:

- 19 `Dense miss -> Hybrid rescue` cases
- 19 `source hit -> fusion loss` cases

The study is diagnostic and intentionally selected around rescue/loss behavior. It is **not a representative random sample of all 450 TRAIN queries**, so its category proportions must not be extrapolated to the full benchmark.

### Hybrid-rescue cases

Primary manual labels:

- exact lexical: 12 / 19
- chunk boundary: 1 / 19
- weak document hit: 3 / 19
- ambiguous/questionable gold: 3 / 19

Manual evidence-value judgment:

- clearly valuable: 13 / 19
- conditionally valuable: 3 / 19
- questionable/not useful for architecture tuning: 3 / 19

The rescue set strongly supports the claim that BM25 supplies real lexical complementarity, especially for error codes, versions, product identifiers, configuration terminology, and fixed technical strings.

### Fusion-loss cases

Manual evidence-value judgment:

- clearly worth preserving: 9 / 19
- conditionally valuable: 3 / 19
- weak/questionable and not worth protecting solely to improve document-level qrel metrics: 7 / 19

Among the 11 Dense-only fusion-loss cases, seven were judged clearly valuable semantic evidence.

This suggests a future hypothesis narrower than "RRF loses good candidates":

> Fixed-budget equal RRF may suppress a small number of valuable single-source semantic candidates when they compete with stronger or cross-source-supported chunks.

This hypothesis is not sufficient justification for TechQA TRAIN parameter tuning. It should be tested only if it remains meaningful under evidence-level evaluation or an independent dataset.

## Chunk/document evaluation mismatch

TechQA retrieval qrels are document-level while the runtime retrieves chunks.

Therefore:

`gold document hit != answer-bearing evidence hit`

A retrieved chunk can belong to the official gold document while containing little or no information needed to answer the query.

The manual study identified multiple examples of:

- weak chunks inside a valid gold document;
- long gold documents containing many unrelated chunks;
- source hits that satisfy document-level metrics but are poor RAG context.

This means official document Recall/MRR remain valid TechQA benchmark metrics, but they are insufficient by themselves to guide RAG evidence retrieval architecture.

Future evaluation should preserve the official document-level metrics and add a separate evidence-level evaluation layer.

## Questionable qrels

The manual study also identified a small number of cases where the official gold document only weakly or indirectly supports the user question.

These cases must remain in official benchmark reporting.

However, they should not automatically drive retrieval architecture changes.

A benchmark miss is not equivalent to a proven retrieval-system defect.

## Interpretation boundary

Supported conclusions:

1. BM25 provides genuine lexical complementarity to Dense retrieval.
2. C1 improves aggregate candidate coverage and final Recall@20.
3. Fixed Top-100 RRF compression creates a real candidate-retention tradeoff.
4. Most observed compression losses are weak single-source tail hits, not evidence of an RRF bug.
5. A smaller subset of valuable Dense-only semantic candidates is also lost.
6. Chunk/document granularity mismatch can make document-level retrieval metrics overstate answer-evidence quality.
7. Some official qrels are too weak or ambiguous to use blindly for architecture tuning.
8. The additional C1 candidate recall did not produce enough early-rank gain under the frozen reranker to clear the preregistered MRR gate.

Not supported:

- exact paired E1-to-C1 rank-movement claims;
- crowding as the primary C1 root cause;
- RRF parameter tuning as the next justified experiment;
- per-document caps, source quotas, or fusion-weight tuning from the current TRAIN evidence;
- claims that TechQA document-level qrels directly measure answer-bearing evidence retrieval.

## Final decision

**R4 C1 is closed as FAIL under its preregistered success gate.**

No original C2 paid parameter-tuning experiment is admitted.

Do not continue TechQA TRAIN searches over:

- RRF k;
- Dense/BM25 source weights;
- candidate depth;
- per-document caps;
- source quotas;
- gate thresholds.

The next evaluation stage is not another RRF tuning cycle.

The next stage will add a small audited evidence-level evaluation layer while preserving the official TechQA document-level benchmark.

Only evidence-supported failure classes may admit a subsequent retrieval architecture experiment.

Current candidate hypotheses for future gated testing are:

- early fusion/compression retention;
- evidence-aware reranking;
- document-aware local evidence retrieval;
- contextual/section-aware chunk representation;
- query expansion for symptom-to-mechanism gaps;
- iterative retrieval only if simpler expansion remains insufficient.

These are hypotheses, not an implementation backlog.

## R4 C1 closure statement

C1 demonstrated that Hybrid retrieval is useful but that retrieval success cannot be reduced to whether an official gold document appears in the candidate set. The next stage therefore shifts from parameter tuning to measurement quality: official document-level IR metrics will remain for benchmark comparability, while an audited evidence-level layer will determine whether retrieved chunks actually support the technical answer.
