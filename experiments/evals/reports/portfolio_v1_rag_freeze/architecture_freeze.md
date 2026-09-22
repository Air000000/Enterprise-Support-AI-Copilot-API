# Portfolio-v1 RAG Architecture Freeze

Status: **FROZEN for portfolio-v1**
Scope: retrieval candidate + context assembly
Runtime integration: **not yet complete**

This document records the engineering decision reached after the TechQA retrieval/context research line. It does not rewrite historical experiment outcomes and it does not claim that the frozen candidate is already deployed online.

## 1. Current runtime vs frozen integration target

Current online/API runtime:

```text
Dense Chroma Retrieval
```

Frozen portfolio-v1 integration target:

```text
Dense chunk Top100
+
BM25 chunk Top100
        ↓
equal-weight chunk-level RRF (k=60)
        ↓
fused Top100
        ↓
qwen3-rerank
        ↓
Flat Top14
```

The target above is an engineering freeze for integration. It is not a statement that the complete Hybrid+Rerank+Flat14 route has received a fresh held-out DEV confirmation.

## 2. Retrieval freeze

Frozen retrieval contract:

- Dense chunk candidates: Top100
- BM25 chunk candidates: Top100
- BM25 implementation: existing TechQA BM25 route
- fusion: equal-weight chunk-level reciprocal-rank fusion
- RRF constant: 60
- fused candidate budget: Top100
- reranker: `qwen3-rerank`
- no Dense Top1 rescue in the frozen context policy
- no further RRF-k, source-weight, candidate-depth, source-quota, or per-document-cap tuning

### Historical R4 C1 status

R4 C1 Hybrid+Rerank remains formally **FAIL** under its preregistered gate.

TRAIN metrics:

| Method | Recall@5 | Recall@20 | MRR@10 |
| --- | ---: | ---: | ---: |
| E1 Dense+rerank | 0.691111 | 0.815556 | 0.567206 |
| C1 Hybrid+rerank | 0.702222 | 0.831111 | 0.570929 |

The C1 route improved all three aggregate TRAIN metrics, but it did not reach the preregistered MRR promotion threshold. The later decision to retain the fixed Hybrid route as a portfolio-v1 engineering candidate is therefore a separate engineering decision, not a rewrite of the formal C1 result.

## 3. Context freeze

Frozen context policy:

```text
policy = flat_rerank_top14_v1
final_context = reranked_chunks[:14]
```

The following are not part of the frozen policy:

- Dense Top1 rescue
- forward-only sibling expansion
- symmetric locality expansion
- second locality rerank
- whole-document expansion
- section expansion
- Parent-Child retrieval
- hard document admission

### Why Top14

On the frozen 54-case TRAIN evidence audit:

| Context budget | Answer-bearing evidence hits | Useful evidence hits |
| --- | ---: | ---: |
| Top14 | 35 / 54 | 43 / 54 |
| Top20 | 35 / 54 | 43 / 54 |

Top14 was selected as the smallest TRAIN-development K that reached the audited Top20 evidence-hit ceiling. It is not claimed to be a universal or production-optimal K.

## 4. Rejected context challengers

### 4.1 rerank-first locality recovery

Frozen Flat Top14 vs locality second-rerank:

- answer-bearing evidence: 35 -> 35
- useful evidence: 43 -> 43
- answer miss -> hit: 0
- answer hit -> miss: 0
- actionable residual recovery: 0 / 7
- second-rerank p50: 661.314 ms
- second-rerank p95: 1163.955 ms
- provider tokens: 659,386

Decision: **REJECTED_NO_GAIN**.

### 4.2 structure-preserving synthesis

Under the same per-query raw-character context budget:

- answer-bearing evidence: 35 -> 32
- useful evidence: 43 -> 38
- answer miss -> hit: 2
- answer hit -> miss: 5
- targeted structural residual recovery: 2 / 6
- all query context budgets respected: yes

Decision: **REJECTED_NET_REGRESSION**.

## 5. Failure attribution

The structure-preserving challenger was audited case-by-case.

Regression attribution:

- 5 / 5 answer regressions: `BUDGET_CROWD_OUT`
- assembly-mapping miss: 0
- span/implementation error: 0
- other: 0

Gain attribution:

- 2 / 2 gains: `WHOLE_DOCUMENT_RECOVERY`
- section recovery: 0
- atomic recovery: 0

Breadth/depth movement:

- baseline median unique documents in context: 12
- challenger median unique documents in context: 4.5
- baseline median atomic chunks: 14
- challenger median processed anchors: 5
- challenger median whole-document parents: 2.5

The supported interpretation is:

> Under the fixed context budget, static parent expansion traded cross-document breadth for within-document depth. Depth helped when an early high-ranked anchor already pointed to the answer document, but it displaced later useful evidence often enough to produce a net regression on the audited TRAIN set.

This does **not** establish that Parent-Child, small-to-big, or document structure are generally ineffective.

## 6. Research closure

For portfolio-v1:

- retrieval parameter research: **CLOSED**
- context assembly research: **CLOSED**

These lines should not be reopened to tune the current artifacts. A future change requires a genuinely new failure mode outside the current portfolio-v1 finalization scope.

## 7. Unresolved before final integration

The following are intentionally not frozen by this document:

- refusal / evidence-sufficiency policy
- final generation acceptance protocol
- final DEV generation validation
- online runtime integration
- Ticket Agent shared retrieval integration

Historical refusal logic based on `dense_top1_distance > 0.9` must not be inherited automatically by the frozen Hybrid+Rerank candidate, because Dense Top1 distance represents only one first-stage retrieval signal.

## 8. Provenance boundary

The final context-budget, locality, structure-forensic, structure-preserving-synthesis, and closure runs were completed in a local closure lineage before publication of this summary.

Known local lineage identifiers:

- base main SHA at closure start: `a39cdae8a3054e61b1133f688befef29c3a6de22`
- final context-budget local commit: `e8e6f04eeca2c63094627d913b5223a1dceaaef1`
- final closure local commit: `6948dfdd5beed1c2664ec72922f70e82b81a46a6`

Those latter local SHAs were not GitHub refs at the time this public summary was created. They are recorded only as provenance, not as clickable published commits.
