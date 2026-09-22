# Portfolio-v1 RAG Evidence Timeline

This timeline is a compact index of the retrieval/context decisions that led to the portfolio-v1 freeze. Each stage records the engineering question, the evidence, and the resulting decision.

## E0 — Dense baseline

**Question:** What is the starting retrieval quality on the frozen TechQA contract?

**Evidence:** Dense retrieval established the baseline for later comparisons.

**Decision:** Retain as the historical comparator.

---

## E1 — Dense + rerank

**Question:** Does reranking improve held-out retrieval?

**Evidence:** On frozen DEV:

- Recall@5: 0.643750 -> 0.725000
- Recall@20: 0.818750 -> 0.843750
- MRR@10: 0.518931 -> 0.560841

**Decision:** Reranking is retained as the strongest held-out-confirmed retrieval improvement in the lineage.

---

## R4 C1 — Hybrid + rerank

**Question:** Does Hybrid+rerank clear the preregistered promotion gates?

**Evidence:** On TRAIN:

- E1 Dense+rerank: Recall@5 0.691111, Recall@20 0.815556, MRR@10 0.567206
- Hybrid+rerank: Recall@5 0.702222, Recall@20 0.831111, MRR@10 0.570929
- Recall@20 gate: PASS
- MRR@10 gate: FAIL

**Decision:** Formal experiment status remains **FAIL**. The route is later retained only as a fixed portfolio-v1 engineering candidate; no further fusion tuning is admitted.

---

## R1 — Evidence-level audit

**Question:** Does retrieving the correct document guarantee that answer-bearing evidence reaches the final context?

**Evidence:** A manually audited TRAIN subset separated useful and answer-bearing chunks from simple document hits.

**Decision:** Keep evidence-level accounting as a diagnostic layer because document hit and answer-evidence hit are not equivalent.

---

## G1 / G2 — Document-local and admission experiments

**Question:** Can broader document-local reconstruction or rerank-informed admission improve Stage2 evidence quality?

**Evidence:** Both routes produced aggregate movement but also preregistered regressions / NO_GO outcomes.

**Decision:** Do not promote either route.

---

## Retrieval frontier closure

**Question:** Is another retrieval-parameter search cycle justified?

**Evidence:** The post-hoc frontier audit showed genuine Dense/other-ranking complementarity but was exploratory and diagnostic only.

**Decision:** Close retrieval parameter research for portfolio-v1. No further RRF-k, depth, source-weight, quota, or per-document-cap tuning.

---

## Final context budget

**Question:** What is the smallest flat rerank K that reaches the audited Top20 evidence-hit ceiling?

**Evidence:** On 54 usable TRAIN evidence-audit cases:

- Top14: answer 35/54, useful 43/54
- Top20: answer 35/54, useful 43/54

**Decision:** Select `flat_rerank_top14_v1`.

---

## Locality second-rerank challenger

**Question:** Can rerank-first local recovery rescue residual answer-bearing chunks?

**Evidence:**

- answer: 35 -> 35
- useful: 43 -> 43
- actionable residual recovery: 0/7
- added second-rerank latency and provider cost

**Decision:** Reject. Keep Flat Top14.

---

## TechQA structure forensic

**Question:** Is there enough document structure to justify a structure-preserving synthesis test?

**Evidence:**

- 28,481 documents
- recognized-structure allowlist coverage: 0.994347
- this coverage is not parser accuracy
- six-case forensic did not reveal a simple universal QUESTION/PROBLEM -> ANSWER/RESOLUTION pattern

**Decision:** Permit one bounded synthesis counterfactual; do not reopen splitter or retrieval chunking.

---

## Structure-preserving synthesis

**Question:** Under the same per-query character budget, does static whole-document / section recovery improve net evidence coverage?

**Evidence:**

- answer: 35 -> 32
- useful: 43 -> 38
- miss -> hit: 2
- hit -> miss: 5
- all context budgets respected

**Decision:** Reject with net regression.

---

## Failure attribution

**Question:** Why did the structure-preserving challenger regress?

**Evidence:**

- 5/5 answer regressions = `BUDGET_CROWD_OUT`
- 2/2 gains = `WHOLE_DOCUMENT_RECOVERY`
- baseline median unique documents in context = 12
- challenger median unique documents in context = 4.5
- implementation invariants passed

**Decision:** The observed trade-off is breadth versus within-document depth under a fixed context budget. On the audited TRAIN set, the regressions outweighed the recoveries.

---

## Portfolio-v1 freeze

**Question:** What retrieval/context contract should be carried into integration?

**Decision:**

```text
Dense Top100 + BM25 Top100
    -> equal-weight chunk RRF (k=60)
    -> fused Top100
    -> qwen3-rerank
    -> Flat Top14
```

Research state:

- retrieval parameter research: **CLOSED**
- context assembly research: **CLOSED**

Still unresolved:

- refusal / evidence-sufficiency policy
- final generation acceptance
- final DEV generation validation
- online runtime integration
- Ticket Agent shared retrieval integration
