# TechQA Evaluation

This directory contains the **primary long-term evaluation harness** for Enterprise Support AI Copilot.

The project uses **TechQA** as its main technical-support data and evaluation backbone. It is not treated as a temporary Phase before migrating to another primary corpus.

Current focus:

- retrieval quality;
- reranking / hybrid retrieval comparison;
- evidence-level failure diagnosis;
- generation correctness and faithfulness;
- abstention / hallucination behavior;
- reproducibility and leakage control.

> Future multi-source, conflict, completeness, or agentic stress tests, if added, are supplemental evaluation surfaces. They do not replace TechQA as the primary corpus or reset the current benchmark lineage.

---

## 1. Why TechQA

TechQA matches the project's **Technical Support** domain and supports a continuous evaluation story across retrieval, evidence quality, generation, and abstention.

### Retrieval corpus

- **28,481** Technote documents;
- **610** answerable retrieval queries;
- **610** deterministic qrels rows;
- each answerable query has exactly **1 relevant document**.

### Generation / abstention dataset

- **610** answerable questions;
- **300** impossible questions;
- **910** QA records total.

This lets the project evaluate the same support domain through the following chain:

```text
Dense Retrieval
      ↓
Rerank / Hybrid Comparison
      ↓
Evidence-level Diagnosis
      ↓
Generation Evaluation
      ↓
Abstention / Hallucination Evaluation
```

Frozen dataset identities and SHA256 values live in:

- `datasets/techqa/manifest.json`
- `datasets/techqa/corpus_manifest.json`

---

## 2. Split Contract

Question IDs preserve original provenance:

| Split | Answerable | Impossible | Usage |
| --- | ---: | ---: | --- |
| `TRAIN_*` | 450 | 150 | development, failure analysis, parameter selection |
| `DEV_*` | 160 | 150 | frozen held-out comparison |

Rules:

- TRAIN is the development surface;
- DEV is the held-out comparison surface;
- once an E0 / E1 comparison is frozen, do not inspect individual DEV failures to tune the compared configuration;
- hard cases are not removed to improve aggregate metrics.

This split discipline is part of the benchmark itself, not optional reporting metadata.

---

## 3. Retrieval Contract

The online RAG retrieves **chunks**, while TechQA qrels are **document-level**.

For formal IR evaluation:

1. preserve the raw chunk ranking;
2. collapse chunks to unique `document_id` values;
3. retain each document at the rank where its first chunk appears;
4. evaluate the collapsed document ranking against qrels.

Because each answerable TechQA query has exactly one relevant Technote:

- Document Recall@K and Hit@K are numerically equivalent on this benchmark;
- MRR additionally measures how early the first relevant document appears.

Primary retrieval metrics:

- Document Recall@5
- Document Recall@20
- MRR@10

This contract prevents a chunk-level implementation detail from being silently compared against document-level gold labels.

---

## 4. Frozen Dense → Rerank Result

Formal held-out DEV comparison:

| Method | Recall@5 | Recall@20 | MRR@10 |
| --- | ---: | ---: | ---: |
| Dense baseline | 0.643750 | 0.818750 | 0.518931 |
| Dense Top-100 + `qwen3-rerank` | **0.725000** | **0.843750** | **0.560841** |

Equivalent presentation:

- Recall@5: **64.4% → 72.5% (+8.1pp)**
- Recall@20: **81.9% → 84.4%**
- MRR@10: **0.519 → 0.561**

The important claim is the improvement on the **same frozen held-out benchmark**, not an isolated cross-benchmark absolute score.

Primary report:

- `reports/e1_rerank/comparison.md`

---

## 5. Controlled Retrieval Experiments

The evaluation line did not stop at a single successful reranker result.

Offline comparisons include:

- Dense Retrieval;
- BM25;
- Dense + BM25 / RRF Hybrid;
- Dense / Hybrid candidate pools + `qwen3-rerank`.

These are **offline evaluation routes**. They are not claims that the online API serving path has switched to Hybrid Retrieval.

### C1 Hybrid + Rerank decision

C1 improved all three TRAIN aggregate metrics, but did not clear the preregistered early-rank MRR gate:

```text
Recall@20 gate: PASS
MRR@10 gate: FAIL
Overall C1 decision: FAIL
```

The route was therefore stopped instead of being described as a successful optimization merely because some metrics increased.

Relevant reports:

- `reports/r4_c1_hybrid_rerank/comparison.md`
- `reports/r4_c1_hybrid_rerank/postmortem_decision.md`

This is an explicit **go / no-go evaluation decision**, not just metric logging.

---

## 6. Evidence-level Audit

Document-level success can hide a more important failure mode:

> The correct document may be retrieved while the actual answer-bearing evidence remains too low in the chunk ranking to reach the final context.

To separate document retrieval from evidence quality, the project adds a small audited evidence layer:

- **60** labeled TRAIN queries;
- **54** formally evaluated queries;
- **187** candidate chunk labels;
- labels distinguish weak / irrelevant, useful, and answer-bearing evidence.

Evidence-level metrics include:

- AnswerEvidenceHit@K
- AnswerEvidenceMRR@10
- UsefulEvidenceHit@K
- GoldDocHitButEvidenceMissRate@K

The audit is used for **failure attribution and route selection**. It does not replace the official TechQA document-level benchmark.

Relevant artifacts:

- `reports/r1_evidence_audit/evidence_labels.jsonl`
- `reports/r1_evidence_audit/evidence_metrics.json`

A key conclusion from this layer is:

> **Document hit != answer evidence hit.**

This explains why improving document-level Recall alone is not sufficient to guarantee better generation context.

---

## 7. Generation / Abstention Evaluation

TechQA also provides the project's main generation and abstention evaluation surface.

Dataset:

- 610 answerable;
- 300 impossible;
- 910 total QA records.

The generation harness tracks:

- correctness;
- faithfulness;
- abstention accuracy;
- hallucination rate;
- end-to-end latency;
- retrieved context identities;
- frozen run identity / manifest / checkpoints.

Current retrieval-to-context policy used by the generation harness:

```text
Dense Top-100
   ↓
qwen3-rerank
   ↓
Top-3 rerank anchors
   + Dense Top-1 rescue anchor
   ↓
forward sibling expansion per unique anchor document
   ↓
deduplicate
   ↓
max 16 context chunks
```

Policy identifier:

```text
document_aware_forward_expansion_v1
```

Implementation:

- `eval_techqa_generation.py`

### G1 document-local evidence experiment

G1 tested a different Stage2 evidence-construction hypothesis without promoting it into the serving path:

```text
historical E0 ranking
   ↓
first 5 unique candidate documents
   ↓
full frozen document-local chunks
   ↓
merged qwen3-rerank
   ↓
Top16 evidence chunks
```

The formal 30-case method-label-blinded comparison against E1 produced:

| Metric | E1 | G1 |
| --- | ---: | ---: |
| COMPLETE | 24 / 30 | **28 / 30** |
| PARTIAL | 1 / 30 | 1 / 30 |
| INSUFFICIENT | 5 / 30 | **1 / 30** |
| Macro claim coverage | 0.825000 | **0.955556** |

Pairwise movement was **6 wins / 22 ties / 2 losses** for G1. However, one loss (`TRAIN_Q346`) met the preregistered catastrophic-regression definition `E1=COMPLETE && G1=INSUFFICIENT`.

Therefore:

```text
aggregate evidence sufficiency: improved
preregistered catastrophic-regression gate: failed
formal G1 decision: NO_GO
reference policy: keep E1
```

A 65-claim transition forensic found 6 newly covered claims and 2 lost claims. The two regressions had different, non-repeating boundaries: one candidate-document admission miss (`TRAIN_Q346`) and one continuity miss (`TRAIN_Q492`). Because neither mechanism repeated elsewhere in the frozen sample, the project does not patch G1 against those already-inspected TRAIN cases.

Full closeout:

- `reports/g1_document_local/final_decision.md`
- `reports/g1_document_local/evidence_sufficiency_30case_preregistration_v1_1.json`

Important boundary:

> The G1 result is a conditional evidence-sufficiency experiment on TRAIN cases under a frozen Stage2 contract. It is **not** a production generation uplift, a full-TechQA retrieval improvement, or evidence that G1 replaced E1.

### G2-A rerank-informed document admission

G2-A tested rerank-informed document admission on a fresh TRAIN sample of 30 cases. G1 admitted the first five unique documents by frozen Dense order; G2-A admitted the first five unique documents by one shared global rerank over the same Dense Top100. G2 changed no downstream document expansion or evidence-selection contract.

| Metric | G1 | G2-A |
| --- | ---: | ---: |
| COMPLETE | 19 | 19 |
| PARTIAL | 2 | 1 |
| INSUFFICIENT | 9 | 10 |
| Macro claim coverage | 0.666667 | 0.650000 |
| Pairwise wins / ties / losses | - | 2 / 25 / 3 |
| Catastrophic regressions | - | 2 |

Final decision: **NO_GO**. E1 remains the reference policy; G1 and G2-A remain experimental NO_GO branches. G2-A is not promoted over E1. This was a fresh TRAIN development experiment, not frozen DEV validation. It does not establish generation accuracy uplift or production latency/cost improvement, and it does not show that global reranking is generally harmful. It only rejects this specific rerank-informed five-document admission policy.

The detailed post-hoc, non-confirmatory movement analysis is in [G2 Rerank-Informed Admission Post-Hoc Forensic Analysis](reports/g2_rerank_informed_admission/post_hoc_forensic_analysis.md). The forensic report records the Task 7/8 execution deviations and confirms that no judgment, analytical logic, or gate logic changed after unblinding.

---

## 8. Leakage Rules

Do not:

- construct the retrieval corpus from NVIDIA gold contexts;
- use gold answers during retrieval;
- substitute gold contexts for actual retrieved context in end-to-end evaluation;
- hard-code question IDs to document IDs;
- tune a frozen comparison from individual DEV failures;
- remove difficult cases to improve reported metrics;
- report TRAIN tuning results as held-out DEV gains;
- patch Q346/Q492-specific rules and present the same 30-case sample as fresh validation.

The goal is to preserve a benchmark that can still falsify an optimization hypothesis.

---

## 9. Reproducibility Contract

Formal runs record enough identity to reproduce or audit the result, including:

- project commit SHA;
- dataset repository and revision;
- raw-file SHA256 values;
- loaded corpus / query counts;
- split rules;
- embedding / reranker / judge identities;
- retrieval and context-policy configuration;
- latency;
- dependency versions;
- checkpoint / manifest identities where applicable.

This prevents an aggregate score from becoming detached from the exact data and configuration that produced it.

---

## 10. Evaluation Lineage

The current evaluation story is intentionally continuous:

```text
Frozen TechQA data contract
        ↓
E0 Dense baseline
        ↓
E1 qwen3-rerank held-out improvement
        ↓
BM25 / RRF / Hybrid controlled comparison
        ↓
C1 preregistered gate → STOP
        ↓
Evidence-level audit
        ↓
Candidate coverage vs evidence-ranking diagnosis
        ↓
E1 document-aware context policy
        ↓
G1 document-local evidence experiment
        ↓
aggregate gain + catastrophic gate failure → NO_GO
        ↓
G2-A rerank-informed 5-document admission
        ↓
2 wins / 25 ties / 3 losses + 2 catastrophic regressions
        ↓
NO_GO
        ↓
keep E1 reference; retain G1/G2-A as experimental evidence
```

The value of this lineage is not that every experiment wins. The value is that each result determines the next engineering question without resetting the corpus, benchmark, or evaluation contract.

---

## 11. Artifact Map

Key entry points:

```text
experiments/evals/
├── README.md
├── datasets/techqa/
│   ├── manifest.json
│   └── corpus_manifest.json
├── reports/
│   ├── e1_rerank/
│   ├── r4_c1_hybrid_rerank/
│   ├── r1_evidence_audit/
│   ├── g0_generation/
│   └── g1_document_local/
│       ├── evidence_sufficiency_30case_preregistration_v1_1.json
│       └── final_decision.md
│   └── g2_rerank_informed_admission/
│       └── post_hoc_forensic_analysis.md
├── eval_techqa_generation.py
├── eval_techqa_hybrid_rerank.py
└── rerankers/
```

For the application-level architecture and Agent workflow, return to the repository root [README](../../README.md).

---

## 12. Current Scope

TechQA is the **long-term primary technical-support corpus and benchmark** for this project.

Current claims are intentionally bounded:

- formal retrieval claims use the frozen TechQA contract;
- online RAG remains Dense Chroma unless runtime code says otherwise;
- Hybrid / RRF / reranking experiments remain offline evaluation evidence unless explicitly promoted into serving;
- evidence-level audit is diagnostic and does not replace official document metrics;
- G1 improved evidence sufficiency in its frozen conditional 30-case experiment but failed its preregistered catastrophic-regression gate and therefore does not replace E1;
- G2-A rerank-informed five-document admission failed its preregistered gates and does not replace E1; its five-case movement analysis is post-hoc diagnostic evidence only;
- generation uplift is not claimed before frozen generation evaluation is complete;
- multi-source / conflict-resolution / autonomous Agentic RAG capability is not claimed by this benchmark.

If new stress sets are added later, they should extend this evaluation system rather than replace the existing TechQA lineage and force a new primary-corpus embedding / benchmark reset.
