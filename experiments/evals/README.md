# TechQA Evaluation

This directory is the primary long-term TechQA evaluation harness for Enterprise Support AI Copilot. It records frozen evidence and decisions; it is not an online-runtime configuration file.

## Portfolio-v1 Frozen Decision

### Current online runtime

The current online/API serving path is **Dense Chroma Retrieval**. BM25, RRF, Hybrid, reranking, and Flat Top14 are not described as deployed serving behavior.

### Held-out validation evidence

The frozen DEV comparison for Dense Top100 plus `qwen3-rerank` is:

| Metric | Dense | Dense + rerank |
| --- | ---: | ---: |
| Recall@5 | 0.643750 | 0.725000 |
| Recall@20 | 0.818750 | 0.843750 |
| MRR@10 | 0.518931 | 0.560841 |

This is a held-out retrieval result, not an online-serving claim.

### Portfolio-v1 frozen engineering candidate

```text
Dense chunk Top100 + BM25 chunk Top100
    -> equal-weight chunk RRF (k=60)
    -> fused Top100
    -> qwen3-rerank
    -> flat_rerank_top14_v1
```

R4 C1 Hybrid+Rerank's formal historical status remains **FAIL** because its preregistered MRR promotion gate failed. All three TRAIN aggregate metrics improved and Dense/BM25 lexical complementarity was demonstrated, but the later retention of this fixed Hybrid route is a portfolio-v1 engineering decision—not a rewrite of that formal experiment outcome. No additional fusion tuning is admitted.

### Unresolved items

- refusal / evidence-sufficiency policy;
- final generation acceptance;
- final DEV generation validation;
- runtime integration;
- Ticket Agent shared retrieval integration.

Historical `dense_top1_distance > 0.9` does not automatically become the portfolio-v1 refusal contract: Dense Top1 distance is only one first-stage signal while the frozen candidate is Hybrid+rereank.

The latest decision entry is [portfolio_v1_rag_freeze/architecture_freeze.md](reports/portfolio_v1_rag_freeze/architecture_freeze.md).

---

## Dataset and Split Contract

TechQA supplies 28,481 Technote documents, 610 answerable retrieval queries with deterministic document-level qrels, and 910 generation/abstention QA records (610 answerable, 300 impossible). Each answerable retrieval query has one relevant document.

| Split | Answerable | Impossible | Usage |
| --- | ---: | ---: | --- |
| `TRAIN_*` | 450 | 150 | development, failure analysis, parameter selection |
| `DEV_*` | 160 | 150 | frozen held-out comparison |

TRAIN is the development surface and DEV is held out. Do not inspect individual DEV failures to tune a frozen comparison, remove difficult cases, or turn TRAIN tuning into a DEV claim. Dataset identities and SHA256 values are in `datasets/techqa/manifest.json` and `datasets/techqa/corpus_manifest.json`.

## Retrieval Metric Contract

The runtime retrieves chunks while TechQA qrels are document-level. For formal IR metrics, preserve the raw chunk ranking, collapse to unique `document_id` values at the first occurrence, then evaluate that document ranking. The primary metrics are Document Recall@5, Document Recall@20, and MRR@10. On single-relevant-document TechQA, Recall@K and Hit@K are numerically equivalent.

## Experiment Decision Table

| Stage | Question | Key evidence | Formal status | Portfolio decision |
| --- | --- | --- | --- | --- |
| E0 Dense | Establish a baseline? | Frozen Dense route. | Baseline | Historical comparator. |
| E1 Dense+rereank | Does reranking improve held-out retrieval? | DEV R@5 .643750 -> .725000; R@20 .818750 -> .843750; MRR .518931 -> .560841. | PASS | Held-out evidence retained. |
| R4 C1 Hybrid+rereank | Does Hybrid+rereank clear the preregistered promotion gates? | TRAIN E1 .691111/.815556/.567206; Hybrid .702222/.831111/.570929. | **FAIL** | Fixed Hybrid route retained later as a portfolio-v1 engineering candidate; no fusion tuning. |
| R1 evidence audit | Are document hits answer evidence? | 60 labels; 54 usable cases. | Diagnostic | Use evidence-hit accounting. |
| G1 document-local | Does this Stage2 context hypothesis promote? | Aggregate gain with catastrophic regression. | NO_GO | Historical experimental branch. |
| G2-A rerank-informed admission | Does rerank-informed admission promote? | 2 wins / 25 ties / 3 losses; 2 catastrophic regressions. | NO_GO | Historical experimental branch. |
| Retrieval frontier freeze | Are more retrieval parameters justified? | Post-hoc frontier audit closed without admitting another retrieval parameter-search cycle. | CLOSED | Retrieval parameter research closed; no RRF-k, candidate-depth, source-weight, quota, or per-doc-cap tuning. |
| Final context budget | What K reaches the audited ceiling? | K14 and K20: answer 35/54; useful 43/54. | Selected | Flat Top14. |
| Final locality recovery | Can second locality rerank recover residuals? | Answer 35 -> 35; useful 43 -> 43; 0/7 actionable recovery. | Rejected | No locality second rerank. |
| TechQA structure forensic | Is structure observable for diagnosis? | 28,481 docs; 0.994347 allowlist coverage, not parser accuracy. | Diagnostic | Bounded synthesis only. |
| Structure-preserving synthesis | Does static parent expansion help net hits? | Answer 35 -> 32; useful 43 -> 38; 2 miss->hit, 5 hit->miss. | Rejected | No parent expansion. |
| Failure attribution | What caused the regressions? | 5/5 budget crowd-out; 2/2 whole-document recovery; median docs 12 -> 4.5. | Established for audited cases | Retain Flat Top14. |
| Portfolio-v1 freeze | What is the contract? | Fixed retrieval route and `flat_rerank_top14_v1`. | CLOSED | Stop retrieval/context research. |

Static parent expansion traded cross-document breadth for within-document depth under a fixed context budget. This supports the audited Flat Top14 decision; it does **not** establish that breadth is always better than depth, Parent-Child is bad, or document structure is generally ineffective.

## Historical Generation Harness

`document_aware_forward_expansion_v1` remains the real **historical generation harness context policy** and its artifacts are retained. It is **not** the frozen portfolio-v1 final context policy.

The frozen portfolio-v1 context is `flat_rerank_top14_v1`. Final refusal and generation contracts remain unresolved. G1 and G2-A are historical TRAIN experiments with formal NO_GO outcomes; see their reports for their detailed evidence rather than treating them as current design.

## Evaluation Lineage

```text
Frozen TechQA contract
  -> E0 Dense
  -> E1 Dense+rereank held-out improvement
  -> BM25/RRF complementarity
  -> R4 C1 formal FAIL
  -> evidence-level audit
  -> G1/G2 NO_GO
  -> retrieval frontier closure
  -> Flat context budget analysis
  -> Top14 selected
  -> locality second-rerank rejected
  -> TechQA structure forensic
  -> structure-preserving synthesis rejected
  -> budget-crowd-out attribution
  -> portfolio-v1 retrieval/context freeze
```

`retrieval_parameter_research = CLOSED` and `context_assembly_research = CLOSED`. Reopen only for a new failure mode outside portfolio-v1 finalization, not to tune current artifacts.

## Artifact Map

- Latest decision: `reports/portfolio_v1_rag_freeze/architecture_freeze.md`
- Frozen contract: `reports/portfolio_v1_rag_freeze/freeze.json`
- Decision timeline: `reports/portfolio_v1_rag_freeze/evidence_timeline.md`
- Flat budget: `reports/final_context_budget/`
- Locality replay: `reports/final_locality_recovery/`
- Structure forensic: `reports/techqa_structure_forensic/`
- Structure-preserving synthesis and attribution: `reports/structure_preserving_synthesis/`
- Held-out E1: `reports/e1_rerank/comparison.md`
- Hybrid R4 C1: `reports/r4_c1_hybrid_rerank/`
- Evidence audit: `reports/r1_evidence_audit/`
- Historical G1/G2: `reports/g1_document_local/`, `reports/g2_rerank_informed_admission/`

## Leakage, Reproducibility, and Scope

Do not use gold answers or contexts for retrieval, hard-code question-to-document mappings, retune from individual frozen DEV failures, or present TRAIN tuning as held-out gains. Formal runs record commit, dataset identity, file SHA256, split, model/configuration, latency, dependency, and checkpoint/manifest identities.

Current non-claims:

- online runtime remains Dense Chroma;
- the Hybrid+rereank route is an evaluated/frozen candidate, not serving;
- R4 C1 remains formal FAIL;
- Flat Top14 is a TRAIN-development selection, not a universal optimum;
- locality and structure-expansion negatives do not generalize to Parent-Child or document structure generally;
- refusal policy and final generation validation are not frozen.

For application architecture and the Agent workflow, see the repository [README](../../README.md).
