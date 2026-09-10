# G2 Rerank-Informed Document Admission Design

## Status

Approved experimental design for the next TechQA evidence-retrieval iteration after G1.

This is a **bounded retrieval/evidence experiment**, not a serving-path migration. The only architecture variable under test is **which ranking determines admission of the first five candidate documents**.

Baseline commit: `7caf1a1245bdc73698c9fb218c863425ae3b7f8f`.

## Problem

G1 improved aggregate evidence sufficiency over E1, but the preregistered replacement gate remained `NO_GO` because one catastrophic regression occurred. The G1 closeout identified `TRAIN_Q346` as a candidate-document admission miss: a decisive chunk was present deep in the historical Dense Top100 and could be promoted by the strong global reranker, but G1 permanently excluded its document before that reranker could influence document admission.

The engineering lesson is broader than that case:

> A weak first-stage ranking should not receive an irreversible document-admission veto before a stronger reranker has had a chance to reorder the already-retrieved candidate pool.

This design must not encode Q346- or Q492-specific logic. Those cases are diagnostic/development evidence only and are excluded from fresh confirmation.

## Existing contracts

### E1 reference

```text
historical E0 Dense Top100
        ↓
one shared qwen3-rerank over Top100
        ↓
Top3 rerank anchors + historical Dense rank1 rescue
        ↓
anchor + up to 3 forward siblings per unique anchor document
        ↓
dedupe
        ↓
max16 context
```

The existing implementation is `retrieve_g0_e1_context()` in `experiments/evals/eval_techqa_generation.py`.

### G1 candidate

```text
historical E0 Dense Top100 order
        ↓
first 5 unique documents by Dense order
        ↓
full frozen document-local chunk expansion
        ↓
one merged qwen3-rerank
        ↓
Top16 selected evidence
        ↓
max16 context
```

The existing reusable G1 helpers in `experiments/evals/eval_techqa_generation.py` are:

- `select_candidate_document_ids()`
- `build_document_local_candidate_pool()`
- `select_document_local_evidence()`
- `assemble_selected_evidence_context()`
- `run_document_local_evidence_diagnostic()`

The reranker implementation is `rerank_candidates()` in `experiments/evals/rerankers/qwen3_reranker.py`.

## G2-A hypothesis

G2-A keeps all G1 downstream evidence-construction behavior fixed and changes only candidate-document admission order:

```text
historical E0 Dense Top100
        ↓
shared global qwen3-rerank over the same Top100
        ↓
first 5 unique documents by GLOBAL RERANK order
        ↓
full frozen document-local chunk expansion
        ↓
one merged qwen3-rerank
        ↓
Top16 selected evidence
        ↓
max16 context
```

The shared global rerank is an injected/replayed result. The G2 diagnostic path must **not** perform a second equivalent global provider call.

### Exact causal variable

G1:

```text
candidate_doc_ids = first 5 unique document IDs in Dense Top100 order
```

G2-A:

```text
candidate_doc_ids = first 5 unique document IDs in shared global-rerank order
```

Everything after `candidate_doc_ids` remains the same G1 mechanism.

## Frozen invariants

The following are fixed for G2-A and must not be tuned after looking at results:

- input candidate pool remains the historical/frozen E0 Dense Top100 chunks;
- global reranker remains `qwen3-rerank`;
- global rerank instruction remains exactly `Rank the candidate passages by relevance to resolving the technical support query.`;
- global rerank must be computed once per case and reused by any method that needs it;
- candidate document limit remains exactly `5`;
- candidate document order is first occurrence in the chosen ranking;
- admitted documents are expanded using the frozen/offline TechQA corpus;
- candidate pool safety ceiling remains `500` chunks;
- merged evidence reranker remains the same `qwen3-rerank` contract;
- final evidence limit remains `16`;
- final context limit remains `16`;
- no query rewrite;
- no BM25/Hybrid addition;
- no alternate reranker;
- no per-document rerank;
- no second-pass rerank;
- no gold labels or gold answers may influence retrieval;
- no case-specific rescue logic;
- no continuity/sibling-preservation change in G2-A.

Q492's continuity miss is intentionally **not** fixed in this experiment. Continuity remains a separate future causal variable.

## Code organization

G2-specific orchestration should live in a focused module instead of further expanding the already-large generation harness:

- create `experiments/evals/eval_techqa_g2_admission.py` for G2-A admission and comparison orchestration;
- create `tests/test_eval_techqa_g2_admission.py` for pure contract tests;
- reuse G1 pool construction, merged evidence selection, and context assembly from `eval_techqa_generation.py`;
- reuse `RerankResult` / `RerankedCandidate` / `rerank_candidates` from `experiments/evals/rerankers/qwen3_reranker.py`;
- do not modify the online API/serving retrieval path.

If the existing `select_candidate_document_ids()` type annotation prevents safe reuse with `RerankedCandidate`, only a behavior-preserving typing generalization is allowed. G1 ordering semantics must stay unchanged and remain covered by its existing tests.

## Fresh evaluation population

The previous G1 30-case set is now development/diagnostic evidence for this architecture and cannot confirm G2-A.

The new preregistered development sample is drawn from answerable `TRAIN_*` cases with:

1. a frozen historical E0 Top100 trace;
2. exactly one formal relevant TechQA document;
3. at least one chunk from that formal relevant document somewhere in the historical E0 Top100.

The eligibility boundary intentionally changes from G1's `formal relevant doc in Dense first5` condition to `formal relevant doc in Dense Top100` because document admission itself is the variable under test. Conditioning on Dense first5 would partially pre-select for the G1 admission policy.

Exclude before sampling:

- the historical 12-case G0 generation pilot;
- all 30 cases from `evidence_sufficiency_30case_preregistration_v1_1.json`;
- all TRAIN question IDs that appear in repository-frozen manual evidence-label artifacts, including `reports/r1_evidence_audit/evidence_labels.jsonl`;
- any additional explicitly enumerated case IDs that were manually inspected during prior G1 forensic work, if not already covered by the sets above.

Selection is deterministic:

```text
seed = techqa-g2-rerank-informed-admission-v1
key(qid) = sha256(seed + ':' + qid).hexdigest()
sort ascending by (key, qid)
select first 30 eligible cases
```

No replacement, resampling, outcome-based exclusion, or parameter change is allowed after the sample is frozen.

This remains a **fresh preregistered TRAIN development confirmation**, not the project's frozen DEV held-out benchmark. A DEV confirmation, if later justified, is a separate experiment and is not part of G2-A implementation.

## Method comparison

Primary comparison: **G1 vs G2-A**.

For each fresh case:

```text
historical E0 Dense Top100
        │
        ├── G1 admission: first5 docs by Dense order
        │       ↓
        │   full-doc expansion
        │       ↓
        │   merged rerank
        │       ↓
        │   Top16
        │
        └── one shared global rerank over Top100
                ↓
            G2-A admission: first5 docs by rerank order
                ↓
            full-doc expansion
                ↓
            merged rerank
                ↓
            Top16
```

The two methods therefore share the same Dense input and all downstream G1 mechanisms; only the ranking used for document admission differs.

Expected real rerank call graph per case during formal comparison:

- shared global Top100 rerank: `1`;
- G1 merged document-local rerank: `1`;
- G2-A merged document-local rerank: `1`;
- total: `3` real rerank calls per case;
- embedding: `0`;
- generation: `0`;
- judge/provider LLM: `0`.

For 30 cases, the hard maximum is therefore `90` real rerank calls. Provider failures stop the run; no retry, candidate truncation, case replacement, or model change is allowed.

## Evidence-sufficiency rubric

Reuse the frozen G1 claim-support semantics rather than inventing a new quality metric:

Claim support labels:

- `EXPLICIT_SUPPORT`
- `COMPOSABLE_SUPPORT`
- `INFERENCE_ONLY`
- `ABSENT`
- `UNRESOLVED`

Covered labels are only `EXPLICIT_SUPPORT` and `COMPOSABLE_SUPPORT`.

Per-context status:

- `COMPLETE`: all frozen material claims covered;
- `PARTIAL`: at least one but not all claims covered;
- `INSUFFICIENT`: zero material claims covered;
- `UNRESOLVED`: support cannot be judged reliably.

Primary aggregate metric is macro mean per-case claim coverage.

Claims must be frozen from question + gold answer **before** opening either method context. Context review is method-label blinded. Because the primary comparison now has two methods, use anonymous labels `A/B` rather than rebuilding the previous three-method `A/B/C` packet.

## Preregistered decision rule

Before any paid G2-A execution, freeze the exact GO/NO-GO gate.

`GO_TO_NEXT_STAGE=yes` only if all are true:

- all 30 cases execute under the frozen provider/capacity contract;
- unresolved evidence-review count is `0`;
- `G2_complete_count >= G1_complete_count`;
- `G2_insufficient_count <= G1_insufficient_count`;
- `G2_macro_claim_coverage >= G1_macro_claim_coverage`;
- pairwise ordinal `G2_wins > G2_losses`;
- catastrophic regression count is `0`, where catastrophic regression is `G1=COMPLETE && G2=INSUFFICIENT`.

If all contexts tie, G2-A does not justify its additional global-rerank cost and therefore does not pass the strict `wins > losses` gate.

A GO only authorizes a next design/evaluation stage. It does not establish production accuracy, latency improvement, statistical significance, or serving-path promotion.

## Cost and latency boundary

G2-A adds a global rerank stage relative to standalone G1. Track provider call count and returned token usage separately for:

- shared global rerank;
- G1 merged rerank;
- G2-A merged rerank.

These are descriptive engineering costs, not quality-gate inputs. Offline wall-clock timing must not be presented as production latency improvement/regression without a dedicated controlled runtime experiment.

## Leakage and anti-overfitting rules

Do not:

- tune `5`, `16`, or `500` from the new sample;
- add Q346-specific document rescue;
- add Q492-specific sibling preservation;
- inspect method labels while freezing claims;
- inspect G2 outcomes before the sample and gates are frozen;
- replace cases after provider or capacity failure;
- call the same global rerank twice to simplify code;
- report this TRAIN experiment as frozen DEV evidence;
- promote G2-A to the serving path from this experiment alone.

## Success interpretation

The experiment is designed to answer one causal question:

> Does moving document admission from Dense order to an already-available stronger global-rerank order preserve or improve G1 evidence sufficiency on fresh cases without introducing new catastrophic regressions?

A result outside that question should not be inferred from G2-A.