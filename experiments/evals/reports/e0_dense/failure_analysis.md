# Task 7 — TechQA TRAIN Failure Analysis v1

**Scope:** TRAIN only  
**Baseline:** E0 Dense Retrieval + E0 Generation  
**Date:** 2026-08-24  
**Purpose:** Use measured failures to select exactly one E1 optimization. DEV remains frozen.

## 1. Evidence used

This report joins the frozen E0 retrieval and generation artifacts and separates deterministic retrieval evidence from diagnostic judge/manual evidence.

E0 retrieval on 450 answerable TRAIN queries:

| Metric | Value |
| --- | ---: |
| Document Recall@5 | 0.613333 |
| Document Recall@20 | 0.740000 |
| MRR@10 | 0.510477 |
| p50 retrieval latency | 991.995 ms |
| p95 retrieval latency | 1819.083 ms |

E0 generation on 600 TRAIN cases produced a mean automatic correctness score of 0.312. That score is used only as a screening signal because fixed-sample manual calibration showed coarse correctness agreement of 14/20 (70%). Faithfulness is auxiliary only because its coarse agreement was 10/20 (50%).

The original exact-string abstention metric is also not treated as a factual hallucination rate. A separate manual audit of all 150 impossible cases found 70 semantic abstentions, 58 corpus-supported answers despite the benchmark impossible label, 21 exact correct abstentions, and 1 confirmed unsafe answer.

## 2. Observable failure buckets

The offline join materialized one observable bucket for every answerable query.

| Observable bucket | Count | Rate |
| --- | ---: | ---: |
| gold_in_context_low_correctness | 132 | 29.33% |
| retrieval_miss_top20 | 117 | 26.00% |
| gold_in_context_high_correctness | 68 | 15.11% |
| retrieval_rank_6_20 | 57 | 12.67% |
| gold_in_context_mid_correctness | 44 | 9.78% |
| retrieval_rank_4_5 | 22 | 4.89% |
| top3_chunk_admission_gap | 6 | 1.33% |
| gold_in_context_model_abstain | 4 | 0.89% |

Totals:

- answerable cases: 450
- gold present in actual generation Top-3 document context: 250 (55.56%)
- gold absent from actual generation Top-3 document context: 200 (44.44%)

These buckets are observable pipeline states. They are not automatically equivalent to causal taxonomy codes such as R2, R4, or G1.

## 3. Cross-run query-normalization drift

The retrieval and generation datasets contain the same question IDs and same substantive question text, but 250/450 answerable questions differ only in trailing whitespace.

Cross-run diagnostic:

- exact question text: 200 cases; Top-3 sequence changed in 0; gold admission changed in 0
- trailing-whitespace-only difference: 250 cases; Top-3 sequence changed in 74 (29.60%); gold admission changed in 5 (2.00%)

The five gold-admission changes were:

- TRAIN_Q091: retrieval Top-3 hit -> generation Top-3 miss
- TRAIN_Q212: retrieval Top-3 miss -> generation Top-3 hit
- TRAIN_Q358: retrieval Top-3 miss -> generation Top-3 hit
- TRAIN_Q500: retrieval Top-3 hit -> generation Top-3 miss
- TRAIN_Q572: retrieval Top-3 hit -> generation Top-3 miss

These five cases are treated as `cross_run_query_normalization_drift`, not as evidence for R2/R3/R4/G1/G2.

After excluding those five cases from causal counting, the stable observable counts are:

| Observable bucket | Stable count |
| --- | ---: |
| retrieval_miss_top20 | 117 |
| retrieval_rank_6_20 | 57 |
| retrieval_rank_4_5 | 20 |
| top3_chunk_admission_gap | 3 |
| gold_in_context_low_correctness | 132 |
| gold_in_context_mid_correctness | 44 |
| gold_in_context_high_correctness | 68 |
| gold_in_context_model_abstain | 4 |
| **Total** | **445** |

Future evaluation calls must use one canonical query string per question. Trailing whitespace is normalized with `rstrip()` before embedding/rerank provider calls; substantive text differences remain errors.

## 4. Manual causal audit

A fixed random diagnostic sample was drawn after excluding the five drift cases:

- 30 / 117 `retrieval_miss_top20`
- 30 / 132 `gold_in_context_low_correctness`

These sample proportions are diagnostic signals for experiment selection. They are not population-level failure rates and must not be used as resume metrics.

### 4.1 Dense Top-20 miss sample

| Manual diagnosis | Count | Sample rate |
| --- | ---: | ---: |
| benchmark qrel / query ambiguity | 17 | 56.7% |
| clear R2 lexical candidate | 7 | 23.3% |
| semantic / indirect miss | 6 | 20.0% |

Interpretation:

- Some exact-token cases are credible R2 failures, especially distinctive error-code/version/CVE queries.
- R2 is not dominant in this diagnostic sample.
- Many apparent Dense misses are contaminated by incomplete/ambiguous qrels or queries where retrieved documents are at least as directly relevant as the single labeled gold document.
- Therefore `retrieval_miss_top20 = 117` is not sufficient evidence to admit Hybrid/BM25 as E1.

### 4.2 Gold-in-context low-correctness sample

| Manual diagnosis | Count | Sample rate |
| --- | ---: | ---: |
| evaluation reference / judge mismatch | 14 | 46.7% |
| R4 context coverage / chunk evidence missing | 12 | 40.0% |
| G1 generation misuse | 2 | 6.7% |
| G2 incomplete answer | 2 | 6.7% |

Interpretation:

- Automatic low correctness substantially overstates clear model-generation failure.
- R4/context coverage is a real issue: in multiple cases the relevant document is present but the decisive evidence chunk is outside actual generation Top-3.
- Clear G1/G2 cases are a minority in the diagnostic sample, so prompt/model changes are not admitted as E1.
- Chunk optimization remains a later candidate, but current evidence is not stronger than the deterministic ranking-error evidence below.

## 5. Actionable failure decision

The strongest deterministic actionable class is R3 Ranking Error.

After excluding cross-run drift:

- relevant document rank 4-5: 20 cases
- relevant document rank 6-20: 57 cases
- deterministic candidate-in-pool ranking cases: **77 / 450**

These cases satisfy the R3 condition directly: the relevant document is already in the larger Dense candidate pool but is not ranked high enough for the final small context window.

Other candidate actions are not admitted now:

- **Hybrid / BM25:** credible R2 cases exist, but the fixed manual miss sample is dominated by qrel/query ambiguity rather than clear lexical misses.
- **Chunk optimization:** R4/context-coverage cases are real, but the evidence currently comes from a 30-case diagnostic sample and is mixed with reference/judge mismatch.
- **Prompt/model optimization:** only 4/30 low-correctness audit cases were clear G1/G2.
- **Grounding/abstention:** the full 150-case impossible audit found only 1 confirmed unsafe answer, so N1/G3 is not the dominant actionable class.

## 6. Approved E1 — qwen3-rerank

Exactly one experiment is admitted:

> **E1 = frozen E0 Dense Top-100 raw chunk candidates + DashScope `qwen3-rerank`.**

The E1 retrieval runner will:

1. reuse the saved E0 raw Top-100 chunk candidate IDs for each TRAIN query;
2. read candidate text from the existing isolated TechQA Chroma collection;
3. normalize only trailing query whitespace before the rerank provider call;
4. rerank the same 100 candidates with `qwen3-rerank`;
5. preserve chunk IDs and document IDs;
6. collapse reranked chunks to document ranking using the existing first-occurrence rule;
7. evaluate with the same qrels/ranx metric definitions;
8. record provider failures and p50/p95 rerank latency.

Fair-ablation constants:

- corpus revision unchanged
- TRAIN query IDs unchanged
- qrels unchanged
- chunk strategy unchanged (`800 / 120 / 150`, paragraph-aware)
- embedding model unchanged
- saved Dense candidate pool unchanged
- generation model/prompt unchanged
- judge contract unchanged
- no BM25/RRF in E1
- no chunk-strategy change in E1
- no DEV failure-driven tuning

E1 is considered useful only if the reranking stage improves the ranking-oriented retrieval metrics (especially Document Recall@5 and MRR@10) while Recall@20, rerank latency, and provider reliability are reported transparently. The frozen DEV ablation remains the final evidence gate.

## 7. Task 7 exit gate

Task 7 exit condition is satisfied:

- weak retrieval/generation cases were materialized;
- observable failure classes were counted;
- cross-run evaluation noise was isolated;
- representative causal samples were manually reviewed;
- competing optimization hypotheses were rejected or deferred with evidence;
- exactly one E1 was selected.

**Next task:** Task 8 Branch B — implement and validate the isolated `qwen3-rerank` E1 runner before any frozen DEV run.
