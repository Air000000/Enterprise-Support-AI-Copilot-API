# Enterprise Support AI Copilot — Evaluation Governance r4 Design

Date: 2026-08-25
Branch: `feat/eval-techqa-baseline`
Status: Approved with corrections; implementation plan pending final user review

## 1. Why r4 exists

The previous evaluation path proved the retrieval-side hypothesis, but also exposed two problems:

1. Generation and LLM-as-Judge can dominate experiment cost without necessarily answering a new causal question.
2. Re-running retrieval inside generation evaluation introduces unnecessary cost and experimental drift. Generator comparisons should hold retrieval context fixed.

r4 therefore changes the execution order and evaluation data flow while preserving already completed results.

## 2. Frozen evidence that must not be invalidated

The following completed results remain valid and are frozen.

### TRAIN retrieval

E0 Dense:
- Recall@5: 0.613333
- Recall@20: 0.740000
- MRR@10: 0.510477

E1 Dense Top-100 + qwen3-rerank:
- Recall@5: 0.691111
- Recall@20: 0.815556
- MRR@10: 0.567206
- rerank p50: 3078.361 ms
- rerank p95: 3409.894 ms
- provider total tokens: 11,972,838

### DEV retrieval

E0 Dense:
- Recall@5: 0.643750
- Recall@20: 0.818750
- MRR@10: 0.518931
- p95 retrieval: 1142.953 ms

E1 Dense Top-100 + qwen3-rerank:
- Recall@5: 0.725000
- Recall@20: 0.843750
- MRR@10: 0.560841
- full retrieval p95: 4657.189 ms
- provider total tokens: 4,163,611

Paired DEV evidence:
- Recall@5: 21 fixed / 8 regressed
- Recall@20: 5 fixed / 1 regressed
- MRR@10: 44 improved / 30 regressed / 86 unchanged

The DEV retrieval result is already consumed and must not be used to tune reranker parameters or later retrieval design decisions.

## 3. New hard experiment-governance rule

Every paid model experiment must answer all seven questions before execution:

1. What hypothesis is being tested?
2. Which existing artifacts can be reused?
3. What is the minimum number of model calls needed?
4. What is the hard cost cap?
5. What condition stops the experiment immediately?
6. What quantitative or auditable result will be produced?
7. Can that result become resume evidence or interview evidence?

No paid run is justified solely because an additional metric would make the report look more complete.

Any threshold described as a meaningful improvement, acceptable trade-off, or admission condition must be numerically pre-registered before the experiment that consumes that threshold is run. Thresholds may be chosen after a preceding diagnostic stage establishes the relevant population size, but they must be frozen before observing the gated experiment result.

## 4. Cost policy and ledger

Cheap/local experiments run first. Paid experiments are admission-gated by positive evidence from cheaper experiments.

Priority order:

`local analysis -> local retrieval experiment -> cheap rerank -> small controlled generation probe -> human evaluation -> final DEV generation`

Bulk LLM-as-Judge is no longer the default evaluation strategy.

Faithfulness remains an auxiliary metric because prior calibration showed weak human agreement. It must not become a hard optimization target without additional calibration evidence.

Provider/region identity is part of the experiment contract. Existing Singapore `qwen3-rerank` results must not be mixed with a cheaper region endpoint under the same experiment label.

A lightweight cost ledger must be maintained for all remaining stages. It does not need per-token accounting, but every paid stage must record estimated cap, actual spend, and the question the spend answered.

Initial ledger:

| Stage | Hypothesis / purpose | API | Estimated cap | Actual cost | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Completed E0/E1 | baseline + rerank evidence | mixed | historical | approximately CNY 41 cumulative | frozen |
| R2 | residual cause | local | CNY 0 | — | pending |
| R3 | lexical complementarity | local | CNY 0 API | — | allowed |
| R4 | Hybrid + Rerank benefit | qwen3-rerank | freeze before run | — | gated |
| G1 | generator bottleneck | candidate LLM | freeze before run | — | gated |
| D1 | final generation validation | final LLM / frozen evaluation | final approval | — | gated |

## 5. Retrieval-to-generation architecture

The target evaluation architecture is:

```text
Question
  -> retrieval once
  -> Frozen Retrieval Snapshot
       -> Retrieval metrics
       -> Generation eval
       -> Generator ablation
       -> Human pairwise evaluation
```

The frozen retrieval snapshot is more important than reusing query embeddings. Reusing embeddings would still repeat vector search; reusing frozen retrieval results removes both repeat search cost and retrieval drift.

A generation-ready retrieval snapshot must contain at least:

- `question_id`
- normalized question
- rank
- `chunk_id`
- `document_id`
- chunk content
- distance / score required by the refusal policy

For a generator ablation, Question, Context and Prompt must be identical. The only main variable may be the generator model.

## 6. Existing-artifact compatibility

The current E0 retrieval artifact for answerable cases contains chunk/document identity and rankings but not all generation-required content/distance fields.

Generation datasets also include impossible cases that are absent from retrieval qrels evaluation:

- TRAIN generation: 450 answerable + 150 impossible
- DEV generation: 160 answerable + 150 impossible

Therefore r4 does not pretend that all old retrieval JSONL files are already generation-ready snapshots.

Instead:

- completed E0 TRAIN generation artifacts are preserved;
- future generator ablations should reuse already persisted TRAIN `retrieval_context` wherever possible;
- future final DEV generation should construct/freeze retrieval context once, including impossible cases, before generation evaluation consumes it.

## 7. DEV governance after partial consumption

Ideal protocol would use DEV only after all TRAIN design decisions are frozen. That is no longer fully possible because E0/E1 retrieval DEV has already been executed and inspected.

The exposure boundary must be described precisely:

- 160 answerable DEV query IDs have already been exposed on the retrieval side through aggregate metrics and paired retrieval analysis;
- DEV generation gold answers have not been used for generator selection or prompt tuning;
- DEV generation outputs have not been produced or inspected;
- DEV generation-side labels, human rubric results, and LLM-as-Judge results have not been used for optimization;
- the 150 DEV impossible cases have not yet been consumed by the final generation evaluation path.

Therefore the final generation experiment must be described as:

> frozen DEV generation validation after prior retrieval-side DEV exposure

It must not be described as a completely untouched held-out DEV evaluation.

From r4 onward:

- retrieval DEV is frozen historical validation evidence;
- no Hybrid decision, generator selection, prompt change, judge rubric change or other optimization may be justified from DEV retrieval outcomes;
- all subsequent design decisions return to TRAIN;
- no failure-driven patching is allowed after final generation DEV begins.

The final report must describe this chronology accurately.

## 8. E1 evidence versioning requirement

Before beginning new retrieval optimization, the already-completed E1 evidence must be made repository-auditable without committing unnecessarily large checkpoints/results.

Create a lightweight versioned directory:

```text
experiments/evals/reports/e1_rerank/
```

At minimum version:

- TRAIN run manifest or a compact frozen manifest containing the same experiment identity;
- TRAIN metrics;
- DEV run manifest or a compact frozen manifest containing the same experiment identity;
- DEV metrics;
- `comparison.md` summarizing E0 -> E1 TRAIN and DEV results.

`comparison.md` must include:

- Recall@5 / Recall@20 / MRR@10 deltas;
- paired fixed/regressed evidence where available;
- rerank incremental latency and full retrieval latency where available;
- provider/model/region identity;
- provider token usage;
- SHA256 references for the local large checkpoint/results artifacts that are intentionally not committed.

Large `*_checkpoint.jsonl` and candidate-heavy `*_results.jsonl` files may remain local when their hashes and provenance are captured in the compact evidence bundle.

## 9. Revised execution sequence

### Stage R1 — Close current E0 DEV generation infrastructure TDD

Purpose: finish the already-started zero-cost code path without launching a paid DEV generation run.

Action:
- verify commit `8d1ed869a0c531961bba7d56910295be9bacc615` locally;
- keep DEV generation CLI/manifest/checkpoint infrastructure available;
- do not run `python -m experiments.evals.eval_techqa_generation --split dev`.

Exit gate:
- focused pytest passes;
- Ruff passes;
- no provider call.

### Stage R1.5 — Version E1 evidence and initialize the cost ledger

Purpose: make already-earned retrieval results auditable before starting another experiment.

Cost: CNY 0 API.

Output:
- compact E1 TRAIN/DEV manifests and metrics;
- E0 vs E1 paired comparison summary;
- provider/region/token evidence;
- hashes for local large artifacts;
- initialized cost ledger.

Exit gate:
- repository contains enough compact evidence to independently trace the resume/interview metrics without committing large payloads.

### Stage R2 — E1 TRAIN residual analysis

Hypothesis: characterize the dominant residual retrieval failures after reranking and quantify where lexical/complementary retrieval might help.

Cost: approximately zero API cost.

Output:
- residual taxonomy/counts;
- lexical-miss share;
- rank failure distribution;
- Dense-miss population relevant to the local Hybrid pilot;
- evidence used to pre-register R3/R4 gates.

R2 does not gate whether a zero-cost R3 Hybrid pilot may run. It informs the pre-registered thresholds and interpretation of R3.

### Stage R3 — Local BM25 + Dense + RRF TRAIN pilot

Hypothesis: lexical retrieval provides complementary relevant-document coverage beyond Dense.

Admission: allowed by default because BM25/RRF is local and approximately zero API cost.

Cost: CNY 0 model API.

Compare at minimum:
- Dense alone;
- BM25 alone;
- Dense + BM25 + RRF.

Main outputs:
- candidate Recall@20 / Recall@100 as applicable;
- Dense-only relevant hits;
- BM25-only relevant hits;
- overlap/complementarity;
- recovered Dense misses;
- local retrieval latency.

Before observing R3 results, freeze numerical interpretation thresholds using the R2 population. At least one pre-registered complementarity/recall condition must be satisfied before paid R4 is admitted.

### Stage R4 — Hybrid + Rerank TRAIN, conditional and paid

Hypothesis: the better Hybrid candidate pool, if R3 establishes one, can be converted by reranking into a better final ranking than E1 Dense + Rerank.

Admission:
- only if R3 satisfies the pre-registered complementarity/candidate-recall gate;
- provider/model/region and rerank contract must be frozen;
- cost cap must be estimated and approved before launch.

Main metrics:
- Recall@5
- Recall@20
- MRR@10
- incremental/full retrieval latency
- provider token usage and cost

Before R4 results are observed, freeze the minimum acceptable retrieval gain and maximum acceptable latency/cost trade-off. No post-hoc definition of meaningful improvement is allowed.

### Stage G1 — Small controlled generator ablation on TRAIN

Hypothesis: generator model capability is a material bottleneck once relevant evidence is already present.

Candidate universe:
- all TRAIN answerable cases for which gold evidence is confirmed to be present in the frozen Top-3 context.

Sampling rule:
- freeze the sample before running the candidate generator;
- do not select only baseline failures;
- stratify approximately 40–60 answerable cases across baseline high / medium / low correctness;
- where practical, preserve diversity in question length and evidence complexity.

Controlled variables:
- same question
- same frozen context
- same prompt
- only generator changes

Primary evaluation:
- blind A/B pairwise human evaluation;
- no bulk Faithfulness run by default;
- optional small Judge probe only when it answers a separate explicit question.

Answerable-case rubric:
1. core factual correctness
2. evidence grounding
3. completeness
4. contradiction
5. over-abstention / failure to answer

Answerable error labels:
- factual error
- key omission
- evidence contradiction
- unsupported extension
- over-abstention / failure to answer

Primary answerable outputs:
- A win / B win / Tie rates;
- reason taxonomy;
- recommendation to keep or switch generator.

#### G1b — Small impossible-case abstention probe

Add approximately 10–15 frozen TRAIN impossible cases as a separate probe rather than mixing them into the answerable pairwise win rate.

Purpose:
- test correct abstention vs unsafe answer behavior under the same frozen evaluation discipline.

Outputs:
- correct-abstention count/rate;
- unsafe-answer count/rate;
- representative reason labels.

### Stage G2 — Evaluation and final-system freeze

Before final generation DEV, freeze:
- final generator model;
- prompt;
- retrieval pipeline;
- frozen retrieval snapshot contract;
- refusal rule;
- human rubric;
- any Judge usage and its role (primary vs auxiliary);
- final DEV metrics to be reported.

No metric definition or system configuration may be changed after final DEV generation output is observed.

### Stage D1 — Final generation DEV

Purpose: validate the frozen final generation configuration after prior retrieval-side DEV exposure.

Requirements:
- construct/freeze retrieval context once for all 310 DEV generation cases;
- default to running only the final generator selected and frozen in G2;
- do not repeat retrieval solely because another metric is being evaluated;
- no DEV-driven patching after the run begins;
- estimate and hard-cap cost immediately before launch.

Baseline-vs-final DEV generator comparison is not the default. If such a comparison is required, both generators and the comparison contract must be pre-registered before any DEV generation output is observed, and the result may not be used to trigger another model/configuration change.

Final outputs must include only metrics and evaluation procedures frozen in G2.

## 10. Pre-registration rule for stage gates

The implementation plan must turn qualitative gates into explicit numeric criteria before the gated result is observed.

Examples of eligible R3 gate signals include:
- absolute candidate Recall@100 gain of at least a pre-registered number of percentage points;
- recovery of at least a pre-registered fraction of Dense misses;
- at least a pre-registered number of BM25-only gold-document hits.

Examples of eligible R4 gate signals include:
- minimum Recall@5 or MRR@10 improvement over E1;
- maximum tolerated full-retrieval latency increase;
- maximum provider cost per evaluation run.

The exact numbers should be chosen from R2 population sizes and operational constraints, then frozen before R3/R4 results are inspected. The plan must not retrofit thresholds to observed outcomes.

## 11. Resume/interview output rule

Each stage must produce a verifiable engineering result, not necessarily an improvement.

A valid result can be:
- measured improvement;
- measured regression/trade-off;
- a failure class distribution;
- a gate decision that rejects unnecessary complexity;
- an evaluation reliability result.

Final resume wording should use only the strongest 2–3 quantitative outcomes. Other stages remain supporting interview evidence explaining why each engineering decision was made.

## 12. Immediate next step

Do not start any new paid experiment.

First close the existing DEV-generation infrastructure TDD locally with both pytest and Ruff evidence. Then version the existing E1 evidence and cost ledger before beginning Stage R2.

After final user review of this corrected R4 design, create the r4 implementation plan and execute in the order:

`R1 -> R1.5 -> R2 -> R3 -> gated R4 -> G1/G1b -> G2 -> D1`.
