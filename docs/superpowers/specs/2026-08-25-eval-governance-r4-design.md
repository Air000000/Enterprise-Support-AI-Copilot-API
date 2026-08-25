# Enterprise Support AI Copilot — Evaluation Governance r4 Design

Date: 2026-08-25
Branch: `feat/eval-techqa-baseline`
Status: Approved design revision; implementation plan pending user review

## 1. Why r4 exists

The previous evaluation path proved the retrieval-side hypothesis, but also exposed two problems:

1. Generation and LLM-as-Judge can dominate experiment cost without necessarily answering a new causal question.
2. Re-running retrieval inside generation evaluation introduces unnecessary cost and experimental drift. Generator comparisons should hold retrieval context fixed.

r4 therefore changes the execution order and evaluation data flow while preserving already completed results.

## 2. Frozen evidence that must not be invalidated

The following completed results remain valid and are frozen:

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

## 4. Cost policy

Cheap/local experiments run first. Paid experiments are admission-gated by positive evidence from cheaper experiments.

Priority order:

`local analysis -> local retrieval experiment -> cheap rerank -> small controlled generation probe -> human evaluation -> final DEV generation`

Bulk LLM-as-Judge is no longer the default evaluation strategy.

Faithfulness remains an auxiliary metric because prior calibration showed weak human agreement. It must not become a hard optimization target without additional calibration evidence.

Provider/region identity is part of the experiment contract. Existing Singapore `qwen3-rerank` results must not be mixed with a cheaper region endpoint under the same experiment label.

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

From r4 onward:

- retrieval DEV is frozen historical validation evidence;
- no Hybrid decision, generator selection, prompt change, judge rubric change or other optimization may be justified from DEV retrieval outcomes;
- all subsequent design decisions return to TRAIN;
- generation DEV remains unconsumed and is reserved for final validation after retrieval/generator/prompt/evaluation rubric are frozen;
- no failure-driven patching is allowed after final generation DEV begins.

The final report must describe this chronology accurately rather than claiming that all DEV data remained untouched until the end.

## 8. Revised execution sequence

### Stage R1 — Close current E0 DEV generation infrastructure TDD

Purpose: finish the already-started zero-cost code path without launching a paid DEV generation run.

Action:
- verify commit `8d1ed869a0c531961bba7d56910295be9bacc615` locally;
- keep DEV generation CLI/manifest/checkpoint infrastructure available;
- do not run `python -m experiments.evals.eval_techqa_generation --split dev`.

Exit gate:
- focused tests pass;
- Ruff passes;
- no provider call.

### Stage R2 — E1 TRAIN residual analysis

Hypothesis: after reranking, identify the dominant residual retrieval failure class before adding another retrieval component.

Cost: approximately zero API cost.

Output:
- residual taxonomy/counts;
- lexical-miss share;
- rank failure distribution;
- explicit recommendation: enter or skip Hybrid.

Gate:
- Hybrid is admitted only if residual evidence shows a meaningful lexical/complementary-recall opportunity.

### Stage R3 — BM25 + Dense + RRF TRAIN, conditional

Hypothesis: lexical retrieval can recover relevant documents missed by Dense.

Cost: local BM25/RRF; approximately zero model API cost.

Main metrics:
- candidate Recall@20 / Recall@100 as applicable;
- overlap/complementarity with Dense;
- latency.

Gate:
- only meaningful candidate-recall improvement admits Hybrid + Rerank.

### Stage R4 — Hybrid + Rerank TRAIN, conditional

Hypothesis: a better candidate pool plus reranking yields a better final ranking than E1 Dense + Rerank.

Cost: hard-capped before execution using the frozen provider/region pricing applicable at run time.

Main metrics:
- Recall@5
- Recall@20
- MRR@10
- incremental/full retrieval latency
- provider token usage and cost

Gate:
- must beat E1 on a meaningful retrieval metric with an acceptable latency/cost trade-off.

### Stage G1 — Small controlled generator ablation on TRAIN

Hypothesis: generator model capability is a material bottleneck once relevant evidence is already present.

Sample:
- approximately 40–60 information-rich answerable cases;
- select cases where gold evidence is already in the frozen Top-3 context;
- use existing persisted TRAIN contexts where possible.

Controlled variables:
- same question
- same frozen context
- same prompt
- only generator changes

Evaluation:
- blind A/B pairwise human evaluation is primary;
- no bulk Faithfulness run by default;
- optional small judge probe only when it answers a separate explicit question.

Human rubric:
1. core factual correctness
2. evidence grounding
3. completeness
4. contradiction
5. abstention behavior

Error labels should include:
- factual error
- key omission
- evidence contradiction
- unsupported extension
- correct abstention
- over-abstention / failure to answer

Outputs:
- A win / B win / Tie rates;
- reason taxonomy;
- recommendation to keep or switch generator.

### Stage G2 — Evaluation rubric freeze

Before final generation DEV:
- freeze generator model;
- freeze prompt;
- freeze retrieval pipeline;
- freeze refusal rule;
- freeze human rubric;
- freeze any Judge usage and its role (primary vs auxiliary).

No metric definition may be changed after final DEV results are observed.

### Stage D1 — Final generation DEV

Purpose: held-out validation of the frozen generation configuration.

Requirements:
- construct/freeze retrieval context once for all 310 DEV generation cases;
- reuse frozen context for generation and any generator comparison;
- no repeated retrieval solely because another metric/model is being evaluated;
- no DEV-driven patching after the run begins.

Cost:
- must be estimated and hard-capped immediately before launch;
- the run may proceed only if the estimated cost is justified by the final validation question.

Final outputs should include only metrics that were frozen in Stage G2.

## 9. Resume/interview output rule

Each stage must produce a verifiable engineering result, not necessarily an improvement.

A valid result can be:
- measured improvement;
- measured regression/trade-off;
- a failure class distribution;
- a gate decision that rejects unnecessary complexity;
- an evaluation reliability result.

Final resume wording should use only the strongest 2–3 quantitative outcomes. Other stages remain supporting interview evidence explaining why each engineering decision was made.

## 10. Immediate next step

Do not start any new paid experiment.

First close the existing DEV-generation infrastructure TDD locally:

```powershell
git pull

pytest `
  tests/test_generation_eval_dev.py `
  tests/test_generation_eval_cli.py `
  tests/test_generation_eval_runner.py `
  tests/test_generation_eval_manifest_lock.py `
  -v

ruff check `
  experiments/evals/eval_techqa_generation.py `
  tests/test_generation_eval_dev.py `
  tests/test_generation_eval_cli.py `
  tests/test_generation_eval_runner.py `
  tests/test_generation_eval_manifest_lock.py
```

After that verification passes, create the r4 implementation plan and begin Stage R2 (E1 TRAIN residual analysis).
