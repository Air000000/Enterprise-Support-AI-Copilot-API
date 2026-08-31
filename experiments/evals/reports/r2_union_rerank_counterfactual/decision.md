# R2 Full-Union Rerank Counterfactual Decision

## Status

**STOP_R2_ROUTE**

Do not expand this counterfactual to the full 450-query TechQA TRAIN set.

This experiment is a targeted diagnostic counterfactual, not a representative benchmark.

## Hypothesis

C1 fixed-budget chunk-level RRF Top100 compression may discard useful
single-source tail information before the reranker can inspect it.

R2 removes that compression step for the preregistered diagnostic cohorts:

- Dense Top100
- BM25 Top100
- stable unique union: Dense order first, then unseen BM25 chunks
- no RRF
- same qwen3-rerank model
- same reranker instruction
- no parameter search
- TRAIN only
- no DEV

## Frozen diagnostic cohorts

### Cohort A: document-level fusion losses

- 19 preregistered cases
- all 19 target gold documents are present in the full union
- R2 GoldDoc Top5: 2/19
- R2 GoldDoc Top20: 8/19
- therefore 11/19 remain outside Top20

### Cohort B: human-confirmed answer-evidence compression losses

- 5 preregistered cases
- all 5 human-confirmed answer-bearing chunks are present in the full union
- R2 AnswerEvidence Top5: 0/5
- R2 AnswerEvidence Top20: 1/5
- therefore 4/5 remain outside Top20

## Interpretation

For these targeted cases, candidate availability is necessary but insufficient.

Removing the RRF Top100 compression restores the missing target information to
the reranker's input, but the same reranker does not reliably promote that
information into the useful early ranks.

The document-level cohort shows partial recovery, while the evidence-level
cohort is substantially weaker. Therefore the current full-union rerank design
does not provide sufficient evidence to justify a 450-query TRAIN expansion.

The remaining failures in these cohorts occur downstream of candidate
availability: the target document/evidence is present in the full union but
usually remains too low after reranking.

This does not establish the prevalence of this failure mode over the entire
TechQA dataset because both cohorts were intentionally selected diagnostics.

## Cost and execution

- formal successful rerank calls: 23
- provider tokens: 1,101,402
- estimated successful-call cost at $0.10 / 1M input tokens: $0.110140
- latency p50: 2078.585 ms
- latency p95: 2674.447 ms
- request IDs present: 23/23
- unique request IDs: 23/23
- DEV opened: false

Before the formal run there was one HTTP 404 routing attempt against an
incorrect shared endpoint:

- HTTP attempts: 1
- successful model calls: 0
- request usage observed: false
- billing for that failed request was not observable

The endpoint correction was an infrastructure routing correction only. It did
not change the frozen query cohort, candidate construction, reranker model,
instruction, or evaluation criteria.

## Decision

Stop this R2 route.

Do not:

- tune RRF further on TechQA TRAIN;
- enlarge the full-union counterfactual to all 450 TRAIN queries;
- invent a post-hoc pass threshold;
- use DEV to rescue or retune this experiment.

Carry forward the narrower finding:

> Hybrid source complementarity is real, but simply exposing the entire
> Dense/BM25 union to the same reranker does not reliably recover
> answer-bearing evidence into early ranks.

The next project stage should move away from additional TechQA TRAIN retrieval
tuning and use the frozen retrieval evidence to support the next evaluation
layer.
