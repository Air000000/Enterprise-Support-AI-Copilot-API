# R4 C1 Hybrid + Rerank Design

Date: 2026-08-27
Branch: `feat/r4-c1-hybrid-rerank`
Status: Approved

## 1. Goal

C1 tests one controlled hypothesis:

> Holding the reranker contract fixed, does replacing E1's Dense-only
> Top-100 chunk candidate pool with a Hybrid Dense + BM25 candidate pool
> improve the final TechQA TRAIN ranking?

C1 is the controlled stage of a two-stage strategy:

- C1: causal comparison against E1 with candidate construction as the main variable.
- C2: optional optimization only if C1 provides positive evidence.

No DEV data may be used in C1.

## 2. Frozen E1 baseline

TRAIN query count: 450.

E1 metrics:

- Recall@5: 0.691111
- Recall@20: 0.815556
- MRR@10: 0.567206

E1 reranker contract:

- model: `qwen3-rerank`
- candidate budget: 100 chunks
- instruction:
  `Rank the candidate passages by relevance to resolving the technical support query.`
- query normalization: `rstrip`
- final document ranking:
  first occurrence of `document_id` after chunk rerank

C1 must preserve this reranker contract.

## 3. Frozen candidate construction

For each TRAIN query:

```text
Frozen Dense Top-100 unique chunks --+
                                     +-- chunk-id RRF(k=60)
Chunk BM25 Top-100 unique chunks -----+
                    |
                    v
          fused Top-100 unique chunks
                    |
                    v
               qwen3-rerank
                    |
                    v
        first-occurrence doc collapse
```

Rules:

1. Dense contributes exactly its frozen Top-100 chunk ranking.
2. BM25 retrieves Top-100 chunks using the already frozen chunk-BM25
   configuration from the R4 audit.
3. Each source ranking must contain unique `chunk_id` values.
4. Cross-source overlap must NOT be removed before RRF.
5. RRF is equal-weight, `k=60`, aggregated by `chunk_id`.
6. A chunk present in both Dense and BM25 receives both rank contributions.
7. RRF output must contain exactly 100 unique chunks.
8. No per-document cap, adaptive BM25 depth, source weighting, query routing,
   or other candidate tuning is allowed in C1.

## 4. Zero-cost frozen candidate snapshot

Candidate construction is completed before any paid rerank call.

All 450 TRAIN queries are materialized into a frozen Hybrid candidate
snapshot.

Each query record must contain at least:

- `question_id`
- question required by the reranker
- `relevant_document_ids`
- Dense Top-100 chunk IDs
- BM25 Top-100 chunk IDs
- fused Top-100 candidates
- fused candidate `chunk_id`
- fused candidate `document_id`
- candidate content required by the reranker

The snapshot must be complete for all 450 queries before paid execution.

The snapshot is hashed with SHA256 and its identity is recorded in the
C1 manifest.

The paid runner consumes this frozen snapshot only. It must not rerun
Dense retrieval, BM25 retrieval, or RRF during paid execution or resume.

If the snapshot is missing or its SHA does not match the manifest,
paid execution must fail before contacting the provider.

A large candidate-heavy snapshot may remain a local experimental
artifact if its SHA256 and provenance are persisted in the compact
versioned manifest.

## 5. Paid rerank contract

Frozen paid configuration:

- split: TRAIN only
- queries: at most 450
- model: `qwen3-rerank`
- provider deployment: same Singapore deployment as E1
- instruction: exactly the E1 instruction
- rerank input: exactly 100 fused chunks per query
- SDK automatic retry: disabled with `max_retries=0`
- DEV: forbidden

The paid runner is resumable.

Completed query IDs already durably persisted in the checkpoint must
never call the provider again.

## 6. Paid-run safety

### Provider-token stop threshold

Normal stop threshold:

`13,500,000 provider-reported tokens`

Token usage is known only after a provider response is received.

Execution order:

```text
provider request
-> successful response
-> persist completed result/checkpoint
-> update cumulative provider tokens
-> if cumulative tokens >= 13.5M:
     stop before sending another request
```

The threshold may be exceeded by the final already-issued request.

### Monetary safety envelope

Approved monetary safety envelope:

`USD 1.50`

The token threshold is the normal application stop control.
The monetary envelope is the outer budget safety bound.

### Provider errors and retries

SDK automatic retries are disabled.

Any provider error stops the run immediately.

To avoid silently replaying a request whose provider-side outcome is
unknown, the runner maintains only the minimum durable attempt state:

- completed: durable checkpoint exists; automatically skip on resume
- never started: eligible to call provider
- inflight/uncertain: automatic retry is forbidden; stop for manual review

This is a paid-run safety mechanism, not a general workflow/state-machine
subsystem.

## 7. Evaluation and success gate

Primary metrics:

- Recall@5
- Recall@20
- MRR@10

Pre-registered C1 success gate:

```text
MRR@10 >= 0.5772063492063492
AND
Recall@20 >= 0.8111111111111111
```

This corresponds to:

- MRR@10 absolute improvement of at least +0.010 over E1
- Recall@20 regression no worse than -0.005 absolute

Recall@5 is reported but is not a hard success condition.

Paired diagnostic evidence is reported only if complete preserved E1
per-query evidence is available for all 450 TRAIN queries.

The repository does not preserve the complete historical E1 per-query
rerank artifact. C1 must therefore NOT rerun E1 solely to recreate paired
counts.

The authoritative C1 comparison is the frozen aggregate E1 baseline and
the pre-registered success gate above.

Any partial historical paired evidence may be reported only as
supplementary evidence and must not be presented as a full-population
450-query comparison.

## 8. Provenance boundary

Historical E1 references an E0 TRAIN results SHA that is no longer
recoverable byte-for-byte.

The current recoverable E0 TRAIN artifact has already been independently
verified against R3 for all 450 TRAIN queries at the query ID, gold
document, and collapsed dense document-ranking level.

C1 therefore records both the recoverable input identity and the
historical provenance note.

C1 must not claim that the historical E1 raw Top-100 chunk artifact has
been proven byte-identical when that evidence is unavailable.

This provenance limitation does not change the C1 execution contract.

## 9. Required artifacts

Zero-cost phase:

- frozen Hybrid candidate snapshot
- snapshot manifest with SHA256
- preflight summary

Paid phase:

- resumable checkpoint/results
- run manifest
- metrics
- aggregate comparison against the frozen E1 baseline; paired comparison only if complete preserved E1 per-query evidence exists
- provider token usage
- latency summary
- final gate decision
- cost-ledger update

## 10. Testing scope

C1 adds only three behavior-level contract scenarios.

### Contract 1 - candidate construction

Validate one complete Hybrid construction behavior:

```text
Dense + BM25
-> source uniqueness
-> RRF(k=60)
-> shared chunks receive both contributions
-> exactly 100 unique fused chunks
```

### Contract 2 - one-query orchestration

Validate:

```text
frozen fused Top-100
-> fake qwen3-rerank result
-> complete reranked permutation
-> document collapse
-> result/token/request identity
```

### Contract 3 - paid-run safety

Validate the paid execution boundary as one scenario family:

- completed checkpoint queries are skipped
- manifest/snapshot identity mismatch fails before provider use
- candidate count != 100 fails before provider use
- token stop threshold prevents the next request
- provider error stops execution
- inflight/uncertain query is not automatically replayed
- DEV cannot enter the C1 paid path
- rerank client disables SDK retries

Existing lower-level RRF, BM25, reranker parsing, tie-breaking, tokenizer,
and checkpoint tests are reused and are not duplicated.

## 11. Acceptance sequence

Before any real provider call:

```text
3 C1 contract tests
-> relevant existing retrieval/rerank tests
-> full pytest
-> Ruff
-> zero-provider-call build of all 450 fused candidate records
-> freeze snapshot + SHA + manifest
-> verify preflight contract
-> explicit human approval
-> paid TRAIN run
```

No paid provider call is allowed during implementation or zero-cost
acceptance.

## 12. Non-goals

C1 will not:

- tune BM25 depth
- add adaptive retrieval
- add per-document chunk caps
- tune RRF k
- weight Dense/BM25 differently
- change reranker model/instruction
- inspect DEV
- optimize generation
- add a general experiment workflow framework
- add tests beyond the minimum contract coverage unless a real bug requires
  a regression test

Any such optimization belongs to C2 and is considered only after C1
results are known.

