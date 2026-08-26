# R4 Chunk-BM25 Candidate Audit Design

Date: 2026-08-26
Branch: `feat/r4-chunk-bm25-audit`
Status: Approved in chat; implementation not started

## 1. Purpose

R3 proved that document-level BM25 provides lexical complementarity beyond Dense retrieval. R4 must now transition toward a production-aligned chunk-level retrieval pipeline without spending reranker budget before the chunk candidate pool is understood.

This audit answers one question only:

> When BM25 is moved from document-level retrieval to the same frozen chunk identity used by Dense retrieval, does candidate diversity or gold-document coverage degrade enough that a diversity control is required before paid Hybrid+Rerank?

This is a zero-provider-call preflight. It is not the paid R4 experiment and it does not tune reranker behavior.

## 2. Frozen architectural boundary

R4 is chunk-centered internally:

```text
Dense unit   = chunk
BM25 unit    = chunk
RRF unit     = chunk
Rerank unit  = chunk

TechQA evaluation adapter only:
chunk ranking -> first-occurrence document collapse -> document-level metrics
```

Document collapse exists only because TechQA qrels are document-level. Production retrieval remains chunk-level.

## 3. Frozen inputs

### Split

- TRAIN only
- 450 answerable TechQA retrieval queries
- DEV must not be opened, inspected, or used for tuning

### Corpus

Use the same frozen TechQA chunk identity as E0 Dense:

- 28,481 source documents
- 172,614 chunks
- same paragraph-aware character chunking
- `chunk_size=800`
- `overlap=120`
- `min_chunk_size=150`
- same deterministic `chunk_id`, `document_id`, `chunk_index`, and `content`

The audit must reuse or deterministically reconstruct the existing frozen chunk corpus and must verify the frozen corpus/chunk identity before running. It must not call an embedding provider.

## 4. Frozen BM25 configuration

Reuse the R3 lexical configuration unless this audit later proves a separate redesign is necessary:

- library: `bm25s==0.3.10`
- method: `lucene`
- `k1=1.5`
- `b=0.75`
- backend: `numpy`
- query normalization: `rstrip`
- tokenizer regex: `[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*`

The indexed unit changes from document to chunk. BM25 parameters do not change in this audit.

## 5. Index lifecycle assumption

Chunk-level BM25 indexing is an offline/startup concern, not a per-request operation.

The future production contract should be:

```text
startup / offline build -> build or load BM25 chunk index once
request -> tokenize query -> search existing index -> Top-K chunks
```

The audit may build the local chunk BM25 index once for the experiment. It must not rebuild the full index once per query.

## 6. Audit K values

Evaluate the raw chunk-BM25 ranking at:

- K=20
- K=50
- K=100

No RRF or rerank is applied in this stage.

## 7. Required quantitative outputs

For every K, compute at minimum:

### 7.1 Candidate diversity

- `unique_document_count@K` per query
- aggregate p50 / p95 or equivalent distribution summaries
- `duplicate_slot_count@K = returned_chunk_count@K - unique_document_count@K`
- `duplicate_ratio@K = 1 - unique_document_count@K / returned_chunk_count@K`

The denominator must use the number of actually returned chunks so queries with fewer than K results are not misreported.

### 7.2 Raw chunk-slot gold coverage

For each raw chunk cutoff K, measure how much of the document-level qrel set is represented by the first K chunk candidates:

`gold_document_recall_within_chunk_k = represented_relevant_document_count / relevant_document_count`

Aggregate this over TRAIN for K=20/50/100.

This is the primary metric for the specific diversity question because it preserves the finite chunk-slot budget. If repeated chunks from one document crowd out other documents, this metric can expose the loss directly.

### 7.3 Collapsed document-ranking comparability

Separately, collapse the raw chunk ranking by first occurrence of `document_id` and compute document-level ranking metrics for comparability with prior TechQA retrieval evidence.

At minimum report:

- Recall@20
- Recall@50
- Recall@100 where the available unique-document ranking supports the cutoff

These are evaluation-adapter metrics only. They must not replace the raw chunk-slot coverage metric in Section 7.2 and they do not change the chunk-level internal pipeline.

### 7.4 Latency

Measure BM25 query latency only:

- p50
- p95

Index build time may be reported separately but must not be mixed into per-query latency.

### 7.5 Diagnostic cases

Produce a deterministic compact case report containing representative high-duplication queries. At minimum capture:

- question_id
- question
- cutoff K
- returned chunk count
- unique document count
- duplicate ratio
- relevant document IDs
- raw chunk-slot gold coverage at K
- enough chunk/document identity to inspect the duplication pattern

Case selection must be deterministic, e.g. sort by duplicate ratio descending then `question_id` ascending. Do not use DEV or model judgment.

## 8. Interpretation rule

This audit does not pre-register a new arbitrary per-document chunk cap.

The audit decides whether the next R4 candidate-construction design should use:

A. raw chunk-BM25 candidates, or
B. a separately designed and pre-registered diversity rule.

A diversity rule is justified only if TRAIN evidence shows material slot concentration or coverage loss. The audit must not silently introduce `max_chunks_per_document`, score normalization, reranking, or any other candidate mutation.

## 9. Provider and cost contract

Hard requirements:

- provider calls = 0
- DashScope calls = 0
- reranker calls = 0
- embedding API calls = 0
- expected model API cost = CNY 0

Unexpected provider-client construction is a stop condition.

## 10. Prohibited actions

During this audit do not:

- open or analyze DEV artifacts
- call `qwen3-rerank`
- call chat/generation models
- rerun paid embedding generation
- change R3 BM25 parameters
- tune tokenizer rules from observed audit results
- run RRF
- add a per-document chunk cap
- modify official qrels

## 11. Artifact and reproducibility contract

The implementation must produce compact, auditable artifacts for TRAIN, including:

- run manifest containing frozen corpus/config identity
- aggregate metrics
- deterministic diagnostic cases
- provider-calls field fixed to zero
- relevant input hashes where existing frozen artifacts are reused

Large local index data need not be committed if its identity and reproducibility inputs are captured by the manifest.

## 12. TDD implementation boundary

Implementation follows strict RED -> GREEN -> verification.

The first RED should define behavior before implementation for:

1. chunk-level candidate identity
2. diversity metrics
3. raw chunk-slot gold coverage
4. first-occurrence document collapse for comparable audit metrics
5. frozen audit manifest / zero-provider contract
6. deterministic diagnostic-case ordering

No production implementation code may be written before the failing test is observed locally.

## 13. Exit condition

The audit is complete when:

- TRAIN 450 queries are evaluated at K=20/50/100
- diversity, raw chunk-slot gold coverage, collapsed document metrics, and latency outputs are produced
- deterministic diagnostic cases are available
- focused tests pass
- relevant/full test suite remains green
- Ruff passes
- provider calls remain zero

The resulting evidence is then used to freeze the actual R4 Hybrid candidate construction and paid rerank contract before any new reranker call.
