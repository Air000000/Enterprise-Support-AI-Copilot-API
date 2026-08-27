# R4 Chunk-BM25 Candidate Audit Design

Date: 2026-08-26
Branch: `feat/r4-chunk-bm25-audit`
Status: Revised after review; implementation not started

## 1. Purpose

R3 proved that document-level BM25 provides lexical complementarity beyond Dense retrieval. R4 must now transition toward a production-aligned chunk-level retrieval pipeline without spending reranker budget before the chunk candidate pool is understood.

This audit answers one question only:

> When BM25 is moved from document-level retrieval to the same frozen chunk identity used by Dense retrieval, does duplicate-chunk pressure materially reduce finite chunk-slot coverage enough that a diversity control is required before paid Hybrid+Rerank?

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

## 3. Frozen inputs and chunk identity

### Split

- TRAIN only
- 450 answerable TechQA retrieval queries
- DEV must not be opened, inspected, or used for tuning

### Corpus

Use the same deterministic TechQA chunk construction as E0 Dense:

- 28,481 source documents
- expected 172,614 chunks
- paragraph-aware character chunking
- `chunk_size=800`
- `overlap=120`
- `min_chunk_size=150`
- deterministic `chunk_id`, `document_id`, `chunk_index`, and `content`

Chunk identity is guarded by the already frozen dataset revision and source `corpus_sha256`, the chunk parameters above, the expected chunk count, and the exact `rag_runtime/text_splitter.py` Git blob identity used by E0/current main (`64026b4434f1eea46b95bfce9f667680a37a2103`).

This audit does **not** add a separate `chunk_corpus_sha256`. That would be redundant for this zero-cost candidate audit as long as the source corpus, splitter implementation, splitter parameters, and chunk count remain unchanged.

The audit must not call an embedding provider.

## 4. Frozen BM25 configuration

Reuse the R3 lexical configuration:

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

```text
startup / offline build -> build or load BM25 chunk index once
request -> tokenize query -> search existing index -> ranked chunks
```

The audit may build the local chunk BM25 index once for the experiment. It must not rebuild the full index once per query.

## 6. Search depth M and metric cutoffs K

Raw chunk-slot cutoffs and audit retrieval depth are separate concepts.

### Raw metric cutoffs

Evaluate finite chunk-slot behavior at:

- K=20
- K=50
- K=100

### Audit retrieval depth

BM25 must retrieve deeper than the largest raw cutoff so that a separate unique-document ranking can be formed. Use a deterministic, qrel-blind depth rule:

1. initial `M=500` chunks;
2. collapse the M raw chunks by first occurrence of `document_id`;
3. if fewer than 100 unique documents are available, increase M deterministically (`500 -> 1000 -> 2000 -> ...`) until at least 100 unique documents are available or the corpus is exhausted.

The expansion rule must not inspect qrels, gold ranks, recall, or any effectiveness metric. M is an audit-depth mechanism, not a tuned retrieval parameter.

This separation ensures that `Top-100 chunks` and `Top-100 unique documents` are treated as different candidate budgets.

## 7. Required quantitative outputs

For every K in 20/50/100, compute at minimum:

### 7.1 Candidate diversity

For the first K raw chunk candidates:

- `returned_chunk_count@K`
- `unique_document_count@K`
- `duplicate_slot_count@K = returned_chunk_count@K - unique_document_count@K`
- `duplicate_ratio@K = 1 - unique_document_count@K / returned_chunk_count@K`

Aggregate tail summaries must emphasize bad cases:

- `unique_document_count@K`: p05 and p50
- `duplicate_ratio@K`: p50 and p95

The denominator uses actually returned chunks so short result lists are not misreported.

### 7.2 Raw chunk-slot gold coverage

For each raw chunk cutoff K, report whether the relevant document is represented by any of the first K chunk candidates.

Primary report name:

- `gold_document_hit_within_chunk_k`

Current TechQA retrieval qrels contain exactly one relevant document per answerable query, so this binary Hit@K is numerically equivalent to document Recall@K on this benchmark. The implementation may keep a generalized multi-qrel calculation internally, but the report must state the single-gold equivalence explicitly.

### 7.3 Paired crowding attribution

For each query derive from the same deep BM25 chunk ranking:

- `first_gold_chunk_rank`: first 1-based raw chunk rank whose `document_id` is relevant;
- `first_gold_document_rank`: 1-based rank of that relevant document after first-occurrence document collapse;
- `crowding_gap = first_gold_chunk_rank - first_gold_document_rank` when both ranks exist;
- `crowding_rescue@K = true` when `first_gold_chunk_rank > K` but `first_gold_document_rank <= K`.

`crowding_rescue@K` is the direct paired indicator that a gold document would fit inside K unique-document slots but is pushed outside K raw chunk slots by repeated chunks ahead of it.

Aggregate at minimum:

- count/rate of `crowding_rescue@20`
- count/rate of `crowding_rescue@50`
- count/rate of `crowding_rescue@100`
- distribution summary of `crowding_gap` for queries where both ranks exist

A high duplicate ratio alone must not be interpreted as causal evidence of coverage loss.

### 7.4 Collapsed document-ranking comparability

Using the qrel-blind audit depth M, collapse the raw ranking by first occurrence of `document_id` and obtain at least 100 unique documents when possible. Then compute document-level ranking metrics for comparability with prior TechQA evidence:

- Document Recall@20
- Document Recall@50
- Document Recall@100

These cutoffs refer to unique-document ranks, not raw chunk slots. They are evaluation-adapter metrics only and must not replace Sections 7.2 or 7.3.

### 7.5 Latency

Measure BM25 query latency separately from index build time:

- p50
- p95

If adaptive M expansion causes more than one retrieve call for a query, the audit latency for that query must include all BM25 retrieval work required by the deterministic depth rule.

Index build time may be reported separately but must not be mixed into per-query latency.

### 7.6 Diagnostic cases

Produce deterministic compact diagnostic outputs in two groups:

1. `high_duplication_cases`: sort by duplicate ratio descending, then `question_id` ascending;
2. `crowding_rescue_cases`: cases with `crowding_rescue@K=true`, sorted deterministically by K, then crowding gap descending, then `question_id` ascending.

Capture at minimum:

- question_id
- question
- cutoff K
- returned chunk count
- unique document count
- duplicate ratio
- relevant document IDs
- `first_gold_chunk_rank`
- `first_gold_document_rank`
- `crowding_gap`
- `crowding_rescue@K`
- enough chunk/document identity to inspect the duplication pattern

Do not use DEV or model judgment for case selection.

## 8. Interpretation rule

This audit does not pre-register an arbitrary per-document chunk cap.

The audit decides whether the next R4 candidate-construction design should use:

A. raw chunk-BM25 candidates, or
B. a separately designed and pre-registered diversity rule.

A diversity rule is justified only if TRAIN evidence shows material finite-slot loss, especially through `crowding_rescue@K` and related coverage evidence. High duplication without paired crowding loss is not sufficient justification by itself.

The audit must not silently introduce `max_chunks_per_document`, score normalization, reranking, or any other candidate mutation.

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
- tune M using qrels or effectiveness results
- run RRF
- add a per-document chunk cap
- modify official qrels

## 11. Artifact and reproducibility contract

The implementation must produce compact TRAIN artifacts including:

- run manifest with frozen dataset/config identity;
- source `corpus_sha256` and dataset revision;
- splitter blob SHA and chunk parameters;
- expected/observed chunk count;
- search-depth rule and metric cutoffs;
- aggregate diversity, coverage, crowding, collapsed-document, and latency metrics;
- deterministic diagnostic cases;
- `provider_calls=0`.

Large local BM25 index data need not be committed.

## 12. TDD implementation boundary

Implementation follows strict RED -> GREEN -> verification.

The RED sequence must define behavior before implementation for:

1. deterministic chunk-level candidate identity;
2. diversity metrics;
3. separation of raw cutoff K from audit depth M;
4. `gold_document_hit_within_chunk_k`;
5. first gold chunk/document ranks and `crowding_rescue@K`;
6. first-occurrence document collapse and document-cutoff metrics;
7. frozen manifest / splitter identity / zero-provider contract;
8. deterministic diagnostic ordering.

No production implementation code may be written before the corresponding failing test is observed locally.

## 13. Exit condition

The audit is complete when:

- TRAIN 450 queries are evaluated;
- raw chunk behavior is reported at K=20/50/100;
- collapsed document metrics are reported at unique-document K=20/50/100;
- diversity, gold hit, paired crowding, and latency outputs are produced;
- deterministic diagnostic cases are available;
- focused tests pass;
- relevant/full test suite remains green;
- Ruff passes;
- provider calls remain zero.

The resulting evidence is then used to freeze the actual R4 Hybrid candidate construction and paid rerank contract before any new reranker call.
