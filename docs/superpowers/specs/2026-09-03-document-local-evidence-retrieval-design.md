# Document-Local Evidence Retrieval Design

## Goal

Replace the current `forward +3` context heuristic with a more general separation between **retrieval**, **document grouping**, **evidence selection**, and **context assembly**.

The design keeps chunk-level retrieval for precise matching, but does not assume that the retrieved chunk itself is the complete answer evidence or that useful evidence lies in a fixed physical direction.

## Design

```text
Query
  -> chunk retrieval / oversampling
  -> document_id grouping
  -> candidate documents
  -> collect chunks from candidate documents
  -> query-relevance evidence selection
  -> bounded evidence chunks
  -> optional minimal local context expansion
  -> LLM
```

### 1. Retrieval unit: chunk

Small chunks remain the primary search unit because they provide precise lexical/semantic matching. Existing Dense / BM25 / rerank work remains useful at this layer.

### 2. Grouping unit: document

Chunk hits are projected to `document_id` so repeated chunks from one document do not consume the whole candidate budget. Candidate construction may oversample chunks before grouping. This follows mature grouped-search / parent-child retrieval patterns rather than assuming one chunk equals one document-level result.

### 3. Evidence selection: query relevance inside candidate documents

After candidate documents are identified, their chunks become a bounded evidence candidate pool. Evidence is selected by relevance to the original query, not by relative chunk position.

The first implementation should prefer one bounded rerank over the merged document-local candidate pool instead of one provider call per document.

### 4. Context assembly: preserve evidence, then continuity

Only after evidence chunks are selected may a small neighbor/parent expansion be used to restore local semantic continuity. Neighbor expansion is not responsible for discovering the answer.

### 5. Evaluation granularity remains separated

- **Formal TechQA retrieval evaluation:** document-level, because qrels identify relevant documents.
- **Internal diagnostics:** raw chunk coverage, duplicate pressure, unique-document depth, evidence/context behavior.
- **Generation evaluation:** evidence completeness, generation correctness, faithfulness, paired regressions.

Do not invent gold-chunk recall when the benchmark has no gold chunk annotation. A diagnostic such as `first_gold_document_chunk_rank` means the first retrieved chunk belonging to the gold document, not a verified gold evidence chunk.

## Why not freeze `forward +3`

The TRAIN pilot proved an important mechanism: Top-ranked chunks can hit the correct document while omitting complete evidence, and restoring document-local context can repair generation. Q055 and Q572 demonstrated this clearly.

However, `forward +3` encodes two case-shaped assumptions that are not yet justified as invariants: useful evidence is downstream of the anchor, and three chunks is the right window. The experiment is therefore retained as a successful diagnostic repair, not treated as the final production policy.

## Scope boundary

This stage does **not** redesign Dense retrieval, tune abstention, add RAPTOR, introduce full document embeddings, or run paid DEV generation. The next implementation should only establish the document-grouping and document-local evidence-selection path with deterministic tests and zero-cost retrieval diagnostics before any paid generation experiment.
