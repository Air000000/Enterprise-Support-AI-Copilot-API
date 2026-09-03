# Document-Local Evidence Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded document-grouping and document-local evidence-selection path that replaces fixed forward-only answer discovery while preserving document-level TechQA evaluation.

**Architecture:** Keep chunk retrieval as the first-stage locator. Collapse/group hits by `document_id`, collect a bounded chunk pool from candidate documents, then run one relevance-selection pass over that pool before final context assembly. Neighbor expansion, if retained, is limited to semantic continuity after evidence selection rather than answer discovery.

**Tech Stack:** Python 3.11, pytest, existing TechQA eval adapters, Chroma, existing Qwen3 reranker integration.

**Spec:** `docs/superpowers/specs/2026-09-03-document-local-evidence-retrieval-design.md`

## Global Constraints

- Do not tune strategy constants from Q055/Q572 geometry alone.
- Do not change formal TechQA qrels or invent gold-chunk labels.
- Do not redesign Dense retrieval, Hybrid retrieval, abstention, or generation prompts in this plan.
- Use zero-cost diagnostics before any provider-backed generation run.
- One rerank over the merged document-local evidence pool is preferred over one provider call per document.
- Preserve the existing Top3 baseline and `pilot_document_aware_v1` artifacts.

---

### Task 1: Add deterministic document grouping primitives

**Files:**
- Modify: `experiments/evals/eval_techqa_generation.py`
- Test: `tests/test_generation_eval_runner.py`

**Interfaces:**
- Consumes: ranked `G0RetrievedChunk` results carrying `chunk_id`, `document_id`, `chunk_index`, `content`, `distance`.
- Produces: a deterministic ordered list/tuple of candidate document IDs and the retained representative chunk(s) for each document.

- [ ] **Step 1: Write failing tests**

Add tests proving that document grouping:

```python
ranked = [
    chunk("A_3", "A", 3, 0.10),
    chunk("A_5", "A", 5, 0.12),
    chunk("B_2", "B", 2, 0.15),
    chunk("C_1", "C", 1, 0.20),
]

assert candidate_document_ids(ranked, limit=3) == ("A", "B", "C")
```

and that repeated chunks from document A do not consume multiple document slots.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_generation_eval_runner.py -k "candidate_document" -v
```

Expected: FAIL because the grouping primitive does not yet exist.

- [ ] **Step 3: Implement the minimal grouping primitive**

Implement one small deterministic helper in `eval_techqa_generation.py`. It must preserve ranking order by first document occurrence and enforce a document-count limit without changing retrieval scores.

- [ ] **Step 4: Verify GREEN**

Run the focused tests, then:

```bash
pytest tests/test_generation_eval_runner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_generation.py tests/test_generation_eval_runner.py
git commit -m "test: define document grouping for generation retrieval"
```

---

### Task 2: Build a bounded document-local chunk candidate pool

**Files:**
- Modify: `experiments/evals/eval_techqa_generation.py`
- Test: `tests/test_generation_eval_runner.py`

**Interfaces:**
- Consumes: candidate document IDs from Task 1 and a chunk loader keyed by `document_id`.
- Produces: a deterministic deduplicated sequence of chunks from those documents, capped by an explicit candidate-pool budget.

- [ ] **Step 1: Write failing tests**

Cover these cases:

```python
# Chunks can come from before or after the original anchor.
# Duplicate chunk IDs are emitted once.
# Candidate-document order is stable.
# Candidate-pool budget is enforced deterministically.
```

Include a case where the useful candidate is earlier than the original anchor to ensure the implementation has no forward-only assumption.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_generation_eval_runner.py -k "document_local" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the minimal pool builder**

Use an injected loader for unit tests. The production loader may read chunks from the existing TechQA Chroma collection by `document_id`; it must not call embedding, generation, or judge providers.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and the full generation runner test file.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_generation.py tests/test_generation_eval_runner.py
git commit -m "feat: build document-local evidence candidate pool"
```

---

### Task 3: Select evidence by query relevance, not chunk position

**Files:**
- Modify: `experiments/evals/eval_techqa_generation.py`
- Test: `tests/test_generation_eval_runner.py`

**Interfaces:**
- Consumes: original query and merged document-local chunk candidate pool from Task 2.
- Produces: bounded selected evidence chunks ordered by reranker relevance.

- [ ] **Step 1: Write failing tests with a fake reranker**

Construct a pool such as:

```python
A1 = chunk("A_1", "A", 1, 0.4)   # correct evidence, before anchor
A5 = chunk("A_5", "A", 5, 0.2)   # original anchor
A9 = chunk("A_9", "A", 9, 0.3)
B2 = chunk("B_2", "B", 2, 0.25)
```

Make the fake reranker rank `A1` highest and assert that `A1` is selected even though it precedes the anchor.

Also assert that the selector issues one reranker invocation over the merged candidate pool, not one call per document.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_generation_eval_runner.py -k "evidence_selection" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the minimal evidence selector**

Reuse the existing reranker adapter and preserve chunk identity/provenance. Do not add a new provider or scoring framework.

- [ ] **Step 4: Verify GREEN**

Run focused tests and the full generation runner tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_generation.py tests/test_generation_eval_runner.py
git commit -m "feat: select document-local evidence by query relevance"
```

---

### Task 4: Assemble bounded context without forward-only answer discovery

**Files:**
- Modify: `experiments/evals/eval_techqa_generation.py`
- Test: `tests/test_generation_eval_runner.py`

**Interfaces:**
- Consumes: selected evidence chunks from Task 3.
- Produces: final context chunks under the existing context budget.

- [ ] **Step 1: Write failing tests**

Prove that:

```python
# selected evidence itself is never dropped;
# final context is deduplicated;
# context budget is enforced;
# optional adjacency cannot introduce a direction-only invariant;
# adjacency is not needed to discover an evidence chunk already selected by relevance.
```

- [ ] **Step 2: Verify RED**

Run the focused assembly tests.

- [ ] **Step 3: Implement minimal bounded assembly**

Keep adjacency minimal and symmetric if used at all. The implementation must make the evidence-selection step, not chunk position, responsible for answer discovery.

- [ ] **Step 4: Verify GREEN**

Run focused tests and the full generation runner tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/evals/eval_techqa_generation.py tests/test_generation_eval_runner.py
git commit -m "feat: assemble bounded evidence context"
```

---

### Task 5: Add zero-cost TechQA diagnostics before paid generation

**Files:**
- Modify: `experiments/evals/eval_techqa_generation.py` or create a focused diagnostic module under `experiments/evals/` if the existing file becomes unwieldy.
- Test: matching test module.

**Interfaces:**
- Consumes: frozen TRAIN diagnostic cases and the new retrieval/context path.
- Produces: structured diagnostics for candidate documents, selected evidence chunk IDs, context chunk IDs, and provider-call boundaries.

- [ ] **Step 1: Write tests for diagnostic payloads**

The payload must expose enough provenance to answer:

```text
Which documents entered stage 2?
Which chunks were considered inside those documents?
Which chunks did the evidence selector retain?
Which chunks entered final context?
```

- [ ] **Step 2: Verify RED**

Run the focused diagnostic tests.

- [ ] **Step 3: Implement diagnostic output**

Do not call generation or judge providers. Make embedding/rerank usage explicit so a diagnostic can be classified as truly zero-cost only when those providers are also stubbed/cached/not invoked.

- [ ] **Step 4: Run targeted Q055/Q572/Q299/Q130 diagnostics**

Interpretation goals:

```text
Q055/Q572: can the architecture recover needed document-local evidence without a forward-only rule?
Q299: does the new evidence stage retain/recover the decisive V8 chunk when its candidate document is present?
Q130: remains a known upstream retrieval miss unless the candidate document actually enters stage 2.
```

Do not optimize constants from these cases.

- [ ] **Step 5: Run regression tests and lint**

```bash
pytest tests/test_generation_eval_runner.py -v
ruff check experiments/evals/eval_techqa_generation.py tests/test_generation_eval_runner.py
```

Then run the full suite if focused verification passes.

- [ ] **Step 6: Commit**

```bash
git add experiments/evals tests
git commit -m "test: add document-local evidence diagnostics"
```

---

### Task 6: Decide whether a paid pilot is justified

**Files:**
- Modify only experiment documentation/manifest definitions if the zero-cost evidence supports a paid run.

**Interfaces:**
- Consumes: zero-cost diagnostic evidence from Task 5.
- Produces: an explicit go/no-go decision and, only on GO, a frozen paid-pilot identity.

- [ ] **Step 1: Review mechanism evidence**

Require evidence that the new path changes the intended failure layer without reopening unrelated retrieval tuning.

- [ ] **Step 2: Pre-register paid evaluation gates**

The gates must include paired case movement and aggregate metrics. Do not use only one average score.

- [ ] **Step 3: Explicitly bound provider cost**

State maximum embedding, rerank, generation, and judge calls before execution.

- [ ] **Step 4: Run a paid pilot only after the above review passes**

Do not run DEV or full TRAIN automatically. Stop on provider failure and inspect checkpoints before recovery.

- [ ] **Step 5: Update `docs/evaluation/rag_experiment_lessons.md` only if the experiment changes an engineering decision**

Do not turn the lessons file into a chronological lab notebook.
