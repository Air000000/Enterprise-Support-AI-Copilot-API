# R4 Chunk-BM25 Candidate Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a zero-provider-call TRAIN-only chunk-level BM25 audit that measures duplicate-slot pressure, gold coverage, paired crowding loss, document-level comparability, and local query latency before any paid R4 rerank experiment.

**Architecture:** Reconstruct the same deterministic TechQA chunk universe used by E0 with the unchanged splitter and frozen `800/120/150` parameters, then build one in-memory `bm25s` index over those chunks. The evaluator consumes only the frozen E0 TRAIN result artifact for questions/qrels, searches chunk BM25 with a qrel-blind adaptive depth rule, computes raw chunk-slot and collapsed-document metrics separately, and writes compact reproducible audit artifacts. No Dense query, embedding call, reranker call, RRF, DEV read, or provider client is part of this plan.

**Tech Stack:** Python 3.11, pytest 9.0.3, Ruff 0.15.21, datasets 5.0.1, bm25s 0.3.10, ranx 0.3.21 only where useful for document-level metric compatibility.

**Spec:** `docs/superpowers/specs/2026-08-26-r4-chunk-bm25-candidate-audit-design.md`

## Global Constraints

- Work on branch `feat/r4-chunk-bm25-audit` in an isolated worktree at execution time.
- Strict TDD for every behavior: RED -> user confirms expected failure -> minimal GREEN -> focused pytest -> Ruff -> next RED.
- TRAIN only: exactly 450 answerable TechQA retrieval queries.
- Use `experiments/evals/reports/e0_dense/train_results.jsonl` as the authoritative query/qrel input so this audit does not load or inspect DEV query rows.
- Validate the E0 TRAIN input SHA against the already-versioned R3 manifest field `dense_source.results_sha256` before the full audit run.
- Source corpus identity remains the frozen TechQA retrieval dataset revision and `corpus_sha256` in `experiments/evals/datasets/techqa/manifest.json`.
- Chunk construction remains `paragraph_aware_character`, `chunk_size=800`, `overlap=120`, `min_chunk_size=150`, expected chunk count `172614`.
- The exact splitter Git blob identity is `64026b4434f1eea46b95bfce9f667680a37a2103`.
- Do **not** add `chunk_corpus_sha256` in this audit.
- BM25 remains `bm25s==0.3.10`, `method="lucene"`, `k1=1.5`, `b=0.75`, `backend="numpy"`, with the existing technical tokenizer regex `[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*`.
- Raw chunk cutoffs are exactly `20, 50, 100`.
- Audit retrieval depth starts at `M=500`; if fewer than 100 unique documents are available, expand qrel-blind as `500 -> 1000 -> 2000 -> ...` until at least 100 unique documents are present or the corpus is exhausted.
- `provider_calls=0`, `DashScope calls=0`, `reranker calls=0`, `embedding API calls=0`.
- Do not run RRF, add a per-document chunk cap, change tokenizer/BM25 parameters, modify qrels, or inspect DEV artifacts.
- Do not use destructive git commands or broad staging such as `git add .`.

## File Structure

- Create `experiments/evals/retrievers/bm25_techqa_chunks.py` — deterministic TechQA chunk reconstruction plus one-time chunk BM25 index/search.
- Create `tests/test_bm25_techqa_chunks.py` — chunk identity and retriever behavior.
- Create `experiments/evals/eval_techqa_chunk_bm25.py` — TRAIN input validation, adaptive depth search, per-query audit observations, aggregation, manifest, diagnostics, artifact writer, CLI.
- Create `tests/test_eval_techqa_chunk_bm25.py` — pure audit metrics, crowding attribution, adaptive depth, frozen identity, ordering, artifact contract.
- Produce `experiments/evals/reports/r4_chunk_bm25_audit/train_manifest.json`.
- Produce `experiments/evals/reports/r4_chunk_bm25_audit/train_metrics.json`.
- Produce `experiments/evals/reports/r4_chunk_bm25_audit/train_results.jsonl`.
- Produce `experiments/evals/reports/r4_chunk_bm25_audit/diagnostic_cases.json`.

---

### Task 1: Deterministic chunk corpus and chunk BM25 retriever

**Files:**
- Create: `experiments/evals/retrievers/bm25_techqa_chunks.py`
- Create: `tests/test_bm25_techqa_chunks.py`

**Interfaces:**
- Consumes: `TechQADocument`, `split_text`, `build_chunk_id`, frozen default splitter constants, and existing `tokenize_technical`.
- Produces:
  - `TechQAChunk(chunk_id: str, document_id: str, chunk_index: int, content: str)`
  - `build_techqa_chunks(documents: Sequence[TechQADocument]) -> list[TechQAChunk]`
  - `TechQAChunkBM25Retriever(chunks: Sequence[TechQAChunk])`
  - `TechQAChunkBM25Retriever.search(query: str, top_k: int) -> list[TechQAChunk]`

- [ ] **Step 1: Write the first RED for deterministic chunk identity**

Create `tests/test_bm25_techqa_chunks.py` with:

```python
from importlib import import_module

import pytest

from experiments.evals.adapters.techqa import TechQADocument


def _load_module():
    try:
        return import_module("experiments.evals.retrievers.bm25_techqa_chunks")
    except ModuleNotFoundError:
        pytest.fail("chunk BM25 retriever is not implemented yet")


def test_build_techqa_chunks_is_deterministic_across_document_order() -> None:
    module = _load_module()
    documents = [
        TechQADocument("b", "beta paragraph"),
        TechQADocument("a", "alpha paragraph"),
    ]

    forward = module.build_techqa_chunks(documents)
    reversed_input = module.build_techqa_chunks(list(reversed(documents)))

    assert forward == reversed_input
    assert [(chunk.document_id, chunk.chunk_index, chunk.chunk_id) for chunk in forward] == [
        ("a", 0, "a_chunk_0"),
        ("b", 0, "b_chunk_0"),
    ]
```

- [ ] **Step 2: Run RED and stop**

Run from the feature worktree:

```powershell
pytest tests/test_bm25_techqa_chunks.py::test_build_techqa_chunks_is_deterministic_across_document_order -v
```

Expected: FAIL because `experiments.evals.retrievers.bm25_techqa_chunks` does not exist.

Do not write implementation until this exact missing-feature failure is observed.

- [ ] **Step 3: Implement the minimal chunk model/builder**

Create `experiments/evals/retrievers/bm25_techqa_chunks.py` beginning with:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import bm25s

from experiments.evals.adapters.techqa import TechQADocument
from experiments.evals.retrievers.bm25_techqa import tokenize_technical
from rag_runtime.text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    build_chunk_id,
    split_text,
)


@dataclass(frozen=True)
class TechQAChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str


def build_techqa_chunks(documents: Sequence[TechQADocument]) -> list[TechQAChunk]:
    chunks: list[TechQAChunk] = []
    for document in sorted(documents, key=lambda item: item.document_id):
        contents = split_text(
            document.text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
            min_chunk_size=MIN_CHUNK_SIZE,
        )
        for chunk_index, content in enumerate(contents):
            chunks.append(
                TechQAChunk(
                    chunk_id=build_chunk_id(document.document_id, chunk_index),
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=content,
                )
            )
    return chunks
```

- [ ] **Step 4: Run GREEN for the builder**

```powershell
pytest tests/test_bm25_techqa_chunks.py::test_build_techqa_chunks_is_deterministic_across_document_order -v
```

Expected: PASS.

- [ ] **Step 5: Add the RED for chunk-level BM25 search**

Append:

```python
def test_chunk_bm25_returns_specific_chunk_identity() -> None:
    module = _load_module()
    chunks = [
        module.TechQAChunk("a_chunk_0", "a", 0, "printer configuration"),
        module.TechQAChunk("b_chunk_0", "b", 0, "permission denied 0x80070005"),
        module.TechQAChunk("b_chunk_1", "b", 1, "unrelated follow up"),
    ]

    retriever = module.TechQAChunkBM25Retriever(chunks)
    results = retriever.search("0x80070005", top_k=3)

    assert results[0].chunk_id == "b_chunk_0"
    assert results[0].document_id == "b"
    assert results[0].chunk_index == 0


def test_chunk_bm25_handles_nonpositive_k_and_empty_tokens() -> None:
    module = _load_module()
    retriever = module.TechQAChunkBM25Retriever(
        [module.TechQAChunk("a_chunk_0", "a", 0, "permission denied")]
    )

    assert retriever.search("permission", top_k=0) == []
    assert retriever.search("--- /// ...", top_k=10) == []
```

- [ ] **Step 6: Run RED for missing retriever**

```powershell
pytest tests/test_bm25_techqa_chunks.py -v
```

Expected: builder test PASS; retriever tests FAIL because `TechQAChunkBM25Retriever` is absent.

- [ ] **Step 7: Implement one-time index construction and search**

Add:

```python
class TechQAChunkBM25Retriever:
    def __init__(self, chunks: Sequence[TechQAChunk]) -> None:
        ordered = sorted(chunks, key=lambda item: (item.document_id, item.chunk_index, item.chunk_id))
        self._chunks = ordered
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in ordered}
        self._chunk_ids = [chunk.chunk_id for chunk in ordered]
        corpus_tokens = [tokenize_technical(chunk.content) for chunk in ordered]
        self._retriever = bm25s.BM25(
            k1=1.5,
            b=0.75,
            method="lucene",
            corpus=self._chunk_ids,
            backend="numpy",
        )
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 100) -> list[TechQAChunk]:
        if top_k <= 0 or not self._chunk_ids:
            return []
        query_tokens = tokenize_technical(query)
        if not query_tokens:
            return []
        limit = min(top_k, len(self._chunk_ids))
        chunk_ids = self._retriever.retrieve(
            [query_tokens],
            k=limit,
            return_as="documents",
            show_progress=False,
            backend_selection="numpy",
        )[0]
        return [self._chunks_by_id[str(chunk_id)] for chunk_id in chunk_ids]
```

The index is built only in `__init__`; `search()` performs retrieval against that existing index.

- [ ] **Step 8: Verify Task 1 and commit**

```powershell
pytest tests/test_bm25_techqa_chunks.py -v
ruff check experiments/evals/retrievers/bm25_techqa_chunks.py tests/test_bm25_techqa_chunks.py
```

Then stage exact files only:

```powershell
git add experiments/evals/retrievers/bm25_techqa_chunks.py tests/test_bm25_techqa_chunks.py
git commit -m "feat(eval): add chunk-level TechQA BM25 retriever"
```

---

### Task 2: Pure diversity, gold-rank, and crowding metrics

**Files:**
- Create: `experiments/evals/eval_techqa_chunk_bm25.py`
- Create: `tests/test_eval_techqa_chunk_bm25.py`

**Interfaces:**
- Produces:
  - `collapse_document_ids(document_ids: Sequence[str]) -> list[str]`
  - `first_relevant_rank(document_ids: Sequence[str], relevant_document_ids: Sequence[str]) -> int | None`
  - `ChunkCutoffObservation`
  - `build_cutoff_observation(raw_document_ids, relevant_document_ids, cutoff) -> ChunkCutoffObservation`

- [ ] **Step 1: RED for first-occurrence collapse and diversity**

Create `tests/test_eval_techqa_chunk_bm25.py`:

```python
from importlib import import_module

import pytest


def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_chunk_bm25")
    except ModuleNotFoundError:
        pytest.fail("chunk BM25 audit evaluator is not implemented yet")


def test_cutoff_observation_measures_duplicate_slot_pressure() -> None:
    module = _load_module()
    observation = module.build_cutoff_observation(
        raw_document_ids=["a", "a", "a", "b", "c", "gold"],
        relevant_document_ids=["gold"],
        cutoff=5,
    )

    assert observation.returned_chunk_count == 5
    assert observation.unique_document_count == 3
    assert observation.duplicate_slot_count == 2
    assert observation.duplicate_ratio == pytest.approx(0.4)
    assert observation.gold_document_hit_within_chunk_k is False
```

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py::test_cutoff_observation_measures_duplicate_slot_pressure -v
```

Expected: FAIL because the evaluator module is absent.

- [ ] **Step 3: Minimal metric implementation**

Create `experiments/evals/eval_techqa_chunk_bm25.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkCutoffObservation:
    cutoff: int
    returned_chunk_count: int
    unique_document_count: int
    duplicate_slot_count: int
    duplicate_ratio: float
    gold_document_hit_within_chunk_k: bool
    crowding_rescue: bool


def collapse_document_ids(document_ids: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    seen: set[str] = set()
    for document_id in document_ids:
        value = str(document_id)
        if value in seen:
            continue
        seen.add(value)
        collapsed.append(value)
    return collapsed


def first_relevant_rank(
    document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
) -> int | None:
    relevant = {str(value) for value in relevant_document_ids}
    for rank, document_id in enumerate(document_ids, start=1):
        if str(document_id) in relevant:
            return rank
    return None


def build_cutoff_observation(
    raw_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
    cutoff: int,
) -> ChunkCutoffObservation:
    if cutoff <= 0:
        raise ValueError("cutoff must be greater than 0")
    raw_prefix = [str(value) for value in raw_document_ids[:cutoff]]
    returned = len(raw_prefix)
    unique_count = len(set(raw_prefix))
    duplicate_count = returned - unique_count
    duplicate_ratio = (duplicate_count / returned) if returned else 0.0
    first_chunk_rank = first_relevant_rank(raw_document_ids, relevant_document_ids)
    collapsed = collapse_document_ids(raw_document_ids)
    first_document_rank = first_relevant_rank(collapsed, relevant_document_ids)
    gold_hit = first_chunk_rank is not None and first_chunk_rank <= cutoff
    crowding_rescue = (
        first_chunk_rank is not None
        and first_document_rank is not None
        and first_chunk_rank > cutoff
        and first_document_rank <= cutoff
    )
    return ChunkCutoffObservation(
        cutoff=cutoff,
        returned_chunk_count=returned,
        unique_document_count=unique_count,
        duplicate_slot_count=duplicate_count,
        duplicate_ratio=duplicate_ratio,
        gold_document_hit_within_chunk_k=gold_hit,
        crowding_rescue=crowding_rescue,
    )
```

- [ ] **Step 4: Run GREEN**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py::test_cutoff_observation_measures_duplicate_slot_pressure -v
```

- [ ] **Step 5: RED for paired crowding attribution**

Append:

```python
def test_paired_ranks_identify_crowding_rescue() -> None:
    module = _load_module()
    raw = ["a", "a", "a", "b", "a", "c", "d", "e", "f", "gold"]
    collapsed = module.collapse_document_ids(raw)

    assert module.first_relevant_rank(raw, ["gold"]) == 10
    assert module.first_relevant_rank(collapsed, ["gold"]) == 7

    observation = module.build_cutoff_observation(raw, ["gold"], cutoff=8)
    assert observation.gold_document_hit_within_chunk_k is False
    assert observation.crowding_rescue is True
```

- [ ] **Step 6: Verify Task 2 and commit**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py -v
ruff check experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
```

```powershell
git add experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
git commit -m "feat(eval): add chunk BM25 crowding metrics"
```

---

### Task 3: Qrel-blind adaptive search depth and per-query audit result

**Files:**
- Modify: `experiments/evals/eval_techqa_chunk_bm25.py`
- Modify: `tests/test_eval_techqa_chunk_bm25.py`

**Interfaces:**
- Produces:
  - `AdaptiveSearchResult`
  - `retrieve_with_unique_document_depth(query, searcher, clock, initial_depth=500, required_unique_documents=100, max_depth=172614)`
  - `TechQAChunkBM25AuditResult`
  - `evaluate_audit_case(case, searcher, clock, max_depth) -> TechQAChunkBM25AuditResult`

- [ ] **Step 1: RED proving M and K are independent**

Append a fake-candidate helper and test:

```python
from experiments.evals.adapters.techqa import TechQARetrievalCase


def test_adaptive_depth_expands_without_qrels_until_100_unique_documents() -> None:
    module = _load_module()
    calls: list[int] = []

    def searcher(query: str, top_k: int):
        calls.append(top_k)
        if top_k == 500:
            return [
                module.TechQAChunkRef(f"a_chunk_{index}", "a", index)
                for index in range(500)
            ]
        return [
            module.TechQAChunkRef(f"doc{index}_chunk_0", f"doc{index}", 0)
            for index in range(100)
        ]

    clock_values = iter([1.000, 1.010, 2.000, 2.030])
    result = module.retrieve_with_unique_document_depth(
        "query",
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=2000,
    )

    assert calls == [500, 1000]
    assert result.final_search_depth == 1000
    assert len(module.collapse_document_ids([item.document_id for item in result.candidates])) == 100
    assert result.latency_ms == pytest.approx(40.0)
```

The production helper deliberately has no qrels/relevant-document argument, making the expansion rule structurally qrel-blind.

- [ ] **Step 2: Run RED**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py::test_adaptive_depth_expands_without_qrels_until_100_unique_documents -v
```

Expected: FAIL because the adaptive-search interfaces do not exist.

- [ ] **Step 3: Implement adaptive depth minimally**

Add a lightweight reference protocol/dataclass used by the evaluator:

```python
from collections.abc import Callable
import time


@dataclass(frozen=True)
class TechQAChunkRef:
    chunk_id: str
    document_id: str
    chunk_index: int


@dataclass(frozen=True)
class AdaptiveSearchResult:
    candidates: tuple[object, ...]
    final_search_depth: int
    latency_ms: float


def retrieve_with_unique_document_depth(
    query: str,
    *,
    searcher: Callable[..., list[object]],
    clock: Callable[[], float] = time.perf_counter,
    initial_depth: int = 500,
    required_unique_documents: int = 100,
    max_depth: int = 172614,
) -> AdaptiveSearchResult:
    if initial_depth <= 0 or required_unique_documents <= 0 or max_depth <= 0:
        raise ValueError("audit depth values must be greater than 0")

    depth = min(initial_depth, max_depth)
    total_latency_ms = 0.0
    latest: list[object] = []

    while True:
        started = clock()
        latest = searcher(query, top_k=depth)
        total_latency_ms += (clock() - started) * 1000.0
        unique_count = len(
            collapse_document_ids([str(item.document_id) for item in latest])
        )
        if unique_count >= required_unique_documents:
            break
        if depth >= max_depth or len(latest) < depth:
            break
        depth = min(depth * 2, max_depth)

    return AdaptiveSearchResult(
        candidates=tuple(latest),
        final_search_depth=depth,
        latency_ms=total_latency_ms,
    )
```

- [ ] **Step 4: GREEN for adaptive depth**

Run the focused test above.

- [ ] **Step 5: RED for complete per-query audit semantics**

Append:

```python
def test_evaluate_audit_case_keeps_raw_chunk_and_unique_document_budgets_separate() -> None:
    module = _load_module()
    case = TechQARetrievalCase(
        question_id="TRAIN_Q1",
        question="Error 0x80070005   \n",
        relevant_document_ids=("gold",),
        split="train",
    )

    candidates = [
        module.TechQAChunkRef("a0", "a", 0),
        module.TechQAChunkRef("a1", "a", 1),
        module.TechQAChunkRef("b0", "b", 0),
        module.TechQAChunkRef("gold0", "gold", 0),
    ] + [
        module.TechQAChunkRef(f"d{index}", f"d{index}", 0)
        for index in range(100)
    ]

    def searcher(query: str, top_k: int):
        assert query == "Error 0x80070005"
        return candidates[:top_k]

    clock_values = iter([1.0, 1.01])
    result = module.evaluate_audit_case(
        case,
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=len(candidates),
    )

    assert result.first_gold_chunk_rank == 4
    assert result.first_gold_document_rank == 3
    assert result.crowding_gap == 1
    assert result.document_top100[:3] == ("a", "b", "gold")
    assert result.cutoff_observations[0].cutoff == 20
```

- [ ] **Step 6: Implement result dataclass/evaluator**

Use exactly one gold document as a frozen TechQA precondition:

```python
RAW_CUTOFFS = (20, 50, 100)


@dataclass(frozen=True)
class TechQAChunkBM25AuditResult:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    audit_search_depth: int
    latency_ms: float
    raw_top100_chunk_ids: tuple[str, ...]
    raw_top100_document_ids: tuple[str, ...]
    document_top100: tuple[str, ...]
    first_gold_chunk_rank: int | None
    first_gold_document_rank: int | None
    crowding_gap: int | None
    cutoff_observations: tuple[ChunkCutoffObservation, ...]
```

`evaluate_audit_case()` must:
1. reject `case.split != "train"`;
2. reject any case where `len(relevant_document_ids) != 1`;
3. call adaptive search with `case.question.rstrip()`;
4. derive the raw and collapsed first-gold ranks from the deepest returned ranking;
5. store only raw Top-100 identities plus collapsed document Top-100 in the result artifact;
6. compute observations for 20/50/100 from the full raw ranking;
7. compute `crowding_gap` only when both ranks exist.

- [ ] **Step 7: Verify Task 3 and commit**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py -v
ruff check experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
```

```powershell
git add experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
git commit -m "feat(eval): add adaptive chunk BM25 audit rows"
```

---

### Task 4: Aggregate metrics and deterministic diagnostics

**Files:**
- Modify: `experiments/evals/eval_techqa_chunk_bm25.py`
- Modify: `tests/test_eval_techqa_chunk_bm25.py`

**Interfaces:**
- Produces:
  - `percentile(values, p) -> float`
  - `build_audit_summary(results) -> dict[str, object]`
  - `build_diagnostic_cases(results, limit_per_group=10) -> dict[str, list[dict[str, object]]]`

- [ ] **Step 1: RED for aggregate tail metrics and document recall**

Append tests that construct two `TechQAChunkBM25AuditResult` rows and assert:

```python
summary = module.build_audit_summary(results)

assert summary["query_count"] == 2
assert summary["cutoffs"]["20"]["unique_document_count_p05"] <= summary["cutoffs"]["20"]["unique_document_count_p50"]
assert summary["cutoffs"]["20"]["duplicate_ratio_p95"] >= summary["cutoffs"]["20"]["duplicate_ratio_p50"]
assert summary["cutoffs"]["20"]["gold_document_hit_rate"] == pytest.approx(0.5)
assert summary["cutoffs"]["20"]["crowding_rescue_count"] == 1
assert summary["collapsed_document_recall"]["recall@20"] == pytest.approx(1.0)
assert set(summary["collapsed_document_recall"]) == {"recall@20", "recall@50", "recall@100"}
```

Use document recall as the mean single-gold hit rate at unique-document cutoffs, which is numerically equivalent to Recall@K for the frozen TechQA qrels contract.

- [ ] **Step 2: RED for deterministic diagnostic ordering**

Append tests asserting:

```python
diagnostics = module.build_diagnostic_cases(results, limit_per_group=10)
assert list(diagnostics) == ["high_duplication_cases", "crowding_rescue_cases"]
assert diagnostics["high_duplication_cases"][0]["question_id"] == "TRAIN_Q2"
assert diagnostics["crowding_rescue_cases"][0]["crowding_rescue"] is True
```

Construct ties so ordering is forced by the spec: duplicate ratio descending then `question_id`; rescue cases by cutoff ascending, crowding gap descending, then `question_id`.

- [ ] **Step 3: Implement aggregation**

`build_audit_summary()` must report:

```text
query_count
cutoffs:
  20/50/100:
    unique_document_count_p05
    unique_document_count_p50
    duplicate_ratio_p50
    duplicate_ratio_p95
    gold_document_hit_count
    gold_document_hit_rate
    crowding_rescue_count
    crowding_rescue_rate
collapsed_document_recall:
  recall@20
  recall@50
  recall@100
crowding_gap:
  observed_count
  p50
  p95
latency_ms:
  p50
  p95
```

For `collapsed_document_recall`, test the gold document against each result's `document_top100[:K]`. Do not use raw chunk cutoffs for these three values.

- [ ] **Step 4: Implement deterministic diagnostic outputs**

Each diagnostic record must include:

```text
question_id
question
cutoff
returned_chunk_count
unique_document_count
duplicate_ratio
relevant_document_ids
first_gold_chunk_rank
first_gold_document_rank
crowding_gap
crowding_rescue
raw_top100_chunk_ids
raw_top100_document_ids
```

For high-duplication cases, emit one row per query using the K=100 observation. For rescue cases, emit every `(query, K)` where `crowding_rescue=True`, then deterministic sort and truncate to `limit_per_group`.

- [ ] **Step 5: Verify Task 4 and commit**

```powershell
pytest tests/test_eval_techqa_chunk_bm25.py -v
ruff check experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
```

```powershell
git add experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py
git commit -m "feat(eval): summarize chunk BM25 crowding evidence"
```

---

### Task 5: Frozen TRAIN preflight, manifest, runner, and artifacts

**Files:**
- Modify: `experiments/evals/eval_techqa_chunk_bm25.py`
- Modify: `tests/test_eval_techqa_chunk_bm25.py`
- Produce: `experiments/evals/reports/r4_chunk_bm25_audit/train_manifest.json`
- Produce: `experiments/evals/reports/r4_chunk_bm25_audit/train_metrics.json`
- Produce: `experiments/evals/reports/r4_chunk_bm25_audit/train_results.jsonl`
- Produce: `experiments/evals/reports/r4_chunk_bm25_audit/diagnostic_cases.json`

**Interfaces:**
- Consumes:
  - `experiments/evals/reports/e0_dense/train_results.jsonl`
  - `experiments/evals/reports/r3_hybrid/train_manifest.json`
  - `experiments/evals/datasets/techqa/manifest.json`
  - frozen TechQA corpus revision
- Produces:
  - `load_frozen_train_cases_from_e0(...) -> list[TechQARetrievalCase]`
  - `run_preflight(...) -> dict[str, object]`
  - `build_run_manifest(...) -> dict[str, object]`
  - `write_audit_artifacts(...) -> None`
  - CLI entry point for the full zero-cost TRAIN audit.

- [ ] **Step 1: RED for TRAIN-only authoritative input**

Append:

```python
def test_load_frozen_train_cases_rejects_dev_and_requires_450(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "train_results.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"question_id":"TRAIN_Q0","question":"q","relevant_document_ids":["g"]}',
                '{"question_id":"DEV_Q0","question":"dev","relevant_document_ids":["g"]}',
            ]
        ) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="TRAIN-only input"):
        module.load_frozen_train_cases_from_e0(path, expected_count=None)
```

A second test must create 449 valid TRAIN rows and expect `TRAIN query count mismatch` when `expected_count=450`.

- [ ] **Step 2: Implement the local JSONL TRAIN loader**

It must parse only these fields from each E0 row:

```text
question_id
question
relevant_document_ids
```

and construct `TechQARetrievalCase(..., split="train")`. It must fail if any QID does not start with `TRAIN_`, if a row has not exactly one gold document, if QIDs duplicate, or if the final count differs from 450.

This intentionally avoids loading the 610-query Hugging Face query/qrel table and therefore avoids reading DEV query rows during the audit.

- [ ] **Step 3: RED for frozen preflight identity**

Create temporary E0/R3/dataset manifests and test that `run_preflight()` rejects each mismatch independently:

```text
E0 TRAIN SHA does not equal R3 dense_source.results_sha256
bm25s version != 0.3.10
splitter blob != 64026b4434f1eea46b95bfce9f667680a37a2103
observed chunk count != 172614
TRAIN query count != 450
```

The passing report must contain:

```python
{
    "split": "train",
    "query_count": 450,
    "chunk_count": 172614,
    "splitter_blob_sha": "64026b4434f1eea46b95bfce9f667680a37a2103",
    "bm25_version": "0.3.10",
    "provider_calls": 0,
    "dev_artifact_opened": False,
}
```

plus the actual frozen source dataset revision/hash and E0 TRAIN result SHA.

- [ ] **Step 4: Implement fail-closed preflight**

Use `hashlib.sha256(Path(...).read_bytes()).hexdigest()` only for the already-versioned E0 TRAIN artifact identity. Do **not** compute a separate chunk corpus hash.

Resolve the splitter Git blob with:

```python
subprocess.run(
    ["git", "hash-object", "rag_runtime/text_splitter.py"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
```

Read `bm25s` with `importlib.metadata.version("bm25s")`.

- [ ] **Step 5: RED for manifest contract**

Assert `build_run_manifest()` freezes:

```text
benchmark = TechQA-RAG-Eval
run = r4_chunk_bm25_candidate_audit
split = train
query_count = 450
provider_calls = 0
chunking = strategy + 800/120/150 + splitter blob + observed chunk count
bm25 = library/version/method/k1/b/backend/tokenizer/query_normalization
raw_chunk_cutoffs = [20, 50, 100]
audit_depth = initial 500, required unique docs 100, growth factor 2, max chunk count
retrieval_dataset = repo/revision/corpus_sha256/queries_sha256/qrels_sha256
input_e0_train_results_sha256
```

No `chunk_corpus_sha256`, rerank model, RRF parameter, or DEV artifact may appear.

- [ ] **Step 6: Implement corpus load and full runner**

Load the frozen corpus directly from the dataset manifest without importing `experiments.evals.build_techqa_index` or embedding code:

```python
from datasets import load_dataset
from experiments.evals.adapters.techqa import build_techqa_documents

rows = load_dataset(
    retrieval["repo"],
    "corpus",
    split="train",
    revision=retrieval["revision"],
)
documents = build_techqa_documents(rows)
```

Then the CLI flow is strictly:

```text
load manifests
-> load/validate 450 local E0 TRAIN cases
-> load frozen corpus
-> build deterministic chunks
-> preflight exact identities/counts/version
-> time one TechQAChunkBM25Retriever construction as index_build_seconds
-> evaluate 450 cases sequentially
-> aggregate metrics
-> build diagnostics
-> write artifacts
```

The runner must instantiate `TechQAChunkBM25Retriever` exactly once and reuse `retriever.search` for all 450 cases.

- [ ] **Step 7: Implement compact artifact writer**

Write JSON with `ensure_ascii=False, indent=2` and JSONL with one `asdict(result)` per line. `train_results.jsonl` contains only compact fields already defined in `TechQAChunkBM25AuditResult`; it must not persist the full adaptive-depth candidate list beyond Top-100 identities.

`train_metrics.json` must include `index_build_seconds` separately from the per-query `latency_ms` p50/p95.

- [ ] **Step 8: Verify all focused tests before the real audit**

```powershell
pytest tests/test_bm25_techqa.py tests/test_bm25_techqa_chunks.py tests/test_eval_techqa_chunk_bm25.py -v
ruff check experiments/evals/retrievers/bm25_techqa_chunks.py experiments/evals/eval_techqa_chunk_bm25.py tests/test_bm25_techqa_chunks.py tests/test_eval_techqa_chunk_bm25.py
```

Expected: all focused tests PASS; Ruff `All checks passed!`.

- [ ] **Step 9: Run the zero-provider audit**

```powershell
python -m experiments.evals.eval_techqa_chunk_bm25
```

Hard stop on any preflight mismatch. The command must not request any DashScope/OpenAI credentials and must not call embedding, rerank, generation, Judge, or DEV evaluation code.

Expected artifacts:

```text
experiments/evals/reports/r4_chunk_bm25_audit/train_manifest.json
experiments/evals/reports/r4_chunk_bm25_audit/train_metrics.json
experiments/evals/reports/r4_chunk_bm25_audit/train_results.jsonl
experiments/evals/reports/r4_chunk_bm25_audit/diagnostic_cases.json
```

- [ ] **Step 10: Inspect evidence before making any diversity decision**

Check these fields only after the run:

```text
unique_document_count@20/50/100 p05/p50
duplicate_ratio@20/50/100 p50/p95
gold_document_hit_rate@20/50/100
crowding_rescue_count/rate@20/50/100
crowding_gap distribution
collapsed Document Recall@20/50/100
BM25 query latency p50/p95
index build time
```

Do **not** add a per-document cap in this task, regardless of observed results. Any diversity rule requires a separate design/pre-registration step.

- [ ] **Step 11: Full regression verification**

```powershell
pytest -q
ruff check .
```

Expected: full suite PASS and Ruff PASS.

- [ ] **Step 12: Commit implementation and compact evidence**

First inspect status explicitly:

```powershell
git status --short
```

Then stage exact intended files only. Do not use `git add .`.

```powershell
git add experiments/evals/retrievers/bm25_techqa_chunks.py tests/test_bm25_techqa_chunks.py experiments/evals/eval_techqa_chunk_bm25.py tests/test_eval_techqa_chunk_bm25.py experiments/evals/reports/r4_chunk_bm25_audit/train_manifest.json experiments/evals/reports/r4_chunk_bm25_audit/train_metrics.json experiments/evals/reports/r4_chunk_bm25_audit/train_results.jsonl experiments/evals/reports/r4_chunk_bm25_audit/diagnostic_cases.json
git commit -m "feat(eval): complete chunk BM25 candidate audit"
```

**Task 5 exit gate:** all 450 TRAIN cases are evaluated with one local chunk BM25 index; K and M remain separate; paired crowding evidence and document comparability metrics are materialized; full tests/Ruff pass; provider calls remain zero.

---

## Final Decision Boundary

Completion of this plan does **not** authorize paid R4 reranking.

After the audit, use the TRAIN evidence to choose one next design:

```text
A. raw chunk-BM25 candidates
or
B. a separately pre-registered chunk-diversity rule
```

Only after that candidate-construction contract, reranker candidate K, provider pricing/cost cap, paid stop condition, resume/manifest identity, and R4 admission threshold are frozen may `qwen3-rerank` be called again.
